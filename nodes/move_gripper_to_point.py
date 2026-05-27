#!/usr/bin/env python3

# Standard Imports
import threading
from tkinter import Y
import traceback
from typing import Union, Generator, List, Optional, Tuple
import cv2
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Header

import rclpy

# Third-Party Imports
import stretch_urdf.urdf_utils as uu
import tf2_ros

from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2

# Local Imports
from nrc_web_teleop.action import MoveGripperToPoint
from nrc_web_teleop_helpers.constants import (
    Joint,
    Frame,
    DEFAULT_ARM_LENGTH,
)
from nrc_web_teleop_helpers.functions import compute_click_point_in_base
from nrc_web_teleop_helpers.conversions import (
    deproject_pixel_to_pointcloud_point,
    depth_img_to_pointcloud,
    remaining_time,
    ros_msg_to_cv2_image,
    tf2_transform,
    tf_lookup_matrix,
)
from nrc_web_teleop_helpers.move_to_action_state import MoveToActionState
from nrc_web_teleop_helpers.stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
)
from nrc_web_teleop_helpers.ggcnn import utils
from nrc_web_teleop_helpers.ggcnn.ggcnn_torch import predict
from nrc_web_teleop_helpers.da.utils import get_pred_depth

class MoveGripperToPointNode(Node):

    def __init__(
        self,
        tf_timeout_secs: float = 0.5,
        action_timeout_secs: float = 60.0,
    ):

        super().__init__("move_gripper_to_point")

        # Initialize TF2
        self.tf_timeout = Duration(seconds=tf_timeout_secs)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.lift_offset: Optional[Tuple[float, float]] = None
        self.wrist_offset: Optional[Tuple[float, float]] = None
        self.camera_offset: Optional[Tuple[float, float]] = None
        self.pan_offset: Optional[Tuple[float, float]] = None
        self.cam_height = None

        # Create the inverse jacobian controller to execute motions
        urdf_fpaths = uu.generate_ik_urdfs("nrc_web_teleop", rigid_wrist_urdf=False)
        urdf_fpath = urdf_fpaths[0]
        self.controller = StretchIKControl(
            self,
            tf_buffer=self.tf_buffer,
            urdf_path=urdf_fpath,
            static_transform_broadcaster=self.static_transform_broadcaster,
        )

        self.cv_bridge = CvBridge()

        # Subscribe to the Realsense camera's CompressedImage and camera info feed
        self.latest_realsense_rgb_lock = threading.Lock()
        self.latest_realsense_rgb: Optional[CompressedImage] = None
        self.camera_rgb_subscriber = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.realsense_rgb_cb,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.latest_realsense_depth_lock = threading.Lock()
        self.latest_realsense_depth: Optional[CompressedImage] = None
        self.depth_image_subscriber = self.create_subscription(
            CompressedImage,
            "/camera/aligned_depth_to_color/image_raw/compressedDepth",
            self.realsense_depth_cb,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        
        self.latest_realsense_info_lock = threading.Lock()
        self.latest_realsense_info: Optional[CameraInfo] = None
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/camera/aligned_depth_to_color/camera_info",
            self.realsense_info_cb,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Subscribe to the Navigation camera's CompressedImage and camera info feed
        self.latest_navigation_camera_image_lock = threading.Lock()
        self.latest_navigation_camera_image: Optional[CompressedImage] = None
        self.navigation_camera_subscriber = self.create_subscription(
            CompressedImage,
            "/navigation_camera/image_raw/rotated/compressed",
            self.navigation_camera_cb,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        
        # Subscribe to the gripper camera's CompressedImage and depth feed
        # Gripper Camera doesn't pulish aligned_depth_to_color
        self.latest_gripper_camera_rgb_image_lock = threading.Lock()
        self.latest_gripper_realsense_depth_image: Optional[CompressedImage] = None
        self.gripper_camera_rgb_subscriber = self.create_subscription(
            CompressedImage,
            "/gripper_camera/image_raw/compressed",
            self.gripper_realsense_rgb_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.latest_gripper_realsense_depth_image_lock = threading.Lock()
        self.latest_gripper_realsense_depth_image: Optional[CompressedImage] = None
        # self.latest_gripper_realsense_depth_image: Optional[Image] = None
        # self.gripper_depth_subscriber = self.create_subscription(
        #     Image,
        #     "/gripper_camera/depth/image_rect_raw",
        #     self.gripper_realsense_depth_cb,
        #     QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        # )
        self.gripper_depth_subscriber = self.create_subscription(
            CompressedImage,
            "/gripper_camera/depth/image_rect_raw/compressedDepth",
            self.gripper_realsense_depth_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.latest_gripper_camera_info_lock = threading.Lock()
        self.latest_gripper_camera_info: Optional[CameraInfo] = None
        self.gripper_camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/gripper_camera/depth/camera_info",
            self.gripper_camera_info_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        
        # Create the action timeout
        self.action_timeout = Duration(seconds=action_timeout_secs)

    def initialize(self) -> bool:
        # Initialize the controller
        ok = self.controller.initialize()
        if not ok:
            self.get_logger().error(
                "Failed to initialize the inverse jacobian controller"
            )
            return False

        # Create the shared resource to ensure that the action server rejects all
        # new goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the action server
        self.action_server = ActionServer(
            self,
            MoveGripperToPoint,
            "move_gripper_to_point",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        # NOTE: `/get_distance` 서비스는 `move_base_to_point.py`가 제공한다.
        # 두 노드가 같은 이름으로 등록할 수 없어 여기서는 제거됨. (DESIGN.md 참고)

        return True
    
    def realsense_rgb_cb(
        self,
        rgb_ros_image: Union[CompressedImage, Image],
    ):
        with self.latest_realsense_rgb_lock:
            self.latest_realsense_rgb = rgb_ros_image


    def realsense_depth_cb(
        self,
        depth_msg: CompressedImage,
    ) -> None:
        with self.latest_realsense_depth_lock:
            self.latest_realsense_depth = depth_msg

    def realsense_info_cb(
        self,
        info_msg: CameraInfo,
    ) -> None:
        with self.latest_realsense_info_lock:
            self.latest_realsense_info = info_msg

    def navigation_camera_cb(self, ros_image: CompressedImage) -> None:
        with self.latest_navigation_camera_image_lock:
            self.latest_navigation_camera_image = ros_image

    def gripper_realsense_depth_cb(
        self,
        depth_msg: Union[CompressedImage, Image, PointCloud2],
    ):
        with self.latest_gripper_realsense_depth_image_lock:
            self.latest_gripper_realsense_depth_image = depth_msg

    def gripper_realsense_rgb_cb(
        self,
        ros_image: Union[CompressedImage, Image],
    ):
        with self.latest_gripper_camera_rgb_image_lock:
            self.latest_gripper_camera_rgb_image = ros_image

    def gripper_camera_info_cb(        
        self,
        info_msg: CameraInfo,
    ) -> None:
        with self.latest_gripper_camera_info_lock:
            self.latest_gripper_camera_info = info_msg

    def goal_callback(self, goal_request: MoveGripperToPoint.Goal) -> GoalResponse:
        self.get_logger().info(f"Received request {goal_request}")

        # Reject the goal if no Navigation Camera RGB image has been received yet
        with self.latest_navigation_camera_image_lock:
            if self.latest_navigation_camera_image is None:
                self.get_logger().info(
                    "Rejecting goal request since no Navigation Camera RGB image message has been received yet"
                )
                return GoalResponse.REJECT

        # Reject the goal if no Realsense camera info has been received
        with self.latest_realsense_info_lock:
            if self.latest_realsense_info is None:
                self.get_logger().info(
                    "Rejecting goal request since no Realsense camera info message has been received yet"
                )
                return GoalResponse.REJECT

        # Reject the goal if no Realsense messages have been received yet
        with self.latest_realsense_depth_lock:
            if self.latest_realsense_depth is None:
                self.get_logger().info(
                    "Rejecting goal request since no Realsense depth message has been received yet"
                )
                return GoalResponse.REJECT
            
        # Reject the goal if no Realsense RGB messages have been received yet
        with self.latest_realsense_rgb_lock:
            if self.latest_realsense_rgb is None:
                self.get_logger().info(
                    "Rejecting goal request since no Realsense RGB message has been received yet"
                )
                return GoalResponse.REJECT
            
        # Reject the goal if no Gripper Realsense depth messages have been received yet
        with self.latest_gripper_realsense_depth_image_lock:
            if self.latest_gripper_realsense_depth_image is None:
                self.get_logger().info(
                    "Rejecting goal request since no Gripper Realsense depth message has been received yet"
                )
                return GoalResponse.REJECT
        
        # Reject the goal if no Gripper Realsense RGB messages have been received yet
        with self.latest_gripper_camera_rgb_image_lock:
            if self.latest_gripper_camera_rgb_image is None:
                self.get_logger().info(
                    "Rejecting goal request since no Gripper Realsense RGB message has been received yet"
                )
                return GoalResponse.REJECT
            
        # Reject the goal if no Gripper Realsense camera info has been received
        with self.latest_gripper_camera_info_lock:
            if self.latest_gripper_camera_info is None:
                self.get_logger().info(
                    "Rejecting goal request since no Gripper Realsense camera info message has been received yet"
                )
                return GoalResponse.REJECT

        # Reject the goal is there is already an active goal
        with self.active_goal_request_lock:
            if self.active_goal_request is not None:
                
                self.get_logger().info(
                    "Rejecting goal request since there is already an active one"
                )
                return GoalResponse.REJECT

        # Accept the goal
        self.get_logger().info("Accepting goal request")
        self.active_goal_request = goal_request
        return GoalResponse.ACCEPT

    def cancel_callback(self, _: ServerGoalHandle) -> CancelResponse:
        """
        Always accept client requests to cancel the active goal.

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> MoveGripperToPoint.Result:
    
        # Start the timer
        start_time = self.get_clock().now()

        # Initialize the feedback
        feedback = MoveGripperToPoint.Feedback()

        # Functions to cleanup the action
        terminate_motion_executors = False
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []

        def cleanup() -> None:
            """
            Clean up before returning from the action.
            """
            nonlocal terminate_motion_executors, motion_executors
            self.active_goal_request = None
            self.get_logger().debug("Setting termination flag to True")
            terminate_motion_executors = True
            # Execute the motion executors once more to process cancellation.
            if len(motion_executors) > 0:
                try:
                    for i, motion_executor in enumerate(motion_executors):
                        _ = next(motion_executor)
                except Exception:
                    self.get_logger().debug(traceback.format_exc())

        def action_error_callback(
            error_msg: str = "Goal failed",
            status: int = MoveGripperToPoint.Result.STATUS_FAILURE,
        ) -> MoveGripperToPoint.Result:
            self.get_logger().error(error_msg)
            goal_handle.abort()
            cleanup()
            return MoveGripperToPoint.Result(status=status)

        def action_success_callback(
            success_msg: str = "Goal succeeded",
        ) -> MoveGripperToPoint.Result:
            self.get_logger().info(success_msg)
            goal_handle.succeed()
            cleanup()
            return MoveGripperToPoint.Result(status=MoveGripperToPoint.Result.STATUS_SUCCESS)

        def action_cancel_callback(
            cancel_msg: str = "Goal canceled",
        ) -> MoveGripperToPoint.Result:
            self.get_logger().info(cancel_msg)
            goal_handle.canceled()
            cleanup()
            return MoveGripperToPoint.Result(status=MoveGripperToPoint.Result.STATUS_CANCELLED)

        goal_point = None
        # Get the initial goal point
        raw_scaled_u, raw_scaled_v = (
            goal_handle.request.scaled_u,
            goal_handle.request.scaled_v,
        )
        goal_point = np.array([raw_scaled_u, raw_scaled_v])
        self.get_logger().debug(f"##### Initial Goal Point: {goal_point}")

        with self.latest_navigation_camera_image_lock:
            image_msg = self.latest_navigation_camera_image
                
        goal_pose = PoseStamped()
        # goal_pose.header.stamp = self.get_clock().now().to_msg()
        # goal_pose.header.frame_id = "base_link"
        # goal_pose.header = image_msg.header

        # Publich_feedback message
        def publish_update_goal_point_feedback():
            self.get_logger().info(f"##### Updated Goal Point: [{feedback.new_scaled_u}, {feedback.new_scaled_v}]")
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            goal_handle.publish_feedback(feedback)
            self.save_images(save_gripper=True)

        # Execute the states
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []
        states = MoveToActionState.get_states_for_move_gripper_to_point()
        self.get_logger().info(f"All States: {states}")

        state_i = 0
        rate = self.create_rate(5.0)  # 5 Hz

        # First, get the offset between the depth and color camera frames
        if self.camera_offset is None:
            T = self.controller.get_transform(
                parent_link=Frame.CAMERA_COLOR_FRAME, child_link=Frame.CAMERA_DEPTH_FRAME
            )
            self.camera_offset = (T[0, 3], T[1, 3])
            self.get_logger().debug(f"Camera offset (x, y): {self.camera_offset}")

        if self.pan_offset is None:
            T = self.controller.get_transform(
                parent_link=Frame.BASE_LINK, child_link=Frame.HEAD_PAN_LINK
            )
            self.pan_offset = (T[0, 3], T[1, 3])
            self.get_logger().debug(f"Pan offset (x, y): {self.pan_offset}")

        if self.cam_height is None:
            T = self.controller.get_transform(
                parent_link=Frame.BASE_LINK, child_link=Frame.CAMERA_COLOR_FRAME
            )
            self.cam_height = T[2, 3]
            self.get_logger().info(f"cam_height: {self.cam_height}")

        # Calculate the goal positions
        initial_head_joint_states = self.controller.get_head_joint_states()
        initial_head_pan = initial_head_joint_states[Joint.HEAD_PAN]
        initial_head_tilt = initial_head_joint_states[Joint.HEAD_TILT]

        target_theta = -1.0 * np.arctan2(raw_scaled_u - 0.5, 0.5) # HFOV 90deg
        beta = np.pi * (127.0/180.0) # VFOV 127deg
        focal_length = 0.5 / np.tan(beta/2.0)
        alpha = -1.0 * np.arctan2(raw_scaled_v - 0.5, focal_length)  # tan(alpha) = (y-0.5) / focal_length
        tilt_theta = initial_head_tilt  + alpha
        
        feedback.new_scaled_u = 0.5
        feedback.new_scaled_v = focal_length * np.tan(tilt_theta - (initial_head_tilt+alpha)) + 0.5
        
        # Head와 ARM을 클릭 좌표 방향으로 향하도록 Base rotation과 head pan 설정
        goal_positions = {}
        if initial_head_pan + target_theta < np.pi/2:
            goal_positions[Joint.BASE_ROTATION] = initial_head_pan + target_theta + np.pi/2
        else:
            goal_positions[Joint.BASE_ROTATION] = initial_head_pan + target_theta - 3*np.pi/2
        goal_positions[Joint.HEAD_PAN] = -np.pi/2 # 그리퍼 방향
        goal_positions[Joint.HEAD_TILT] = tilt_theta

        goal_positions[Joint.ARM_LIFT] = None
        goal_positions[Joint.ARM_L0] = None

        # Base translation: 클릭 픽셀을 TF로 (T0) base_link 의 3D 점(m)으로
        # 환산해, 팔이 닿는 위치(= 클릭점에서 DEFAULT_ARM_LENGTH 만큼 base
        # 쪽) 를 base_origin 의 목표로 설정. odom 으로 변환해 두면
        # translate_base_to_goal_pose 가 world 기준으로 추종한다 (pregrasp 패턴).
        try:
            nav_image_cv = ros_msg_to_cv2_image(image_msg, self.cv_bridge)
            # IK URDF는 head joints가 fixed라 controller.get_transform 으로는
            # 실 head_tilt/head_pan 반영이 안 된다. 실 TF 트리 조회 사용.
            # stamp = 이미지 헤더 시점 (캡처 순간의 head 자세와 동기화).
            T_base_from_nav_link = tf_lookup_matrix(
                self.tf_buffer, Frame.BASE_LINK.value,
                Frame.NAV_CAM_LINK.value, self.tf_timeout,
                stamp=image_msg.header.stamp,
            )
            if T_base_from_nav_link is None:
                raise RuntimeError(
                    "Failed to lookup base_link <- link_head_nav_cam TF"
                )
            result = compute_click_point_in_base(
                nav_image_cv, raw_scaled_u, raw_scaled_v, T_base_from_nav_link,
            )
            click_xyz = None if result is None else result[0]
        except Exception:
            self.get_logger().error(traceback.format_exc())
            click_xyz = None
        if click_xyz is None:
            self.get_logger().warning(
                "Failed to compute click point; skipping base translation"
            )
            # goal_pose 를 현재 base_link 원점으로 두면 controller 가 즉시 종료.
            goal_pose.header.frame_id = Frame.BASE_LINK.value
            goal_pose.header.stamp = image_msg.header.stamp
            goal_pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
            goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        else:
            cx, cy = float(click_xyz[0]), float(click_xyz[1])
            d = float(np.hypot(cx, cy))  # m
            if d == 0.0:
                self.get_logger().warning("Click point at base origin")
                arm_frac = 0.0
            else:
                arm_frac = (d - DEFAULT_ARM_LENGTH) / d
            goal_base_x = cx * arm_frac
            goal_base_y = cy * arm_frac
            goal_pose.header.frame_id = Frame.BASE_LINK.value
            goal_pose.header.stamp = image_msg.header.stamp
            goal_pose.pose.position = Point(x=goal_base_x, y=goal_base_y, z=0.0)
            goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self.get_logger().info(
                f"##### gripper goal in T0 base: x={goal_base_x:.3f} m, "
                f"y={goal_base_y:.3f} m (click d={d:.3f} m)"
            )
        # T0 base_link → odom (world 고정 frame) 변환. 이후 MOVE_BASE state 가
        # 매 iteration 현재 base_link 로 다시 변환해 err 를 계산한다.
        ok, goal_pose_odom = tf2_transform(
            self.tf_buffer, goal_pose, Frame.ODOM.value, self.tf_timeout,
        )
        if ok:
            goal_pose.header = goal_pose_odom.header
            goal_pose.pose = goal_pose_odom.pose
        else:
            self.get_logger().warning(
                "Failed to transform goal_pose to odom; using base_link frame"
            )

        def update_feedback_and_publish_feedback(distance_error: float):
            nonlocal tilt_theta, beta, focal_length
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            alpha = np.arctan2(distance_error, self.cam_height) - beta/2
            feedback.new_scaled_v = 0.5 - focal_length * np.tan(alpha)
            feedback.new_scaled_u = 0.5
            # self.get_logger().info(f"##### Feedback: {distance_error}")
            goal_handle.publish_feedback(feedback)

        def get_depth_of_center_pixel() -> Optional[float]:
            with self.latest_realsense_depth_lock:
                depth_msg = self.latest_realsense_depth
            if depth_msg is None:
                self.get_logger().error("No depth message received yet")
                return 0.0
            depth_image = ros_msg_to_cv2_image(depth_msg, self.cv_bridge)
            center_depth = depth_image[depth_image.shape[0]//2, depth_image.shape[1]//2]
            center_depth = center_depth / 1000.0 # Convert from mm to m
            self.get_logger().info(f"Depth of center pixel: {center_depth} m")
            
            return center_depth
        
        def get_height_of_center_pixel(theta_tilt = 0.0) -> Optional[float]:
            with self.latest_realsense_depth_lock:
                depth_msg = self.latest_realsense_depth
            if depth_msg is None:
                self.get_logger().error("No depth message received yet")
                return 0.0
            depth_image = ros_msg_to_cv2_image(depth_msg, self.cv_bridge)
            center_depth = depth_image[depth_image.shape[0]//2, depth_image.shape[1]//2]
            center_depth = center_depth / 1000.0 # Convert from mm to m
            center_height = self.cam_height + center_depth* np.tan(tilt_theta)
            self.get_logger().info(f"Center Depth: {center_depth} m, Height of center pixel: {center_height} m")

            return center_height

        # Loop
        while rclpy.ok():
            concurrent_states = states[state_i]
            # self.get_logger().info(
            #     f"Executing States: {concurrent_states}", throttle_duration_sec=1.0
            # )
            # Check if a cancel has been requested   
            if goal_handle.is_cancel_requested:
                return action_cancel_callback("Goal canceled")
            # Check if the action has timed out
            if (self.get_clock().now() - start_time) > self.action_timeout:
                return action_error_callback("Goal timed out", MoveGripperToPoint.Result.STATUS_TIMEOUT)

            # Move the robot
            if len(motion_executors) == 0:
                for state in concurrent_states:
                    motion_executor = state.get_motion_executor(
                        controller=self.controller,
                        goal_pose=goal_pose,
                        ik_solution=goal_positions,
                        timeout_secs=remaining_time(
                            self.get_clock().now(),
                            start_time,
                            self.action_timeout,
                            return_secs=True,
                        ),
                        check_cancel=lambda: terminate_motion_executors,
                        err_callback=[update_feedback_and_publish_feedback, get_height_of_center_pixel, get_depth_of_center_pixel],
                        success_callback=[publish_update_goal_point_feedback],
                    )
                    if motion_executor is None:
                        return action_success_callback("Goal succeeded")
                    motion_executors.append(motion_executor)
            # Check if the robot is done moving
            else:
                try:
                    for i, motion_executor in enumerate(motion_executors):
                        retval = next(motion_executor)
                        if retval == MotionGeneratorRetval.SUCCESS:
                            motion_executors.pop(i)
                            self.get_logger().info(
                                f"##### Success (State Num {state_i}:{concurrent_states}"
                            )
                            break
                        elif retval == MotionGeneratorRetval.FAILURE:
                            raise Exception("Failed to move to goal pose")
                        else:  # CONTINUE
                            pass
                    if len(motion_executors) == 0:
                        state_i += 1
                except Exception as e:
                    self.get_logger().error(traceback.format_exc())
                    return action_error_callback(
                            f"Error executing the motion generator: {e}",
                        MoveGripperToPoint.Result.STATUS_FAILURE,
                    )

            # Sleep
            rate.sleep()
        
        # Failed to execute MoveGripperToPoint
        return action_error_callback("Failed to execute MoveGripperToPoint")

    def get_optimal_grasp(self):
        # Navigation Cam image
        # GR conv
        return

    def save_images(self, save_gripper: bool = False, grip_pred: bool = False):
        # D435 30 cm 이상부터 측정 가능. 최대 3 m
        with self.latest_realsense_rgb_lock:
            rgb_msg = self.latest_realsense_rgb
            rgb_image = utils.get_rgb_image_from_msg(rgb_msg, self.cv_bridge)
        with self.latest_realsense_depth_lock:
            depth_msg = self.latest_realsense_depth
            depth_image = utils.get_depth_image_from_msg(
                depth_msg, self.cv_bridge, measurement_max=3.0, measurement_min=0.3)
        # D405 7 cm 이상부터 측정 가능. 최대 50 cm
        with self.latest_gripper_camera_rgb_image_lock:
            gripper_rgb_msg = self.latest_gripper_camera_rgb_image
            gripper_rgb_image = utils.get_rgb_image_from_msg(gripper_rgb_msg, self.cv_bridge)
        with self.latest_gripper_realsense_depth_image_lock:
            gripper_depth_msg = self.latest_gripper_realsense_depth_image
            gripper_depth_image = utils.get_depth_image_from_msg(
                gripper_depth_msg, self.cv_bridge, measurement_max=0.5, measurement_min=0.07)
        
        if rgb_image is None or depth_image is None:
            self.get_logger().error("No images received yet")
            return
            
        utils.save_image(depth_image, filename_prefix="depth", sub_dir="head")
        utils.save_image(rgb_image, filename_prefix="color", sub_dir="head")
        
        if save_gripper:
            if gripper_rgb_image is None or gripper_depth_image is None:
                self.get_logger().error("No gripper images received yet")
                return
            utils.save_image(gripper_depth_image, "gripper_depth", sub_dir="gripper")
            utils.save_image(gripper_rgb_image, "gripper_rgb", sub_dir="gripper")
            gripper_pred_depth = get_pred_depth(gripper_rgb_image)
            self.save_ggcnn_results(gripper_pred_depth, sub_dir="gripper")
            utils.save_image(gripper_pred_depth, filename_prefix="pred_depth", sub_dir="gripper")
                    
        # if grip_pred:
        pred_depth = get_pred_depth(rgb_image)
        self.save_ggcnn_results(pred_depth, sub_dir="head")
        out_path = utils.save_image(pred_depth, filename_prefix="pred_depth", sub_dir="head")
        self.get_logger().info(f"##### Saved images in {out_path}")


    def save_ggcnn_results(self, depth_image, sub_dir: str = "head"):
        depth_nan_mask = depth_image == 0
        depth_image = (depth_image/255.0).astype(np.float32)
        q_out, ang_out, width_out, depth_out = predict(
            depth_image, process_depth=True, crop_size=None, out_size=300,
            depth_nan_mask=depth_nan_mask, crop_y_offset=0,
            filters= (False, False, False) # (2.0, 1.0, 1.0)
        )
        image_size = (depth_image.shape[1], depth_image.shape[0])
        q_out = (q_out - q_out.min()) / (q_out.max() - q_out.min())
        q_out = cv2.resize((255.0*np.clip(q_out, 0.0, 1.0)).astype(np.uint8), image_size, cv2.INTER_AREA)
        ang_out = (ang_out + np.pi/2)/np.pi
        ang_out = cv2.resize((255.0*np.clip(ang_out, 0.0, 1.0)).astype(np.uint8), image_size, cv2.INTER_AREA)
        width_out = (width_out - width_out.min()) / (width_out.max() - width_out.min())
        width_out = cv2.resize((255.0*np.clip(width_out, 0.0, 1.0)).astype(np.uint8), image_size, cv2.INTER_AREA)
        depth_out = cv2.resize((255.0*np.clip(depth_out, 0.0, 1.0)).astype(np.uint8), image_size, cv2.INTER_AREA)
        utils.save_image(q_out, "points_out", sub_dir=sub_dir)
        utils.save_image(ang_out, "ang_out", sub_dir=sub_dir)
        utils.save_image(width_out, "width_out", sub_dir=sub_dir)
        utils.save_image(depth_out, "processed_depth", sub_dir=sub_dir)

    def get_clicked_pixel(
                self, request: MoveGripperToPoint.Goal
    ) -> Optional[Tuple[float, float, float, Header]]:
        """
        Get the 3D coordinates of the clicked pixel in camera frame.

        Parameters
        ----------
        goal_handle: The goal handle.

        Returns
        -------
        Optional[Tuple[float, float, float, Header]]: The clicked pixel, and the header of
            the depth message, or None if the clicked pixel could not be deprojected.
        """
        # Get the latest Realsense messages
        with self.latest_realsense_depth_lock:
            depth_msg = self.latest_realsense_depth
        with self.latest_realsense_info_lock:
            camera_info_msg = self.latest_realsense_info
        depth_image = ros_msg_to_cv2_image(depth_msg, self.cv_bridge)
        pointcloud = depth_img_to_pointcloud(
            depth_image,
            f_x=camera_info_msg.k[0],
            f_y=camera_info_msg.k[4],
            c_x=camera_info_msg.k[2],
            c_y=camera_info_msg.k[5],
        )  # N x 3 array

        # Undo any transformation that were applied to the raw camera image before sending it
        # to the web app
        raw_scaled_u, raw_scaled_v = (
            request.scaled_u,
            request.scaled_v,
        )
        if (
            "realsense" in self.image_params
            and "default" in self.image_params["realsense"]
        ):
            params = self.image_params["realsense"]["default"]
        else:
            params = None
        u, v = self.inverse_transform_pixel(
            raw_scaled_u, raw_scaled_v, params, camera_info_msg
        )
        self.get_logger().debug(
            f"Clicked pixel after inverse transform (camera frame): {(u, v)}"
        )

        # Deproject the clicked pixel to get the 3D coordinates of the clicked point
        retval = deproject_pixel_to_pointcloud_point(
            u, v, pointcloud, np.array(camera_info_msg.p).reshape(3, 4)
        )
        if retval is None:
            self.get_logger().error("Failed to deproject clicked pixel")
            return None
        x, y, z = retval
        self.get_logger().debug(
            f"Closest point to clicked pixel (camera frame): {(x, y, z)}"
        )

        return x, y, z, depth_msg.header


def main(args: Optional[List[str]] = None):
    rclpy.init(args=args)

    move_to_point = MoveGripperToPointNode()
    move_to_point.get_logger().info("Created!")

    # Use a MultiThreadedExecutor so that subscriptions, actions, etc. can be
    # processed in parallel.
    executor = MultiThreadedExecutor()

    # Spin in the background, as the node initializes
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(move_to_point,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Initialize the node
    move_to_point.initialize()

    # Spin in the foreground
    spin_thread.join()

    move_to_point.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()