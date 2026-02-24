#!/usr/bin/env python3

# Standard Imports
import sys
import threading
import traceback
from typing import Callable, Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt
import rclpy

# Third-Party Imports
import stretch_urdf.urdf_utils as uu
import tf2_ros
import yaml

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Quaternion, Transform, TransformStamped, Vector3
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header
from tf2_geometry_msgs import PoseStamped
from tf_transformations import quaternion_about_axis, quaternion_multiply

# Local Imports
from nrc_web_teleop.action import MoveToPoint
from stretch_web_teleop_helpers.constants import (
    Frame,
    Joint,
    adjust_arm_lift_for_base_collision,
    get_pregrasp_wrist_configuration,
    get_stow_configuration,
)
from stretch_web_teleop_helpers.conversions import (
    deproject_pixel_to_pointcloud_point,
    depth_img_to_pointcloud,
    remaining_time,
    ros_msg_to_cv2_image,
    tf2_transform,
)
from stretch_web_teleop_helpers.move_to_point_state import MoveToPointState
from stretch_web_teleop_helpers.stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
)


class MoveToPointNode(Node):
    """
    The MoveToPoint node exposes an action server that takes in the
    (x, y) pixel coordinates of an operator's click on the Wide-Angle
    camera feed. It then moves the robot so its base is
    located on the clicked pixel.
    """

    def __init__(
        self,
        tf_timeout_secs: float = 0.5,
        action_timeout_secs: float = 60.0,
    ):
        """
        Initialize the MoveToPointNode

        Parameters
        ----------
        tf_timeout_secs: The timeout in seconds for TF lookups.
        action_timeout_secs: The timeout in seconds for the action server.
        """
        super().__init__("move_to_point")

        # Initialize TF2
        self.tf_timeout = Duration(seconds=tf_timeout_secs)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.lift_offset: Optional[Tuple[float, float]] = None
        self.wrist_offset: Optional[Tuple[float, float]] = None

        # Create the inverse jacobian controller to execute motions
        urdf_fpaths = uu.generate_ik_urdfs("nrc_web_teleop", rigid_wrist_urdf=False)
        urdf_fpath = urdf_fpaths[0]
        self.controller = StretchIKControl(
            self,
            tf_buffer=self.tf_buffer,
            urdf_path=urdf_fpath,
            static_transform_broadcaster=self.static_transform_broadcaster,
        )
        # self.get_logger().info(
        #     "### MoveToPointNode initialized ###"
        # )

        self.cv_bridge = CvBridge()
        # Subscribe to the Navigation camera's CompressedImage and camera info feed
        self.latest_navigation_camera_image = None
        self.latest_navigation_camera_image_lock = threading.Lock()
        self.navigation_camera_subscriber = self.create_subscription(
            CompressedImage,  # ConfigureVideoStreamsNode publishes usb cam's CompressedImage
            "/navigation_camera/image_raw/rotated/compressed",
            self.navigation_camera_cb,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.latest_navigation_info_lock = threading.Lock()
        self.latest_navigation_info: Optional[CameraInfo] = None
        self.navigation_camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/navigation_camera/camera_info",
            self.navigation_info_cb,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Create the action timeout
        self.action_timeout = Duration(seconds=action_timeout_secs)

    def initialize(self) -> bool:
        """
        Initialize the MoveToPointNode.

        This is necessary because ROS must be spinning while the controller
        is being initialized.

        Returns
        -------
        bool: Whether the initialization was successful.
        """
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
            MoveToPoint,
            "move_to_point",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        return True

    def navigation_camera_cb(self, ros_image: CompressedImage) -> None:
        """
        Callback for the Navigation camera feed subscriber. 
        Save the latest Navigation Camera RGB image.
        """
        with self.latest_navigation_camera_image_lock:
            self.latest_navigation_camera_image = ros_image

    def navigation_info_cb(
        self,
        info_msg: CameraInfo,
    ) -> None:
        """
        Callback for the Navigation camera info subscriber. Save the latest camera info.

        Parameters
        ----------
        info_msg: The camera info message
        """
        with self.latest_navigation_info_lock:
            self.latest_navigation_info = info_msg

    def goal_callback(self, goal_request: MoveToPoint.Goal) -> GoalResponse:
        """
        Accept a goal if this action does not already have an active goal, else reject.

        Parameters
        ----------
        goal_request: The goal request message.
        """
        self.get_logger().info(f"Received request {goal_request}")

        # Reject the goal if no Navigation Camera RGB image has been received yet
        with self.latest_navigation_camera_image_lock:
            if self.latest_navigation_camera_image is None:
                self.get_logger().info(
                    "Rejecting goal request since no Navigation Camera RGB image message has been received yet"
                )
                return GoalResponse.REJECT

        # Reject the goal is there is already an active goal
        # with self.active_goal_request_lock:
        #     if self.active_goal_request is not None:
                
        #         self.get_logger().info(
        #             "Rejecting goal request since there is already an active one"
        #         )
        #         return GoalResponse.REJECT

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
    ) -> MoveToPoint.Result:

        def cleanup():
            """Clean up resources after action completion"""
            with self.active_goal_request_lock:
                self.active_goal_request = None

        def action_error_callback(
            error_msg: str = "Goal failed",
            status: int = MoveToPoint.Result.STATUS_FAILURE,
        ) -> MoveToPoint.Result:
            self.get_logger().error(error_msg)
            goal_handle.abort()
            cleanup()
            return MoveToPoint.Result(status=status)

        def action_success_callback(
            success_msg: str = "Goal succeeded",
        ) -> MoveToPoint.Result:
            self.get_logger().info(success_msg)
            goal_handle.succeed()
            cleanup()
            return MoveToPoint.Result(status=MoveToPoint.Result.STATUS_SUCCESS)

        def action_cancel_callback(
            cancel_msg: str = "Goal canceled",
        ) -> MoveToPoint.Result:
            self.get_logger().info(cancel_msg)
            goal_handle.canceled()
            cleanup()
            return MoveToPoint.Result(status=MoveToPoint.Result.STATUS_CANCELLED)

        # Start the timer
        start_time = self.get_clock().now()

        # Initialize the feedback
        feedback = MoveToPoint.Feedback()
        feedback.new_scaled_u = -1.0
        feedback.new_scaled_v = -1.0

        # Initialize ORB + Opticalflow tracker
        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        prev_gray = None
        prev_keypoints = None
        prev_descriptors = None
        frame_count = 0
        reinit_interval = 10
        goal_point = None
        tracking_failure_count = 0
        max_failure_count = 5
        image_width = self.latest_navigation_info.width
        image_height = self.latest_navigation_info.height
        self.get_logger().debug(
            f"##### Image dimensions (WxH): {(image_width, image_height)}"
        )

        def process_navigation_camera_image():
            navigation_camera_image = self.latest_navigation_camera_image
            navigation_camera_image = ros_msg_to_cv2_image(navigation_camera_image, self.cv_bridge)
            gray_image = cv2.cvtColor(navigation_camera_image, cv2.COLOR_RGB2GRAY)
            return gray_image
        
        # Get the initial featrues of the goal point
        with self.latest_navigation_camera_image_lock:
            gray_image = process_navigation_camera_image()
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)
            prev_gray = gray_image
            prev_keypoints = keypoints
            prev_descriptors = descriptors
            raw_scaled_u, raw_scaled_v = (
                goal_handle.request.scaled_u,
                goal_handle.request.scaled_v,
            )
            feedback.new_scaled_u = raw_scaled_u
            feedback.new_scaled_v = raw_scaled_v

            goal_point = np.array([raw_scaled_u * image_width, raw_scaled_v * image_height], dtype=np.float32).reshape(1, 1, 2)

        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []
        states: List[List[MoveToPointState]] = []
        states.append([MoveToPointState.HEAD_PAN])
        states.append([MoveToPointState.TERMINAL])
        # self.get_logger().info(f"All States: {states}")
        state_i = 0
        concurrent_states = states[state_i]
        for state in concurrent_states:
            motion_executor = state.get_motion_executor(
                        controller=self.controller,
                        timeout_secs=remaining_time(
                            self.get_clock().now(),
                            start_time,
                            self.action_timeout,
                            return_secs=True,
                        ),
                    )
        motion_executors.append(motion_executor)
        while rclpy.ok():
            # Check if a cancel has been requested   
            if goal_handle.is_cancel_requested:
                return action_cancel_callback("Goal canceled")
            # Check if the action has timed out
            if (self.get_clock().now() - start_time) > self.action_timeout:
                return action_error_callback("Goal timed out", MoveToPoint.Result.STATUS_TIMEOUT)

            with self.latest_navigation_camera_image_lock:
                gray_image = process_navigation_camera_image()
                frame_count += 1

                try:
                    for i, motion_executor in enumerate(motion_executors):
                        retval = next(motion_executor)
                        if retval == MotionGeneratorRetval.SUCCESS:
                            motion_executors.pop(i)
                            self.get_logger().info(
                                f"##### New point (State {state_i}): {goal_point}"
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
                        MoveToPoint.Result.STATUS_FAILURE,
                    )

                # Optical Flow (LK tracker)
                new_point, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_image, goal_point, None, winSize=(15, 15), maxLevel=2)

                if st[0][0] == 1 and err[0][0] < 50.0:
                    # Optical flow succeeded, update the goal point
                    goal_point = new_point
                    feedback.new_scaled_u = goal_point[0][0][0] / image_width
                    feedback.new_scaled_v = goal_point[0][0][1] / image_height
                else:
                    # Optical flow failed, ORB matching fallback
                    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
                    if prev_descriptors is not None and descriptors is not None and len(descriptors) > 0:
                        matches = bf.match(prev_descriptors, descriptors)
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        # 적응적 검색 반경으로 매치 찾기
                        valid_matches = []
                        search_radii = [50.0, 100.0, 200.0, 300.0]
                        
                        for radius in search_radii:
                            valid_matches = []
                            for m in matches:
                                pt = keypoints[m.trainIdx].pt
                                distance = np.linalg.norm(np.array(pt) - goal_point[0][0])
                                if distance < radius:
                                    valid_matches.append((m, distance))
                            
                            if valid_matches:
                                self.get_logger().info(f"Found {len(valid_matches)} matches within radius {radius}")
                                break

                        if valid_matches:
                            # descriptor 거리가 가장 작은 매치 선택 (이미 정렬됨)
                            best_match = valid_matches[0][0]
                            pt = keypoints[best_match.trainIdx].pt
                            goal_point = np.array(pt, dtype=np.float32).reshape(1, 1, 2)
                            feedback.new_scaled_u = goal_point[0][0][0] / image_width
                            feedback.new_scaled_v = goal_point[0][0][1] / image_height
                            tracking_failure_count = 0  # 성공 시 카운터 리셋
                            # self.get_logger().info(
                            #     f"##### New point (ORB matching fallback): {goal_point}"
                            # )
                        else:
                            # ORB 매칭 실패 시 템플릿 매칭 시도
                            tracking_failure_count += 1
                            self.get_logger().warn(f"ORB matching failed {tracking_failure_count}/{max_failure_count}")
                            
                            # 템플릿 매칭 백업
                            template_size = 50
                            x, y = int(goal_point[0][0][0]), int(goal_point[0][0][1])
                            
                            # 경계 확인 후 템플릿 추출
                            if (x >= template_size//2 and x < image_width - template_size//2 and
                                y >= template_size//2 and y < image_height - template_size//2):
                                
                                template = prev_gray[y-template_size//2:y+template_size//2,
                                                   x-template_size//2:x+template_size//2]
                                
                                if template.size > 0:
                                    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                                    
                                    if max_val > 0.6:  # 템플릿 매칭 임계값
                                        new_x = max_loc[0] + template_size // 2
                                        new_y = max_loc[1] + template_size // 2
                                        goal_point = np.array([new_x, new_y], dtype=np.float32).reshape(1, 1, 2)
                                        # feedback.new_scaled_u = goal_point[0][0][0] / image_width
                                        # feedback.new_scaled_v = goal_point[0][0][1] / image_height
                                        tracking_failure_count = 0  # 성공 시 카운터 리셋
                                        self.get_logger().info(
                                            f"##### New point (Template matching): {goal_point}, confidence: {max_val:.3f}"
                                        )
                                    else:
                                        self.get_logger().warn(f"Template matching confidence too low: {max_val:.3f}")
                            
                            # 연속 실패 시에만 에러 반환
                            if tracking_failure_count >= max_failure_count:
                                return action_error_callback("Failed to track goal point after multiple attempts", MoveToPoint.Result.STATUS_FAILURE)
                            else:
                                self.get_logger().info("Maintaining last known goal position")
                                # goal_point는 그대로 유지
                    
                    # 프레임 업데이트
                    prev_gray = gray_image
                    prev_keypoints = keypoints
                    prev_descriptors = descriptors

            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            goal_handle.publish_feedback(feedback)

        # Failed to execute MoveToPoint
        return action_error_callback("Failed to execute MoveToPoint")



def main(args: Optional[List[str]] = None):
    rclpy.init(args=args)

    move_to_point = MoveToPointNode()
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
