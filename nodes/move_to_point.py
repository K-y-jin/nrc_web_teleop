#!/usr/bin/env python3

# Standard Imports
import sys
import threading
from tkinter import Y
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

    def __init__(
        self,
        tf_timeout_secs: float = 0.5,
        action_timeout_secs: float = 60.0,
    ):

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

        self.cv_bridge = CvBridge()

        # Subscribe to the Navigation camera's CompressedImage and camera info feed
        self.latest_navigation_camera_image = None
        self.latest_navigation_camera_image_lock = threading.Lock()
        self.navigation_camera_subscriber = self.create_subscription(
            CompressedImage,
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
        with self.latest_navigation_camera_image_lock:
            self.latest_navigation_camera_image = ros_image

    def navigation_info_cb(
        self,
        info_msg: CameraInfo,
    ) -> None:
        with self.latest_navigation_info_lock:
            self.latest_navigation_info = info_msg

    def goal_callback(self, goal_request: MoveToPoint.Goal) -> GoalResponse:
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
        
        # Functions to cleanup the action
        terminate_motion_executors = False
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []

        def cleanup():
            """Clean up resources after action completion"""
            with self.active_goal_request_lock:
                self.active_goal_request = None

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
        feedback.new_scaled_x = -1.0
        feedback.new_scaled_y = -1.0

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

        def get_keypoints_and_descriptors():
            with self.latest_navigation_camera_image_lock:
                navigation_camera_image = self.latest_navigation_camera_image
            navigation_camera_image = ros_msg_to_cv2_image(navigation_camera_image, self.cv_bridge)
            gray_image = cv2.cvtColor(navigation_camera_image, cv2.COLOR_RGB2GRAY)
            # 
            return orb.detectAndCompute(image=gray_image, mask=None)
        
        # Update the goal point
        def update_goal_point_from_yaw_error(yaw_error: float, head_pan: float):
            self.get_logger().info(f"##### Head Pan: {yaw_error}, {head_pan}")
            if yaw_error + head_pan > -np.pi/4 and yaw_error + head_pan < np.pi/4:
                raw_scaled_u = 0.5 * np.tan(yaw_error + head_pan) + 0.5
                self.get_logger().info(f"##### Updated Goal: [{raw_scaled_u}, {raw_scaled_v}]")
                # Update the feedback
                feedback.new_scaled_x = raw_scaled_u
                feedback.new_scaled_y = raw_scaled_v
                feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
                goal_handle.publish_feedback(feedback)
            else:
                self.get_logger().info(f"##### Fail Update: {yaw_error}")

        def update_goal_point() -> bool:
            # Convert into format for Homography transformation: [x, y, 1]
            goal_point = np.array([raw_scaled_u * image_width, raw_scaled_v * image_height, 1.0], dtype=np.float32)
            keypoints, descriptors = get_keypoints_and_descriptors()
            if keypoints is not None and descriptors is not None:           
                matches = bf.match(descriptors, prev_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # 매칭 품질 확인
                MAX_DISTANCE = 50.0  # ORB descriptor distance 임계값
                MIN_MATCHES = 10     # 최소 필요한 매칭 개수
                
                # 좋은 매칭만 필터링
                good_matches = [m for m in matches if m.distance < MAX_DISTANCE]
                
                self.get_logger().info(f"##### Total matches: {len(matches)}, Good matches: {len(good_matches)}")
                if len(good_matches) > 0:
                    avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
                    max_distance = max(m.distance for m in good_matches)
                    min_distance = min(m.distance for m in good_matches)
                    self.get_logger().info(f"##### Match quality - Avg: {avg_distance:.2f}, Min: {min_distance:.2f}, Max: {max_distance:.2f}")
                
                # 충분한 좋은 매칭이 없으면 실패
                if len(good_matches) < MIN_MATCHES:
                    self.get_logger().warn(f"##### Insufficient good matches: {len(good_matches)} < {MIN_MATCHES}")
                    return False
                
                # 좋은 매칭만 사용하여 keypoints 추출
                pts1 = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([prev_keypoints[m.trainIdx].pt for m in good_matches])

                H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)

                if H is not None:
                    # Homography 품질 확인
                    inlier_ratio = np.sum(inliers) / len(inliers) if len(inliers) > 0 else 0
                    self.get_logger().info(f"##### Homography quality - Inliers: {np.sum(inliers)}/{len(inliers)} ({inlier_ratio:.2%})")
                    
                    # 인라이어 비율이 너무 낮으면 신뢰할 수 없음
                    MIN_INLIER_RATIO = 0.3  # 30% 이상의 인라이어 필요
                    if inlier_ratio < MIN_INLIER_RATIO:
                        self.get_logger().warn(f"##### Low inlier ratio: {inlier_ratio:.2%} < {MIN_INLIER_RATIO:.2%}")
                        return False
                    # Apply homography transformation
                    current_goal_point = H @ goal_point
                    # Normalize homogeneous coordinates
                    current_goal_point = current_goal_point / current_goal_point[2]
                    # Extract x, y coordinates
                    current_x, current_y = current_goal_point[0], current_goal_point[1]
                    self.get_logger().info(f"##### Updated Goal Point: [{current_x}, {current_y}]")
                    # Update the feedback
                    feedback.new_scaled_x = current_x / image_width
                    feedback.new_scaled_y = current_y / image_height
                    feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
                    goal_handle.publish_feedback(feedback)
                    return True
            return False

        # Get the camera info
        with self.latest_navigation_info_lock:
            image_width = self.latest_navigation_info.width
            image_height = self.latest_navigation_info.height
            # k_matrix = self.latest_navigation_info.k
            # self.get_logger().debug(
            #     f"##### Image dimensions (WxH): {(image_width, image_height)}"
            # )
            # not in valid use
            # self.get_logger().debug(
            #     f"##### K Matrix: {k_matrix}"
            # )


        # Get keypoints and descriptors from the initial image
        prev_keypoints, prev_descriptors = get_keypoints_and_descriptors()

        # Get the initial goal point
        raw_scaled_u, raw_scaled_v = (
            goal_handle.request.scaled_u,
            goal_handle.request.scaled_v,
        )
            
        self.get_logger().debug(f"##### Initial Goal Point: {goal_point}")
        # Execute the states
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []
        states = MoveToPointState.get_state_machine(setup_mode=True)
        self.get_logger().info(f"All States: {states}")

        state_i = 0
        rate = self.create_rate(5.0)  # 5 Hz
        ik_solution = self.controller.get_current_joints()
        pan_theta = np.arctan2(0.5, raw_scaled_u - 0.5) - np.pi/2 # HFOV 90deg
        ik_solution[Joint.BASE_ROTATION] = ik_solution[Joint.HEAD_PAN] + pan_theta
        ik_solution[Joint.HEAD_PAN] = 0.0
        self.get_logger().info(f"##### pan_theta: {pan_theta}")
        
        while rclpy.ok():
            concurrent_states = states[state_i]
            self.get_logger().info(
                f"Executing States: {concurrent_states}", throttle_duration_sec=1.0
            )
            # Check if a cancel has been requested   
            if goal_handle.is_cancel_requested:
                return action_cancel_callback("Goal canceled")
            # Check if the action has timed out
            if (self.get_clock().now() - start_time) > self.action_timeout:
                return action_error_callback("Goal timed out", MoveToPoint.Result.STATUS_TIMEOUT)

            # Move the robot
            if len(motion_executors) == 0:
                for state in concurrent_states:
                    motion_executor = state.get_motion_executor(
                        controller=self.controller,
                        ik_solution=ik_solution,
                        timeout_secs=remaining_time(
                            self.get_clock().now(),
                            start_time,
                            self.action_timeout,
                            return_secs=True,
                        ),
                        check_cancel=lambda: terminate_motion_executors,
                        err_callback=update_goal_point_from_yaw_error,
                        success_callback=None,
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
                                f"##### Success (State: {state_i})"
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

            # Sleep
            rate.sleep()
        
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
