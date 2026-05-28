#!/usr/bin/env python3

# Standard Imports
import os
import sys
import threading
import time
import traceback
from typing import Union, Generator, List, Optional, Tuple
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

import rclpy
import torch

# GR-ConvNet 모델 클래스는 grconv/ 내부 상대 경로 (inference.models.grconvnet3
# 등) 로 저장되어 있어 sys.path 에 grconv 디렉토리를 미리 추가해야 torch.load
# 가 성공한다.
import nrc_web_teleop_helpers as _helpers_pkg
_GRCONV_DIR = os.path.abspath(
    os.path.join(os.path.dirname(_helpers_pkg.__file__), "grconv")
)
if _GRCONV_DIR not in sys.path:
    sys.path.insert(0, _GRCONV_DIR)
from inference.post_process import post_process_output  # noqa: E402

GRCONV_PATH = os.path.join(
    _GRCONV_DIR,
    "trained-models",
    "jacquard-rgbd-grconvnet3-drop0-ch32",
    "epoch_42_iou_0.93",
)
GRCONV_OUTPUT_DIR = "output/grconv"

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
    OPTIMAL_DISTANCE,
    BASE_DISTANCE_TOLERANCE,
    BASE_MARGIN,
)
from nrc_web_teleop_helpers.functions import (
    compute_click_point_in_base,
    project_base_point_to_nav_pixel,
)
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

        # GR-ConvNet (grasp 예측) 모델 로드. __init__ 단계에서 한 번 로드해
        # 매 액션 실행 시 PRED_GRASP 콜백이 즉시 사용 가능하도록 한다.
        self.grconv_device: Optional[torch.device] = None
        self.grconv_model: Optional[torch.nn.Module] = None
        self._load_grconv_model()

    def _load_grconv_model(self) -> None:
        try:
            self.grconv_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.grconv_model = torch.load(
                GRCONV_PATH,
                map_location=self.grconv_device,
                weights_only=False,
            )
            self.grconv_model.eval()
            in_ch = self.grconv_model.conv1.in_channels
            self.get_logger().info(
                f"GR-ConvNet loaded from {GRCONV_PATH} on "
                f"{self.grconv_device} (input_channels={in_ch})"
            )
        except Exception:
            self.get_logger().error(
                f"Failed to load GR-ConvNet from {GRCONV_PATH}:\n"
                f"{traceback.format_exc()}"
            )
            self.grconv_model = None

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
        """
        액션 실행. 재설계 흐름:
            PRED_READY (depth pred 자세) → PRED_GOAL (클릭→goal_xyz_in_odom 계산)
            → ROTATE_BASE+HEAD_PAN → TRANSLATE_BASE → GRASP_READY+HEAD_PAN
            → TERMINAL.

        OPTIMAL_DISTANCE 와 BASE_DISTANCE_TOLERANCE 로 base 재배치 필요성을
        판단; 필요 없으면 goal_pose 를 현재 base 위치로 두어 ROTATE/TRANSLATE
        가 즉시 종료한다.
        """
        start_time = self.get_clock().now()
        feedback = MoveGripperToPoint.Feedback()

        terminate_motion_executors = False
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []

        def cleanup() -> None:
            nonlocal terminate_motion_executors, motion_executors
            self.active_goal_request = None
            terminate_motion_executors = True
            if len(motion_executors) > 0:
                try:
                    for motion_executor in motion_executors:
                        _ = next(motion_executor)
                except Exception:
                    self.get_logger().debug(traceback.format_exc())

        # 액션 종료 시 모든 마커를 hide 한 final feedback 을 한 번 발행.
        # marker_state / publish_feedback 가 아직 정의되기 전에 호출될 수도
        # 있어 (예: 카메라 이미지 미수신으로 즉시 error 반환), 존재 여부 가드.
        def hide_markers_and_publish_final_feedback() -> None:
            try:
                marker_state["show_click"] = False
                marker_state["show_gripper_goal"] = False
                marker_state["show_base_goal"] = False
                publish_feedback()
            except (NameError, UnboundLocalError):
                # marker_state / publish_feedback 아직 미정의 — 무시.
                pass

        def action_error_callback(
            error_msg: str = "Goal failed",
            status: int = MoveGripperToPoint.Result.STATUS_FAILURE,
        ) -> MoveGripperToPoint.Result:
            self.get_logger().error(error_msg)
            hide_markers_and_publish_final_feedback()
            goal_handle.abort()
            cleanup()
            return MoveGripperToPoint.Result(status=status)

        def action_success_callback(
            success_msg: str = "Goal succeeded",
        ) -> MoveGripperToPoint.Result:
            self.get_logger().info(success_msg)
            hide_markers_and_publish_final_feedback()
            goal_handle.succeed()
            cleanup()
            return MoveGripperToPoint.Result(
                status=MoveGripperToPoint.Result.STATUS_SUCCESS
            )

        def action_cancel_callback(
            cancel_msg: str = "Goal canceled",
        ) -> MoveGripperToPoint.Result:
            self.get_logger().info(cancel_msg)
            hide_markers_and_publish_final_feedback()
            goal_handle.canceled()
            cleanup()
            return MoveGripperToPoint.Result(
                status=MoveGripperToPoint.Result.STATUS_CANCELLED
            )

        raw_scaled_u = goal_handle.request.scaled_u
        raw_scaled_v = goal_handle.request.scaled_v

        with self.latest_navigation_camera_image_lock:
            image_msg = self.latest_navigation_camera_image
        if image_msg is None:
            return action_error_callback("No navigation camera image received yet")

        # goal_pose 는 MoveToActionState.get_motion_executor signature 호환용
        # placeholder. MoveToActionState 의 모든 state 가 자체 PoseStamped 를
        # 만들어 쓰므로 본 노드에서는 채우지 않는다.
        goal_pose = PoseStamped()

        # goal_positions: BASE_ROTATION 과 BASE_TRANSLATION 은 PRED_GOAL 에서
        # 계산해 갱신. HEAD_PAN 은 ROTATE/TRANSLATE 단계에서 0 (정면),
        # GRASP_READY 단계에서 -π/2 (그리퍼 방향) 로 갱신.
        goal_positions = {
            Joint.BASE_ROTATION: 0.0,
            Joint.BASE_TRANSLATION: 0.0,
            Joint.HEAD_PAN: 0.0,
        }

        # 마커 상태(피드백으로 UI 전달).
        # - click: 사용자 클릭 위치 (operator 측 자체 보유). 액션 진행 중에는
        #   숨김.
        # - gripper_goal: goal_pose (= 클릭점, world 고정) 을 현재 nav view 에
        #   투영한 픽셀 (red '+', 매 iter 갱신).
        # - base_goal: base_goal_pose (= base 도달 world 점) 을 현재 nav view
        #   에 투영한 픽셀 (lime '+', 매 iter 갱신).
        marker_state = {
            "show_click": True,
            "show_gripper_goal": False,
            "gripper_goal_u": 0.0,
            "gripper_goal_v": 0.0,
            "show_base_goal": False,
            "base_goal_u": 0.0,
            "base_goal_v": 0.0,
        }
        # compute_pred_goal 에서 odom 으로 변환해 저장하는 world-fixed 점들.
        # None 이면 아직 미계산 / RELOCATE 불필요.
        click_xyz_in_odom_holder: List[Optional[np.ndarray]] = [None]
        base_goal_xyz_in_odom_holder: List[Optional[np.ndarray]] = [None]

        def _project_odom_point_to_nav_pixel(
            point_in_odom: np.ndarray,
        ) -> Optional[tuple]:
            """odom 의 3D 점을 현재 base_link 와 nav_cam_link TF 로
            정규화 nav 픽셀로 투영. 실패 시 None."""
            T_base_from_odom = tf_lookup_matrix(
                self.tf_buffer, Frame.BASE_LINK.value,
                Frame.ODOM.value, self.tf_timeout,
            )
            T_nav_link_from_base = tf_lookup_matrix(
                self.tf_buffer, Frame.NAV_CAM_LINK.value,
                Frame.BASE_LINK.value, self.tf_timeout,
            )
            if T_base_from_odom is None or T_nav_link_from_base is None:
                return None
            point_h = np.append(point_in_odom, 1.0)
            point_in_base = (T_base_from_odom @ point_h)[:3]
            return project_base_point_to_nav_pixel(
                point_in_base, T_nav_link_from_base,
            )

        def publish_feedback(_distance_error: float = 0.0) -> None:
            # gripper_goal 픽셀 갱신.
            if (
                marker_state["show_gripper_goal"]
                and click_xyz_in_odom_holder[0] is not None
            ):
                try:
                    pixel = _project_odom_point_to_nav_pixel(
                        click_xyz_in_odom_holder[0]
                    )
                    if pixel is not None:
                        marker_state["gripper_goal_u"] = pixel[0]
                        marker_state["gripper_goal_v"] = pixel[1]
                except Exception:
                    self.get_logger().error(traceback.format_exc())
            # base_goal 픽셀 갱신.
            if (
                marker_state["show_base_goal"]
                and base_goal_xyz_in_odom_holder[0] is not None
            ):
                try:
                    pixel = _project_odom_point_to_nav_pixel(
                        base_goal_xyz_in_odom_holder[0]
                    )
                    if pixel is not None:
                        marker_state["base_goal_u"] = pixel[0]
                        marker_state["base_goal_v"] = pixel[1]
                except Exception:
                    self.get_logger().error(traceback.format_exc())
            feedback.show_click_marker = marker_state["show_click"]
            feedback.new_gripper_goal_scaled_u = float(
                marker_state["gripper_goal_u"]
            )
            feedback.new_gripper_goal_scaled_v = float(
                marker_state["gripper_goal_v"]
            )
            feedback.show_gripper_goal_marker = marker_state["show_gripper_goal"]
            feedback.new_base_goal_scaled_u = float(marker_state["base_goal_u"])
            feedback.new_base_goal_scaled_v = float(marker_state["base_goal_v"])
            feedback.show_base_goal_marker = marker_state["show_base_goal"]
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            goal_handle.publish_feedback(feedback)

        publish_feedback()

        # PRED_GOAL state 의 success_callback. 클릭 픽셀을 base→nav_cam TF 로
        # 환산해, base ↔ 클릭 거리가 OPTIMAL_DISTANCE 가 되도록
        # goal_positions[BASE_ROTATION], [BASE_TRANSLATION] 을 채운다. 이미
        # OPTIMAL_DISTANCE 이내면 RELOCATE (ROTATE_BASE+TRANSLATE_BASE) 단계
        # 전체를 skip. 그렇지 않으면 click_xyz_in_odom (gripper_goal, red 마커)
        # 와 base_goal_xyz_in_odom (base_goal, lime 마커) 을 holder 에 저장.
        pred_goal_state = {
            "failed": False,
            "error_msg": "",
            "skip_relocate": False,
        }

        def compute_pred_goal() -> None:
            try:
                with self.latest_navigation_camera_image_lock:
                    pred_image_msg = self.latest_navigation_camera_image
                if pred_image_msg is None:
                    raise RuntimeError("No navigation camera image at PRED_GOAL")
                nav_image = ros_msg_to_cv2_image(pred_image_msg, self.cv_bridge)
                T_base_from_nav_link = tf_lookup_matrix(
                    self.tf_buffer, Frame.BASE_LINK.value,
                    Frame.NAV_CAM_LINK.value, self.tf_timeout,
                    stamp=pred_image_msg.header.stamp,
                )
                if T_base_from_nav_link is None:
                    raise RuntimeError("Failed to lookup base<-nav_cam TF")
                result = compute_click_point_in_base(
                    nav_image, raw_scaled_u, raw_scaled_v, T_base_from_nav_link,
                )
                if result is None:
                    raise RuntimeError("Failed to deproject click pixel")
                click_xyz, _z = result
                cx, cy = float(click_xyz[0]), float(click_xyz[1])
                raw_distance = float(np.hypot(cx, cy))  # m
                if raw_distance == 0.0:
                    raise RuntimeError("Click point at base origin")

                if abs(raw_distance - OPTIMAL_DISTANCE) < BASE_DISTANCE_TOLERANCE:
                    # RELOCATE_BASE 전체 skip (ROTATE_BASE+TRANSLATE_BASE).
                    pred_goal_state["skip_relocate"] = True
                    self.get_logger().info(
                        f"PRED_GOAL: base↔target xy-distance = "
                        f"{raw_distance:.3f} m ≈ OPTIMAL_DISTANCE "
                        f"({OPTIMAL_DISTANCE:.3f} m); skipping RELOCATE_BASE"
                    )
                    return

                # base_rotation: base x-축이 클릭 방향을 향하도록.
                # base_translation: 실제 base 가 클릭점에서 OPTIMAL_DISTANCE
                # 만큼 떨어진 지점에 정지하도록 raw_distance - OPTIMAL_DISTANCE
                # 만큼 전진. (BASE_MARGIN 차감 안 함 — gripper 잡기 거리 보존.)
                goal_positions[Joint.BASE_ROTATION] = float(np.arctan2(cy, cx))
                goal_positions[Joint.BASE_TRANSLATION] = (
                    raw_distance - OPTIMAL_DISTANCE
                )
                # UI lime 마커 추종용 base_goal 위치 (T0 base 기준).
                # - Forward case (raw_distance >= marker_distance): 클릭점에서
                #   (OPTIMAL_DISTANCE - BASE_MARGIN) 만큼 base 쪽으로 떨어진
                #   지점 = final base 보다 BASE_MARGIN 만큼 클릭 쪽 (안전 buffer
                #   시각화).
                # - Backup case (raw_distance < marker_distance): 위 공식이면
                #   marker 가 초기 base 뒤쪽이 되어 nav 카메라 뒤로 투영됨.
                #   이 경우 marker 를 final base 위치 자체에 둠 (= 그리퍼가
                #   도달할 때 base 가 멈추는 위치를 그대로 표시).
                marker_distance = OPTIMAL_DISTANCE - BASE_MARGIN
                if raw_distance >= marker_distance:
                    frac = 1.0 - marker_distance / raw_distance
                    marker_mode = "forward (BASE_MARGIN buffer)"
                else:
                    # final base 위치 = raw - OPTIMAL (음수, 초기 base 뒤쪽).
                    frac = (raw_distance - OPTIMAL_DISTANCE) / raw_distance
                    marker_mode = "backup (at final base)"
                goal_base_x = cx * frac
                goal_base_y = cy * frac
                self.get_logger().info(
                    f"PRED_GOAL: base↔target xy-distance = "
                    f"{raw_distance:.3f} m, base will stop at "
                    f"OPTIMAL_DISTANCE ({OPTIMAL_DISTANCE:.3f} m); "
                    f"UI base_goal marker [{marker_mode}] in T0 base: "
                    f"x={goal_base_x:.3f}, y={goal_base_y:.3f}"
                )

                # click_xyz (gripper_goal) 과 (goal_base_x, goal_base_y, 0)
                # (base_goal) 두 점을 base_link → odom 으로 한 번씩 변환해
                # holder 에 저장. 매 iter publish_feedback 가 현재 TF 로
                # 재투영해 red/lime 마커 픽셀을 갱신한다.
                def _base_point_to_odom(bx: float, by: float, bz: float):
                    pose_in_base = PoseStamped()
                    pose_in_base.header.frame_id = Frame.BASE_LINK.value
                    pose_in_base.header.stamp = pred_image_msg.header.stamp
                    pose_in_base.pose.position.x = bx
                    pose_in_base.pose.position.y = by
                    pose_in_base.pose.position.z = bz
                    pose_in_base.pose.orientation.w = 1.0
                    ok, pose_in_odom = tf2_transform(
                        self.tf_buffer, pose_in_base, Frame.ODOM.value,
                        self.tf_timeout,
                    )
                    if not ok:
                        return None
                    return np.array([
                        pose_in_odom.pose.position.x,
                        pose_in_odom.pose.position.y,
                        pose_in_odom.pose.position.z,
                    ])

                gripper_in_odom = _base_point_to_odom(cx, cy, 0.0)
                base_in_odom = _base_point_to_odom(goal_base_x, goal_base_y, 0.0)
                if gripper_in_odom is None or base_in_odom is None:
                    self.get_logger().warning(
                        "Failed to transform gripper/base goal to odom; "
                        "marker tracking disabled"
                    )
                else:
                    click_xyz_in_odom_holder[0] = gripper_in_odom
                    base_goal_xyz_in_odom_holder[0] = base_in_odom
                    self.get_logger().info(
                        f"gripper_goal_in_odom={gripper_in_odom}, "
                        f"base_goal_in_odom={base_in_odom}"
                    )
            except Exception as e:
                self.get_logger().error(traceback.format_exc())
                pred_goal_state["failed"] = True
                pred_goal_state["error_msg"] = str(e)

        # PRED_GRASP 콜백: 현재 nav 카메라 RGB + DEPTH_REF 기반 pred_depth 로
        # GR-ConvNet 실행해 grasp 후보(q_out, ang_out, width_out) 산출.
        # 결과는 output/grconv/ 에 timestamped 이미지로 저장 (테스트용).
        # GET_FRONTVIEW / GET_TOPVIEW 단계에서 두 번 호출됨.
        pred_grasp_counter = [0]

        def pred_grasp_callback() -> bool:
            """nav RGB + pred_depth 로 GR-ConvNet grasp 예측. 성공 시 True,
            어떤 단계든 실패하면 False 를 반환해 state 가 FAILURE 로 종료
            되도록 한다."""
            pred_grasp_counter[0] += 1
            tag = f"pred_grasp_{pred_grasp_counter[0]}"
            try:
                with self.latest_navigation_camera_image_lock:
                    image_msg = self.latest_navigation_camera_image
                if image_msg is None:
                    self.get_logger().error(
                        f"[{tag}] No nav camera image available"
                    )
                    return False
                nav_image = ros_msg_to_cv2_image(image_msg, self.cv_bridge)
                # head_tilt=-45° 상태에서 base 가 보이는 nav 이미지 기반
                # metric depth 예측 (mm uint16).
                from nrc_web_teleop_helpers.constants import DEPTH_REF as _REF
                pred_depth = get_pred_depth(nav_image, ref_rcz=_REF)
                # new_gripper_goal_scaled (현재 nav 카메라 기준 재투영) 을
                # 중심으로 (400, 400) crop. 경계 밖은 좌우/상하 대칭 padding.
                scaled_uv: Optional[Tuple[float, float]] = None
                if click_xyz_in_odom_holder[0] is not None:
                    scaled_uv = _project_odom_point_to_nav_pixel(
                        click_xyz_in_odom_holder[0]
                    )
                if scaled_uv is None:
                    scaled_uv = (
                        marker_state["gripper_goal_u"],
                        marker_state["gripper_goal_v"],
                    )
                H, W = nav_image.shape[:2]
                CROP = 400
                HALF = CROP // 2
                cu = int(round(scaled_uv[0] * W))
                cv_c = int(round(scaled_uv[1] * H))
                u0, v0 = cu - HALF, cv_c - HALF
                u1, v1 = u0 + CROP, v0 + CROP
                pad_left = max(0, -u0)
                pad_top = max(0, -v0)
                pad_right = max(0, u1 - W)
                pad_bottom = max(0, v1 - H)
                nav_padded = np.pad(
                    nav_image,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="symmetric",
                )
                depth_padded = np.pad(
                    pred_depth,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="symmetric",
                )
                cu0 = u0 + pad_left
                cv0 = v0 + pad_top
                crop_nav_image = nav_padded[cv0:cv0 + CROP, cu0:cu0 + CROP]
                crop_pred_depth = depth_padded[cv0:cv0 + CROP, cu0:cu0 + CROP]
                # GR-ConvNet 입력용 정규화 depth (0~1 float).
                max_v = (
                    float(crop_pred_depth.max()) if crop_pred_depth.size else 0.0
                )
                if max_v <= 0:
                    self.get_logger().error(
                        f"[{tag}] Invalid crop_pred_depth (max={max_v})"
                    )
                    return False
                norm_crop_pred_depth = (crop_pred_depth / max_v).astype(np.float32)
                result = self._run_grconv(norm_crop_pred_depth, crop_nav_image)
                if result is None:
                    self.get_logger().error(
                        f"[{tag}] grconv inference failed"
                    )
                    return False
                q_out, depth_out, ang_out, width_out = result
                self._save_grconv_outputs(
                    crop_nav_image, q_out, depth_out, tag=tag,
                )
                # 최고 q 위치 로그 (디버깅 / 후속 GRASP 연결 준비).
                peak_idx = int(np.argmax(q_out))
                pr, pc = divmod(peak_idx, q_out.shape[1])
                self.get_logger().info(
                    f"[{tag}] grconv peak at (r={pr}, c={pc}), "
                    f"q={int(q_out[pr, pc])}, "
                    f"ang={float(ang_out[pr, pc]):+.3f} rad, "
                    f"width={float(width_out[pr, pc]):.2f} px"
                )
                return True
            except Exception:
                self.get_logger().error(traceback.format_exc())
                return False

        # GRASP 콜백 (placeholder).
        def grasp_callback() -> bool:
            self.get_logger().info("GRASP callback (TODO: implement)")
            return True

        # 상태 머신.
        states = MoveToActionState.get_states_for_move_gripper_to_point()
        grasp_ready_state_i = next(
            (i for i, cs in enumerate(states)
             if MoveToActionState.GRASP_READY in cs),
            None,
        )
        # skip_relocate 시 TRANSLATE_BASE 만 건너뜀. ROTATE_BASE 는 base 가
        # 클릭점을 향하도록 회전해야 하므로 거리에 무관하게 항상 실행.
        relocate_states = (
            MoveToActionState.TRANSLATE_BASE,
        )
        rotate_state_i = next(
            (i for i, cs in enumerate(states)
             if MoveToActionState.ROTATE_BASE in cs),
            None,
        )
        translate_state_i = next(
            (i for i, cs in enumerate(states)
             if MoveToActionState.TRANSLATE_BASE in cs),
            None,
        )
        state_i = 0
        rate = self.create_rate(5.0)  # 5 Hz

        while rclpy.ok():
            concurrent_states = states[state_i]

            if goal_handle.is_cancel_requested:
                return action_cancel_callback("Goal canceled")
            if (self.get_clock().now() - start_time) > self.action_timeout:
                return action_error_callback(
                    "Goal timed out", MoveGripperToPoint.Result.STATUS_TIMEOUT
                )

            if len(motion_executors) == 0:
                # RELOCATE_BASE skip (PRED_GOAL 에서 결정).
                if pred_goal_state["skip_relocate"] and any(
                    s in relocate_states for s in concurrent_states
                ):
                    self.get_logger().info(
                        f"Skipping state {state_i}: {concurrent_states} "
                        f"(RELOCATE_BASE skipped)"
                    )
                    state_i += 1
                    rate.sleep()
                    continue
                # ROTATE_BASE 진입: 모든 마커 숨김 (회전 중 시각적 혼란 방지).
                if state_i == rotate_state_i:
                    marker_state["show_click"] = False
                    marker_state["show_gripper_goal"] = False
                    marker_state["show_base_goal"] = False
                    publish_feedback()
                # TRANSLATE_BASE 진입: gripper_goal (red '+') 과 base_goal
                # (lime '+') 표시 시작. 매 iter publish_feedback 가 현재 TF 로
                # 두 마커 픽셀 모두 갱신.
                if state_i == translate_state_i:
                    marker_state["show_gripper_goal"] = (
                        click_xyz_in_odom_holder[0] is not None
                    )
                    marker_state["show_base_goal"] = (
                        base_goal_xyz_in_odom_holder[0] is not None
                    )
                    self.get_logger().info(
                        f"TRANSLATE_BASE entry: show_gripper_goal="
                        f"{marker_state['show_gripper_goal']}, "
                        f"show_base_goal={marker_state['show_base_goal']}"
                    )
                    publish_feedback()
                # GRASP_READY 진입: head_pan 을 그리퍼 방향(-π/2) 으로 갱신.
                # (base 는 GRASP_READY state 내부에서 +π/2 delta 회전한다.)
                # TRANSLATE_BASE 완료 시점이므로 두 동적 마커 숨김.
                if state_i == grasp_ready_state_i:
                    goal_positions[Joint.HEAD_PAN] = -np.pi / 2.0
                    marker_state["show_gripper_goal"] = False
                    marker_state["show_base_goal"] = False
                    publish_feedback()
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
                        err_callback=[publish_feedback],
                        success_callback=[
                            compute_pred_goal,
                            pred_grasp_callback,
                            grasp_callback,
                        ],
                    )
                    if motion_executor is None:
                        return action_success_callback("Goal succeeded")
                    motion_executors.append(motion_executor)
                if pred_goal_state["failed"]:
                    return action_error_callback(
                        f"PRED_GOAL failed: {pred_goal_state['error_msg']}",
                        MoveGripperToPoint.Result.STATUS_DEPROJECTION_FAILURE,
                    )
            else:
                try:
                    for i, motion_executor in enumerate(motion_executors):
                        retval = next(motion_executor)
                        if retval == MotionGeneratorRetval.SUCCESS:
                            motion_executors.pop(i)
                            self.get_logger().info(
                                f"##### Success (State Num {state_i}: {concurrent_states})"
                            )
                            break
                        elif retval == MotionGeneratorRetval.FAILURE:
                            raise Exception("Failed to move to goal pose")
                    if len(motion_executors) == 0:
                        state_i += 1
                except Exception as e:
                    self.get_logger().error(traceback.format_exc())
                    return action_error_callback(
                        f"Error executing the motion generator: {e}",
                        MoveGripperToPoint.Result.STATUS_FAILURE,
                    )

            rate.sleep()

        return action_error_callback("Failed to execute MoveGripperToPoint")

    def get_optimal_grasp(self):
        # Navigation Cam image
        # GR conv
        return

    def _run_grconv(
        self, depth_image: np.ndarray, rgb_image: Optional[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Depth(+RGB) 를 GR-ConvNet 에 입력해 (q_out, depth_out, ang_out,
        width_out) 를 원본 이미지 크기로 반환. 실패 시 None.

        - depth_image: float32, 정규화된 depth (0~1 권장).
        - rgb_image: BGR uint8 (HxWx3) 또는 None (depth-only 모델일 때).
        """
        if self.grconv_model is None:
            self.get_logger().warning(
                "GR-ConvNet model is not loaded; skip grconv inference"
            )
            return None
        try:
            input_channels = self.grconv_model.conv1.in_channels
            depth_proc = np.clip(
                depth_image.astype(np.float32) - depth_image.mean(), -1, 1
            )
            if input_channels >= 4 and rgb_image is not None:
                rgb_proc = rgb_image.astype(np.float32) / 255.0
                rgb_proc -= rgb_proc.mean()
                rgb_channels = rgb_proc.transpose((2, 0, 1))
                x = np.concatenate(
                    [depth_proc[np.newaxis, :, :], rgb_channels], axis=0
                )
            else:
                x = depth_proc[np.newaxis, :, :]
            x = torch.from_numpy(
                x[np.newaxis, :, :, :].astype(np.float32)
            ).to(self.grconv_device)
            with torch.no_grad():
                pred = self.grconv_model.predict(x)
            q_img, ang_img, width_img = post_process_output(
                pred["pos"], pred["cos"], pred["sin"], pred["width"]
            )
            image_size = (depth_image.shape[1], depth_image.shape[0])
            q_img = (q_img - q_img.min()) / (q_img.max() - q_img.min() + 1e-8)
            q_out = cv2.resize(
                (255.0 * np.clip(q_img, 0.0, 1.0)).astype(np.uint8),
                image_size, cv2.INTER_AREA,
            )
            ang_out = cv2.resize(
                ang_img.astype(np.float32), image_size, cv2.INTER_LINEAR
            )
            width_out = cv2.resize(
                width_img.astype(np.float32), image_size, cv2.INTER_LINEAR
            )
            depth_vis = (depth_image - depth_image.min()) / (
                depth_image.max() - depth_image.min() + 1e-8
            )
            depth_out = (255.0 * np.clip(depth_vis, 0.0, 1.0)).astype(np.uint8)
            return q_out, depth_out, ang_out, width_out
        except Exception:
            self.get_logger().error(traceback.format_exc())
            return None

    def _save_grconv_outputs(
        self,
        rgb_image: np.ndarray,
        q_out: np.ndarray,
        depth_out: np.ndarray,
        tag: str = "",
    ) -> None:
        """grconv 결과를 rgb / overlay / depth_out 3장만 저장."""
        try:
            os.makedirs(GRCONV_OUTPUT_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            prefix = f"{ts}_{tag}" if tag else ts
            cv2.imwrite(
                os.path.join(GRCONV_OUTPUT_DIR, f"{prefix}_rgb.png"),
                rgb_image,
            )
            cv2.imwrite(
                os.path.join(GRCONV_OUTPUT_DIR, f"{prefix}_depth_out.png"),
                depth_out,
            )
            q_color = cv2.applyColorMap(q_out, cv2.COLORMAP_JET)
            if q_color.shape[:2] != rgb_image.shape[:2]:
                q_color = cv2.resize(
                    q_color, (rgb_image.shape[1], rgb_image.shape[0])
                )
            overlay = cv2.addWeighted(rgb_image, 0.5, q_color, 0.5, 0)
            cv2.imwrite(
                os.path.join(GRCONV_OUTPUT_DIR, f"{prefix}_overlay.png"),
                overlay,
            )
            self.get_logger().info(
                f"GR-ConvNet outputs saved to {GRCONV_OUTPUT_DIR}/{prefix}_*.png"
            )
        except Exception:
            self.get_logger().error(traceback.format_exc())

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