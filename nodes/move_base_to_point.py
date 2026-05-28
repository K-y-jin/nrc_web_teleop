#!/usr/bin/env python3

# Standard Imports
import threading
import traceback
from typing import Generator, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped

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
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger

# Local Imports
from nrc_web_teleop.action import MoveBaseToPoint
from nrc_web_teleop.srv import GetDistance
from nrc_web_teleop_helpers.constants import (
    Joint,
    Frame,
    TILT_READY,
    TILT_READY_TOLERANCE,
    DEPTH_REF,
    BASE_MARGIN,
    NAVIGABLE_LINE_SAMPLES,
    NAVIGABLE_MAX_STEP_RATIO,
)
from nrc_web_teleop_helpers.conversions import (
    remaining_time,
    ros_msg_to_cv2_image,
    tf_lookup_matrix,
)
from nrc_web_teleop_helpers.move_to_action_state import MoveToActionState
from nrc_web_teleop_helpers.stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
)
from nrc_web_teleop_helpers.functions import compute_click_point_in_base


class MoveBaseToPointNode(Node):
    """
    내비게이션 카메라에서 클릭한 지점으로 베이스를 이동시키는 노드.

    - `/get_distance` 서비스(`GetDistance`): 거리 미리보기. head_tilt가
      `TILT_READY`가 아니면 head를 그 자세로 이동시키고 `success=False`로 응답.
      `TILT_READY`이면 `get_pred_depth`(Depth-Anything) + `DEPTH_REF`로 클릭
      픽셀까지의 직선거리(mm)를 반환한다.
    - `move_base_to_point` 액션(`MoveBaseToPoint`): 클릭 픽셀을 받아 팔을 stow
      한 뒤, 베이스를 클릭 방위각으로 회전(`pan_tilt_offset_to_center`의
      `delta_pan`)하고 `head_pan=0.0`으로 두고, 거리 미리보기와 같은 방식으로
      계산한 직선거리만큼 베이스를 전진시킨다.
    """

    def __init__(
        self,
        tf_timeout_secs: float = 0.5,
        action_timeout_secs: float = 60.0,
    ):

        super().__init__("move_base_to_point")

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

        # Subscribe to the Navigation camera's CompressedImage feed
        self.latest_navigation_camera_image: Optional[CompressedImage] = None
        self.latest_navigation_camera_image_lock = threading.Lock()
        self.navigation_camera_subscriber = self.create_subscription(
            CompressedImage,
            "/navigation_camera/image_raw/rotated/compressed",
            self.navigation_camera_cb,
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
            MoveBaseToPoint,
            "move_base_to_point",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        # Create the GetDistance service server (거리 미리보기, move_gripper 용)
        self.get_distance_service = self.create_service(
            GetDistance,
            "get_distance",
            self.get_distance_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        # /is_head_pred_ready — UI 클릭 시 호출. head_tilt 가 TILT_READY 면
        # success=True, 아니면 TILT_READY 로 이동 후 success=False 반환.
        self.is_head_pred_ready_service = self.create_service(
            Trigger,
            "is_head_pred_ready",
            self.is_head_pred_ready_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        return True

    def navigation_camera_cb(self, ros_image: CompressedImage) -> None:
        with self.latest_navigation_camera_image_lock:
            self.latest_navigation_camera_image = ros_image

    def goal_callback(self, goal_request: MoveBaseToPoint.Goal) -> GoalResponse:
        self.get_logger().info(f"Received request {goal_request}")

        # Reject the goal if no Navigation Camera RGB image has been received yet
        with self.latest_navigation_camera_image_lock:
            if self.latest_navigation_camera_image is None:
                self.get_logger().info(
                    "Rejecting goal request since no Navigation Camera RGB image message has been received yet"
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
        """Always accept client requests to cancel the active goal."""
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> MoveBaseToPoint.Result:
        """
        액션 실행. 재설계 흐름:
            PRED_READY (depth pred 자세) → PRED_GOAL (클릭→goal_xyz_in_odom 계산)
            → ROTATE_BASE+HEAD_PAN → TRANSLATE_BASE → TERMINAL.
        """
        start_time = self.get_clock().now()
        feedback = MoveBaseToPoint.Feedback()

        terminate_motion_executors = False
        motion_executors: List[Generator[MotionGeneratorRetval, None, None]] = []

        def cleanup() -> None:
            nonlocal terminate_motion_executors, motion_executors
            self.active_goal_request = None
            terminate_motion_executors = True
            # Execute the motion executors once more to process cancellation.
            if len(motion_executors) > 0:
                try:
                    for motion_executor in motion_executors:
                        _ = next(motion_executor)
                except Exception:
                    self.get_logger().debug(traceback.format_exc())

        def action_error_callback(
            error_msg: str = "Goal failed",
            status: int = MoveBaseToPoint.Result.STATUS_FAILURE,
        ) -> MoveBaseToPoint.Result:
            self.get_logger().error(error_msg)
            goal_handle.abort()
            cleanup()
            return MoveBaseToPoint.Result(status=status)

        def action_success_callback(
            success_msg: str = "Goal succeeded",
        ) -> MoveBaseToPoint.Result:
            self.get_logger().info(success_msg)
            goal_handle.succeed()
            cleanup()
            return MoveBaseToPoint.Result(status=MoveBaseToPoint.Result.STATUS_SUCCESS)

        def action_cancel_callback(
            cancel_msg: str = "Goal canceled",
        ) -> MoveBaseToPoint.Result:
            self.get_logger().info(cancel_msg)
            goal_handle.canceled()
            cleanup()
            return MoveBaseToPoint.Result(
                status=MoveBaseToPoint.Result.STATUS_CANCELLED
            )

        # Goal: the clicked normalized pixel on the navigation camera view.
        raw_scaled_u = goal_handle.request.scaled_u
        raw_scaled_v = goal_handle.request.scaled_v

        # nav 이미지 수신 확인 (goal_callback 에서 이미 했지만 방어).
        with self.latest_navigation_camera_image_lock:
            image_msg = self.latest_navigation_camera_image
        if image_msg is None:
            return action_error_callback("No navigation camera image received yet")

        # goal_pose 는 MoveToActionState.get_motion_executor signature 호환용
        # placeholder. MoveToActionState 의 모든 state 가 자체 PoseStamped 를
        # 만들어 쓰므로 본 노드에서는 채우지 않는다.
        goal_pose = PoseStamped()

        # goal_positions: BASE_ROTATION / BASE_TRANSLATION 은 PRED_GOAL 에서
        # 계산해 갱신, head_pan 은 항상 0 (base 가 클릭 방향을 향한 뒤 head 는
        # 정면).
        goal_positions = {
            Joint.BASE_ROTATION: 0.0,
            Joint.BASE_TRANSLATION: 0.0,
            Joint.HEAD_PAN: 0.0,
        }

        # 마커 상태(피드백으로 UI 전달).
        # - PRED_READY/PRED_GOAL: click 마커 표시.
        # - ROTATE_BASE: 둘 다 숨김.
        # - TRANSLATE_BASE: base_goal 마커 (red '+') 표시 + base 전진에 따라
        #   픽셀 갱신.
        # - TERMINAL: 둘 다 숨김.
        marker_state = {
            "show_click": True,
            "show_base_goal": False,
            "base_goal_u": 0.0,
            "base_goal_v": 0.0,
            "translation_active": False,
        }

        # 정규화 ref_u/v (DEPTH_REF 픽셀 ÷ image 크기). PRED_GOAL 에서 채워짐.
        ref_u_holder = [0.0]
        ref_v_holder = [0.0]

        def publish_feedback(_distance_error: float = 0.0) -> None:
            """진행 피드백. 마커는 marker_state 에 따라 표시/갱신."""
            if marker_state["show_base_goal"] and marker_state.get("translation_active"):
                raw = marker_state["raw_distance"]  # m
                if raw != 0.0:
                    remaining = float(_distance_error) if _distance_error else 0.0
                    ratio = (BASE_MARGIN + remaining) / raw
                    ratio = max(min(ratio, 1.0), 0.0)
                    marker_state["base_goal_u"] = ref_u_holder[0] + ratio * (
                        marker_state["post_click_u"] - ref_u_holder[0]
                    )
                    marker_state["base_goal_v"] = ref_v_holder[0] + ratio * (
                        marker_state["post_click_v"] - ref_v_holder[0]
                    )
            feedback.show_click_marker = marker_state["show_click"]
            feedback.new_base_goal_scaled_u = float(marker_state["base_goal_u"])
            feedback.new_base_goal_scaled_v = float(marker_state["base_goal_v"])
            feedback.show_base_goal_marker = marker_state["show_base_goal"]
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            goal_handle.publish_feedback(feedback)

        # 액션 수락 직후 click 마커 표시 상태로 피드백 전송 (UI 유지).
        publish_feedback()

        # PRED_GOAL state 의 success_callback. 클릭 픽셀을 base→nav_cam TF 로
        # 환산해 goal_xyz_in_odom 을 계산하고 goal_pose 와 base_rotation 을 채운다.
        pred_goal_state = {"failed": False, "error_msg": ""}

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

                # base_rotation: base x-축이 클릭 방향을 향하도록.
                # base_translation: ROTATE_BASE 후 base x-축이 클릭 방향이
                # 되므로, 남은 전진 거리는 raw_distance - BASE_MARGIN.
                goal_positions[Joint.BASE_ROTATION] = float(np.arctan2(cy, cx))
                goal_positions[Joint.BASE_TRANSLATION] = (
                    raw_distance - BASE_MARGIN
                )

                # 마커 보간용: 이미지에서 DEPTH_REF↔click 직선을 image-forward
                # 에 정렬한 좌표(post_click).
                h_img, w_img = nav_image.shape[:2]
                ref_u_holder[0] = DEPTH_REF[1] / float(w_img)
                ref_v_holder[0] = DEPTH_REF[0] / float(h_img)
                dx = raw_scaled_u - ref_u_holder[0]
                dy = raw_scaled_v - ref_v_holder[0]
                theta = float(np.arctan2(dx, -dy))
                cos_t, sin_t = np.cos(-theta), np.sin(-theta)
                marker_state["post_click_u"] = float(
                    ref_u_holder[0] + cos_t * dx - sin_t * dy
                )
                marker_state["post_click_v"] = float(
                    ref_v_holder[0] + sin_t * dx + cos_t * dy
                )
                marker_state["raw_distance"] = raw_distance  # m
                marker_state["translation_total"] = raw_distance - BASE_MARGIN  # m

                self.get_logger().info(
                    f"PRED_GOAL: base↔target xy-plane distance = "
                    f"{raw_distance:.3f} m, "
                    f"base_rotation = "
                    f"{goal_positions[Joint.BASE_ROTATION]:.3f} rad"
                )
            except Exception as e:
                self.get_logger().error(traceback.format_exc())
                pred_goal_state["failed"] = True
                pred_goal_state["error_msg"] = str(e)

        # 상태 머신.
        states = MoveToActionState.get_states_for_move_base_to_point()
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
                    "Goal timed out", MoveBaseToPoint.Result.STATUS_TIMEOUT
                )

            # Start the motion executors for the current concurrent states.
            if len(motion_executors) == 0:
                # state 진입 시 마커 표시 상태 전환.
                if state_i == rotate_state_i:
                    # ROTATE_BASE 진입: click/base_goal 마커 둘 다 숨김.
                    marker_state["show_click"] = False
                    marker_state["show_base_goal"] = False
                    publish_feedback()
                elif state_i == translate_state_i:
                    # TRANSLATE_BASE 진입: base_goal 마커 표시 시작 (post_click 위치).
                    marker_state["show_click"] = False
                    marker_state["show_base_goal"] = True
                    marker_state["translation_active"] = True
                    marker_state["base_goal_u"] = marker_state["post_click_u"]
                    marker_state["base_goal_v"] = marker_state["post_click_v"]
                    publish_feedback(marker_state["translation_total"])
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
                        success_callback=[compute_pred_goal],
                    )
                    if motion_executor is None:
                        return action_success_callback("Goal succeeded")
                    motion_executors.append(motion_executor)
                # PRED_GOAL 실패 시 즉시 abort.
                if pred_goal_state["failed"]:
                    return action_error_callback(
                        f"PRED_GOAL failed: {pred_goal_state['error_msg']}",
                        MoveBaseToPoint.Result.STATUS_DEPROJECTION_FAILURE,
                    )
            # Step the motion executors until the current concurrent states finish.
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
                        MoveBaseToPoint.Result.STATUS_FAILURE,
                    )

            rate.sleep()

        return action_error_callback("Failed to execute MoveBaseToPoint")

    @staticmethod
    def is_line_navigable(
        pred_depth: np.ndarray,
        click_row: int,
        click_col: int,
    ) -> bool:
        """DEPTH_REF 픽셀에서 클릭 픽셀까지의 직선상 pred_depth가 스무스하게
        단조 증가하면 True (base가 이동 가능한 ground), 도중에 급격한 증가가
        있으면 False.

        - 단조성: 클릭 픽셀이 기준 픽셀보다 깊지 않으면 False.
        - 스무스함: 가장 큰 한 스텝의 깊이 변화가 평균 스텝의
          NAVIGABLE_MAX_STEP_RATIO 배를 넘으면 False.
        """
        ref_r, ref_c, _ = DEPTH_REF
        h, w = pred_depth.shape[:2]
        rs = np.clip(
            np.linspace(ref_r, click_row, NAVIGABLE_LINE_SAMPLES).astype(int),
            0, h - 1,
        )
        cs = np.clip(
            np.linspace(ref_c, click_col, NAVIGABLE_LINE_SAMPLES).astype(int),
            0, w - 1,
        )
        line = pred_depth[rs, cs].astype(np.float32)
        total = float(line[-1] - line[0])
        if total <= 0.0:
            return False
        steps = np.diff(line)
        expected = total / (NAVIGABLE_LINE_SAMPLES - 1)
        return float(np.max(np.abs(steps))) <= expected * NAVIGABLE_MAX_STEP_RATIO

    def head_tilt_is_ready(self) -> bool:
        """현재 head_tilt가 TILT_READY 허용 오차 내인지."""
        head_tilt = self.controller.get_head_joint_states()[Joint.HEAD_TILT]
        return abs(head_tilt - TILT_READY) <= TILT_READY_TOLERANCE

    def move_head_tilt_to_ready(self) -> bool:
        """head_tilt를 TILT_READY로 이동시키고 완료 시 True."""
        executor = self.controller.move_to_joint_positions(
            joint_positions={Joint.HEAD_TILT: TILT_READY},
            timeout_secs=10.0,
        )
        rate = self.create_rate(15.0)
        for retval in executor:
            if retval == MotionGeneratorRetval.SUCCESS:
                return True
            if retval == MotionGeneratorRetval.FAILURE:
                return False
            rate.sleep()
        return False

    def is_head_pred_ready_callback(self, request, response):
        """
        `/is_head_pred_ready` 서비스 콜백. UI 가 nav 카메라 클릭 시 호출.

        head_tilt 가 TILT_READY (-45°) 면 success=True 반환 (UI 가 마커 표시).
        아니면 head_tilt 를 TILT_READY 로 이동시키고 success=False 반환
        (UI 는 마커를 표시하지 않고 사용자가 head 정렬 후 다시 클릭).
        """
        _ = request  # Trigger 는 request 가 비어 있다.
        if self.head_tilt_is_ready():
            response.success = True
            response.message = "head_tilt at TILT_READY"
        else:
            self.get_logger().info(
                "head_tilt not at TILT_READY; moving head and asking for re-click"
            )
            self.move_head_tilt_to_ready()
            response.success = False
            response.message = "head_tilt moved to TILT_READY; re-click on the image"
        return response

    def get_distance_callback(self, request, response):
        """
        `/get_distance` 서비스 콜백 — 거리 미리보기 (DESIGN (A)).

        head_tilt가 TILT_READY가 아니면 TILT_READY로 이동시키고 `success=False`로
        응답한다. TILT_READY이면 클릭 좌표까지의 직선거리(mm)를 반환한다.
        """
        self.get_logger().info(
            f"GetDistance request: scaled_u={request.scaled_u}, "
            f"scaled_v={request.scaled_v}"
        )
        try:
            if not self.head_tilt_is_ready():
                self.get_logger().info("head_tilt not at TILT_READY; moving head")
                self.move_head_tilt_to_ready()
                response.success = False
                return response

            with self.latest_navigation_camera_image_lock:
                image_msg = self.latest_navigation_camera_image
            if image_msg is None:
                self.get_logger().error("No navigation camera image received yet")
                response.success = False
                return response

            nav_image = ros_msg_to_cv2_image(image_msg, self.cv_bridge)
            T_base_from_nav_link = tf_lookup_matrix(
                self.tf_buffer, Frame.BASE_LINK.value,
                Frame.NAV_CAM_LINK.value, self.tf_timeout,
                stamp=image_msg.header.stamp,
            )
            if T_base_from_nav_link is None:
                self.get_logger().error(
                    "Failed to lookup base_link <- link_head_nav_cam TF"
                )
                response.success = False
                return response
            result = compute_click_point_in_base(
                nav_image, request.scaled_u, request.scaled_v,
                T_base_from_nav_link,
            )
            if result is None:
                response.success = False
                return response
            click_xyz, _z = result
            raw_distance = float(np.hypot(click_xyz[0], click_xyz[1]))  # m
            # 라벨에는 Move Base 액션이 실제로 base_translation 으로 사용할
            # 값(= raw 거리 − BASE_MARGIN, m) 을 표시. 타깃이 안전 마진보다
            # 가까우면 음수(= 후진 거리)도 그대로 노출.
            response.distance = raw_distance - BASE_MARGIN
            response.success = True
            # ground 판정은 일단 사용하지 않음. 추후 활성화 시
            # self.is_line_navigable(...)로 채워 넣을 수 있도록 함수는 보존.
            response.is_navigable = True
            # base 정지 예상 위치 = DEPTH_REF 픽셀에서 클릭 픽셀로 향하는
            # nav 이미지 직선상의, (raw - BASE_MARGIN)/raw 비율 지점.
            h, w = nav_image.shape[:2]
            ref_u = DEPTH_REF[1] / float(w)
            ref_v = DEPTH_REF[0] / float(h)
            frac = ((raw_distance - BASE_MARGIN) / raw_distance
                    if raw_distance != 0.0 else 0.0)
            response.stop_scaled_u = ref_u + frac * (request.scaled_u - ref_u)
            response.stop_scaled_v = ref_v + frac * (request.scaled_v - ref_v)
        except Exception as e:
            self.get_logger().error(f"GetDistance failed: {e}")
            response.success = False
        return response


def main(args: Optional[List[str]] = None):
    rclpy.init(args=args)

    move_to_point = MoveBaseToPointNode()
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
