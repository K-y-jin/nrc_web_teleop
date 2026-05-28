from __future__ import annotations  # Required for type hinting a class within itself

# Standard imports
from enum import Enum
from builtin_interfaces.msg import Time
from typing import Callable, Dict, Generator, List, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Point, Quaternion, PoseStamped

# Local imports
from .constants import (
    Joint,
    get_pred_ready_configuration,
    get_frontview_configuration,
)
from .stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
    TerminationCriteria,
)
from tf_transformations import quaternion_about_axis


class MoveToActionState(Enum):
    """
    `move_base_to_point.py` / `move_gripper_to_point.py` 전용 state machine.

    Pregrasp 액션의 state machine 은 `move_to_pregrasp_state.py` 의
    `MoveToPregraspState` 를 사용한다 (서로 독립).
    """

    # 공통 상태.
    HEAD_PAN = 4
    TERMINAL = 12

    # Move Base / Move Gripper 공통 단계 (재설계 후 흐름).
    # PRED_READY: depth prediction 용 자세 (get_pred_ready_configuration).
    # PRED_GOAL: 클릭 픽셀 → goal_xyz_in_odom 계산 (execute_callback 에 위임
    #            via success_callback). 모션 없음.
    # ROTATE_BASE: head_pan=0 일 때 nav 카메라가 goal 을 향하도록 base 회전.
    # TRANSLATE_BASE: BASE_MARGIN / OPTIMAL_DISTANCE 고려해 goal_xyz_in_odom
    #                 까지 전진/후진.
    PRED_READY = 20
    PRED_GOAL = 21
    ROTATE_BASE = 22
    TRANSLATE_BASE = 23
    # Move Gripper 전용: RELOCATE_BASE 후 그리퍼가 타깃을 향하도록 base 를
    # 추가로 +π/2 회전한다 (그리퍼가 base 측면에 있으므로). 동시에 head_pan 도
    # -π/2 로 (gripper 방향) 이동.
    GRASP_READY = 24
    # GRASP_READY 이후 grasp 시퀀스. GET_FRONTVIEW → PRED_GRASP (frontview
    # 이미지로 1차 예측) → GET_TOPVIEW → PRED_GRASP (topview 이미지로 2차
    # 예측) → GRASP (실행).
    GET_FRONTVIEW = 27
    GET_TOPVIEW = 28
    PRED_GRASP = 25
    GRASP = 26

    @staticmethod
    def get_states_for_move_base_to_point() -> List[List[MoveToActionState]]:
        # 새 설계: depth prediction 자세 → goal 계산 → 회전 → 전진/후진.
        states = []
        states.append([MoveToActionState.PRED_READY])
        states.append([MoveToActionState.PRED_GOAL])
        states.append([MoveToActionState.ROTATE_BASE, MoveToActionState.HEAD_PAN])
        states.append([MoveToActionState.TRANSLATE_BASE])
        states.append([MoveToActionState.TERMINAL])
        return states

    @staticmethod
    def get_states_for_move_gripper_to_point() -> List[List[MoveToActionState]]:
        # 새 설계: PRED_READY → PRED_GOAL → RELOCATE(ROTATE+TRANSLATE) →
        # GRASP_READY → GET_FRONTVIEW → PRED_GRASP → GET_TOPVIEW → PRED_GRASP
        # → GRASP → TERMINAL.
        # PRED_GOAL 에서 OPTIMAL_DISTANCE 기준으로 base 이동이 필요한지 결정;
        # 필요 없으면 RELOCATE 단계 skip. PRED_GRASP 는 frontview / topview
        # 두 번 호출되어 각각 이미지로 grasp 예측 (의도된 중복).
        states = []
        states.append([MoveToActionState.PRED_READY])
        states.append([MoveToActionState.PRED_GOAL])
        states.append([MoveToActionState.ROTATE_BASE, MoveToActionState.HEAD_PAN])
        states.append([MoveToActionState.TRANSLATE_BASE])
        states.append([MoveToActionState.PRED_GRASP])
        states.append([MoveToActionState.GRASP_READY, MoveToActionState.HEAD_PAN])
        states.append([MoveToActionState.GET_FRONTVIEW])
        states.append([MoveToActionState.PRED_GRASP])
        states.append([MoveToActionState.GET_TOPVIEW])
        states.append([MoveToActionState.PRED_GRASP])
        states.append([MoveToActionState.GRASP])
        states.append([MoveToActionState.TERMINAL])
        return states

    def get_motion_executor(
        self,
        controller: StretchIKControl,
        goal_pose: PoseStamped,
        ik_solution: Dict[Joint, float],
        horizontal_grasp: bool = True,
        timeout_secs: float = 30.0,
        check_cancel: Callable[[], bool] = lambda: False,
        err_callback: Optional[Callable[[npt.NDArray[np.float64]], None]] = None,
        success_callback: Optional[Callable[[npt.NDArray[np.float64]], None]] = None,
    ) -> Optional[Generator[MotionGeneratorRetval, None, None]]:

        # The parameters that are state-dependant
        joints_for_velocity_control = []
        joint_position_overrides = {}
        joints_for_position_control = {}
        velocity_overrides = {}
        error_callback_temp = None
        success_callback_temp = None

        # Configure the parameters depending on the state
        if self == MoveToActionState.TERMINAL:
            return None
        elif self == MoveToActionState.PRED_READY:
            # depth prediction 자세 (팔/손목 수납).
            joints_for_position_control.update(get_pred_ready_configuration())
        elif self == MoveToActionState.PRED_GOAL:
            # 모션 없음. execute_callback 의 success_callback 으로 click →
            # goal_xyz_in_odom 계산을 위임하고 즉시 SUCCESS yield.
            compute_goal = success_callback[0] if success_callback else None

            def _pred_goal_gen():
                if compute_goal is not None:
                    compute_goal()
                yield MotionGeneratorRetval.SUCCESS
            return _pred_goal_gen()
        elif self == MoveToActionState.GET_FRONTVIEW:
            # TODO: head_pan / head_tilt 를 frontview 자세로 이동.
            # 현재는 stub — 즉시 SUCCESS.
            def _get_frontview_gen():
                yield MotionGeneratorRetval.SUCCESS
            return _get_frontview_gen()
        elif self == MoveToActionState.GET_TOPVIEW:
            # TODO: head_pan / head_tilt 를 topview 자세로 이동.
            # 현재는 stub — 즉시 SUCCESS.
            def _get_topview_gen():
                yield MotionGeneratorRetval.SUCCESS
            return _get_topview_gen()
        elif self == MoveToActionState.PRED_GRASP:
            # 모션 없음. execute_callback 의 success_callback[1] 로 nav 카메라
            # 이미지 + pred_depth 기반 grasp 예측 위임. frontview / topview
            # 단계에서 두 번 호출.
            # 콜백이 False 를 반환하면 FAILURE, True / None 이면 SUCCESS.
            pred_grasp_cb = (
                success_callback[1]
                if success_callback and len(success_callback) > 1
                else None
            )

            def _pred_grasp_gen():
                if pred_grasp_cb is not None:
                    ok = pred_grasp_cb()
                    if ok is False:
                        yield MotionGeneratorRetval.FAILURE
                        return
                yield MotionGeneratorRetval.SUCCESS
            return _pred_grasp_gen()
        elif self == MoveToActionState.GRASP:
            # 그리퍼 잡기 실행. execute_callback 의 success_callback[2] 로 위임.
            # 콜백이 False 를 반환하면 FAILURE, True / None 이면 SUCCESS.
            grasp_cb = (
                success_callback[2]
                if success_callback and len(success_callback) > 2
                else None
            )

            def _grasp_gen():
                if grasp_cb is not None:
                    ok = grasp_cb()
                    if ok is False:
                        yield MotionGeneratorRetval.FAILURE
                        return
                yield MotionGeneratorRetval.SUCCESS
            return _grasp_gen()
        elif self == MoveToActionState.HEAD_PAN:
            joints_for_position_control[Joint.HEAD_PAN] = ik_solution[Joint.HEAD_PAN]
            # Cap the head pan at the base's max rotation speed, so the base and head pan
            # camera roughly track each other.
            velocity_overrides[Joint.HEAD_PAN] = controller.joint_vel_abs_lim[
                Joint.BASE_ROTATION
            ][1]
        elif self == MoveToActionState.TRANSLATE_BASE:
            error_callback_temp = err_callback[0]
            # 이동용 별도 PoseStamped 를 생성한다 (ROTATE_BASE 의 rotation_pose
            # 와 같은 패턴). 호출자가 ik_solution[BASE_TRANSLATION] 에 base
            # x-축 방향 전진 거리(m, 부호 포함)를 넣어 준다. controller 가
            # 진입 시 base_link → odom 변환을 한 번 수행해 world 기준 목표
            # 로 추종한다.
            base_translation = ik_solution[Joint.BASE_TRANSLATION]
            translation_pose = PoseStamped()
            translation_pose.header.stamp = Time()  # latest
            translation_pose.header.frame_id = "base_link"
            translation_pose.pose.position = Point(
                x=base_translation, y=0.0, z=0.0
            )
            translation_pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0
            )
            controller.node.get_logger().info(
                f"[translate-state] ik_solution[BASE_TRANSLATION]="
                f"{base_translation:+.4f} m → translation_pose in base_link="
                f"({base_translation:+.4f}, 0.0, 0.0)"
            )
            return controller.translate_base_to_goal_pose(
                goal=translation_pose,
                termination=TerminationCriteria.ZERO_ERR,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )
        elif (
            self == MoveToActionState.ROTATE_BASE
            or self == MoveToActionState.GRASP_READY
        ):
            # 회전용 별도 PoseStamped 를 생성한다. 호출자가 넘긴 goal_pose 는
            # TRANSLATE_BASE 에서 odom 좌표로 그대로 재사용되므로 절대로
            # 덮어쓰지 않는다 (frame_id/position/orientation 모두).
            # GRASP_READY 는 ik_solution 에 무관하게 +π/2 delta 회전.
            if self == MoveToActionState.GRASP_READY:
                base_rotation = np.pi / 2.0
            else:
                base_rotation = ik_solution[Joint.BASE_ROTATION]
            r = quaternion_about_axis(base_rotation, [0, 0, 1])
            rotation_pose = PoseStamped()
            rotation_pose.header.stamp = Time()  # latest
            rotation_pose.header.frame_id = "base_link"
            rotation_pose.pose.orientation = Quaternion(
                x=r[0], y=r[1], z=r[2], w=r[3]
            )

            joints_for_velocity_control += [Joint.BASE_ROTATION]
            # BASE_TRANSLATION 은 TRANSLATE_BASE 전용 스칼라이므로 fixed
            # joint position 으로 넘기지 않는다.
            joint_position_overrides.update(
                {
                    joint: position
                    for joint, position in ik_solution.items()
                    if joint not in (
                        Joint.BASE_ROTATION,
                        Joint.BASE_TRANSLATION,
                    )
                }
            )
            # GRASP_READY 진입 시 그리퍼가 nav 카메라 정면을 향하도록
            # frontview 자세(WRIST_YAW) 를 함께 적용.
            if self == MoveToActionState.GRASP_READY:
                joint_position_overrides.update(get_frontview_configuration())
            return controller.rotate_base_to_goal_pose(
                goal=rotation_pose,
                articulated_joints=joints_for_velocity_control,
                termination=TerminationCriteria.ZERO_VEL,
                joint_position_overrides=joint_position_overrides,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )

        # Create the motion executor
        if len(joints_for_velocity_control) > 0:
            return controller.move_to_ee_pose_inverse_jacobian(
                goal=goal_pose,
                articulated_joints=joints_for_velocity_control,
                termination=TerminationCriteria.ZERO_VEL,
                joint_position_overrides=joint_position_overrides,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )
        if len(joints_for_position_control) > 0:
            return controller.move_to_joint_positions(
                joint_positions=joints_for_position_control,
                velocity_overrides=velocity_overrides,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                success_callback=success_callback_temp,
            )
        return None
