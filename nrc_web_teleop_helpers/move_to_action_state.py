from __future__ import annotations  # Required for type hinting a class within itself

# Standard imports
from enum import Enum
from builtin_interfaces.msg import Time
from typing import Callable, Dict, Generator, List, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Quaternion, PoseStamped

# Local imports
from .constants import (
    Joint,
    get_pred_ready_configuration,
    get_stow_configuration,
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

    # 공통 / Move Gripper 용 상태
    STOW_ARM_LENGTH_FULL = 0
    STOW_WRIST = 1
    STOW_ARM_LIFT = 2
    HEAD_PAN = 4
    HEAD_TILT = 8
    ROTATE_BASE_SIMPLE = 9
    MOVE_BASE = 11
    TERMINAL = 12
    CHECK_AND_LIFT_ARM = 13

    # Move Base 전용 신규 상태 (재설계 후 흐름).
    # PRED_READY: depth prediction 용 자세(get_pred_ready_configuration).
    # PRED_GOAL: 클릭 픽셀 → goal_xyz_in_odom 계산 (execute_callback 에 위임
    #            via success_callback). 모션 없음.
    # ROTATE_BASE: head_pan=0 일 때 nav 카메라가 goal 을 향하도록 base 회전.
    # TRANSLATE_BASE: BASE_MARGIN 고려해 goal_xyz_in_odom 까지 전진/후진.
    PRED_READY = 20
    PRED_GOAL = 21
    ROTATE_BASE = 22
    TRANSLATE_BASE = 23

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
        states = []
        states.append([MoveToActionState.STOW_ARM_LENGTH_FULL])
        states.append([MoveToActionState.STOW_WRIST])
        states.append([MoveToActionState.STOW_ARM_LIFT])
        states.append([MoveToActionState.HEAD_TILT])
        states.append([MoveToActionState.ROTATE_BASE_SIMPLE, MoveToActionState.HEAD_PAN])
        # base를 타깃 근처(팔 길이 남김)까지 전진시켜 팔이 닿는 위치로 이동.
        states.append([MoveToActionState.MOVE_BASE])
        states.append([MoveToActionState.CHECK_AND_LIFT_ARM])
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
        elif self == MoveToActionState.STOW_ARM_LENGTH_FULL:
            joints_for_position_control.update(get_stow_configuration([Joint.ARM_L0]))
        elif self == MoveToActionState.STOW_ARM_LIFT:
            joints_for_position_control.update(get_stow_configuration([Joint.ARM_LIFT]))
        elif self == MoveToActionState.STOW_WRIST:
            joints_for_position_control.update(
                get_stow_configuration(Joint.get_wrist_joints() + [Joint.GRIPPER_LEFT])
            )
        elif self == MoveToActionState.HEAD_PAN:
            joints_for_position_control[Joint.HEAD_PAN] = ik_solution[Joint.HEAD_PAN]
            # Cap the head pan at the base's max rotation speed, so the base and head pan
            # camera roughly track each other.
            velocity_overrides[Joint.HEAD_PAN] = controller.joint_vel_abs_lim[
                Joint.BASE_ROTATION
            ][1]
        elif self == MoveToActionState.CHECK_AND_LIFT_ARM:
            success_callback[0]()  # Head Cam Image Processing
        elif self == MoveToActionState.HEAD_TILT:
            joints_for_position_control[Joint.HEAD_TILT] = ik_solution[Joint.HEAD_TILT]
            velocity_overrides[Joint.HEAD_TILT] = 0.5
        elif self == MoveToActionState.MOVE_BASE or self == MoveToActionState.TRANSLATE_BASE:
            error_callback_temp = err_callback[0]
            # goal_pose는 호출자가 사전에 odom(또는 다른 고정 frame)으로
            # 설정해 둔다. 컨트롤러가 매 iteration 현재 base_link 로 변환해
            # err 를 계산하므로, base가 움직여도 world 기준 목표를 추종한다.
            return controller.translate_base_to_goal_pose(
                goal=goal_pose,
                termination=TerminationCriteria.ZERO_ERR,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )
        elif (
            self == MoveToActionState.ROTATE_BASE_SIMPLE
            or self == MoveToActionState.ROTATE_BASE
        ):
            # 회전용 별도 PoseStamped 를 생성한다. 호출자가 넘긴 goal_pose 는
            # TRANSLATE_BASE 에서 odom 좌표로 그대로 재사용되므로 절대로
            # 덮어쓰지 않는다 (frame_id/position/orientation 모두).
            base_rotation = ik_solution[Joint.BASE_ROTATION]
            r = quaternion_about_axis(base_rotation, [0, 0, 1])
            rotation_pose = PoseStamped()
            rotation_pose.header.stamp = Time()  # latest
            rotation_pose.header.frame_id = "base_link"
            rotation_pose.pose.orientation = Quaternion(
                x=r[0], y=r[1], z=r[2], w=r[3]
            )

            joints_for_velocity_control += [Joint.BASE_ROTATION]
            joint_position_overrides.update(
                {
                    joint: position
                    for joint, position in ik_solution.items()
                    if joint != Joint.BASE_ROTATION
                }
            )
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
