from __future__ import annotations  # Required for type hinting a class within itself

# Standard imports
from enum import Enum
from builtin_interfaces.msg import Time
from typing import Callable, Dict, Generator, List, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from nrc_web_teleop_helpers.move_to_pregrasp_state import MoveToPregraspState
from std_msgs.msg import Header
# Local imports
from .constants import (
    Joint,
    get_gripper_configuration,
    get_pregrasp_wrist_configuration,
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
    Determine the goal point is reachable.
    First, the robot stow the arm.
    Second, the robot rotate its base heading to the goal point, keeping the camera view fixed.
    Third, the robot move its base to the goal point.

    The below states can be strung together to form a state machine that moves the robot
    to a pregrasp position. The general principle we follow is that the robot should only
    rotate its base and move the lift when its arm is within the base footprint of the robot
    (i.e., the arm length is fully in and the wrist is stowed).
    """

    # Stow the arm until the gripper would collide with the base if the gripper were vertically down.
    STOW_ARM_LENGTH_PARTIAL = -1
    # Stow the arm fully.
    STOW_ARM_LENGTH_FULL = 0
    STOW_WRIST = 1
    STOW_ARM_LIFT = 2
    ROTATE_BASE = 3
    HEAD_PAN = 4 
    LIFT_ARM = 5
    MOVE_WRIST = 6
    LENGTHEN_ARM = 7
    HEAD_TILT = 8
    ROTATE_BASE_SIMPLE = 9
    LENGTHEN_ARM_SIMPLE = 10
    MOVE_BASE = 11
    TERMINAL = 12
    @staticmethod
    def get_state_for_pregrasp_action(
        horizontal_grasp: bool,
        init_lift_near_base: bool,
        goal_lift_near_base: bool,
        init_length_near_mast: bool,
    ) -> List[List[MoveToActionState]]:
        """
        Get the default state for the move to action state.

        Parameters
        ----------
        horizontal_grasp: Whether the robot will be grasping the object horizontally
            (True) or vertically (False).
        init_lift_near_base: Whether the robot's arm is near the base at the start.
        goal_lift_near_base: Whether the robot's arm should be near the base at the end.
        init_length_near_mast: Whether the robot's arm length is near the mast at the start.

        Returns
        -------
        List[List[MoveToActionState]]: The default state machine. Each list of states
            (axis 0) will be executed sequentially. Within a list of states (axis 1), the
            states will be executed in parallel.
        """
        states = []
        # If the current arm lift is near the base, and the length is not already near the mast,
        # move the arm to the stow height before fully stowing the arm length. This is to account
        # for the case where the wrist is vertically down and may collide with the base.
        if init_lift_near_base:
            if not init_length_near_mast:
                states.append([MoveToActionState.STOW_ARM_LENGTH_PARTIAL])
            states.append([MoveToActionState.STOW_ARM_LIFT])
            states.append([MoveToActionState.STOW_ARM_LENGTH_FULL])
        else:
            states.append([MoveToActionState.STOW_ARM_LENGTH_FULL])
        states.append([MoveToActionState.STOW_WRIST])
        # If the goal arm lift is near the base and we haven't already stowed the arm lift, stow the arm lift.
        if goal_lift_near_base and MoveToActionState.STOW_ARM_LIFT not in states:
            states.append([MoveToActionState.STOW_ARM_LIFT])
        states.append([MoveToActionState.ROTATE_BASE, MoveToActionState.HEAD_PAN])
        # If the goal is near the base and we're doing a vertical grasp, lengthen the arm before deploying the wrist.
        if goal_lift_near_base:
            states.append([MoveToActionState.LENGTHEN_ARM])
            states.append([MoveToActionState.MOVE_WRIST])
            states.append([MoveToActionState.LIFT_ARM])
        else:
            states.append([MoveToActionState.LIFT_ARM])
            states.append([MoveToActionState.MOVE_WRIST])
            states.append([MoveToActionState.LENGTHEN_ARM])
        states.append([MoveToActionState.TERMINAL])

        return states
    @staticmethod
    def get_state_for_move_base_to_point() -> List[List[MoveToActionState]]:
        states = []
        states.append([MoveToActionState.STOW_ARM_LENGTH_FULL])
        states.append([MoveToActionState.STOW_WRIST, MoveToActionState.STOW_ARM_LIFT])
        states.append([MoveToActionState.HEAD_TILT])
        states.append([MoveToActionState.ROTATE_BASE_SIMPLE, MoveToActionState.HEAD_PAN])
        states.append([MoveToActionState.MOVE_BASE])
        states.append([MoveToActionState.TERMINAL])
        return states

    def get_state_for_move_gripper_to_point() -> List[List[MoveToActionState]]:
        states = []
        states.append([MoveToActionState.STOW_ARM_LENGTH_FULL])
        states.append([MoveToActionState.STOW_WRIST, MoveToActionState.STOW_ARM_LIFT])
        states.append([MoveToActionState.HEAD_TILT])
        states.append([MoveToActionState.ROTATE_BASE_SIMPLE, MoveToActionState.HEAD_PAN])
        states.append([MoveToActionState.LIFT_ARM])
        states.append([MoveToActionState.MOVE_WRIST])
        states.append([MoveToActionState.LENGTHEN_ARM_SIMPLE])
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
        get_cartesian_mask = None
        error_callback_temp = None
        success_callback_temp = None

        # Configure the parameters depending on the state
        if self == MoveToActionState.TERMINAL:
            return None
        elif self == MoveToActionState.STOW_ARM_LENGTH_FULL:
            joints_for_position_control.update(get_stow_configuration([Joint.ARM_L0]))
        elif self == MoveToActionState.STOW_ARM_LENGTH_PARTIAL:
            joints_for_position_control.update(
                get_stow_configuration([Joint.ARM_L0], partial=True)
            )
        elif self == MoveToActionState.STOW_ARM_LIFT:
            joints_for_position_control.update(get_stow_configuration([Joint.ARM_LIFT]))
        elif self == MoveToActionState.STOW_WRIST:
            joints_for_position_control.update(
                get_stow_configuration(Joint.get_wrist_joints() + [Joint.GRIPPER_LEFT])
            )
        elif self == MoveToActionState.ROTATE_BASE:
            joints_for_velocity_control += [Joint.BASE_ROTATION]
            joint_position_overrides.update(
                {
                    joint: position
                    for joint, position in ik_solution.items()
                    if joint != Joint.BASE_ROTATION
                }
            )
            joint_position_overrides.update(
                get_pregrasp_wrist_configuration(horizontal_grasp)
            )

            # Care about yaw error when error is large, but for final positioning,
            # care about x error. This is because our yaw is slightly off, so if
            # we care about both yaw and x for final positioning, it will stop
            # at a slightly off position.
            def get_cartesian_mask(err: npt.NDArray[float]):
                cartesian_mask = np.array([True, False, False, False, False, False])
                err_yaw = err[5]
                if np.abs(err_yaw) > np.pi / 4.0:
                    cartesian_mask[5] = True
                return cartesian_mask

        elif self == MoveToActionState.HEAD_PAN:
            joints_for_position_control[Joint.HEAD_PAN] = ik_solution[Joint.HEAD_PAN] # The head should rotate in the opposite direction of the base, to keep the field of view roughly the same
            # Cap the head pan at the base's max rotation speed, so the base and head pan
            # camera roughly track each other.
            velocity_overrides[Joint.HEAD_PAN] = controller.joint_vel_abs_lim[
                Joint.BASE_ROTATION
            ][1]
        elif self == MoveToActionState.LIFT_ARM:
            if ik_solution[Joint.ARM_LIFT] is None:
                ik_solution[Joint.ARM_LIFT] = err_callback[1]()
            joints_for_position_control[Joint.ARM_LIFT] = ik_solution[Joint.ARM_LIFT]
        elif self == MoveToActionState.MOVE_WRIST:
            # Simulation에서 gripper position control이 잘 안 됨.
            # joints_for_position_control.update(get_gripper_configuration(closed=False))
            joints_for_position_control.update(
                get_pregrasp_wrist_configuration(horizontal_grasp)
            )
        elif self == MoveToActionState.LENGTHEN_ARM:
            joints_for_position_control[Joint.ARM_L0] = ik_solution[Joint.ARM_L0]
        elif self == MoveToActionState.LENGTHEN_ARM_SIMPLE:
            if ik_solution[Joint.ARM_L0] is None:
                ik_solution[Joint.ARM_L0] = err_callback[2]() - 0.5 # gripper length 고려 
            joints_for_position_control[Joint.ARM_L0] = ik_solution[Joint.ARM_L0]
        elif self == MoveToActionState.HEAD_TILT:
            joints_for_position_control[Joint.HEAD_TILT] = ik_solution[Joint.HEAD_TILT] # 33deg down
            velocity_overrides[Joint.HEAD_TILT] = 0.5
        elif self == MoveToActionState.MOVE_BASE:
            error_callback_temp = err_callback[0]
            # move base to the goal point
            goal_pose.header.stamp = Time() # controller.node.get_clock().now().to_msg()
            goal_pose.header.frame_id = "base_link"
            goal_pose.pose.position = Point(x=ik_solution[Joint.BASE_TRANSLATION], y=0.0, z=0.0)
            goal_pose.pose.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
            # Joint.BASE_TRANSLATION is not included in the controllable joints
            # So, we cannot use the ZERO_VEL termination criteria
            return controller.translate_base_to_goal_pose(
                goal=goal_pose,
                termination=TerminationCriteria.ZERO_ERR,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
                err_callback=error_callback_temp,
                success_callback=success_callback_temp,
            )
        elif self == MoveToActionState.ROTATE_BASE_SIMPLE:
            success_callback_temp = success_callback[0]
            goal_pose.header.stamp = Time() # latest
            goal_pose.header.frame_id = "base_link"

            goal_pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
            base_rotation = ik_solution[Joint.BASE_ROTATION]
            r = quaternion_about_axis(base_rotation, [0, 0, 1])
            goal_pose.pose.orientation = Quaternion(x=r[0], y=r[1], z=r[2], w=r[3])

            joints_for_velocity_control += [Joint.BASE_ROTATION]
            joint_position_overrides.update(
                {
                    joint: position
                    for joint, position in ik_solution.items()
                    if joint != Joint.BASE_ROTATION
                }
            )
            return controller.rotate_base_to_goal_pose(
                goal=goal_pose,
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
                get_cartesian_mask=get_cartesian_mask,
                success_callback=success_callback_temp,
            )
        if len(joints_for_position_control) > 0:
            return controller.move_to_joint_positions(
                joint_positions=joints_for_position_control,
                velocity_overrides=velocity_overrides,
                timeout_secs=timeout_secs,
                check_cancel=check_cancel,
            )
        return None