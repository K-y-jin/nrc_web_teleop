from __future__ import annotations  # Required for type hinting a class within itself

# Standard imports
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import Header
# Local imports
from .constants import (
    Joint,
    get_stow_configuration,
)
from .stretch_ik_control import (
    MotionGeneratorRetval,
    StretchIKControl,
    TerminationCriteria,
)
from tf_transformations import quaternion_about_axis

class MoveGripperToPointState(Enum):
    """
    Determine the user assistant state.
    First, the robot stow the arm.
    Second, the robot rotate its base heading to the user assistant, keeping the camera view fixed.
    Third, the robot move its base to the user assistant.
    """

    STOW_ARM = 0

    TERMINAL = 8

    @staticmethod
    def get_state_machine(setup_mode: bool = True) -> List[List[MoveGripperToPointState]]:
        states = []
        if setup_mode:
            states.append([MoveGripperToPointState.STOW_ARM])
        states.append([MoveGripperToPointState.TERMINAL])
        return states

    def get_motion_executor(
        self,
        controller: StretchIKControl,
        ik_solution: Dict[Joint, float],
        timeout_secs: float,
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
        goal_pose = PoseStamped(
            header=Header(frame_id="base_link"),
            pose=PoseStamped().pose,
        )

        # Configure the parameters depending on the state
        if self == MoveGripperToPointState.TERMINAL:
            return None
        elif self == MoveGripperToPointState.STOW_ARM:
            joints_for_position_control.update(
                get_stow_configuration([Joint.ARM_L0, Joint.ARM_LIFT, Joint.WRIST_PITCH],
                grip_stuff=True)
            )
        
        # Create the motion executor
        if len(joints_for_velocity_control) > 0:
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
        if len(joints_for_position_control) > 0:
            return controller.move_to_joint_positions(
                joint_positions=joints_for_position_control,
                velocity_overrides=velocity_overrides,
                timeout_secs=timeout_secs,
            )
        return None