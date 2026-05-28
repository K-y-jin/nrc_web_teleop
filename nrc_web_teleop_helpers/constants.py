"""
This file contains various constants relevant to stretch:
joint names, hard-coded configurations, speed profiles, control modes,
frames of reference, etc.
"""

# Standard Imports
from enum import Enum
from typing import Dict, List

# Third-Party Imports
import numpy as np


class Joint(Enum):
    """
    Joint names of Stretch.
    """

    BASE_TRANSLATION = "joint_mobile_base_translation"
    BASE_ROTATION = "joint_mobile_base_rotation"
    ARM_LIFT = "joint_lift"
    ARM_L0 = "joint_arm_l0"
    ARM_L1 = "joint_arm_l1"
    ARM_L2 = "joint_arm_l2"
    ARM_L3 = "joint_arm_l3"
    COMBINED_ARM = "joint_arm"
    WRIST_EXTENSION = "wrist_extension"
    WRIST_YAW = "joint_wrist_yaw"
    WRIST_PITCH = "joint_wrist_pitch"
    WRIST_ROLL = "joint_wrist_roll"
    GRIPPER_RIGHT = "joint_gripper_finger_right"
    GRIPPER_LEFT = "joint_gripper_finger_left"
    RIGHT_WHEEL = "joint_right_wheel"
    LEFT_WHEEL = "joint_left_wheel"
    HEAD_PAN = "joint_head_pan"
    HEAD_TILT = "joint_head_tilt"

    @staticmethod
    def get_arm_joints():
        """
        Get the list of telescoping arm joints.
        """
        return [Joint.ARM_L0, Joint.ARM_L1, Joint.ARM_L2, Joint.ARM_L3]

    @staticmethod
    def get_wrist_joints():
        """
        Get the list of wrist joints.
        """
        return [Joint.WRIST_YAW, Joint.WRIST_PITCH, Joint.WRIST_ROLL]


class Frame(Enum):
    """
    Key frame names of reference for Stretch.
    """

    BASE_LINK = "base_link"
    END_EFFECTOR_LINK = "link_grasp_center"
    LIFT_LINK = "link_lift"
    L0_LINK = "link_arm_l0"
    WRIST_PITCH_LINK = "link_wrist_pitch"
    ODOM = "odom"
    CAMERA_COLOR_FRAME = "camera_color_frame"
    CAMERA_DEPTH_FRAME = "camera_depth_frame"
    HEAD_PAN_LINK = "link_head_pan"
    NAV_CAM_LINK = "link_head_nav_cam"


# Navigation camera calibration for the /navigation_camera/image_raw/rotated/
# compressed topic. Shared by move_base_to_point.py, move_gripper_to_point.py
# and depth_helper.py; update here only.
NAV_CAMERA_K = np.array([
    [389.93427177,   0.0,          278.48238639],
    [  0.0,          389.0470267,  363.26460058],
    [  0.0,            0.0,          1.0         ],
])
NAV_CAMERA_D = np.array([
    -0.29785045, 0.09304576, -0.00081603, -0.00068536, -0.01281908
])

# Navigation image size (px) of the /navigation_camera/image_raw/rotated/
# compressed topic. Matches NAV_CAMERA_K's principal point (cx≈278, cy≈363).
NAV_IMAGE_WIDTH = 600
NAV_IMAGE_HEIGHT = 800

# Rotation from the link_head_nav_cam URDF frame to the navigation camera's
# optical frame (OpenCV convention: +z forward, +x right, +y down) that
# NAV_CAMERA_K / NAV_CAMERA_D are calibrated in. link_head_nav_cam is +z
# forward, +x down, +y left, so a valid optical frame is a 90 deg roll about
# the forward axis.
# NOTE: the physical sensor mounting and the ROTATE_90_CCW applied to the
# /navigation_camera/image_raw/rotated topic may add another 90/180/270 deg
# in-plane roll. Verify on-robot and, if the anchor pixel looks rotated or
# mirrored, replace this with one of the other in-plane rolls:
#   Rz(0):   [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#   Rz(90):  [[0, -1, 0], [1, 0, 0], [0, 0, 1]]   (default below)
#   Rz(180): [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
#   Rz(270): [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
NAV_OPTICAL_FROM_LINK = np.array([
    [-1.0,  0.0, 0.0],
    [ 0.0, -1.0, 0.0],
    [ 0.0,  0.0, 1.0],
])


# move_base_to_point.py 거리 미리보기 / 베이스 이동 기준값.
# TILT_READY: 이 head_tilt 각도(rad)에서 nav 카메라에 로봇 자신의 base가 보인다.
# TILT_READY_TOLERANCE: TILT_READY 도달 판정 허용 오차(rad).
# DEPTH_REF: TILT_READY 자세에서 nav 카메라에 보이는 base 기준점 (row, col,
#   depth_mm). get_pred_depth(Depth-Anything) 예측의 metric 스케일을 고정한다.
#   row/col은 nav 이미지(NAV_IMAGE_WIDTH x NAV_IMAGE_HEIGHT) 픽셀.
# TODO: TILT_READY는 실제 로봇에서 base가 화면에 들어오는 각도로 캘리브레이션할 것.
TILT_READY = -45.0 * np.pi / 180.0 # -45.0 deg
TILT_READY_TOLERANCE = 0.05
# row, col 픽셀 + 그 픽셀의 실측 depth(mm). depth_mm 은 get_pred_depth 의
# uint16 스케일과 일치시키기 위해 mm 단위로만 사용되는 내부 상수다. 코드의
# 나머지 부분은 모두 m 단위로 통일되어 있다.
DEPTH_REF = (695, 300, 1084)  # row, col, depth_mm

# Move Gripper 액션: 그리퍼로 물체를 조작하기에 적당한 base ↔ 타깃 거리(m).
# 현재 base ↔ 타깃 거리가 OPTIMAL_DISTANCE 와 BASE_DISTANCE_TOLERANCE 이내면
# RELOCATE_BASE 를 생략, 그렇지 않으면 OPTIMAL_DISTANCE 가 되도록 base 이동.
OPTIMAL_DISTANCE = 0.7  # m
BASE_DISTANCE_TOLERANCE = 0.05  # m
# Move Base 액션: base 자체를 타깃 근처까지 보낼 때, 충돌 회피용으로
# 남겨두는 안전 마진(m). raw 거리 - BASE_MARGIN 만큼 전진한다.
BASE_MARGIN = 0.1  # m

# Ground 판정 — DEPTH_REF 픽셀에서 클릭 픽셀까지 직선상의 pred_depth가
# 스무스하게 증가하는지를 판정할 때, 한 스텝의 깊이 변화가 평균 스텝의
# NAVIGABLE_MAX_STEP_RATIO배를 초과하면 ground가 아닌 것으로 본다.
NAVIGABLE_LINE_SAMPLES = 64
NAVIGABLE_MAX_STEP_RATIO = 3.0


class ControlMode(Enum):
    """
    The control modes for the Stretch Driver.
    """

    POSITION = "position"
    NAVIGATION = "navigation"

    def get_service_name(self):
        """
        Get the service name for switching to this control mode.
        """
        return f"switch_to_{self.value}_mode"


class SpeedProfile(Enum):
    """
    The speed profile to use to get max velocities and accelerations. This
    should correspond to the speed profile in robot params.
    """

    SLOW = "slow"
    DEFAULT = "default"
    FAST = "fast"
    MAX = "max"


def get_stow_configuration(
    joints: List[Joint], partial: bool = False
) -> Dict[Joint, float]:
    """
    Get the joint configuration for stowing the arm.

    Note that in practice, commanding all these joints at the same time can create
    a motion with a larger footprint than desired, resulting in collisions. It is
    typically best practice to first command arm length, then command wrist/gripper,
    then command lift.

    Parameters
    ----------
    joints: The joints return.
    partial: If True, make the arm length stop slightly before the robot base, so that if the
        wrist is vertically down, it won't collide with the base.

    Returns
    -------
    Dict[Joint, float]: The joint configuration.
    """
    retval = {}
    for joint in joints:
        if joint == Joint.ARM_L0:
            retval[joint] = 0.1675 if partial else 0.0
        elif joint == Joint.ARM_LIFT:
            retval[
                joint
            ] = 0.40 # 0 to about 1.1m
             # This is chosen so even when the gripper is pointing down, it doesn't hit the base.
        elif joint == Joint.WRIST_YAW:
            retval[joint] = 0.7854 # 3.19579  # Should match src/shared/util.tsx
        elif joint == Joint.WRIST_PITCH:
            retval[joint] = -0.497 # Should match src/shared/util.tsx
        elif joint == Joint.WRIST_ROLL:
            retval[joint] = 0.0  # Should match src/shared/util.tsx
        elif joint == Joint.GRIPPER_LEFT:
            retval[
                joint
            ] = 0.0  # close gripper when stowed. An open gripper can sometimes get caught in the mast.
    return retval


def get_pred_ready_configuration() -> Dict[Joint, float]:
    """
    Depth prediction(`get_pred_depth` + `DEPTH_REF`) 를 위해 base 가 카메라에
    선명히 잡히고, base 회전/이동 시 주변 충돌이 최소화되도록 팔/손목을
    수납한 자세. `move_base_to_point` 액션의 PRED_READY state 에서 사용.
    """
    return {
        Joint.ARM_L0: 0.0,
        Joint.ARM_LIFT: 0.9,
        Joint.WRIST_YAW: 0.7854,  # 45 deg nav 카메라에 안 잡힘.
        Joint.WRIST_PITCH: -0.0,
        Joint.WRIST_ROLL: 0.0,
    }


def get_frontview_configuration() -> Dict[Joint, float]:
    """
    Frontview grasp 예측 자세. GRASP_READY 의 base +π/2 회전과 함께 적용해
    그리퍼가 nav 카메라 정면을 향하도록 한다.
    """
    return {
        Joint.WRIST_YAW: 0.0,  # 90 deg
    }


def adjust_arm_lift_for_base_collision(
    ik_solution: Dict[Joint, float],
    horizontal_grasp: bool,
) -> None:
    """
    Modifies the arm lift to avoid collision with the base.

    Parameters
    ----------
    ik_solution: The current IK solution.
    horizontal_grasp: Whether the robot will be grasping the object horizontally
    """
    if horizontal_grasp:
        # If the arm length is less than 10cm and the arm lift is less than 11cm,
        # a horizontal wrist will collide with the base. Raise the arm lift.
        if ik_solution[Joint.ARM_L0] < 0.10:
            if ik_solution[Joint.ARM_LIFT] < 0.11:
                ik_solution[Joint.ARM_LIFT] = 0.11
    else:
        # If the arm length is less than 10cm and the arm lift is less than 33cm,
        # a vertical wrist will collide with the base. Raise the arm lift.
        if ik_solution[Joint.ARM_L0] < 0.10:
            if ik_solution[Joint.ARM_LIFT] < 0.33:
                ik_solution[Joint.ARM_LIFT] = 0.33


def get_pregrasp_wrist_configuration(horizontal: bool) -> Dict[Joint, float]:
    """
    Get the wrist rotation for the pregrasp position.

    TODO: Add the option to specify 0 or 90 degree roll.

    Parameters
    ----------
    horizontal_grasp: Whether the pregrasp position is horizontal.

    Returns
    -------
    Dict[Joint, float]: The joint configuration.
    """
    if horizontal:
        return {
            Joint.WRIST_YAW: 0.0,
            Joint.WRIST_PITCH: 0.0,
            Joint.WRIST_ROLL: 0.0,
        }
    else:
        return {
            Joint.WRIST_YAW: 0.0,
            Joint.WRIST_PITCH: -np.pi / 2.0,
            Joint.WRIST_ROLL: 0.0,
        }


def get_gripper_configuration(closed: bool) -> Dict[Joint, float]:
    """
    Get the gripper configuration.

    Parameters
    ----------
    closed: Whether the gripper is closed. If false, returns the configuration
        where the gripper is fully open.

    Returns
    -------
    Dict[Joint, float]: The joint configuration.
    """
    return {
        # We only need to command one gripper joint, as the other is coupled.
        Joint.GRIPPER_LEFT: 0.0
        if closed
        else 0.55 #0.84, # -0.6에서 0.6 가능인데. 양쪽을 합하는 것인가?
    }
