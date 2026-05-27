"""내비게이션 카메라 뷰 기반 헤드 조준(pan/tilt) 헬퍼 함수 모음."""

from typing import Optional, Tuple

import cv2
import numpy as np

from nrc_web_teleop_helpers.constants import (
    DEPTH_REF,
    NAV_CAMERA_K,
    NAV_CAMERA_D,
    NAV_IMAGE_WIDTH,
    NAV_IMAGE_HEIGHT,
    NAV_OPTICAL_FROM_LINK,
)
from nrc_web_teleop_helpers.da.utils import get_pred_depth

# head_tilt 관절 가동 범위 (output/stretch.urdf, joint_head_tilt 기준, rad)
HEAD_TILT_MIN = -1.53
HEAD_TILT_MAX = 0.79


# 선택한 픽셀을 내비게이션 카메라 뷰 중앙으로 가져오는 pan/tilt 각도 변화량 계산
def pan_tilt_offset_to_center(
    nav_scaled_u: float,
    nav_scaled_v: float,
) -> Tuple[float, float]:
    """
    선택한 픽셀을 내비게이션 카메라 뷰 중앙에 오도록 하는 데 필요한
    head pan / tilt 각도 변화량(라디안)을 계산한다.

    픽셀을 왜곡 보정해 정규화 ray (x_n, y_n, 1)로 만든 뒤, 그 ray를
    카메라 광축 (0, 0, 1)에 정렬시키는 회전을 pan(수직축 회전) →
    tilt(수평축 회전) 순서로 분해한다. 카메라 회전은 ray 방향을 ray
    방향으로 매핑하므로 이 계산에는 깊이값이 필요 없다.

    부호 규약은 move_base_to_point.py / move_gripper_to_point.py와 같다.
    즉 반환값을 현재 head pan / tilt 관절값에 그대로 더하면 된다.

    매개변수
    --------
    nav_scaled_u, nav_scaled_v : float
        내비게이션 카메라 뷰상의 정규화 픽셀 좌표. 각각 [0, 1] 범위이며
        좌상단이 (0, 0), 우하단이 (1, 1)이다 (웹 teleop의 scaled_u/scaled_v).

    반환값
    ------
    Tuple[float, float]
        (delta_pan, delta_tilt) 라디안. 현재 head pan / tilt 관절값에
        더하면 해당 픽셀이 화면 중앙에 오도록 카메라가 회전한다. 픽셀이
        이미 정확히 중앙에 있으면 (0.0, 0.0).
    """
    # 정규화 좌표를 실제 픽셀 좌표로 변환 (intrinsic은 픽셀 단위로 보정됨).
    pixel_uv = np.array(
        [[[nav_scaled_u * NAV_IMAGE_WIDTH, nav_scaled_v * NAV_IMAGE_HEIGHT]]],
        dtype=np.float64,
    )

    # 왜곡 보정 → 광학 프레임의 정규화 ray (x_n, y_n, 1).
    undistorted = cv2.undistortPoints(pixel_uv, NAV_CAMERA_K, NAV_CAMERA_D)
    x_n = float(undistorted[0, 0, 0])
    y_n = float(undistorted[0, 0, 1])

    # ray를 광축에 정렬: pan으로 x 성분을 0으로, 이어서 tilt로 y 성분을 0으로.
    # tilt 분모의 sqrt(x_n^2 + 1)은 pan 회전 뒤 늘어난 전방(z) 성분을 반영한다.
    delta_pan = -np.arctan2(x_n, 1.0)
    delta_tilt = -np.arctan2(y_n, np.sqrt(x_n * x_n + 1.0))

    return delta_pan, delta_tilt


# 클릭 픽셀의 3D 좌표를 base_link 프레임에서 m 단위로 반환. 실패 시 None.
def compute_click_point_in_base(
    nav_image: np.ndarray,
    scaled_u: float,
    scaled_v: float,
    T_base_from_nav_link: np.ndarray,
) -> Optional[np.ndarray]:
    """
    클릭 픽셀(scaled_u, scaled_v)을 base_link 프레임의 3D 점(m)으로 환산.

    1. nav RGB → `get_pred_depth(rgb, ref_rcz=DEPTH_REF)` 로 metric depth(mm,
       내부에서 m 로 변환).
    2. 클릭 픽셀을 `undistortPoints` 로 정규화 ray (x_n, y_n, 1) 로 변환,
       3D 점(광학 프레임, m): (x_n·z, y_n·z, z).
    3. `NAV_OPTICAL_FROM_LINK` 전치로 `link_head_nav_cam` URDF 프레임으로 변환.
    4. `T_base_from_nav_link` (4x4, m) 로 base_link 프레임으로 변환.

    매개변수
    --------
    T_base_from_nav_link : (4, 4) ndarray
        base_link ← link_head_nav_cam 동차 변환. m 단위 (URDF/TF 표준).
        호출자가 `tf_lookup_matrix(BASE_LINK, NAV_CAM_LINK)` 로 얻는다.
    """
    h, w = nav_image.shape[:2]
    pred_depth = get_pred_depth(nav_image, ref_rcz=DEPTH_REF)

    px = int(np.clip(scaled_u * w, 0, w - 1))
    py = int(np.clip(scaled_v * h, 0, h - 1))
    # get_pred_depth 는 mm(uint16) 로 반환되므로 m 로 변환.
    z = float(pred_depth[py, px]) / 1000.0
    if z <= 0.0:
        return None

    pixel_uv = np.array([[[scaled_u * w, scaled_v * h]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pixel_uv, NAV_CAMERA_K, NAV_CAMERA_D)
    x_n = float(undistorted[0, 0, 0])
    y_n = float(undistorted[0, 0, 1])

    # 광학 프레임(m) → URDF link 프레임(m). NAV_OPTICAL_FROM_LINK 는
    # link → optical 회전이므로 역(전치)을 곱한다.
    p_optical = np.array([x_n * z, y_n * z, z], dtype=np.float64)
    p_link = NAV_OPTICAL_FROM_LINK.T @ p_optical

    # link → base_link.
    p_link_h = np.array([p_link[0], p_link[1], p_link[2], 1.0])
    p_base = T_base_from_nav_link @ p_link_h
    return p_base[:3].astype(np.float64), z


# nav 카메라 tilt 변경 후 같은 점이 찍히는 nav_scaled_v 위치 계산
def nav_scaled_v_after_tilt(
    nav_scaled_u: float,
    nav_scaled_v: float,
    initial_tilt: float,
    delta_tilt: float,
) -> float:
    """
    head_tilt를 initial_tilt에서 delta_tilt만큼 바꿨을 때 같은 점이
    이동하는 nav_scaled_v 값을 반환한다.

    new_tilt = initial_tilt + delta_tilt는 head_tilt 관절 한계로 clamp되어
    실제 적용된 변화량만 반영된다. 카메라 회전은 ray 방향만 바꾸므로
    깊이값은 불필요하다. 반환값이 [0, 1] 밖이면 점이 화면을 벗어난 것이고,
    점이 카메라 뒤로 넘어가면 nan을 반환한다.
    """
    # 관절 한계로 clamp한 뒤 실제 적용되는 tilt 변화량.
    new_tilt = min(max(initial_tilt + delta_tilt, HEAD_TILT_MIN), HEAD_TILT_MAX)
    applied = new_tilt - initial_tilt

    # 픽셀 → 왜곡 보정한 정규화 ray (x_n, y_n, 1).
    pixel_uv = np.array(
        [[[nav_scaled_u * NAV_IMAGE_WIDTH, nav_scaled_v * NAV_IMAGE_HEIGHT]]],
        dtype=np.float64,
    )
    x_n, y_n = cv2.undistortPoints(pixel_uv, NAV_CAMERA_K, NAV_CAMERA_D)[0, 0]

    # tilt 회전(광학 x축)을 ray에 적용. 부호 규약은 pan_tilt_offset_to_center와 같다.
    c, s = np.cos(applied), np.sin(applied)
    ray = np.array([x_n, c * y_n + s, c - s * y_n])
    if ray[2] <= 0.0:
        return float("nan")

    # 회전된 ray를 다시 nav 이미지에 투영 → nav_scaled_v.
    uv, _ = cv2.projectPoints(
        ray.reshape(1, 3), np.zeros(3), np.zeros(3), NAV_CAMERA_K, NAV_CAMERA_D
    )
    return float(uv[0, 0, 1]) / NAV_IMAGE_HEIGHT
