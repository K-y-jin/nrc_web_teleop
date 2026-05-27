"""Helpers for moving 2D points between the navigation and RealSense cameras."""

from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from nrc_web_teleop_helpers.constants import NAV_OPTICAL_FROM_LINK


# 내비게이션 카메라 이미지의 픽셀을 RealSense aligned-depth 이미지의 픽셀로 변환하는 함수.
def transform_xy_from_nav_to_realsense(
    nav_u: float,
    nav_v: float,
    nav_depth_mm: float,
    nav_K: npt.ArrayLike,
    nav_D: npt.ArrayLike,
    realsense_K: npt.ArrayLike,
    T_color_from_navlink: npt.NDArray,
    nav_optical_from_link: npt.NDArray = NAV_OPTICAL_FROM_LINK,
    realsense_size: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    ``MoveGripperToPointNode.align_realsense_ref_to_navigation``의 역변환이다.
    
    내비게이션 픽셀을 왜곡 보정해 정규화 ray로 만들고, ``nav_depth_mm``를
    이용해 3D 점으로 펼친 뒤, 내비게이션 카메라 링크로 회전시키고, TF를 통해
    ``camera_color_optical_frame``으로 변환한 다음, RealSense intrinsic으로
    투영한다. RealSense depth 스트림은 color 스트림에 정합(aligned)되어
    있으므로, 반환된 픽셀은 aligned-depth 이미지에도 그대로 사용할 수 있다.

    매개변수
    --------
    nav_u, nav_v : float
        내비게이션 카메라 이미지상의 픽셀 좌표 (열, 행).
    nav_depth_mm : float
        ``(nav_u, nav_v)``에서 내비게이션 카메라 광축(+z) 방향 깊이값(mm).
        예: ``get_pred_depth``의 출력값. 0보다 커야 한다.
    nav_K, nav_D : array_like
        내비게이션 카메라의 intrinsic 행렬(3x3 또는 길이 9의 1차원 배열)과
        왜곡 계수.
    realsense_K : array_like
        RealSense color / aligned-depth intrinsic 행렬(3x3 또는 길이 9).
        예: ``/camera/aligned_depth_to_color/camera_info``.
    T_color_from_navlink : npt.NDArray
        ``link_head_nav_cam``의 점을 ``camera_color_optical_frame``으로
        변환하는 4x4 동차 변환 행렬. ``tf2_get_transform(
        target_frame="camera_color_optical_frame",
        source_frame="link_head_nav_cam")`` 결과에
        ``transform_stamped_to_matrix``를 적용해 얻는다.
    nav_optical_from_link : npt.NDArray
        ``link_head_nav_cam``에서 내비게이션 카메라 광학 프레임으로의 3x3
        회전 행렬. 기본값은 문서화된 Rz(90) 규약.
    realsense_size : Optional[Tuple[int, int]]
        RealSense 이미지의 ``(너비, 높이)`` (선택). 지정하면 투영된 픽셀이
        이미지 범위를 벗어날 경우 ``None``을 반환한다.

    반환값
    ------
    Optional[Tuple[float, float, float]]
        RealSense aligned-depth 이미지상의 ``(u, v, depth_mm)``. 여기서
        ``depth_mm``은 RealSense 광축 방향의 깊이값이다. 깊이값이 유효하지
        않거나, 점이 두 카메라 중 하나의 뒤쪽에 위치하거나,
        ``realsense_size`` 범위를 벗어나면 ``None``을 반환한다. ``u`` / ``v``는
        float이므로 이미지 인덱싱에 쓰기 전에 반올림해야 한다.
    """
    if nav_depth_mm <= 0.0 or not np.isfinite(nav_depth_mm):
        return None

    nav_K = np.asarray(nav_K, dtype=np.float64).reshape(3, 3)
    nav_D = np.asarray(nav_D, dtype=np.float64).reshape(-1)
    realsense_K = np.asarray(realsense_K, dtype=np.float64).reshape(3, 3)

    # Undistort the navigation pixel into a normalized ray (x_n, y_n, 1) in the
    # navigation camera's optical frame.
    pixel_uv = np.array([[[nav_u, nav_v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pixel_uv, nav_K, nav_D)
    x_n = float(undistorted[0, 0, 0])
    y_n = float(undistorted[0, 0, 1])

    # Push the ray out to its 3D point at the given optical-axis depth (meters).
    z_nav = nav_depth_mm / 1000.0
    p_opt = np.array([x_n * z_nav, y_n * z_nav, z_nav])

    # Optical frame -> link_head_nav_cam (a pure rotation, so inverse == T).
    p_navlink = nav_optical_from_link.T @ p_opt

    # link_head_nav_cam -> camera_color_optical_frame.
    p_color = (T_color_from_navlink @ np.append(p_navlink, 1.0))[:3]
    z_color = float(p_color[2])
    if z_color <= 0.0:
        # Point is behind the RealSense camera; not observable.
        return None

    # Project onto the RealSense image (pinhole; the aligned-depth stream is
    # treated as undistorted, matching align_realsense_ref_to_navigation).
    f_x, f_y = realsense_K[0, 0], realsense_K[1, 1]
    c_x, c_y = realsense_K[0, 2], realsense_K[1, 2]
    u_rs = f_x * p_color[0] / z_color + c_x
    v_rs = f_y * p_color[1] / z_color + c_y

    if realsense_size is not None:
        width, height = realsense_size
        if not (0.0 <= u_rs < width and 0.0 <= v_rs < height):
            return None

    return float(u_rs), float(v_rs), z_color * 1000.0
