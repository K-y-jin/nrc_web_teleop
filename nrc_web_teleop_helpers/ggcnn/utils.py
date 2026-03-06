import cv2, os
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple
import pcl
from cv_bridge import CvBridge
import ros2_numpy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from nrc_web_teleop_helpers.conversions import (
    ros_msg_to_cv2_image,
)
from rclpy.node import Node

def get_depth_image_from_msg(
    depth_msg: Union[CompressedImage],
    cvBridge: CvBridge,
    measurement_min: float = 0.3, 
    measurement_max: float = 3.0,
    )-> npt.NDArray[np.uint8]:
    """
    D435 30 cm 이상부터 측정 가능. 최대 3 m
    D405 7 cm 이상부터 측정 가능. 최대 50 cm
    PointCloud2 메시지 타입인 경우, 측정 불가능(값이 0)인 점은 마스크 아웃되어 이미지 크기가 작음.
    CompressedImage는 encoding=16UC1. 보통 mm
    """
    depth_image = ros_msg_to_cv2_image(depth_msg, cvBridge)
    measurement_min = measurement_min*1000.0
    measurement_max = measurement_max*1000.0
    depth_image = (depth_image - measurement_min)/(measurement_max - measurement_min)*255.0
    depth_image = np.clip(depth_image, 0, 255)
    depth_image = (depth_image).astype(np.uint8)
    return depth_image

def get_rgb_image_from_msg(
    rgb_msg: Union[CompressedImage],
    cvBridge: CvBridge,
    )-> npt.NDArray[np.uint8]:
    return ros_msg_to_cv2_image(rgb_msg, cvBridge)

def save_image(
        image: npt.NDArray[np.uint8],
        filename_prefix: str,
        output_dir: str = 'src/nrc_web_teleop/output',
    ):

    output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f'{filename_prefix}_1.png'
    i = 1
    output_path = os.path.join(output_dir, filename)
    while os.path.exists(output_path):
        filename = f'{filename_prefix}_{i}.png'
        output_path = os.path.join(output_dir, filename)
        i += 1

    cv2.imwrite(output_path, image)