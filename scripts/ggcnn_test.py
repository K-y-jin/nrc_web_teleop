import cv2
import os
from nrc_web_teleop_helpers.ggcnn import utils
from nrc_web_teleop_helpers.ggcnn.ggcnn_torch import predict
from nrc_web_teleop_helpers.da.utils import get_pred_depth
import numpy as np

def get_ggcnn_results(depth_image, sub_dir: str = "test"):
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
    # utils.save_image(q_out, "points_out", sub_dir=sub_dir)
    # utils.save_image(ang_out, "ang_out", sub_dir=sub_dir)
    # utils.save_image(width_out, "width_out", sub_dir=sub_dir)
    # utils.save_image(depth_out, "processed_depth", sub_dir=sub_dir)
    return q_out, ang_out, width_out, depth_out


if __name__ == "__main__":
    # image_dir = "output/samples/head"
    image_dir = "output/samples/gripper"
    rotate_cw90 = image_dir.endswith("head")
    image_names = sorted(
        name for name in os.listdir(image_dir)
        if name.startswith("color_") and name.endswith(".png")
    )
    for name in image_names:
        image_path = os.path.join(image_dir, name)
        print(image_path)
        color_image = cv2.imread(image_path)
        depth_image = get_pred_depth(color_image)
        print(color_image.shape)
        print(depth_image.shape)
        q_out, ang_out, width_out, depth_out = get_ggcnn_results(depth_image)

        # Color image와 오버레이
        if color_image.shape[:2] != q_out.shape[:2]:
            color_image = cv2.resize(color_image, (q_out.shape[1], q_out.shape[0]))
        q_color = cv2.applyColorMap(q_out, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color_image, 0.5, q_color, 0.5, 0)

        # Depth image와 오버레이
        # if depth_image.shape[:2] != q_out.shape[:2]:
        #     depth_image = cv2.resize(depth_image, (q_out.shape[1], q_out.shape[0]))
        # depth_bgr = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        # q_color = cv2.applyColorMap(q_out, cv2.COLORMAP_JET)
        # overlay = cv2.addWeighted(depth_bgr, 0.5, q_color, 0.5, 0)
        if rotate_cw90:
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("q_out overlay", overlay)
        cv2.waitKey(2000)
    cv2.destroyAllWindows()
