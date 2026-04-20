import argparse
import cv2
import numpy as np
import numpy.typing as npt
import os
import torch

from nrc_web_teleop_helpers.da.depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_model(encoder='vits'):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    pretrained_dir = os.path.join(os.path.dirname(__file__), 'depth_anything_v2', 'pretrained')
    depth_anything.load_state_dict(torch.load(os.path.join(pretrained_dir, f'depth_anything_v2_{encoder}.pth'), map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    return depth_anything
    
model = get_model()

def get_pred_depth(rgb_image: npt.NDArray, ref_rcz=None):
    depth = model.infer_image(rgb_image, input_size=256)
    depth = 1./(depth+1)
    print("pred depth shape: ", depth.shape)

    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535.0
    if ref_rcz is not None:
        r, c, z = ref_rcz
        if depth[r, c] > 0:
            print(z, depth[r, c])
            scale_factor = z / depth[r, c]
        else:
            scale_factor = 0.0
        depth = scale_factor * depth
        print(z, depth[r, c])
        depth = np.clip(depth, 0, 65535)
    depth = depth.astype(np.uint16)
    return depth