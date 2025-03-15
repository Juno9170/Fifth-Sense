import argparse
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import time
import gc

from depth_anything.dpt import DepthAnythingV2

def extract_depths(frame, model, input_size, mode='base', max_depth=20):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # if mode == 'base':
    #     depth_anything = DepthAnythingV2(**model_configs[encoder])
    #     depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    #     depth_anything = depth_anything.to(device).eval()
    # elif mode == 'metric':
    #     depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    #     depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    #     depth_anything = depth_anything.to(device).eval()

    with torch.no_grad():
        depth = model.infer_image(frame, input_size)
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    return depth