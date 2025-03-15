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

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
CLEAR_CACHE_RATE = 50

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
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    
    cap = cv2.VideoCapture(0)

    fps_array = []

    frame_count = 0

    while cap.isOpened():

        frame_count += 1

        if len(fps_array) > 750:
            break

        start_time = time.time()

        ret, raw_image = cap.read()
        if not ret:
            print('No frame')
            continue

        raw_image = cv2.resize(raw_image, (0, 0), fx=0.5, fy=0.5)
        
        with torch.no_grad():
            depth = depth_anything.infer_image(raw_image, args.input_size)
        
        if DEVICE == 'mps':
            torch.mps.synchronize()
            if frame_count % CLEAR_CACHE_RATE == 0:
                gc.collect()
                torch.mps.empty_cache()
        elif DEVICE == 'cuda':
            torch.cuda.synchronize()
            if frame_count % CLEAR_CACHE_RATE == 0:
                gc.collect()
                torch.cuda.empty_cache()

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        cv2.imshow('Depth', depth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        fps_array.append(fps)
        print(f'FPS: {round(fps, 2):.4f} | Driver Mem: {round(torch.mps.driver_allocated_memory()/1000/1000, 2)}MB | Current Mem: {round(torch.mps.current_allocated_memory()/1000/1000, 2)}MB')

    cap.release()
    cv2.destroyAllWindows()

    # create a plot of the fps array
    plt.plot(fps_array)
    plt.show()
