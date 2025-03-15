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
FRAME_LIMIT = 100
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    # parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./metric_test_files')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vits.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    
    cap = cv2.VideoCapture(1)

    fps_array = []
    min_distance_array = []
    max_distance_array = []
    mean_distance_array = []

    frame_count = 0

    while cap.isOpened():

        frame_count += 1

        if len(fps_array) > FRAME_LIMIT:
            break

        start_time = time.time()

        ret, raw_image = cap.read()
        if not ret:
            print('No frame')
            continue

        # raw_image = cv2.resize(raw_image, (0, 0), fx=0.5, fy=0.5)
        
        with torch.no_grad():
            depth = depth_anything.infer_image(raw_image, args.input_size)
            print(f'Min: {depth.flatten().min():6.2f}m | Max: {depth.flatten().max():6.2f}m')

            min_distance_array.append(depth.flatten().min())
            max_distance_array.append(depth.flatten().max())
            mean_distance_array.append(depth.flatten().mean())
            if args.save_numpy:
                output_path = os.path.join(args.outdir, str(frame_count) + '_raw_depth_meter.npy')
                np.save(output_path, depth)
        
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
        # print(f'FPS: {fps:6.2f} | Driver Mem: {torch.mps.driver_allocated_memory()/1000/1000:6.2f}MB | Current Mem: {torch.mps.current_allocated_memory()/1000/1000:6.2f}MB')

    cap.release()
    cv2.destroyAllWindows()

    # create a plot of the fps array
    # plt.plot(fps_array)
    # plt.show()

    # plt.plot(min_distance_array)
    # plt.show()

    # plt.plot(max_distance_array)
    # plt.show()

    plt.plot(mean_distance_array)
    plt.show()

    print("Average FPS: ", sum(fps_array) / len(fps_array))
