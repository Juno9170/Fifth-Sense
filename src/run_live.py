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

from ultralytics import YOLO
from extract_depths_base import extract_depths
from extract_objects import extract_objects
from depth_anything.dpt import DepthAnythingV2

import sys
sys.path.append('../Gemini')
from imgsearch import analyze_image_with_gemini

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
CAMERA_FOV = 70
CLEAR_CACHE_RATE = 10000000000
FRAME_LIMIT = 20000000
CONF_THRESHOLD = 0.6
N_CLOSEST_OBJECTS = 3
RESIZE_FACTOR = 1
YOLO_MODEL_NAME = 'yolo11s.pt'
GEMINI_THROTTLE = 8 #8s

import socket



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    # parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=350)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
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
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    s = socket.socket()
    print ("Socket successfully created")

    port = 12346

    s.bind(('0.0.0.0', port))
    print ("socket binded to %s" %(port))

    s.listen(5)	 
    print ("socket is listening")

    c, addr = s.accept()
    print ('Got connection from', addr )

    
    cap = cv2.VideoCapture(1)

    fps_array = []

    frame_count = 0

    yolo_model = YOLO(YOLO_MODEL_NAME, verbose=False)

    clear = lambda: os.system('cls')
    cooldown = time.time()

    while cap.isOpened():
        
        c, addr = s.accept()

        mode = c.recv(1024).decode()
        start_time = time.time()

        frame_count += 1

        if len(fps_array) > FRAME_LIMIT:
            break

        ret, raw_image = cap.read()
        if not ret:
            print('No frame')
            continue

        raw_image = cv2.resize(raw_image, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        
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

        # extract depths
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        # Normalize depth values to 0-1 range before applying colormap
        depth = (depth - depth.min()) / (depth.max() - depth.min())


        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # extract objects
        # Perform object detection on the frame    
        results = yolo_model(raw_image)

        boxes_array = []
        labels_array = []
        confidence_array = []

        # Iterate through each detection and get the bounding box, class, and confidence
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if conf < CONF_THRESHOLD:
                continue

            # Get coordinates and convert to int
            x1, y1, x2, y2 = map(int, box)
            # Get the class name using model.names
            label = yolo_model.names[int(cls)]
            # Format label with confidence (optional)
            text = f"{label}: {conf:.2f}"

            # Draw the bounding box on the blank image
            cv2.rectangle(depth, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Determine text size & create a filled rectangle as background for readability
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(depth, (x1, y1 - text_height - baseline),
                          (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
            # Put the label text on the image
            cv2.putText(depth, text, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            boxes_array.append([x1, y1, x2, y2])
            labels_array.append(label)
            confidence_array.append(conf)

        if (len(boxes_array) == 0):
            print("No objects detected")
            continue

        # ------------------------------------------------------------
        # FIND THE CLOSEST N OBJECTS TO THE USER
        # ------------------------------------------------------------
        avg_depths = []
        # use depth to find the closest 5 objects to the user
        # get the depth values of the objects
        for box in boxes_array:
            x1, y1, x2, y2 = box
            object_depth = depth[y1:y2, x1:x2]
            avg_depths.append(np.mean(object_depth))

        # find the N smallest depth values
        closest_depths = np.sort(avg_depths)[:min(N_CLOSEST_OBJECTS, len(avg_depths))]
        # get the indices of the closest depths
        closest_indices = np.argsort(avg_depths)[:min(N_CLOSEST_OBJECTS, len(avg_depths))]

        deg_per_pixel = CAMERA_FOV / np.sqrt(raw_image.shape[1] ** 2 + raw_image.shape[0] ** 2)

        closest_objects = []

        # for each of the closest objects, calculate the distance from the center of the image
        for i in closest_indices:
            x1, y1, x2, y2 = boxes_array[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x_distance_from_center = center_x - raw_image.shape[1] / 2
            y_distance_from_center = center_y - raw_image.shape[0] / 2

            x_distance_from_center_in_degrees = x_distance_from_center * deg_per_pixel
            y_distance_from_center_in_degrees = y_distance_from_center * deg_per_pixel

            closest_objects.append({
                'label': labels_array[i],
                'depth': avg_depths[i],
                'pitch': y_distance_from_center_in_degrees,
                'yaw': x_distance_from_center_in_degrees
            })

        #print(closest_objects)
        #print(mode)
        if mode.count("0") == 1:
            if (mode[0] == '0' and time.time() - cooldown > GEMINI_THROTTLE):
                cooldown = time.time()
                output = analyze_image_with_gemini(raw_image, "Can you describe what it feels like to be in this image? Speak like you are currently talking to a blind friend next to you, dont use Imagine the-. Describe briefly where things are located be as consice and objective as possible dont make a list just a couple sentences.")
                print(output)
            elif (time.time() - cooldown <= GEMINI_THROTTLE):
                print("GEMINI ON COOLDOWN")

        cv2.imshow('Depth', depth)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        end_time = time.time()
        '''
        fps = 1 / (end_time - start_time)
        fps_array.append(fps)
        
        if DEVICE == "mps":
            print(f'FPS: {fps:6.2f} | Driver Mem: {torch.mps.driver_allocated_memory()/1000/1000:6.2f}MB | Current Mem: {torch.mps.current_allocated_memory()/1000/1000:6.2f}MB')
        elif DEVICE == "cuda":
            print(f'FPS: {fps:6.2f} | Driver Mem: {torch.cuda.driver_allocated_memory()/1000/1000:6.2f}MB | Current Mem: {torch.cuda.current_allocated_memory()/1000/1000:6.2f}MB')
        '''
        #print(f'FPS: {fps:6.2f}')
    cap.release()
    cv2.destroyAllWindows()

    # create a plot of the fps array
    #plt.plot(fps_array)
    #plt.show()

    #print("Average FPS: ", sum(fps_array) / len(fps_array))
