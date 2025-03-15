import cv2
import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.6

def extract_objects(frame):
    # Load the model
    model = YOLO("yolo11n.pt")

    # Perform object detection on the frame
    results = model(frame)

    boxes_array = []
    labels_array = []
    confidence_array = []

    # Iterate through each detection; assuming detection results are in results[0].boxes
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if conf < CONF_THRESHOLD:
            continue

        # Get coordinates and convert to int
        x1, y1, x2, y2 = map(int, box)
        # Get the class name using model.names
        label = model.names[int(cls)]
        # Format label with confidence (optional)
        text = f"{label}: {conf:.2f}"

        boxes_array.append([x1, y1, x2, y2])
        labels_array.append(label)
        confidence_array.append(conf)

    return boxes_array, labels_array, confidence_array