import cv2
import numpy as np
from ultralytics import YOLO
import time
CONF_THRESHOLD = 0.6

if __name__ == "__main__":
    # Load the model
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("No Frame")
            continue

        # Perform object detection on the frame
        results = model(frame)

        
        h, w, _ = frame.shape
        blank_image = np.zeros((h, w, 3), dtype=np.uint8)

        i = 0

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

            # take a slice of the frame and save it
            cv2.imwrite(f"slice_{i}.png", frame[y1:y2, x1:x2])
            i += 1

            # Draw the bounding box on the blank image
            cv2.rectangle(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Determine text size & create a filled rectangle as background for readability
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(blank_image, (x1, y1 - text_height - baseline),
                          (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
            # Put the label text on the image
            cv2.putText(blank_image, text, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Display the blank image with bounding boxes and labels
        cv2.imshow('Bounding Boxes with Labels', blank_image)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()
