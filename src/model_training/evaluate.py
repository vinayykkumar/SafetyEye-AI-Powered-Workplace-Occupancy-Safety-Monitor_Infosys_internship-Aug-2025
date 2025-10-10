from ultralytics import YOLO
import cv2
import numpy as np
import time
import base64

# ---- Optional: DeepSORT tracker ----
# pip install deep_sort_realtime
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------
# Load your trained YOLOv8 model
model = YOLO("../../models/best.pt")  # Replace with your trained weights

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Unsafe classes to trigger alerts
unsafe_classes = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # max_age = frames to keep lost tracks

# Store alerted IDs to avoid repeated alerts
alerted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame, conf=0.5)  # adjust confidence threshold
    detections = []

    # Prepare detections for tracker
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = r
        cls_name = results[0].names[int(cls_id)]
        detections.append(([x1, y1, x2, y2], conf, cls_name))

    # Run tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        track_id = track.track_id
        bbox = track.to_ltrb()  # left, top, right, bottom
        cls_name = track.get_det_class()  # class name of detection

        x1, y1, x2, y2 = map(int, bbox)

        # Check if violation
        if cls_name in unsafe_classes and track_id not in alerted_ids:
            alerted_ids.add(track_id)
            print(f"⚠️ ALERT! {cls_name} detected for person ID {track_id}")

            # Crop the violator image
            cropped_person = frame[y1:y2, x1:x2]

            # Convert to base64 (send to dashboard or save)
            _, buffer = cv2.imencode('.jpg', cropped_person)
            img_str = base64.b64encode(buffer).decode('utf-8')

            # Example: Print length of base64 string
            print(f"Person ID {track_id} image ready for dashboard (len={len(img_str)})")

        # Draw bounding box and label
        color = (0, 0, 255) if cls_name in unsafe_classes else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} ID:{track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show annotated frame
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
