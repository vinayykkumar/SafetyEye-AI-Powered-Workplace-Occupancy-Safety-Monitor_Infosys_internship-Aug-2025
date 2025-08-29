# test_detections.py

import cv2
from ultralytics import YOLO
from ppe_rules import check_violations
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "runs/detect/train/best.pt"  # Path to your trained YOLOv8 model
IMAGE_PATH = "example.jpg"                # Test image (can switch to video)
VIDEO_PATH = None                         # Optional: set path to a video file

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Helper function to visualize detections
# -----------------------------
def visualize_results(frame, results):
    if frame is None:
        raise ValueError("Input frame is None!")

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    violations = check_violations(detected_classes)

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name in violations:
            color = (0, 0, 255)  # Red for violation
            label = violations[violations.index(cls_name)]
        else:
            color = (0, 255, 0)  # Green for safe
            label = cls_name

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, violations

# -----------------------------
# Process Image
# -----------------------------
if IMAGE_PATH:
    img_path = Path(IMAGE_PATH)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    frame = cv2.imread(str(img_path))
    results = model(str(img_path))
    frame, violations = visualize_results(frame, results)

    print("Detected Classes:", [model.names[int(box.cls)] for box in results[0].boxes])
    print("Violations:", violations)

    cv2.imshow("PPE Violations", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# Process Video
# -----------------------------
elif VIDEO_PATH:
    video_path = Path(VIDEO_PATH)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame, violations = visualize_results(frame, results)

        cv2.imshow("PPE Violations", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
