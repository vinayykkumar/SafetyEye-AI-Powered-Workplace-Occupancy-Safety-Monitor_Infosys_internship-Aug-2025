import cv2
from ultralytics import YOLO
from ppe_rules import check_violations
from pathlib import Path
import os
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(r"C:\Users\gaikw\OneDrive\Desktop\SafetyEye")
MODEL_PATH = BASE_DIR / r"runs\detect\ppe_yolov8s\weights\best.pt"
OUTPUT_DIR = BASE_DIR / "outputs"
CONF_THRESH = 0.01

# Alert settings
ENABLE_EMAIL = False
ALERT_EMAIL_FROM = "youremail@gmail.com"
ALERT_EMAIL_TO = "receiver@gmail.com"
EMAIL_PASSWORD = "your-app-password"

# -----------------------------
# Setup output folder
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model = YOLO(str(MODEL_PATH))
print("Model Classes:", model.names)

# -----------------------------
# Email alert function
# -----------------------------
def send_email_alert(violations):
    if not ENABLE_EMAIL:
        return
    msg = MIMEText(f"Safety Violation Detected: {', '.join(violations)}")
    msg["Subject"] = "PPE Violation Alert"
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ALERT_EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        print("📧 Email alert sent!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# -----------------------------
# Helper function to visualize detections
# -----------------------------
def visualize_results(frame, results):
    if frame is None:
        return frame, []

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    detected_scores = [float(box.conf) for box in results[0].boxes]
    violations = check_violations(detected_classes)

    person_count = 0
    for box, cls_name, score in zip(results[0].boxes, detected_classes, detected_scores):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Count people
        # Count unique people based on Hardhat or Safety Vest detections
        person_count = len([cls for cls in detected_classes if cls in ["NO-Mask","Hardhat", "Safety Vest"]])


        # Choose color for violations
        if cls_name in violations:
            color = (0, 0, 255)  # Red
            label = f"{cls_name} (Violation) {score:.2f}"
        else:
            color = (0, 255, 0)  # Green
            label = f"{cls_name} {score:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Draw label above the box
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay top-left info box
    info_text = f"People detected: {person_count} | Violations: {len(violations)}"
    cv2.rectangle(frame, (5,5), (400,60), (50,50,50), -1)  # dark background box
    cv2.putText(frame, info_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Overlay violations list if any
    if violations:
        violation_text = "Missing: " + ", ".join(violations)
        cv2.putText(frame, violation_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return frame, violations

# -----------------------------
# Webcam Detection
# -----------------------------
print("🎥 Trying to open Webcam...")
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Webcam opened successfully (index {i})")
        break
else:
    print("❌ Could not open any webcam. Exiting...")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

fourcc = getattr(cv2, "VideoWriter_fourcc")(*'mp4v')
output_path = OUTPUT_DIR / "output_ppe_webcam.mp4"
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

print("🚀 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    results = model(frame, conf=CONF_THRESH)
    frame, violations = visualize_results(frame, results)

    # Send alert if violations
    if violations:
        print("⚠️ Violations Detected:", violations)
        send_email_alert(violations)

    out.write(frame)
    cv2.imshow("PPE Detection - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Webcam output video saved at: {output_path}")

