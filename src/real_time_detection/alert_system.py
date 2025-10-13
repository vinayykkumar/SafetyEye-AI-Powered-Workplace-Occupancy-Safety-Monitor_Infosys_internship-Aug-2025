import cv2
from ultralytics import YOLO
import time
from plyer import notification  # cross-platform pop-up

# -----------------------------
# 1️⃣ Load trained model
# -----------------------------
model = YOLO(r"C:\Users\mkr19\Desktop\SafetyEye\src\real_time_detection\bestt.pt")

# -----------------------------
# 2️⃣ Class names & colors
# -----------------------------
class_names = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest', 'cone']
class_colors = {
    'person': (255, 0, 0),
    'helmet': (0, 255, 0),
    'no_helmet': (0, 0, 255),
    'vest': (0, 255, 255),
    'no_vest': (255, 0, 255),
    'cone': (255, 165, 0)
}

# -----------------------------
# 3️⃣ Open video or webcam
# -----------------------------
video_source = r"C:\Users\mkr19\Downloads\sample1.mp4"  # replace with 0 for webcam
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# -----------------------------
# Frame sizes
# -----------------------------
process_width = 480
process_height = 270
display_scale = 1.5
display_width = int(process_width * display_scale)
display_height = int(process_height * display_scale)

# -----------------------------
# Temporary fix function
# -----------------------------
def fix_vest_label(label, conf):
    if label == "vest" and conf < 0.6:
        return "no_vest"
    return label

# -----------------------------
# Alert cooldown
# -----------------------------
last_alert_time = 0
alert_cooldown = 3  # seconds

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or cannot fetch frame.")
        break

    # Resize for processing
    small_frame = cv2.resize(frame, (process_width, process_height))

    # Run YOLO detection
    results = model.predict(small_frame, conf=0.25, verbose=False)[0]

    violation_detected = False
    violations_list = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        # Apply vest fix
        label = fix_vest_label(label, conf)

        # Print every detection
        print(f"Detected: {label} | Confidence: {conf:.2f}")

        # Check violations
        if label in ["no_helmet", "no_vest"]:
            violation_detected = True
            violations_list.append(label)
            print(f">>> VIOLATION DETECTED: {label}")

        # Draw bounding boxes (optional)
        color = class_colors[label]
        cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(small_frame, f"{label} {conf:.2f}", (x1, max(15, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Pop-up notification for violations
    if violation_detected and (time.time() - last_alert_time) > alert_cooldown:
        last_alert_time = time.time()
        violation_str = ", ".join(set(violations_list))
        notification.notify(
            title="SafetyEye Alert",
            message=f"Violations detected: {violation_str}",
            timeout=5
        )

    # Display frame
    display_frame = cv2.resize(small_frame, (display_width, display_height))
    cv2.imshow("SafetyEye - Real-Time Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
