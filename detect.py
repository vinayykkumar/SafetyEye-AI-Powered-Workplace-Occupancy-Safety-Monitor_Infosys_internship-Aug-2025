import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = r"C:\safetyeye\runs\helmet_augmented_final\yolo_augmented_m_fresh\weights\best.pt" 
CAMERA_INDEX = 0  # Logitech external camera
VIOLATIONS = {"no_helmet", "no_mask", "no_vest"}  # violation labels
SAFE_COLOR = (0, 255, 0)  # green
VIOLATION_COLOR = (0, 0, 255)  # red

# Email settings
EMAIL_SENDER = "minimeeee44@gmail.com"
EMAIL_PASSWORD = "nptn **** ****"  # use Gmail app password
EMAIL_RECEIVER = "minimeeee44@gmail.com"

# ---------------------------
# INITIALIZE
# ---------------------------
print("🚀 Loading model...")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise Exception("⚠ Camera not accessible. Check connection.")

violations_detected = set()
print("✅ Camera opened. Press 'q' to quit.")

# ---------------------------
# REAL-TIME DETECTION LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to grab frame.")
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in VIOLATIONS:
            violations_detected.add(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), VIOLATION_COLOR, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, VIOLATION_COLOR, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), SAFE_COLOR, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, SAFE_COLOR, 2)

    cv2.imshow("Safety Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------
# SEND EMAIL WITH VIOLATIONS
# ---------------------------
if violations_detected:
    try:
        msg = MIMEText(
            f"Safety Detection Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            f"Violations detected:\n" + "\n".join(sorted(violations_detected))
        )
        msg['Subject'] = "Safety Violation Report"
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent with violations.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
else:
    print("No violations detected. No email sent.")
