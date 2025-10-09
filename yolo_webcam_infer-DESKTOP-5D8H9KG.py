import cv2
from ultralytics import YOLO
import csv
import datetime
import time

# Cross-platform beep sound function
try:
    import winsound  # Windows
    beep = lambda: winsound.Beep(1000, 500)
except ImportError:
    import os  # Linux/Mac
    beep = lambda: os.system('play -nq -t alsa synth 0.5 sine 1000')

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

def log_violation(violation_type, box):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    x1, y1, x2, y2 = map(int, box)
    with open('violation_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, violation_type, x1, y1, x2, y2])

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

IOU_THRESHOLD = 0.1
alert_cooldown = 3  # seconds between sound alerts
last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    results = model(frame)
    detections = results[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy()
    class_names = model.names

    person_boxes = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "person"]
    helmet_boxes = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "helmet"]
    vest_boxes = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "vest"]

    violations = []

    for p_box in person_boxes:
        helmet_found = any(iou(p_box, h_box) > IOU_THRESHOLD for h_box in helmet_boxes)
        vest_found = any(iou(p_box, v_box) > IOU_THRESHOLD for v_box in vest_boxes)

        print(f"Helmet found: {helmet_found}, Vest found: {vest_found}")  # Debug line

        if not helmet_found:
            log_violation("Helmet Missing", p_box)
            violations.append((p_box, "Helmet Missing"))
        if not vest_found:
            log_violation("Vest Missing", p_box)
            violations.append((p_box, "Vest Missing"))

    current_time = time.time()
    if violations and (current_time - last_alert_time > alert_cooldown):
        beep()
        last_alert_time = current_time

    for i, (box, label) in enumerate(violations):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, label, (x1, y1 - 10 - 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
