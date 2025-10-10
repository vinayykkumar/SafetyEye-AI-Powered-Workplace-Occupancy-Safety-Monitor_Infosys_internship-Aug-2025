from ultralytics import YOLO
import cv2
import winsound

# --- Load YOLOv8 Model ---
model_path = r"D:\SAFETYEYE 2\source_file\best.pt"  # Your model path
model = YOLO(model_path)

# --- Open Video File ---
video_path = r"C:\Users\HP\Downloads\8964296-uhd_3840_2160_25fps.mp4"
cap = cv2.VideoCapture(video_path)

# --- Alert Sound Function ---
def play_alert():
    duration = 500
    freq = 1000
    winsound.Beep(freq, duration)

# --- Define Colors for Classes (BGR) ---
class_colors = {
    'person': (0, 0, 255),  # Red
    'helmet': (0, 255, 0),  # Green
    'vest': (255, 0, 0),    # Blue
}

# --- Resize video frame while keeping aspect ratio ---
def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (width, new_h))
    return resized_frame

# --- Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing and better display
    frame = resize_frame(frame, width=640)

    # Run YOLO detection
    results = model(frame)
    boxes = results[0].boxes
    names = model.names

    person_detected = False
    helmet_detected = False
    vest_detected = False

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id].lower()

        # Set flags
        if label == 'person':
            person_detected = True
        elif label == 'helmet':
            helmet_detected = True
        elif label == 'vest':
            vest_detected = True

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Select color for this class
        color = class_colors.get(label, (255, 255, 255))  # Default white if unknown

        # Draw bounding box and label with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'{label.capitalize()} {conf:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Check for violations only if person detected
    violation_detected = False
    message = ""

    if person_detected:
        if not helmet_detected:
            message += "⚠️ No Helmet! "
            violation_detected = True
        if not vest_detected:
            message += "⚠️ No Vest! "
            violation_detected = True

    # Display alert message on frame
    if violation_detected:
        cv2.putText(frame, message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        print(message)
        play_alert()

    # Show the frame
    cv2.imshow("YOLOv8 PPE Detection", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
