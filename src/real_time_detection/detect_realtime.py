import cv2
from ultralytics import YOLO

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
# 3️⃣ Load video
# -----------------------------
video_path = r"C:\Users\mkr19\Downloads\sample1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# -----------------------------
# Small frame for processing
# -----------------------------
process_width = 480
process_height = 270

# Slightly bigger display size
display_scale = 1.5  # scale up display
display_width = int(process_width * display_scale)
display_height = int(process_height * display_scale)

# Optional: save smaller output video
out = cv2.VideoWriter('output_small.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      20, 
                      (process_width, process_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Resize for processing (zoomed out)
    # -----------------------------
    small_frame = cv2.resize(frame, (process_width, process_height))

    # -----------------------------
    # Run detection
    # -----------------------------
    results = model.predict(small_frame, conf=0.25, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        # Temporary fix for vest/no_vest
        if label == 'vest' and conf < 0.5:
            label = 'no_vest'
            color = class_colors['no_vest']
        else:
            color = class_colors[label]

        # Draw box + label
        cv2.rectangle(small_frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(small_frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # -----------------------------
    # Resize frame for display only
    # -----------------------------
    display_frame = cv2.resize(small_frame, (display_width, display_height))
    cv2.imshow("Video Detection", display_frame)

    # Save the small processed video
    out.write(small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()