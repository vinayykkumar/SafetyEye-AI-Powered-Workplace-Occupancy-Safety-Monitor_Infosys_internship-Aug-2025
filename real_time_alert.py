from ultralytics import YOLO
import cv2
from tkinter import messagebox, Tk

# Load trained YOLO model
model = YOLO(r"C:\Users\Shreya\Desktop\infosys_project\best.pt")  # <-- your model path

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

# Create a hidden Tkinter root for popup alerts
root = Tk()
root.withdraw()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Check for violations
    for r in results:
        for cls, conf, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
            cls_name = model.names[int(cls)]
            if cls_name == "no_helmet":
                messagebox.showwarning("ALERT", "No Helmet Detected!")

    # Display video
    cv2.imshow("Real-Time Detection", frame)

    # Stop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
