import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  # replace with your model path

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Check for violations
    for det in results:
        for cls, conf, box in zip(det.boxes.cls, det.boxes.conf, det.boxes.xyxy):
            cls_name = model.names[int(cls)]
            if cls_name == "no_helmet":
                print("⚠️ ALERT: No Helmet Detected!")

    # Display frame with boxes
    cv2.imshow("YOLO Live Detection", results[0].plot())  # plot draws boxes

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
