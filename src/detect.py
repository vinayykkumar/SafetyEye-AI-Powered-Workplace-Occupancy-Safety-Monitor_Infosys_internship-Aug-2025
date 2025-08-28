import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("models/safetyeye.pt")

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw results on frame
    annotated = results[0].plot()

    # Count people
    people = sum(1 for r in results[0].boxes.cls if r == 0)

    # Show occupancy
    cv2.putText(annotated, f"Occupancy: {people}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SafetyEye", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
