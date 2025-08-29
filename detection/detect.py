from ultralytics import YOLO
import cv2

# Detection: Real-time PPE violation spotting
def detect_realtime(model_path, source=0):
    model = YOLO(model_path)
    for result in model.predict(source=source, stream=True):
        frame = result.orig_img
        cv2.imshow('YOLOv8 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Example usage:
# detect_realtime('runs/detect/train/weights/best.pt')
