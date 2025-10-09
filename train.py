print("Starting YOLOv8 Training...")

from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='ppe_data.yaml',  # Corrected path to dataset config file
    epochs=50,
    imgsz=640,
    batch=16
)

print("Training complete.")
