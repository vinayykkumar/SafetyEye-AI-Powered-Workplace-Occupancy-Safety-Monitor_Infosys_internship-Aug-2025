from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/content/processed/data.yaml",
    epochs=50,
    patience=5,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.005,
    weight_decay=0.01,
    workers=4,
    project="/content/drive/MyDrive/YOLOv8_training",
    name="ppe_yolov8_small_run_3",
    exist_ok=True,
    augment=True
)