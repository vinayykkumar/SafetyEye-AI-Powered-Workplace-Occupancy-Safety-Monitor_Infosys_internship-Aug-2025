from ultralytics import YOLO
import os


model = YOLO("yolov8s.pt")   # use yolov8m.pt if GPU allows, but s is safe for Colab free

project_folder = "/content/drive/MyDrive/YOLOv8_train"
run_name = "ppe_yolov8_adamw"

model.train(
    data="/content/drive/MyDrive/dataset/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.002,
    lrf=0.1,
    weight_decay=0.01,
    workers=4,
    project=project_folder,  # Drive path
    name=run_name,
    exist_ok=True,
    augment=True,
    patience=10
)

print(f"✅ Training complete! Best model saved in {project_folder}/{run_name}/best.pt")
