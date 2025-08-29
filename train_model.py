from ultralytics import YOLO
import shutil
import os

def train_model(data_path="configs/data.yaml", epochs=50, img_size=640):
    model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 nano

    results = model.train(data=data_path, epochs=epochs, imgsz=img_size)

    weights_path = results.best if hasattr(results, 'best') else None

    if weights_path and os.path.exists(weights_path):
        os.makedirs("models", exist_ok=True)
        dst = os.path.join("models", "trained_model.pt")
        shutil.copy(weights_path, dst)
        print(f"[INFO] Training completed. Best model saved at: {dst}")
    else:
        print("[WARNING] Could not find trained weights after training.")

if __name__ == "__main__":
    train_model()
