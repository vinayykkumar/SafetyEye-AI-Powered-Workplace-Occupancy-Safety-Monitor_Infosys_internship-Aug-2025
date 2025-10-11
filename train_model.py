from ultralytics import YOLO
import shutil
import os

def train_model(
    data_path="configs/data.yaml",
    epochs=25,
    img_size=640,
    batch_size=16,
    project_dir="runs/train",
    exp_name="exp"
):
    model = YOLO("yolov8n.pt")


    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_dir,
        name=exp_name,
        lr0=0.01,               # learning rate
        momentum=0.937,         # momentum
        weight_decay=0.0005,    # weight decay
        warmup_epochs=3.0       # warmup period

    )

    weights_path = results.best if hasattr(results, 'best') else None

    if weights_path and os.path.exists(weights_path):
        os.makedirs("models", exist_ok=True)
        dst = os.path.join("models", "trained_model.pt")
        shutil.copy(weights_path, dst)
        print(f"[INFO] Training completed. Best model saved at: {dst}")
    else:
        print("[WARNING] Could not find trained weights after training.")

if __name__ == "__main__":
    train_model(
        data_path="../configs/data.yaml",
        epochs=25,
        img_size=640,
        batch_size=16,
        project_dir="runs/train",
        exp_name="my_exp1"
    )
