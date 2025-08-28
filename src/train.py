from ultralytics import YOLO

def main():
    # Load a YOLO model (pre-trained weights, you can pick yolov8n, yolov8s, etc.)
    model = YOLO("yolov8n.pt")  # YOLOv8 nano for speed, switch to yolov8s.pt or yolov8m.pt for accuracy

    # Train the model
    model.train(
        data="data/data.yaml",  # path to your dataset YAML
        epochs=50,              # number of epochs (increase if needed)
        imgsz=640,              # image size for training
        batch=16,               # adjust depending on your GPU/CPU
        name="safetyeye_yolov8" # experiment name (saves runs in runs/detect/)
    )

if __name__ == "__main__":
    main()
