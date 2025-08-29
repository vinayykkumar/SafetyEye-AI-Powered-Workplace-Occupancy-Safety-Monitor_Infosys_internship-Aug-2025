from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")  # or your model path
    model.train(
        data="C:/Users/gaikw/OneDrive/Desktop/SafetyEye/processed/safetyeye_v1/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        name="ppe_yolov8s",
        val=True
    )
