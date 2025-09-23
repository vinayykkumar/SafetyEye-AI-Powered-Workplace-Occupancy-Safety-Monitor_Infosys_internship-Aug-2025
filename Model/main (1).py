import os
from ultralytics import YOLO

# Path to dataset.yaml
DATA_YAML = r"C:/Users/yuvan/SAFETYEYE/Dataset/dataset.yaml"

def main():
    # Load YOLOv8 model (pretrained on COCO)
    model = YOLO("yolov8s.pt")  # you can change to yolov8m.pt or yolov8n.pt

    # Train the model
    model.train(
        data=DATA_YAML,       # dataset.yaml path
        epochs=50,            # increase for better accuracy
        imgsz=640,            # image size
        batch=16,             # adjust based on GPU/CPU memory
        name="helmet_detection",  # results saved in runs/detect/helmet_detection
        verbose=True
    )

    # After training, evaluate
    model.val()

    # Example prediction (optional)
    results = model.predict(source="C:/Users/yuvan/SAFETYEYE/test.jpg")
    print("Prediction Results:", results)

if __name__ == "__main__":
    main()
