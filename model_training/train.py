from ultralytics import YOLO

# Model Training: Train YOLOv8 for PPE detection
def train_yolov8(data_yaml, model='yolov8n.pt', epochs=50, imgsz=640):
    model = YOLO(model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

# Example usage:
# train_yolov8('data.yaml')
