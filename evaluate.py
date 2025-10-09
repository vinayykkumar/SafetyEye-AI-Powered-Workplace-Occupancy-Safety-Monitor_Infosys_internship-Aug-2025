import sys
from ultralytics import YOLO

def evaluate_model(data_path, model_path="best.pt"):
    # Load model
    model = YOLO(model_path)
    # Run evaluation on data.yaml or directory of images
    metrics = model.val(data=data_path)
    print("Evaluation Metrics:")
    print(metrics)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <data_path> [model_path]")
    else:
        data_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "best.pt"
        evaluate_model(data_path, model_path)
