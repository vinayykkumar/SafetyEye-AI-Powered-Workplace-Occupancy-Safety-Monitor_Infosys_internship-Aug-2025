import sys
import cv2
from ultralytics import YOLO

def predict_image(image_path, model_path="best.pt"):
    # Load model
    model = YOLO(model_path)
    # Read image
    img = cv2.imread(image_path)
    # Run detection
    results = model(img)
    # Show results
    results.show()
    # Save results
    results.save("output.jpg")
    print("Prediction complete and saved as output.jpg.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
    else:
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "best.pt"
        predict_image(image_path, model_path)
