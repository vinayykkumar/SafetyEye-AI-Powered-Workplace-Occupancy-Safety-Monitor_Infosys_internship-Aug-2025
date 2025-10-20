from ultralytics import YOLO

# Load your trained model (use last.pt if best.pt not saved yet)
model = YOLO("C:/Users/yuvan/runs/detect/ppe_model3/weights/last.pt")

# Run prediction on your test images
results = model.predict(
    source="C:/Users/yuvan/SAFETYEYE/test_images",  # your folder with test images
    save=True,        # save results with bounding boxes
    imgsz=320,        # smaller = faster
    device="cpu"
)
