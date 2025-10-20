from ultralytics import YOLO

# Load the last trained model instead of best.pt
model = YOLO(r"C:/Users/yuvan/runs/detect/ppe_model3/weights/last.pt")

results = model.predict(
    source=r"C:/Users/yuvan/SAFETYEYE/Dataset/images/val",
    save=True,
    imgsz=640,
    conf=0.25,
    device="cpu"
)

for r in results:
    print(r)
