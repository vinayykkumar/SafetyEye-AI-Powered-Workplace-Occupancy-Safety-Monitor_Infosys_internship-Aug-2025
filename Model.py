from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("yolov8s.pt")

print("Running inference on image...")
results = model("bus.jpg")

print("Inference complete. Saving results...")
results[0].save("output.jpg")
print("Output saved as output.jpg")
