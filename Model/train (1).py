from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on your dataset
results = model.train(
    data=r"C:\Users\yuvan\SAFETYEYE\data.yaml",  # path to your dataset yaml
    epochs=50,        
    imgsz=640,        
    batch=8,          # reduce for CPU
    workers=0,        # must be 0 for CPU
    name="ppe_model", 
    device="cpu"      # <-- FIXED
)

# Validate after training
val_results = model.val(
    data=r"C:\Users\yuvan\SAFETYEYE\data.yaml",
    device="cpu"
)

print("✅ Training completed. Best model saved at:")
print(model.ckpt_path)
