import os
from glob import glob
from ultralytics import YOLO

# Path to runs folder
runs_dir = r"C:\Users\yuvan\SAFETYEYE\runs"

# Find all best.pt files under runs/
best_files = glob(os.path.join(runs_dir, "**", "weights", "best.pt"), recursive=True)

if not best_files:
    raise FileNotFoundError("No best.pt found in runs folder. Train the model first!")

# Pick the latest one (by modified time)
latest_best = max(best_files, key=os.path.getmtime)
print(f"Using model: {latest_best}")

# Load and validate
model = YOLO(latest_best)
metrics = model.val()
print(metrics)
