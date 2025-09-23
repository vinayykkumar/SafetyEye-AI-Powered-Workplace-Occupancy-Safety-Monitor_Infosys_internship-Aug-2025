import os
from pathlib import Path

# Base runs directory
runs_dir = Path("C:/Users/yuvan/SAFETYEYE/runs/detect")

# Get all train folders (train, train1, train2, ...)
train_folders = [f for f in runs_dir.iterdir() if f.is_dir() and f.name.startswith("train")]

if not train_folders:
    print("❌ No training folders found in:", runs_dir)
else:
    # Sort by modification time (latest last)
    train_folders.sort(key=lambda x: x.stat().st_mtime)

    latest_train = train_folders[-1]  # last one is latest
    best_path = latest_train / "weights" / "best.pt"

    if best_path.exists():
        print("✅ Found best.pt at:", best_path)
    else:
        print("❌ best.pt not found in:", latest_train)
