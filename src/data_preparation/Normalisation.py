import os

# Dynamically set dataset root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "archive", "data")

label_folders = {
    "train": os.path.join(DATASET_PATH, "train", "labels"),
    "valid": os.path.join(DATASET_PATH, "valid", "labels"),
    "test": os.path.join(DATASET_PATH, "test", "labels"),
}

all_ok = True

for split, folder in label_folders.items():
    if not os.path.exists(folder):
        print(f"[⚠] {split}/labels folder missing, skipping...")
        continue

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)
            with open(file_path) as f:
                for line_num, line in enumerate(f, start=1):
                    parts = line.strip().split()

                    # Format check
                    if len(parts) != 5:
                        print(f"❌ Format error in [{split}] {file}, line {line_num}: {line.strip()}")
                        all_ok = False
                        continue

                    try:
                        class_id = int(parts[0])  # class should be integer
                        x, y, w, h = map(float, parts[1:])
                    except ValueError:
                        print(f"❌ Non-numeric value in [{split}] {file}, line {line_num}: {line.strip()}")
                        all_ok = False
                        continue

                    # Range check
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"❌ Out-of-range value in [{split}] {file}, line {line_num}: {line.strip()}")
                        all_ok = False

if all_ok:
    print("✅ All YOLOv8 labels are normalized and correctly formatted!")
else:
    print("⚠️ Some label errors found. Please fix them before training.")