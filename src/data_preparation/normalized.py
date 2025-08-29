import os

# Folders containing label files
label_folders = [
    "../../archive/data/train/labels",
    "../../archive/data/valid/labels",
    "../../archive/data/test/labels"
]

all_ok = True  # Flag to track if everything is correct

for folder in label_folders:
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)
            with open(file_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"❌ Incorrect format in {file_path}: {line}")
                        all_ok = False
                        break
                    try:
                        class_id, x, y, w, h = map(float, parts)
                    except ValueError:
                        print(f"❌ Non-numeric value in {file_path}: {line}")
                        all_ok = False
                        break
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"❌ Out-of-range value in {file_path}: {line}")
                        all_ok = False
                        break

if all_ok:
    print("✅ All YOLOv8 labels are normalized and correctly formatted!")
