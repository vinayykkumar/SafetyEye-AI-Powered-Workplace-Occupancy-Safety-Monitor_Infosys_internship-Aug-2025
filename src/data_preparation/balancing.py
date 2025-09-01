import os
import cv2
import yaml
from collections import Counter


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "archive", "data")
YAML_FILE = os.path.join(DATASET_PATH, "data.yaml")

with open(YAML_FILE, "r") as f:
    data_yaml = yaml.safe_load(f)
nc = data_yaml["nc"]
class_names = data_yaml.get("names", [str(i) for i in range(nc)])

def check_dataset(split):
    print(f"\n[🔎 Checking {split}]")
    img_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}

    missing_labels = img_files - label_files
    missing_images = label_files - img_files

    if missing_labels:
        print(f"[⚠] Missing labels for: {missing_labels}")
    if missing_images:
        print(f"[⚠] Missing images for: {missing_images}")
    if not missing_labels and not missing_images:
        print("[✅] All images and labels are matched")


    class_counts = Counter()

    for label_file in os.listdir(label_dir):
        path = os.path.join(label_dir, label_file)
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"[❌] Wrong format in {label_file}: {line}")
                    continue

                cls, x, y, w, h = parts
                cls = int(cls)
                if cls < 0 or cls >= nc:
                    print(f"[❌] Invalid class {cls} in {label_file}")
                else:
                    class_counts[cls] += 1

                try:
                    vals = list(map(float, [x, y, w, h]))
                    if not all(0 <= v <= 1 for v in vals):
                        print(f"[❌] Invalid bbox in {label_file}: {vals}")
                except:
                    print(f"[❌] Non-numeric values in {label_file}: {line}")


    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[❌] Corrupted image: {img_file}")
        except:
            print(f"[❌] Error opening image: {img_file}")

    total = sum(class_counts.values())
    print("\n[📊 Class Distribution]")
    for i in range(nc):
        count = class_counts[i]
        percent = (count / total * 100) if total > 0 else 0
        print(f" - {class_names[i]}: {count} ({percent:.2f}%)")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        check_dataset(split)

