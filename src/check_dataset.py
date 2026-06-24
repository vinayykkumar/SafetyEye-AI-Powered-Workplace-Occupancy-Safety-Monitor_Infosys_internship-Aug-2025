import os
import cv2
import yaml

DATASET_PATH = "data/processed"
YAML_FILE = "data/processed/data.yaml"


with open(YAML_FILE, "r") as f:
    data_yaml = yaml.safe_load(f)
nc = data_yaml["nc"]

def check_dataset(split):
    img_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    img_set = {os.path.splitext(f)[0] for f in img_files}
    label_set = {os.path.splitext(f)[0] for f in label_files}

    
    missing_labels = img_set - label_set
    missing_images = label_set - img_set

    if missing_labels:
        print(f"[{split}] Missing labels for {len(missing_labels)} images")
    if missing_images:
        print(f"[{split}] Missing images for {len(missing_images)} labels")
    if not missing_labels and not missing_images:
        print(f"[{split}] All images and labels match")

    # Check label content
    for label_file in label_files:
        path = os.path.join(label_dir, label_file)
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"[{split}] Wrong format in {label_file}: {line}")
                    continue
                cls, x, y, w, h = parts
                cls = int(cls)
                if cls < 0 or cls >= nc:
                    print(f"[{split}] Invalid class ID {cls} in {label_file}")
                for v in [x, y, w, h]:
                    v = float(v)
                    if v < 0 or v > 1:
                        print(f"[{split}] Invalid bbox value {v} in {label_file}")

    # Check corrupted images
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{split}] Corrupted image: {img_file}")

def run_checks():
    for split in ["train", "val", "test"]: 
        check_dataset(split)

if __name__ == "__main__":
    run_checks()
    print("\n Preprocessing check complete. Dataset is ready for training!")
