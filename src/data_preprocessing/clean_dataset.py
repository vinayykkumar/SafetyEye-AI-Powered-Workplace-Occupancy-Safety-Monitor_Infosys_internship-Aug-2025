import os
from PIL import Image
import hashlib
import shutil
import yaml
# ------------------- Paths -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_IMAGES = os.path.join(BASE_DIR, "raw_dataset", "images")
RAW_LABELS = os.path.join(BASE_DIR, "raw_dataset", "labels")
CLEAN_IMAGES = os.path.join(BASE_DIR, "dataset", "images")
CLEAN_LABELS = os.path.join(BASE_DIR, "dataset", "labels")
os.makedirs(CLEAN_IMAGES, exist_ok=True)
os.makedirs(CLEAN_LABELS, exist_ok=True)
# ------------------- Load classes from data.yaml -------------------
with open(os.path.join(BASE_DIR, "data.yaml"), "r") as f:
    data = yaml.safe_load(f)
valid_classes = list(range(data['nc']))  # e.g., 0 to 9
# ------------------- Helper Functions -------------------
def is_image_valid(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False
def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
def check_bounding_boxes(label_path, valid_classes):
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            cls, x_center, y_center, w, h = parts
            cls = int(cls)
            x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
            # Validate class ID
            if cls not in valid_classes:
                return False
            # Validate bounding box coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                return False
        return True
    except:
        return False
def resize_image(img_path, target_size=(416, 416)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img.save(img_path)
# ------------------- Cleaning -------------------
hashes = set()
for img_file in os.listdir(RAW_IMAGES):
    img_path = os.path.join(RAW_IMAGES, img_file)
    label_file = img_file.rsplit('.', 1)[0] + ".txt"
    label_path = os.path.join(RAW_LABELS, label_file)
    # 1. Check label exists
    if not os.path.exists(label_path):
        print(f"Skipping {img_file}: missing label")
        continue
    # 2. Check image validity
    if not is_image_valid(img_path):
        print(f"Skipping {img_file}: corrupted")
        continue
    # 3. Check duplicates
    img_hash = hash_file(img_path)
    if img_hash in hashes:
        print(f"Skipping {img_file}: duplicate")
        continue
    hashes.add(img_hash)
    # 4. Check bounding boxes & class IDs
    if not check_bounding_boxes(label_path, valid_classes):
        print(f"Skipping {img_file}: invalid class ID or bounding box")
        continue
    # 5. Resize image
    temp_path = os.path.join(CLEAN_IMAGES, img_file)  # copy first
    shutil.copy(img_path, temp_path)
    resize_image(temp_path, target_size=(416, 416))
    # 6. Copy label
    shutil.copy(label_path, os.path.join(CLEAN_LABELS, label_file))

print("✅ Dataset fully cleaned and saved in /dataset/ (raw dataset intact)")
