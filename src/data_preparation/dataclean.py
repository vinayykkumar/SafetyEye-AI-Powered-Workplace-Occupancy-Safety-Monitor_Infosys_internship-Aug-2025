import os
import hashlib

# Dynamically get project root (go 2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "archive", "data")

datasets = {
    "train": os.path.join(DATASET_PATH, "train", "images"),
    "valid": os.path.join(DATASET_PATH, "valid", "images"),
    "test":  os.path.join(DATASET_PATH, "test", "images")
}

image_exts = [".jpg", ".jpeg", ".png"]

for split, img_folder in datasets.items():
    if not os.path.exists(img_folder):
        print(f"[⚠] {split} image folder missing, skipping...")
        continue

    print(f"\n🔍 Checking {split} images for duplicates...")
    img_files = [f for f in os.listdir(img_folder) if os.path.splitext(f)[1].lower() in image_exts]

    seen_hashes = {}
    for f in img_files:
        img_path = os.path.join(img_folder, f)
        with open(img_path, "rb") as img_file:
            img_hash = hashlib.md5(img_file.read()).hexdigest()

        if img_hash in seen_hashes:
            print(f"🗑 Removing duplicate image: {img_path}")
            os.remove(img_path)
        else:
            seen_hashes[img_hash] = f

print("\n✅ Duplicate image removal complete!")
