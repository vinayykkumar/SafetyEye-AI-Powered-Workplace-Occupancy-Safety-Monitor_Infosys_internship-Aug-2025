import os
import hashlib

datasets = {
    "train": "../../archive/data/train/images",
    "valid": "../../archive/data/valid/images",
    "test": "../../archive/data/test/images"
}

image_exts = [".jpg", ".jpeg", ".png"]
seen_hashes = {}
duplicates = []

for split, img_folder in datasets.items():
    if not os.path.exists(img_folder):
        continue
    for f in os.listdir(img_folder):
        if os.path.splitext(f)[1].lower() not in image_exts:
            continue
        img_path = os.path.join(img_folder, f)
        with open(img_path, "rb") as img_file:
            img_hash = hashlib.md5(img_file.read()).hexdigest()
        if img_hash in seen_hashes:
            orig_split, orig_name = seen_hashes[img_hash]
            duplicates.append((orig_split, orig_name, split, f))
        else:
            seen_hashes[img_hash] = (split, f)

if duplicates:
    print("Duplicate images found across datasets:")
    for orig_split, orig_name, dup_split, dup_name in duplicates:
        print(f"Original: [{orig_split}] {orig_name} | Duplicate: [{dup_split}] {dup_name}")
else:
    print("No duplicate images found across datasets.")