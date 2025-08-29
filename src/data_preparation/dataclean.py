import os
import hashlib

# Paths to images and labels
datasets = {
    "train": ["../../archive/data/train/images", "../../archive/data/train/labels"],
    "valid": ["../../archive/data/valid/images", "../../archive/data/valid/labels"],
    "test": ["../../archive/data/test/images", "../../archive/data/test/labels"]
}

image_exts = [".jpg", ".jpeg", ".png"]  # supported image extensions

for split, (img_folder, label_folder) in datasets.items():
    if not os.path.exists(img_folder) or not os.path.exists(label_folder):
        continue

    # Get lists of image and label filenames without extension
    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_folder) if os.path.splitext(f)[1].lower() in image_exts}
    label_files = {os.path.splitext(f)[0]: f for f in os.listdir(label_folder) if f.endswith(".txt")}

    # 1️⃣ Remove empty label files
    for name, f in list(label_files.items()):
        label_path = os.path.join(label_folder, f)
        if os.path.getsize(label_path) == 0:
            print(f"Removing empty label: {label_path}")
            os.remove(label_path)
            label_files.pop(name)

    # 2️⃣ Remove images without labels
    for name, f in list(img_files.items()):
        if name not in label_files:
            img_path = os.path.join(img_folder, f)
            print(f"Removing image without label: {img_path}")
            os.remove(img_path)
            img_files.pop(name)

    # 3️⃣ Remove labels without images
    for name, f in list(label_files.items()):
        if name not in img_files:
            label_path = os.path.join(label_folder, f)
            print(f"Removing label without image: {label_path}")
            os.remove(label_path)
            label_files.pop(name)

    # 4️⃣ Remove duplicate images and corresponding labels
    seen_hashes = {}
    for name, f in list(img_files.items()):
        img_path = os.path.join(img_folder, f)
        with open(img_path, "rb") as img_file:
            img_hash = hashlib.md5(img_file.read()).hexdigest()
        if img_hash in seen_hashes:
            # Duplicate found, remove image and label
            print(f"Removing duplicate image: {img_path}")
            os.remove(img_path)
            label_path = os.path.join(label_folder, f"{name}.txt")
            if os.path.exists(label_path):
                print(f"Removing corresponding label: {label_path}")
                os.remove(label_path)
            img_files.pop(name)
        else:
            seen_hashes[img_hash] = name

print("✅ Data cleaning and duplicate removal done!")
