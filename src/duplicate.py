import os
import hashlib
from PIL import Image

def get_hash(image_path):
    """Return md5 hash of an image file."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # ensure consistency
        return hashlib.md5(img.tobytes()).hexdigest()

def remove_duplicates(image_dir, label_ext=".txt"):
    seen_hashes = {}
    removed_files = []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                file_hash = get_hash(img_path)

                if file_hash in seen_hashes:
                    # Duplicate → remove image and its label
                    os.remove(img_path)
                    removed_files.append(img_path)

                    label_path = os.path.splitext(img_path)[0] + label_ext
                    if os.path.exists(label_path):
                        os.remove(label_path)
                        removed_files.append(label_path)
                else:
                    seen_hashes[file_hash] = img_path

    return removed_files

# Example usage
dataset_root = "SAFETEYE/data/processed"  # adjust path if needed
folders = ["train", "train_augmented", "val", "test"]

for folder in folders:
    path = os.path.join(dataset_root, folder)
    removed = remove_duplicates(path)
    print(f"{folder}: Removed {len(removed)} duplicate files")
