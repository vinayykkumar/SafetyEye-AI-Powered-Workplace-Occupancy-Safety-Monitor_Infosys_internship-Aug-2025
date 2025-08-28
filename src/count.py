import os

# Base preprocessed dataset folder
base_path = r"C:\safetyeye\preprocessed"

# Common image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Subfolders to check
splits = ["train", "val", "test"]

for split in splits:
    folder = os.path.join(base_path, split, "images")
    if os.path.exists(folder):
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
        print(f"{split.capitalize()} set: {len(image_files)} images")
    else:
        print(f"{split.capitalize()} set: Folder not found -> {folder}")
