import os

# Paths
DATASET_PATH = "data/processed"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    img_dir = os.path.join(DATASET_PATH, split, "images")
    lbl_dir = os.path.join(DATASET_PATH, split, "labels")

    for img_file in os.listdir(img_dir):
        if not (img_file.endswith(".jpg") or img_file.endswith(".png")):
            continue

        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # 1. Remove if label file missing
        if not os.path.exists(lbl_path):
            print(f"🗑 Removing {img_file} (no label)")
            os.remove(img_path)
            continue

        # 2. Remove if label file empty
        if os.path.getsize(lbl_path) == 0:
            print(f"🗑 Removing {img_file} (empty label)")
            os.remove(img_path)
            os.remove(lbl_path)
            continue

    # 3. Remove label files without images
    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        img_file_jpg = lbl_file.replace(".txt", ".jpg")
        img_file_png = lbl_file.replace(".txt", ".png")

        if not (os.path.exists(os.path.join(img_dir, img_file_jpg)) or os.path.exists(os.path.join(img_dir, img_file_png))):
            print(f"🗑 Removing {lbl_file} (no image)")
            os.remove(os.path.join(lbl_dir, lbl_file))

print("✅ Dataset cleaned successfully!")
