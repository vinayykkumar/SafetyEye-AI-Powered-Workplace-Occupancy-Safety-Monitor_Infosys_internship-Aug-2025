import os
import cv2
import glob
import albumentations as A
import hashlib

# Project root (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
root_path = os.path.join(PROJECT_ROOT, "archive", "data")

# Paths
image_dir = os.path.join(root_path, "train", "images")
label_dir = os.path.join(root_path, "train", "labels")
output_img_dir = os.path.join(root_path, "train", "images_aug")
output_lbl_dir = os.path.join(root_path, "train", "labels_aug")

# Create output folders
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Mapping file (append mode so you don’t overwrite on re-runs)
mapping_file = os.path.join(output_lbl_dir, "filename_mapping.txt")
mapping_f = open(mapping_file, "a")

# Define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=20, p=0.5),
    A.MotionBlur(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Collect images
images = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
print(f"🔎 Found {len(images)} images in {image_dir}")

processed, skipped = 0, 0

for img_path in images:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n➡ Processing {filename}")

    # Generate a unique hash name
    short_name = hashlib.md5(filename.encode()).hexdigest()

    out_img_path = os.path.join(output_img_dir, short_name + "_aug.jpg")
    out_lbl_path = os.path.join(output_lbl_dir, short_name + "_aug.txt")

    # ✅ Skip if this augmented file already exists
    if os.path.exists(out_img_path) and os.path.exists(out_lbl_path):
        print(f"⏩ Skipping {filename} (already augmented)")
        skipped += 1
        continue

    mapping_f.write(f"{short_name} -> {filename}\n")

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠ Could not read image: {img_path}")
        skipped += 1
        continue

    # Label path
    label_path = os.path.join(label_dir, filename + ".txt")
    if not os.path.exists(label_path):
        print(f"⚠ Skipping {filename} (label not found)")
        skipped += 1
        continue

    bboxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, bw, bh = parts
            bboxes.append([float(x), float(y), float(bw), float(bh)])
            class_labels.append(int(cls))

    if not bboxes:
        print(f"⚠ Skipping {filename} (no bounding boxes)")
        skipped += 1
        continue

    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f"⚠ Augmentation failed for {filename}: {e}")
        skipped += 1
        continue

    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # Save augmented image & label
    cv2.imwrite(out_img_path, aug_img)
    with open(out_lbl_path, "w") as f:
        for cls, (x, y, bw, bh) in zip(aug_labels, aug_bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ Saved {out_img_path} and {out_lbl_path}")
    processed += 1

mapping_f.close()
print(f"\n🎉 Augmentation finished. Processed: {processed}, Skipped: {skipped}")
print(f"📑 Filename mapping saved in: {mapping_file}")