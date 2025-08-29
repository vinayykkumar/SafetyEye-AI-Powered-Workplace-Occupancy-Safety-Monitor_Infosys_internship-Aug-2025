import cv2
import os
import glob
import albumentations as A

# Paths (only augment training set)
image_dir = "../../archive/data/train/images"
label_dir = "../../archive/data/train/labels"
output_img_dir = "../../archive/data/train/images_aug"
output_lbl_dir = "../../archive/data/train/labels_aug"

# Create output folders if not exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=20, p=0.5),
    A.MotionBlur(p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Collect all image files (.jpg and .png)
images = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
print(f"🔎 Found {len(images)} images in {image_dir}")

processed, skipped = 0, 0

# Loop over images
for img_path in images:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n➡️ Processing {filename}")

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Could not read image: {img_path}")
        skipped += 1
        continue

    # Load label (YOLO format: class x_center y_center width height)
    label_path = os.path.join(label_dir, filename + ".txt")
    if not os.path.exists(label_path):
        print(f"⚠️ Skipping {filename} (label not found)")
        skipped += 1
        continue

    bboxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            cls, x, y, bw, bh = parts
            bboxes.append([float(x), float(y), float(bw), float(bh)])
            class_labels.append(int(cls))

    if not bboxes:
        print(f"⚠️ Skipping {filename} (no bounding boxes)")
        skipped += 1
        continue

    # Apply augmentation
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f"⚠️ Augmentation failed for {filename}: {e}")
        skipped += 1
        continue

    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # Save augmented image
    out_img_path = os.path.join(output_img_dir, filename + "_aug.jpg")
    cv2.imwrite(out_img_path, aug_img)

    # Save augmented labels
    out_lbl_path = os.path.join(output_lbl_dir, filename + "_aug.txt")
    with open(out_lbl_path, "w") as f:
        for cls, (x, y, bw, bh) in zip(aug_labels, aug_bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ Saved {out_img_path} and {out_lbl_path}")
    processed += 1

print(f"\n🎉 Augmentation finished. Processed: {processed}, Skipped: {skipped}")
