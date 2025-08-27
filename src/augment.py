import os
import cv2
import albumentations as A

# Paths
DATASET_PATH = "data/processed"
SRC_SPLIT = "train"
AUG_SPLIT = "train_augmented"

IMG_DIR = os.path.join(DATASET_PATH, SRC_SPLIT, "images")
LBL_DIR = os.path.join(DATASET_PATH, SRC_SPLIT, "labels")

AUG_IMG_DIR = os.path.join(DATASET_PATH, AUG_SPLIT, "images")
AUG_LBL_DIR = os.path.join(DATASET_PATH, AUG_SPLIT, "labels")

# Create new folders
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LBL_DIR, exist_ok=True)


def save_augmented(img, aug_bboxes, aug_labels, img_name, aug_type):
    """Save augmented image and updated YOLO labels"""
    aug_img_name = img_name.replace(".jpg", f"_{aug_type}.jpg").replace(".png", f"_{aug_type}.png")
    aug_lbl_name = img_name.replace(".jpg", f"_{aug_type}.txt").replace(".png", f"_{aug_type}.txt")

    cv2.imwrite(os.path.join(AUG_IMG_DIR, aug_img_name), img)
    with open(os.path.join(AUG_LBL_DIR, aug_lbl_name), "w") as f:
        for cls, (x, y, bw, bh) in zip(aug_labels, aug_bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")


print("🚀 Starting augmentation...")

for img_file in os.listdir(IMG_DIR):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)

        # Load YOLO labels
        bboxes, class_labels = [], []
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = parts
                bboxes.append([float(x), float(y), float(bw), float(bh)])
                class_labels.append(int(cls))

        if not bboxes:
            continue

        # ✅ Define augmentations with bbox_params (for YOLO format)
        transforms = {
            "hflip": A.Compose([A.HorizontalFlip(p=1)], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
            "vflip": A.Compose([A.VerticalFlip(p=1)], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
            "bright": A.Compose([A.RandomBrightnessContrast(p=1)], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
            "blur": A.Compose([A.Blur(blur_limit=5, p=1)], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
            "rot": A.Compose([A.Rotate(limit=15, p=1)], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
        }

        # Apply each augmentation separately
        for aug_type, transform in transforms.items():
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            save_augmented(aug_img, aug_bboxes, aug_labels, img_file, aug_type)

print("✅ Augmentation complete! Images saved in:", AUG_IMG_DIR)
