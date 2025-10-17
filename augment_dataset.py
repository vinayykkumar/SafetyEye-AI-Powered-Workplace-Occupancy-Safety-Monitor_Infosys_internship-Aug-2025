import os
import cv2
import albumentations as A

# ------------------- Paths -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_IMAGES = os.path.join(BASE_DIR, "normalized_dataset", "images")
INPUT_LABELS = os.path.join(BASE_DIR, "normalized_dataset", "labels")
OUTPUT_IMAGES = os.path.join(BASE_DIR, "augmented_dataset", "images")
OUTPUT_LABELS = os.path.join(BASE_DIR, "augmented_dataset", "labels")

# Create output folders
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# ------------------- Albumentations Transform -------------------
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5)
], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.2
    ))

# ------------------- Helpers -------------------
def clip_value(val):
    """Clip single value to [0,1]."""
    return max(0.0, min(1.0, val))

def clip_yolo_bbox(x, y, w, h):
    """Clip YOLO bbox center-format to keep within [0,1]."""
    x = clip_value(x)
    y = clip_value(y)
    w = clip_value(w)
    h = clip_value(h)
    return x, y, w, h

# ------------------- Augmentation -------------------
for img_file in os.listdir(INPUT_IMAGES):
    if img_file.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(INPUT_IMAGES, img_file)
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(INPUT_LABELS, label_file)

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)

        # Load YOLO labels
        bboxes = []
        class_labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                # Clip YOLO coords to [0,1] BEFORE giving to Albumentations
                x, y, w, h = clip_yolo_bbox(x, y, w, h)
                bboxes.append([x, y, w, h])
                class_labels.append(cls)

        # Skip if no valid boxes
        if not bboxes:
            continue

        try:
            # Apply augmentation
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except ValueError as e:
            print(f"⚠️ Skipping {img_file} due to invalid bbox after augmentation: {e}")
            continue

        # Save augmented image
        aug_img = augmented["image"]
        cv2.imwrite(os.path.join(OUTPUT_IMAGES, img_file), aug_img)

        # Save augmented labels
        with open(os.path.join(OUTPUT_LABELS, label_file), "w") as f:
            for cls, bbox in zip(augmented["class_labels"], augmented["bboxes"]):
                x, y, w, h = clip_yolo_bbox(*bbox)
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("✅ Augmentation done! Clean images + labels saved in /augmented_dataset/")
