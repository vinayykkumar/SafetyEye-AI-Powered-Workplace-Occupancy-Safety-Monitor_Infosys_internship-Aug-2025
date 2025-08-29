import os
import cv2
import albumentations as A

# Paths — update to your project
images_dir = r"C:\Users\mkr19\Desktop\SafetyEye\SafetyEye\data\train\images"
labels_dir = r"C:\Users\mkr19\Desktop\SafetyEye\SafetyEye\data\train\labels"

aug_images_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\aug_images"
aug_labels_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\aug_labels"

os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(p=0.2),
    A.ToGray(p=0.2),                   # grayscale
   A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.3),  # crop + resize

    A.Resize(640, 640),                # final resize (important!)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def load_labels(label_path):
    bboxes, class_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                bboxes.append([x, y, w, h])
                class_labels.append(int(cls))
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# Loop through all images
for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

    image = cv2.imread(img_path)
    if image is None:
        continue

    bboxes, class_labels = load_labels(label_path)

    # Apply augmentation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_classes = transformed['class_labels']

    # Save augmented image
    aug_img_path = os.path.join(aug_images_dir, f"aug_{img_file}")
    cv2.imwrite(aug_img_path, aug_img)

    # Save augmented labels
    aug_label_path = os.path.join(aug_labels_dir, f"aug_{os.path.splitext(img_file)[0]}.txt")
    save_labels(aug_label_path, aug_bboxes, aug_classes)

print("✅ Augmentation finished. All images are now 640x640.")
