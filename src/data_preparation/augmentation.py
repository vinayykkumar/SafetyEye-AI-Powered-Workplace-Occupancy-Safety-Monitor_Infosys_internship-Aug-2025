# import os
# import cv2
# import random
# from collections import Counter

# # ===================== CONFIG =====================
# images_dir = "../../archive/data/train/images"    # original images
# labels_dir = "../../archive/data/train/labels"    # original labels

# output_images_dir = "../../archive/data/train/images_aug3"  # save augmented images
# output_labels_dir = "../../archive/data/train/labels_aug3"  # save augmented labels

# target_count = 5000   # desired minimum per class
# skip_classes = [5]    # 🚫 classes to skip from augmentation
# os.makedirs(output_images_dir, exist_ok=True)
# os.makedirs(output_labels_dir, exist_ok=True)
# # ==================================================


# # ----------- Helper Functions --------------------
# def load_labels(path):
#     """Load YOLO labels and return [cls, x, y, w, h]"""
#     boxes = []
#     with open(path) as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue  # skip malformed lines
#             cls, x, y, bw, bh = map(float, parts)
#             boxes.append([int(cls), x, y, bw, bh])
#     return boxes

# def yolo_to_xyxy(cls, x, y, w, h, img_w, img_h):
#     """Convert YOLO (cx,cy,w,h) to (x1,y1,x2,y2)"""
#     x1 = int((x - w/2) * img_w)
#     y1 = int((y - h/2) * img_h)
#     x2 = int((x + w/2) * img_w)
#     y2 = int((y + h/2) * img_h)
#     return cls, x1, y1, x2, y2

# def xyxy_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
#     """Convert (x1,y1,x2,y2) to YOLO (cx,cy,w,h) string"""
#     x = ((x1 + x2) / 2) / img_w
#     y = ((y1 + y2) / 2) / img_h
#     w = (x2 - x1) / img_w
#     h = (y2 - y1) / img_h
#     return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
# # --------------------------------------------------


# # Step 1: Count current class distribution
# class_counts = Counter()
# for label_file in os.listdir(labels_dir):
#     if label_file.endswith(".txt"):
#         with open(os.path.join(labels_dir, label_file)) as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) != 5:
#                     continue
#                 cls = int(parts[0])
#                 class_counts[cls] += 1

# print("Class distribution before augmentation:", class_counts)

# # Step 2: Find deficit classes
# deficit_classes = {cls: target_count - count for cls, count in class_counts.items() if count < target_count}
# print("Classes needing augmentation:", deficit_classes)


# # Step 3: Augmentation loop
# aug_counter = Counter()

# while any(deficit > 0 for deficit in deficit_classes.values()):
#     for img_file in os.listdir(images_dir):
#         if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
#             continue

#         label_file = os.path.splitext(img_file)[0] + ".txt"
#         label_path = os.path.join(labels_dir, label_file)
#         if not os.path.exists(label_path):
#             continue

#         img = cv2.imread(os.path.join(images_dir, img_file))
#         h, w = img.shape[:2]
#         boxes = load_labels(label_path)

#         # Collect minority crops
#         crops = []
#         for cls, x, y, bw, bh in boxes:
#             if cls in skip_classes:
#                 continue  # 🚫 skip unwanted classes
#             if cls in deficit_classes and deficit_classes[cls] > 0:
#                 cls, x1, y1, x2, y2 = yolo_to_xyxy(cls, x, y, bw, bh, w, h)
#                 crop = img[y1:y2, x1:x2]
#                 if crop.size > 0:
#                     crops.append((cls, crop))

#         if not crops:
#             continue

#         # Pick random background
#         bg_file = random.choice(os.listdir(images_dir))
#         bg_img = cv2.imread(os.path.join(images_dir, bg_file))
#         bh, bw = bg_img.shape[:2]

#         bg_label_file = os.path.splitext(bg_file)[0] + ".txt"
#         bg_label_path = os.path.join(labels_dir, bg_label_file)
#         new_labels = []
#         if os.path.exists(bg_label_path):
#             with open(bg_label_path) as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         new_labels.append(line.strip() + "\n")

#         for cls, crop in crops:
#             if deficit_classes[cls] <= 0:
#                 continue

#             ch, cw = crop.shape[:2]
#             if ch == 0 or cw == 0 or ch > bh or cw > bw:
#                 continue

#             # Random paste location
#             x_offset = random.randint(0, bw - cw)
#             y_offset = random.randint(0, bh - ch)

#             # Paste crop
#             bg_img[y_offset:y_offset+ch, x_offset:x_offset+cw] = crop
#             new_labels.append(xyxy_to_yolo(cls, x_offset, y_offset, x_offset+cw, y_offset+ch, bw, bh))

#             deficit_classes[cls] -= 1
#             aug_counter[cls] += 1

#         # Save augmented result
#         out_img = os.path.join(output_images_dir, f"aug_{random.randint(0,999999)}.jpg")
#         out_lbl = os.path.join(output_labels_dir, os.path.basename(out_img).replace(".jpg", ".txt"))
#         cv2.imwrite(out_img, bg_img)

#         with open(out_lbl, "w") as f:
#             for lbl in new_labels:  # ✅ ensures one line per object
#                 f.write(lbl)

#     print("Remaining deficit:", deficit_classes)

# print("✅ Final augmentation counts:", aug_counter)
# print("🎯 All classes now at least ~5000 instances.")







import os
import random

# List of label directories (YOLO .txt annotation folders)
label_dirs = [
    "../../archive/data/train/labels",
    "../../archive/data/train/labels_aug3"
]

target_class = 5                # Class to reduce
max_instances = 10000           # Desired max number of objects for target_class

# Step 1: Collect all target_class annotations across both dirs
annotations = []  # (filepath, line_index, line_content)
for labels_dir in label_dirs:
    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            filepath = os.path.join(labels_dir, file)
            with open(filepath, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    class_id = int(line.split()[0])
                    if class_id == target_class:
                        annotations.append((filepath, idx, line))

print(f"🔎 Found {len(annotations)} annotations of class {target_class} in all dirs.")

# Step 2: Randomly shuffle and select which annotations to keep
random.shuffle(annotations)
to_keep = set(annotations[:max_instances])

# Step 3: Rewrite label files (remove only extra target_class lines)
for labels_dir in label_dirs:
    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            filepath = os.path.join(labels_dir, file)
            with open(filepath, "r") as f:
                lines = f.readlines()

            new_lines = []
            for idx, line in enumerate(lines):
                class_id = int(line.split()[0])
                if class_id != target_class:
                    new_lines.append(line)  # keep all other classes
                else:
                    if (filepath, idx, line) in to_keep:
                        new_lines.append(line)
                        to_keep.remove((filepath, idx, line))
                    # else -> skip this class 5 line

            with open(filepath, "w") as f:
                f.writelines(new_lines)

print(f"✅ Finished! Reduced class {target_class} to {max_instances} instances across all dirs.")
