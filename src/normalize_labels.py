import os

# Paths
DATASET_PATH = "data/processed"

# Iterate over splits
for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        img_file = os.path.join(img_dir, label_file.replace(".txt", ".jpg"))
        if not os.path.exists(img_file):
            img_file = os.path.join(img_dir, label_file.replace(".txt", ".png"))
        if not os.path.exists(img_file):
            continue  # skip if no matching image

        # Get image size
        import cv2
        img = cv2.imread(img_file)
        h, w = img.shape[:2]

        # Read label file
        label_path = os.path.join(label_dir, label_file)
        new_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)

                # Check if values are >1 (pixel coords)
                if x > 1 or y > 1 or bw > 1 or bh > 1:
                    # Convert pixel → YOLO format
                    x_center = (x + bw/2) / w
                    y_center = (y + bh/2) / h
                    width = bw / w
                    height = bh / h
                    new_lines.append(f"{int(cls)} {x_center} {y_center} {width} {height}\n")
                else:
                    # Already normalized
                    new_lines.append(line)

        # Overwrite with normalized labels
        with open(label_path, "w") as f:
            f.writelines(new_lines)

print(" Normalization complete for all splits.")
