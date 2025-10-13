import cv2
import os

# Paths (update these if needed)
images_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\images"
labels_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\labels"
output_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\visual_check"
os.makedirs(output_dir, exist_ok=True)

# Correct class names and colors matching YAML
class_names = ["Person", "Helmet", "Vest", "Machinery", "Cone"]
colors = {
    0: (255, 0, 0),      # Person - Blue
    1: (0, 255, 0),      # Helmet - Green
    2: (255, 255, 0),    # Vest - Cyan
    3: (0, 165, 255),    # Machinery - Orange
    4: (255, 0, 255)     # Cone - Magenta
}

def visualize_labels(images_dir, labels_dir, output_dir, num_images=None):
    img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if num_images:
        img_files = img_files[:num_images]

    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:5])

                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - bw/2) * w)
                    y1 = int((y_center - bh/2) * h)
                    x2 = int((x_center + bw/2) * w)
                    y2 = int((y_center + bh/2) * h)

                    color = colors.get(cls_id, (255, 255, 255))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save image for visual check
        save_path = os.path.join(output_dir, img_file)
        cv2.imwrite(save_path, image)

    print(f"✅ Visualization complete. Check images in {output_dir}")

# Run visualization (optional: set num_images=20 for quick check)
visualize_labels(images_dir, labels_dir, output_dir, num_images=10)
