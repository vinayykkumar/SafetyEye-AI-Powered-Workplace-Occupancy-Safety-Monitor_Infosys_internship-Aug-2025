import os
import cv2
import numpy as np

#  Paths 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_IMAGES = os.path.join(BASE_DIR, "dataset", "images")
OUTPUT_IMAGES = os.path.join(BASE_DIR, "normalized_dataset", "images")
OUTPUT_LABELS = os.path.join(BASE_DIR, "normalized_dataset", "labels")

# Create output folders
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

#  Normalize Function 
def normalize_image(img_path, save_path):
    img = cv2.imread(img_path).astype(np.float32) / 255.0  # scale to [0,1]
    # Convert back to 0–255 for saving (otherwise OpenCV won’t save correctly)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(save_path, img)

#run normallization
for img_file in os.listdir(INPUT_IMAGES):
    if img_file.endswith((".jpg", ".png", ".jpeg")):
        input_img_path = os.path.join(INPUT_IMAGES, img_file)
        output_img_path = os.path.join(OUTPUT_IMAGES, img_file)

        normalize_image(input_img_path, output_img_path)

        # Copy labels 
        label_file = img_file.rsplit('.', 1)[0] + ".txt"
        input_label_path = os.path.join(BASE_DIR, "dataset", "labels", label_file)
        if os.path.exists(input_label_path):
            output_label_path = os.path.join(OUTPUT_LABELS, label_file)
            with open(input_label_path, "r") as f_in, open(output_label_path, "w") as f_out:
                f_out.writelines(f_in.readlines())

print("✅ Normalization done! Images saved in /normalized_dataset/")

