import cv2
import os

# Paths to your dataset
TRAIN_DIR = r"C:\Users\yuvan\SAFETYEYE\Dataset\images\train"
VAL_DIR = r"C:\Users\yuvan\SAFETYEYE\Dataset\images\val"

def preprocess_image_cv2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    # Resize for example (YOLO usually works with 640x640)
    img = cv2.resize(img, (640, 640))
    return img

def process_dataset(folder):
    print(f"\nProcessing images in {folder}")
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, file)
            try:
                cv2_img = preprocess_image_cv2(path)
                print(f"Processed: {file} ✅")
            except Exception as e:
                print(f"Skipping {file}: {e}")

def main():
    process_dataset(TRAIN_DIR)
    process_dataset(VAL_DIR)

if __name__ == "__main__":
    main()
