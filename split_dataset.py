import os
import shutil
from sklearn.model_selection import train_test_split

# ------------------- Paths -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Correct path to cleaned dataset
dataset_images = os.path.join(BASE_DIR, "Dataset", "images")
dataset_labels = os.path.join(BASE_DIR, "Dataset", "labels")

# Output folder for splitted data
output_base = os.path.join(BASE_DIR, "data")
train_images = os.path.join(output_base, "train", "images")
train_labels = os.path.join(output_base, "train", "labels")
val_images = os.path.join(output_base, "valid", "images")
val_labels = os.path.join(output_base, "valid", "labels")
test_images = os.path.join(output_base, "test", "images")
test_labels = os.path.join(output_base, "test", "labels")

# Create directories
for path in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
    os.makedirs(path, exist_ok=True)

# ------------------- Split Dataset -------------------
image_files = [f for f in os.listdir(dataset_images) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort for consistency
image_files.sort()

# Split into train (70%), val (20%), test (10%)
train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

def copy_files(file_list, src_img, src_lbl, dst_img, dst_lbl):
    for file in file_list:
        img_src = os.path.join(src_img, file)
        lbl_src = os.path.join(src_lbl, file.rsplit('.', 1)[0] + ".txt")

        if os.path.exists(lbl_src):  # Only copy if label exists
            shutil.copy(img_src, dst_img)
            shutil.copy(lbl_src, dst_lbl)

# Copy files
copy_files(train_files, dataset_images, dataset_labels, train_images, train_labels)
copy_files(val_files, dataset_images, dataset_labels, val_images, val_labels)
copy_files(test_files, dataset_images, dataset_labels, test_images, test_labels)

print("✅ Dataset successfully split into train, val, and test sets inside 'data' folder.")
print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

