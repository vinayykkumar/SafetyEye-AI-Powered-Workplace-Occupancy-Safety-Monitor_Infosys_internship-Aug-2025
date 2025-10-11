from PIL import Image
import os
import shutil

input_image_folder = 'data/unique_cleaned/images'
output_image_folder = 'data/normalized/images'
input_label_folder = 'data/unique_cleaned/labels'
output_label_folder = 'data/normalized/labels'

target_size = (640, 640)

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

for fname in os.listdir(input_image_folder):
    img_path = os.path.join(input_image_folder, fname)
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        img = img.resize(target_size)
        img.save(os.path.join(output_image_folder, fname))

for label_file in os.listdir(input_label_folder):
    shutil.copy(os.path.join(input_label_folder, label_file), os.path.join(output_label_folder, label_file))

print("Resizing normalization complete. Images and labels saved in data/normalized/")
