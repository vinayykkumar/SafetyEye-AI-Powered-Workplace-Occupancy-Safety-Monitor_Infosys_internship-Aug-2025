from PIL import Image
import os

input_folder = 'data/cleaned/images'
output_folder = 'data/rgb/images'
os.makedirs(output_folder, exist_ok=True)
for fname in os.listdir(input_folder):
    img_path = os.path.join(input_folder, fname)
    img = Image.open(img_path).convert('RGB')
    img.save(os.path.join(output_folder, fname))
