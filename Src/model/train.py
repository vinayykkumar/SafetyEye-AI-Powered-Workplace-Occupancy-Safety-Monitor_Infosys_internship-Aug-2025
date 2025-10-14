import os
from ultralytics import YOLO
import cv2

train_images_folder = r"D:/CompanyProject/SafetyEye/preprocessing/splitted_dataset/train/images"

first_image = sorted(os.listdir(train_images_folder))[0]
img_path = os.path.join(train_images_folder, first_image)

model = YOLO("best.pt")  

results = model(img_path, conf=0.5) 

results[0].show()   

save_path = "output_detected.jpg"
results.save(save_path)
print(f"✅ Detection saved at {save_path}")
