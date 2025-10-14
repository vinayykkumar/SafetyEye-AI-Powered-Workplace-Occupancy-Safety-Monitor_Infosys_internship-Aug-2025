import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def preprocess_image_cv2(image_path, target_size=(224, 224)):
    """
    Load and preprocess image using OpenCV.
    Steps: read -> resize -> normalize -> convert to RGB
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)

    img = img.astype(np.float32) / 255.0

    return img

torch_transform = transforms.Compose([
    transforms.Resize((224, 224)),             
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),           
    transforms.RandomHorizontalFlip(p=0.5),     
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) 
])

def preprocess_image_torch(image_path):
    """
    Preprocess image using Torchvision transforms (deep learning friendly).
    """
    img = Image.open(image_path).convert("RGB")
    return torch_transform(img)
##neagtive class
## no helmet 
