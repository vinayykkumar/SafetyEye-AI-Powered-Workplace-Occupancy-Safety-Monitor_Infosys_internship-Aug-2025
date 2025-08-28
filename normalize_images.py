import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def normalize_and_save_training_images(processed_dir):
    print("\n--- Phase 5: Manually Normalizing Training Images ---")
    
    train_images_path = Path(processed_dir) / "images" / "train"
    normalized_output_path = Path(processed_dir) / "normalized_images" / "train"
    
    normalized_output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(train_images_path.glob("*.*"))
    
    for img_path in tqdm(image_files, desc="Normalizing images"):
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32)
        normalized_image = image_float / 255.0
        
        output_npy_path = normalized_output_path / (img_path.stem + ".npy")
        np.save(output_npy_path, normalized_image)
        
    print(f"\nSuccessfully normalized {len(image_files)} images.")
    print(f"Normalized arrays saved in: {normalized_output_path}")

if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "../dataset/processed"
    normalize_and_save_training_images(OUTPUT_DIR)
