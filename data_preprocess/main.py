
import os
import cv2
import numpy as np
import pickle
import shutil
from data_utils import get_image_label_paths, remove_duplicates
from image_ops import preprocess_image_and_label

if __name__ == '__main__':
    sets = ['train', 'valid', 'test']

    for data_set in sets:
        print(f"\n--- Preprocessing {data_set} set ---")
        
        # Step 1: Get image-label pairs
        raw_base_path = os.path.join('data', data_set)
        data_paths = get_image_label_paths(raw_base_path)
        
        # Step 2: Remove duplicates only for the training set (as per your original code)
        if data_set == 'train':
            data_paths = remove_duplicates(data_paths)
            
        print(f"Found {len(data_paths)} unique image-label pairs.")
        
        # Define and create output directories
        output_base_path = os.path.join('processed_data', data_set)
        output_image_dir = os.path.join(output_base_path, 'images')
        output_label_dir = os.path.join(output_base_path, 'labels')
        
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # Step 3: Loop, preprocess, and save
        for img_path, label_path in data_paths:
            preprocessed_img, preprocessed_labels = preprocess_image_and_label(img_path, label_path)
            
            if preprocessed_img is not None:
                img_filename = os.path.basename(img_path)
                label_filename = os.path.splitext(img_filename)[0] + '.txt'

                # Save the preprocessed image by converting it back to 8-bit unsigned integers
                cv2.imwrite(os.path.join(output_image_dir, img_filename), (preprocessed_img * 255).astype(np.uint8))
                
                # Save the preprocessed labels
                with open(os.path.join(output_label_dir, label_filename), 'w') as f:
                    for label in preprocessed_labels:
                        f.write(' '.join(map(str, label)) + '\n')

        print(f"Preprocessed data stored in: {output_base_path}")