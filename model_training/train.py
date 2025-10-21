# train.py

import os
from ultralytics import YOLO

def main():
    # --- Configuration ---
    # Update this path to your 'data.yaml' file.
    data_yaml_path = 'data/raw/data.yaml' 

    model_variant = 'yolov8n.pt'
    num_epochs = 50
    image_size = 640
    
    # Let's use the GPU! A batch size of 16 is good for a 4GB GPU.
    batch_size = 16   

    num_workers = 4

    # --- End of Configuration ---

    if not os.path.exists(data_yaml_path):
        print(f"Error: The file '{data_yaml_path}' does not exist.")
        return

    model = YOLO(model_variant)

    print("GPU acceleration is active! Starting training...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=num_epochs,
        imgsz=image_size,
        batch=batch_size,
        workers = num_workers,
        name='yolov8n_ppe_gpu_run_1' # New name for our first GPU run
    )
    
    print("Training finished!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()