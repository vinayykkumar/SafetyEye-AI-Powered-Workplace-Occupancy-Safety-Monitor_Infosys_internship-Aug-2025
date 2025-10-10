import os
import logging
from splitter import create_data_yaml
from stats_and_graphs import compute_class_statistics
from visualization import plot_class_distribution, plot_samples_with_and_without_boxes
from config import CFG
from train import train
from preprocessing import preprocess_images

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CFG.OUTPUT_PATH, 'main.log')),
        logging.StreamHandler()
    ]
)

def main():
    # Verify dataset directory structure
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(CFG.DATASET_ROOT, split, 'images')
        labels_dir = os.path.join(CFG.DATASET_ROOT, split, 'labels')
        if not os.path.exists(images_dir):
            logging.warning(f"Images directory {images_dir} does not exist. Skipping {split} split.")
            continue
        if not os.path.exists(labels_dir):
            logging.warning(f"Labels directory {labels_dir} does not exist. Skipping {split} split.")
            continue
    CFG.ensure_paths()
    
    # Preprocess images
    logging.info("Preprocessing images...")
    try:
        processed_root, processed_splits = preprocess_images()
    except RuntimeError as e:
        logging.error(f"Preprocessing failed: {e}")
        raise
    
    if not processed_splits:
        raise RuntimeError("No valid splits available after preprocessing. Cannot proceed.")
    
    CFG.DATASET_ROOT = processed_root
    logging.info(f"Updated CFG.DATASET_ROOT to {CFG.DATASET_ROOT}")
    # Create data.yaml for processed dataset
    logging.info("Updating data.yaml for processed dataset...")
    create_data_yaml(dataset_root=processed_root)
    
    # Check for example image and verify processed directories
    train_img_path = os.path.join(CFG.DATASET_ROOT, "train", "images")
    if os.path.exists(train_img_path) and len(os.listdir(train_img_path)) > 0:
        example_img = os.path.join(train_img_path, os.listdir(train_img_path)[0])
        logging.info(f"Example image: {example_img}")
    else:
        logging.warning("No images found in train/images!")
        example_img = None
    
    # Compute and visualize dataset statistics on processed dataset
    logging.info("Computing dataset statistics...")
    stats_df = compute_class_statistics()
    if stats_df is not None and not stats_df.empty and stats_df['Total_Files'].sum() > 0:
        logging.info("Generating class distribution plot...")
        plot_class_distribution(stats_df)
        logging.info("Showing sample images...")
        if os.path.exists(train_img_path):
            plot_samples_with_and_without_boxes(split="train", num_samples=4, class_list=CFG.CLASSES)
        else:
            logging.warning("Train images not found, skipping sample visualization.")
    else:
        logging.warning(f"No statistics to plot! Check labels in {CFG.DATASET_ROOT}/[split]/labels.")
    
    # Start training
    logging.info("Initiating training process...")
    for split in processed_splits:
        images_dir = os.path.join(CFG.DATASET_ROOT, split, 'images')
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            logging.error(f"Images directory {images_dir} is missing or empty. Cannot proceed with training.")
            raise FileNotFoundError(f"Images directory {images_dir} is missing or empty.")
    train()

if __name__ == "__main__":
    main()