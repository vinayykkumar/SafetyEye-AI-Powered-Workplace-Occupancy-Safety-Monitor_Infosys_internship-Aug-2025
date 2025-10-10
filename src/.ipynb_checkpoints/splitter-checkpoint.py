import os
import yaml
from config import CFG
import logging

def create_data_yaml(dataset_root=None):
    if dataset_root is None:
        dataset_root = CFG.DATASET_ROOT
    yaml_dict = {
        'nc': len(CFG.CLASSES),
        'names': CFG.CLASSES
    }
    for split in ['train', 'val', 'test']:
        split_dir = 'train' if split == 'val' else split  # Map 'val' to 'valid' for directory check
        images_path = os.path.join(dataset_root, split_dir, 'images')
        if os.path.exists(images_path):
            yaml_dict[split] = os.path.abspath(images_path)  # Use absolute path
            logging.info(f"Added {split} path: {yaml_dict[split]} to data.yaml")
    yaml_file_path = os.path.join(CFG.OUTPUT_PATH, 'data.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_dict, f)
    logging.info(f"YAML created at {yaml_file_path}")