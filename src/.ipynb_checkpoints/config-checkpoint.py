import os

class CFG:
    RANDOM_SEED = 42
    CLASSES = [
        'Fall-Detected', 'Gloves', 'Goggles', 'Hardhat',
        'Ladder', 'Mask', 'NO-Gloves', 'NO-Goggles',
        'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'Safety Cone', 'Safety Vest'
    ]
    
    DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    RESIZE_DIM = 640
    PARALLEL_PROCESSING = True

    @staticmethod
    def ensure_paths():
        os.makedirs(CFG.OUTPUT_PATH, exist_ok=True)
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(CFG.DATASET_ROOT, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(CFG.DATASET_ROOT, split, 'labels'), exist_ok=True)
