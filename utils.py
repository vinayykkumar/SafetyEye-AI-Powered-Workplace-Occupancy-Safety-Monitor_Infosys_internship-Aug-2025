import os

def get_data_split_paths(root='data/processed', split='train'):
    """
    Returns paths for images and labels folder for the given split.
    """
    images_path = os.path.join(root, split, 'images')
    labels_path = os.path.join(root, split, 'labels')
    return images_path, labels_path

def count_images_labels(images_path, labels_path):
    num_images = len([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
    num_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
    return num_images, num_labels
