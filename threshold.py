# remove_tiny_boxes.py
import os, shutil
from pathlib import Path

IMAGES_DIR = Path("C:/safetyeye/preprocessed/val/images")
LABELS_DIR = Path("C:/safetyeye/preprocessed/val/labels")
BAD_DIR = Path("C:/safetyeye/preprocessed/bad_small_boxes")
BAD_DIR.mkdir(parents=True, exist_ok=True)

# set threshold (fraction of image area). e.g., 0.0005 = 0.05% of image area
MIN_BOX_AREA_FRAC = 0.0005

from PIL import Image

moved = 0
for txt in LABELS_DIR.glob("*.txt"):
    img_path = IMAGES_DIR / (txt.stem + ".jpg")
    if not img_path.exists():
        img_path = IMAGES_DIR / (txt.stem + ".png")
        if not img_path.exists():
            continue
    w,h = Image.open(img_path).size
    remove_flag = False
    for line in txt.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5: 
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        box_area = (bw * w) * (bh * h)
        if box_area < (MIN_BOX_AREA_FRAC * (w*h)):
            remove_flag = True
            break
    if remove_flag:
        shutil.move(str(txt), str(BAD_DIR / txt.name))
        if img_path.exists():
            shutil.move(str(img_path), str(BAD_DIR / img_path.name))
        moved += 1

print(f"Moved {moved} image+label pairs with tiny boxes to {BAD_DIR}")
