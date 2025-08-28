# data/src/preprocess.py
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# ---------- Config ----------
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}   # your raw set uses 'valid'
IMG_EXTS = [".jpg", ".jpeg", ".png"]


def find_image(imgs_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = imgs_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_aug_pipeline(imgsz: int):
    """Photometric-only (bbox-safe) pipeline."""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.4),
        A.CLAHE(p=0.2),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.ImageCompression(quality_lower=65, p=0.3),
        A.Resize(height=imgsz, width=imgsz, interpolation=cv2.INTER_AREA),  # resize last for consistency
    ])


def resize_only(img, imgsz):
    return cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def process_split(split_key, raw_root: Path, out_root: Path, imgsz: int, aug_per_image: int):
    """Process one split; returns (n_images, n_labels, n_aug_images)."""
    raw_split = SPLIT_MAP[split_key]  # 'train' -> 'train', 'val' -> 'valid'
    src_img_dir = raw_root / raw_split / "images"
    src_lbl_dir = raw_root / raw_split / "labels"

    dst_img_dir = out_root / split_key / "images"
    dst_lbl_dir = out_root / split_key / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted([p for p in src_lbl_dir.glob("*.txt")])
    n_base, n_labels, n_aug = 0, 0, 0

    aug = build_aug_pipeline(imgsz)
    for lbl_path in tqdm(label_files, desc=f"[{split_key}] preprocessing"):
        stem = lbl_path.stem
        img_path = find_image(src_img_dir, stem)
        if img_path is None:
            # no matching image; skip
            continue

        # read & resize base image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base_resized = resize_only(img, imgsz)

        # save resized base image
        base_out_path = dst_img_dir / f"{stem}.jpg"
        cv2.imwrite(str(base_out_path), cv2.cvtColor(base_resized, cv2.COLOR_RGB2BGR))

        # copy the label 1:1
        shutil.copy(lbl_path, dst_lbl_dir / f"{stem}.txt")

        n_base += 1
        n_labels += 1

        # augment only on train split
        if split_key == "train" and aug_per_image > 0:
            for k in range(aug_per_image):
                aug_img = aug(image=img)["image"]
                aug_name = f"{stem}_aug{k}"
                aug_out_path = dst_img_dir / f"{aug_name}.jpg"
                cv2.imwrite(str(aug_out_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                # labels unchanged (photometric-only aug)
                shutil.copy(lbl_path, dst_lbl_dir / f"{aug_name}.txt")
                n_aug += 1

    return n_base, n_labels, n_aug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/css-data", help="raw dataset root")
    parser.add_argument("--out", type=str, default="preprocessed", help="output root")
    parser.add_argument("--imgsz", type=int, default=640, help="target size (square)")
    parser.add_argument("--augs", type=int, default=1, help="augmentations per train image")
    parser.add_argument("--clear", action="store_true", help="wipe output folder before processing")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)

    if args.clear:
        ensure_empty_dir(out_root)
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    totals = {}
    for split_key in ["train", "val", "test"]:
        n_base, n_labels, n_aug = process_split(split_key, raw_root, out_root, args.imgsz, args.augs)
        totals[split_key] = (n_base, n_labels, n_aug)

    print("\n✅ Done.")
    for k, (nb, nl, na) in totals.items():
        print(f"  {k:>5}: {nb} base imgs, {nl} labels, {na} augmented imgs")
    total_imgs = sum(nb for nb, _, _ in totals.values()) + sum(na for _, _, na in totals.values())
    print(f"\n📊 Total output images: {total_imgs}")
    print(f"📁 Output located at: {out_root.resolve()}")


if __name__ == "__main__":
    main()
