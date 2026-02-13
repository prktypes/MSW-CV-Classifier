"""Sanity-check utilities for the YOLO dataset.

Checks included:
- missing label files for images
- empty label files (files with zero bytes or no lines)
- corrupted/unopenable images

Usage examples (PowerShell):
    python checks.py --data dataset
"""

from pathlib import Path
from PIL import Image
import argparse


def find_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')]


def check_dataset(root: Path):
    images_train = root / 'images' / 'train'
    images_val = root / 'images' / 'val'
    labels_train = root / 'labels' / 'train'
    labels_val = root / 'labels' / 'val'

    problems = False

    for img_dir, lbl_dir in [(images_train, labels_train), (images_val, labels_val)]:
        if not img_dir.exists():
            print(f"Missing folder: {img_dir}")
            problems = True
            continue

        for img in find_images(img_dir):
            lbl = lbl_dir / (img.stem + '.txt')
            if not lbl.exists():
                print(f"Missing label for image: {img}")
                problems = True
            else:
                if lbl.stat().st_size == 0:
                    print(f"Empty label file: {lbl}")
                    problems = True
                else:
                    content = lbl.read_text().strip()
                    if content == '':
                        print(f"Label has no content: {lbl}")
                        problems = True

            # Try opening the image
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                print(f"Corrupted or unreadable image: {img}")
                problems = True

    if not problems:
        print("No problems found. Dataset looks healthy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset')
    args = parser.parse_args()

    check_dataset(Path(args.data))


if __name__ == '__main__':
    main()
