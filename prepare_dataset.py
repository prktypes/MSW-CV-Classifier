"""Prepare YOLOv8 dataset from a classification-style folder structure.

Assumptions and behavior:
- Source dataset is `dataset-resized/` with subfolders named after classes.
- If there are existing YOLO-format .txt label files in the same folders, they will be copied.
- If no label files exist, the script will create a single-object label that covers the whole image
  (class_id x_center y_center width height => 0.5 0.5 1.0 1.0). This is a best-effort conversion
  from classification to detection datasets. If you want precise bounding boxes, label images
  with LabelImg or Roboflow before training.

Usage (PowerShell / Windows):
    python prepare_dataset.py --src "dataset-resized" --dst "dataset" --val-split 0.2

The script will create:
 dataset/
   images/train/
   images/val/
   labels/train/
   labels/val/
and will copy/create label files alongside images.
"""

from pathlib import Path
import random
import shutil
import argparse
from PIL import Image


CLASSES = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash',
]


def is_image_file(p: Path):
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def prepare(src: Path, dst: Path, val_split: float = 0.2, seed: int = 42):
    random.seed(seed)
    src = Path(src)
    dst = Path(dst)

    images_train = dst / 'images' / 'train'
    images_val = dst / 'images' / 'val'
    labels_train = dst / 'labels' / 'train'
    labels_val = dst / 'labels' / 'val'

    for p in (images_train, images_val, labels_train, labels_val):
        p.mkdir(parents=True, exist_ok=True)

    # Iterate class folders
    for class_id, class_name in enumerate(CLASSES):
        class_dir = src / class_name
        if not class_dir.exists():
            print(f"Warning: expected class folder {class_dir} not found. Skipping.")
            continue

        files = [p for p in class_dir.iterdir() if is_image_file(p)]
        random.shuffle(files)

        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for dataset_files, img_out_dir, lbl_out_dir in [
            (train_files, images_train, labels_train),
            (val_files, images_val, labels_val),
        ]:
            for img_path in dataset_files:
                # Copy image
                out_img = img_out_dir / img_path.name
                shutil.copyfile(img_path, out_img)

                # If a corresponding label file exists in source folder, copy it.
                src_label = img_path.with_suffix('.txt')
                out_label = lbl_out_dir / src_label.name
                if src_label.exists():
                    shutil.copyfile(src_label, out_label)
                else:
                    # Create a default full-image bbox (class_id center_x center_y w h)
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                    except Exception:
                        # If image can't be opened, still write label assuming full image
                        w, h = 1, 1

                    # YOLO format uses normalized center x,y and width,height (0..1)
                    line = f"{class_id} 0.5 0.5 1.0 1.0\n"
                    out_label.write_text(line)

    print(f"Dataset prepared under {dst}\nImages: {images_train} and {images_val}\nLabels: {labels_train} and {labels_val}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='dataset-resized')
    parser.add_argument('--dst', type=str, default='dataset')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    prepare(Path(args.src), Path(args.dst), val_split=args.val_split, seed=args.seed)


if __name__ == '__main__':
    main()
