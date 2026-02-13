"""Fine-tune YOLOv8 model from existing weights (Windows-friendly).

This script will:
 - load your existing weights from 'weights/best.pt'
 - fine-tune on the dataset described in 'data.yaml'
 - use a lower learning rate for fine-tuning
 - save trained weights under 'runs/finetune'

Notes:
 - Make sure you have run `prepare_dataset.py` first to create dataset/ images/labels folders.
 - This script uses the ultralytics package (YOLOv8).
"""

from pathlib import Path
import shutil
import glob
import sys
from ultralytics import YOLO

# Paths (Windows-compatible). Adjust if your project root is different.
ROOT = Path(__file__).parent.resolve()
WEIGHTS_DIR = ROOT / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)
EXISTING_WEIGHTS = WEIGHTS_DIR / 'best.pt'  # your current trained model

DATA_YAML = ROOT / 'data.yaml'  # should be created by prepare_dataset.py

if not EXISTING_WEIGHTS.exists():
    print(f"ERROR: expected weights at {EXISTING_WEIGHTS}. Place your best.pt there.")
    sys.exit(1)

if not DATA_YAML.exists():
    print(f"ERROR: expected data yaml at {DATA_YAML}. Run prepare_dataset.py first.")
    sys.exit(1)

# Load model from existing weights to fine-tune
model = YOLO(str(EXISTING_WEIGHTS))

# Training hyperparameters tuned for fine-tuning
FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-4  # lower learning rate for fine-tuning
IMGSZ = 640
BATCH = 16

print("Starting fine-tuning from:", EXISTING_WEIGHTS)

results = model.train(
    data=str(DATA_YAML),
    epochs=FINETUNE_EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    lr=FINETUNE_LR,
    project=str(ROOT / 'runs'),
    name='finetune',
    exist_ok=True,  # overwrite if run exists
)

print("Training finished. Validating model...")
val_results = model.val()

# Attempt to copy the best weights from the run folder to weights/finetuned_best.pt
run_weights_glob = str(ROOT / 'runs' / 'finetune' / '*/weights/best.pt')
matches = glob.glob(run_weights_glob)
if matches:
    dest = WEIGHTS_DIR / 'finetuned_best.pt'
    shutil.copyfile(matches[-1], dest)
    print(f"Saved fine-tuned best weights to: {dest}")
else:
    print("Could not locate best.pt in the run folder. Check runs/finetune/ for outputs.")

print("Done.")
