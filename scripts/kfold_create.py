import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Path to dataset root
DATASET_PATH = Path("data/dataset")

# Collect all image paths
image_paths = list((DATASET_PATH / "images" / "train").glob("*.jpg"))

# Number of folds
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"\n========== Fold {fold+1} ==========\n")

    fold_dir = Path(f"data/kfold_{fold}")
    train_img_dir = fold_dir / "images/train"
    val_img_dir = fold_dir / "images/val"
    train_lbl_dir = fold_dir / "labels/train"
    val_lbl_dir = fold_dir / "labels/val"

    # Create directories
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split images
    for idx in train_idx:
        img_path = image_paths[idx]
        shutil.copy(img_path, train_img_dir)

        label_path = DATASET_PATH / "labels/train" / (img_path.stem + ".txt")
        shutil.copy(label_path, train_lbl_dir)

    for idx in val_idx:
        img_path = image_paths[idx]
        shutil.copy(img_path, val_img_dir)

        label_path = DATASET_PATH / "labels/train" / (img_path.stem + ".txt")
        shutil.copy(label_path, val_lbl_dir)

    # Create dataset yaml for this fold
    fold_yaml = fold_dir / "dataset.yaml"
    yaml_data = {
        "path": str(fold_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pothole"},
    }

    with open(fold_yaml, "w") as f:
        yaml.dump(yaml_data, f)
