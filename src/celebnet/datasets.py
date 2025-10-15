# src/celebnet/datasets.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import json, re


def _clean_label(s: str) -> str:
    """
    Convert folder names like 'images__4561_' -> '4561'.
    If no digits are found, returns the original string.
    """
    nums = re.findall(r"\d+", s)
    return nums[-1] if nums else s


def make_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return train_tf, eval_tf


def build_loaders(processed_root: Path, batch=64, num_workers=2, pin=True):
    """
    Builds train/val/test loaders from outputs/processed/* and
    writes a cleaned labels.json mapping {<numeric_id>: index}.
    """
    processed_root = Path(processed_root)
    train_tf, eval_tf = make_transforms()

    train_ds = datasets.ImageFolder(processed_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(processed_root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(processed_root / "test",  transform=eval_tf)

    # Raw map from ImageFolder uses folder names verbatim.
    # Clean them to only keep the numeric ID.
    raw_map = train_ds.class_to_idx                   # e.g., {'images__4561_': 23, ...}
    label2idx = {_clean_label(k): v for k, v in raw_map.items()}

    # Save cleaned mapping next to models
    models_root = processed_root.parent / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    with open(models_root / "labels.json", "w") as f:
        json.dump(label2idx, f, indent=2)

    loaders = dict(
        train=DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=pin),
        val  =DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin),
        test =DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin),
    )
    return loaders, label2idx
