import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict

def load_config(config_path: str = "dataset_config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect_samples(config: dict, datasets_root: str = ".") -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []

    for dataset in config["datasets"]:
        if not dataset.get("enabled", True):
            print(f"[SKIP dataset]  {dataset['name']}")
            continue

        for sub in dataset["subdatasets"]:
            if not sub.get("enabled", True):
                print(f"[SKIP subdataset]  {dataset['name']} / {sub['name']}")
                continue

            base = Path(datasets_root) / sub["base_path"]

            for class_key, class_cfg in sub["classes"].items():
                if not class_cfg.get("enabled", True):
                    print(f"[SKIP class]  {dataset['name']} / {sub['name']} / {class_key}")
                    continue

                class_dir = base / class_cfg["path"]
                label     = class_cfg["label"]

                if not class_dir.exists():
                    print(f"[WARNING] Directory not found: {class_dir}")
                    continue

                found = 0
                for img_path in class_dir.rglob("*"):
                    if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        samples.append((str(img_path), label))
                        found += 1

                print(f"[OK]  {dataset['name']:40s} / {sub['name']:25s} / {class_key:20s}  →  {label:12s}  ({found} images)")

    if not samples:
        raise RuntimeError(
            "No images found — all directories were missing.\n"
            f"  datasets_root = '{datasets_root}'\n"
            "  Check that this folder contains your dataset subfolders, "
            "and that base_path values in the JSON do NOT repeat the root prefix."
        )

    return samples

def build_label_maps(samples: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Returns (label_to_idx, idx_to_label) from the active samples."""
    labels = sorted({label for _, label in samples})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

class WasteDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, str]],
        label_to_idx: dict[str, int],
        transform=None,
    ):
        self.samples      = samples
        self.label_to_idx = label_to_idx
        self.transform    = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_idx[label]

def print_summary(samples: list[tuple[str, str]], label_to_idx: dict) -> None:
    counts: dict[str, int] = defaultdict(int)
    for _, label in samples:
        counts[label] += 1

    print("\n" + "=" * 45)
    print(f"{'CLASS':<20} {'IDX':>4}  {'IMAGES':>8}")
    print("=" * 45)
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"{label:<20} {idx:>4}  {counts[label]:>8}")
    print("=" * 45)
    print(f"{'TOTAL':<20}       {len(samples):>8}")
    print("=" * 45 + "\n")

def _set_dataset(config: dict, dataset_name: str, enabled: bool) -> None:
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            ds["enabled"] = enabled
            print(f"[{'ON ' if enabled else 'OFF'}] dataset    → {dataset_name}")
            return
    print(f"[WARNING] Dataset not found: {dataset_name}")

def _set_subdataset(config: dict, dataset_name: str, subdataset_name: str, enabled: bool) -> None:
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            for sub in ds["subdatasets"]:
                if sub["name"] == subdataset_name:
                    sub["enabled"] = enabled
                    print(f"[{'ON ' if enabled else 'OFF'}] subdataset → {dataset_name} / {subdataset_name}")
                    return
    print(f"[WARNING] Subdataset not found: {dataset_name} / {subdataset_name}")

def _set_class(config: dict, dataset_name: str, subdataset_name: str, class_key: str, enabled: bool) -> None:
    for ds in config["datasets"]:
        if ds["name"] == dataset_name:
            for sub in ds["subdatasets"]:
                if sub["name"] == subdataset_name:
                    if class_key in sub["classes"]:
                        sub["classes"][class_key]["enabled"] = enabled
                        print(f"[{'ON ' if enabled else 'OFF'}] class      → {dataset_name} / {subdataset_name} / {class_key}")
                        return
    print(f"[WARNING] Class not found: {dataset_name} / {subdataset_name} / {class_key}")

def _set_label(config: dict, label: str, enabled: bool) -> None:
    """Enable/disable every class that maps to a given unified label (e.g. 'Plastic')."""
    found = False
    for ds in config["datasets"]:
        for sub in ds["subdatasets"]:
            for class_key, class_cfg in sub["classes"].items():
                if class_cfg["label"] == label:
                    class_cfg["enabled"] = enabled
                    found = True
    if found:
        print(f"[{'ON ' if enabled else 'OFF'}] label      → {label} (all sources)")
    else:
        print(f"[WARNING] Label not found: {label}")


# Convenience: enable / disable
def enable_dataset(config, name):             _set_dataset(config, name, True)
def disable_dataset(config, name):            _set_dataset(config, name, False)

def enable_subdataset(config, ds, sub):       _set_subdataset(config, ds, sub, True)
def disable_subdataset(config, ds, sub):      _set_subdataset(config, ds, sub, False)

def enable_class(config, ds, sub, cls):       _set_class(config, ds, sub, cls, True)
def disable_class(config, ds, sub, cls):      _set_class(config, ds, sub, cls, False)

def enable_label(config, label):              _set_label(config, label, True)
def disable_label(config, label):             _set_label(config, label, False)

def status(config: dict) -> None:
    """Print a tree of every dataset/subdataset/class with its ON/OFF state."""
    for ds in config["datasets"]:
        ds_on = ds.get("enabled", True)
        print(f"[{'ON ' if ds_on else 'OFF'}] {ds['name']}")
        for sub in ds["subdatasets"]:
            sub_on = ds_on and sub.get("enabled", True)
            print(f"       [{'ON ' if sub_on else 'OFF'}] {sub['name']}")
            for class_key, cls in sub["classes"].items():
                cls_on = sub_on and cls.get("enabled", True)
                print(f"              [{'ON ' if cls_on else 'OFF'}] {class_key:25s} → {cls['label']}")

def active_labels(config: dict) -> list[str]:
    """Return a sorted list of unified labels that have at least one enabled source."""
    labels = set()
    for ds in config["datasets"]:
        if not ds.get("enabled", True):
            continue
        for sub in ds["subdatasets"]:
            if not sub.get("enabled", True):
                continue
            for cls in sub["classes"].values():
                if cls.get("enabled", True):
                    labels.add(cls["label"])
    result = sorted(labels)
    print("Active labels:", result)
    return result

def save_config(config: dict, config_path: str = "dataset_config.json") -> None:
    """Persist the modified config back to disk."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Config written to {config_path}")