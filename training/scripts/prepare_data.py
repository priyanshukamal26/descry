"""
Data preparation script.
Splits datasets into train/val/test (70/20/10).
Run: python training/scripts/prepare_data.py
"""

import shutil
import random
from pathlib import Path

random.seed(42)


def split_classification_dataset(
    source_dir: Path,
    output_dir: Path,
    splits: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> None:
    """
    Expects source_dir with subfolders = class names.
    Outputs output_dir/train/, output_dir/val/, output_dir/test/
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)

        n = len(images)
        train_end = int(n * splits[0])
        val_end = train_end + int(n * splits[1])

        subsets = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for subset_name, files in subsets.items():
            dest = output_dir / subset_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in files:
                shutil.copy(img_path, dest / img_path.name)

        print(f"  {class_dir.name}: {train_end} train / {val_end-train_end} val / {n-val_end} test")

    print(f"Split complete → {output_dir}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "datasets"

    # Product category dataset (Product-10K)
    split_classification_dataset(
        source_dir=base / "product10k" / "train",
        output_dir=base / "product_category_split",
    )
