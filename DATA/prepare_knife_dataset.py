import os
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNIFE_SOURCE_TRAIN = HERE / "knife" / "train"
KNIFE_SOURCE_TEST = HERE / "knife" / "test"
NON_KNIFE_SOURCE = HERE / "images"
OUTPUT_DIR = HERE / "knife_dataset"

SEED = 123
VALIDATION_SPLIT = 0.15
TRAIN_SPLIT = 0.85

CLASS_NAMES = ["knife", "non_knife"]


def gather_non_knife_images():
    image_paths = []
    for subdir in NON_KNIFE_SOURCE.iterdir():
        if not subdir.is_dir():
            continue
        for file in subdir.iterdir():
            if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(file)
    return image_paths


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_and_create(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_and_copy(source_paths, dest_base, train_ratio, val_ratio):
    random.shuffle(source_paths)
    total = len(source_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_paths = source_paths[:train_end]
    val_paths = source_paths[train_end:val_end]
    test_paths = source_paths[val_end:]

    for split_name, split_paths in [("train", train_paths), ("validation", val_paths), ("test", test_paths)]:
        dest_dir = dest_base / split_name
        make_dir(dest_dir)
        for src in split_paths:
            shutil.copy(src, dest_dir / src.name)

    return len(train_paths), len(val_paths), len(test_paths)


def main():
    random.seed(SEED)

    if not KNIFE_SOURCE_TRAIN.exists() or not KNIFE_SOURCE_TEST.exists():
        raise FileNotFoundError("Expected DATA/knife/train and DATA/knife/test directories")

    if not NON_KNIFE_SOURCE.exists():
        raise FileNotFoundError("Expected DATA/images directory for non_knife examples")

    clear_and_create(OUTPUT_DIR)

    # Knife: use train for train+validation and test for test
    knife_train_paths = [p for p in KNIFE_SOURCE_TRAIN.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    knife_test_paths = [p for p in KNIFE_SOURCE_TEST.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    non_knife_paths = gather_non_knife_images()

    if len(knife_train_paths) == 0 or len(non_knife_paths) == 0:
        raise RuntimeError("Insufficient source images for dataset preparation")

    knife_dest = OUTPUT_DIR / "knife"
    non_knife_dest = OUTPUT_DIR / "non_knife"
    clear_and_create(knife_dest)
    clear_and_create(non_knife_dest)

    # Train/validation split for knife examples
    random.shuffle(knife_train_paths)
    knife_val_size = max(1, int(len(knife_train_paths) * VALIDATION_SPLIT))
    knife_val = knife_train_paths[:knife_val_size]
    knife_train = knife_train_paths[knife_val_size:]

    # Preserve original test set unchanged
    knife_test = knife_test_paths

    # Non-knife split from images
    random.shuffle(non_knife_paths)
    non_knife_val_size = max(1, int(len(non_knife_paths) * VALIDATION_SPLIT))
    non_knife_val = non_knife_paths[:non_knife_val_size]
    non_knife_train = non_knife_paths[non_knife_val_size:]

    # Use 20% of non-knife train for test if there is enough data, otherwise 15%.
    non_knife_test_size = max(1, int(len(non_knife_train) * 0.15))
    non_knife_test = non_knife_train[-non_knife_test_size:]
    non_knife_train = non_knife_train[:-non_knife_test_size]

    # Write files
    def copy_split(paths, split):
        dest_dir = OUTPUT_DIR / split
        make_dir(dest_dir)
        make_dir(dest_dir / "knife")
        make_dir(dest_dir / "non_knife")
        return dest_dir

    # Ensure required output directories exist before copying.
    copy_split([], "train")
    copy_split([], "validation")
    copy_split([], "test")

    for path in knife_train:
        shutil.copy(path, OUTPUT_DIR / "train" / "knife" / path.name)
    for path in knife_val:
        shutil.copy(path, OUTPUT_DIR / "validation" / "knife" / path.name)
    for path in knife_test:
        shutil.copy(path, OUTPUT_DIR / "test" / "knife" / path.name)

    for path in non_knife_train:
        shutil.copy(path, OUTPUT_DIR / "train" / "non_knife" / path.name)
    for path in non_knife_val:
        shutil.copy(path, OUTPUT_DIR / "validation" / "non_knife" / path.name)
    for path in non_knife_test:
        shutil.copy(path, OUTPUT_DIR / "test" / "non_knife" / path.name)

    print("Prepared knife_dataset with splits:")
    print(f"  knife train={len(knife_train)} validation={len(knife_val)} test={len(knife_test)}")
    print(f"  non_knife train={len(non_knife_train)} validation={len(non_knife_val)} test={len(non_knife_test)}")
    print(f"Dataset root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
