import random
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import BrainMRIDataset
from src.paths import SPLIT_DIR


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

LEGACY_SPLIT_DIR = ROOT / "splits"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8-sig"
    ) as f:

        return yaml.safe_load(f)


def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def collect_main_dataset(config):

    dataset_dir = ROOT / config["dataset"]["main_dir"]

    classes = config["dataset"]["classes"]

    records = []

    print("\nLoading MAIN dataset...")

    for label_idx, class_name in enumerate(classes):

        class_dir = dataset_dir / class_name

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        images = sorted([

            p for p in class_dir.rglob("*")

            if p.is_file()

            and p.suffix.lower() in IMAGE_EXTENSIONS

        ])

        if not images:

            raise RuntimeError(
                f"No images found in {class_dir}"
            )

        for image_path in images:

            records.append({

                "path": image_path.relative_to(
                    ROOT
                ).as_posix(),

                "label": class_name,

                "label_idx": label_idx

            })

        print(
            f"{class_name}: {len(images)} images"
        )

    df = pd.DataFrame(records)

    print("\nTotal images:", len(df))

    return df


def create_or_load_splits(
    config,
    force_new=False
):

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = SPLIT_DIR / "train.csv"

    val_file = SPLIT_DIR / "val.csv"

    test_file = SPLIT_DIR / "test.csv"

    summary_file = SPLIT_DIR / "split_summary.json"

    files = [
        train_file,
        val_file,
        test_file,
        summary_file
    ]

    # Preserve the already-created split from the legacy folder.
    # This avoids silently generating a different train/validation/test split.
    legacy_csv_files = [
        LEGACY_SPLIT_DIR / "train.csv",
        LEGACY_SPLIT_DIR / "val.csv",
        LEGACY_SPLIT_DIR / "test.csv"
    ]
    legacy_summary = LEGACY_SPLIT_DIR / "split_summary.json"

    if (
        not all(p.exists() for p in files)
        and all(p.exists() for p in legacy_csv_files)
        and not force_new
    ):
        for source, destination in zip(legacy_csv_files, files[:3]):
            shutil.copy2(source, destination)

        if legacy_summary.exists():
            shutil.copy2(legacy_summary, summary_file)
        else:
            migrated = {
                "seed": config["split"]["random_state"],
                "total_images": 0,
                "ratios": {
                    "train": config["split"]["train"],
                    "validation": config["split"]["validation"],
                    "test": config["split"]["test"]
                },
                "classes": config["dataset"]["classes"],
                "splits": {}
            }

            for split_name, split_path in zip(
                ["train", "val", "test"],
                files[:3]
            ):
                split_df = pd.read_csv(split_path)
                migrated["total_images"] += len(split_df)
                migrated["splits"][split_name] = {
                    "total": len(split_df),
                    "class_counts": {
                        class_name: int(
                            (split_df["label"] == class_name).sum()
                        )
                        for class_name in config["dataset"]["classes"]
                    }
                }

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(migrated, f, indent=4)

        print("\nMigrated the existing split into Results/Shared/Splits.")
    if all(p.exists() for p in files) and not force_new:

        print("\nExisting split found.")

        print(
            "Reusing the same split for all models."
        )

        train_df = pd.read_csv(train_file)

        val_df = pd.read_csv(val_file)

        test_df = pd.read_csv(test_file)

        return train_df, val_df, test_df

    df = collect_main_dataset(config)

    split_config = config["split"]

    seed = split_config["random_state"]

    train_ratio = split_config["train"]

    val_ratio = split_config["validation"]

    test_ratio = split_config["test"]

    total_ratio = (
        train_ratio + val_ratio + test_ratio
    )

    if not np.isclose(total_ratio, 1.0):

        raise ValueError(
            "Train, validation and test ratios must sum to 1."
        )

    train_df, remaining_df = train_test_split(

        df,

        test_size=(val_ratio + test_ratio),

        random_state=seed,

        stratify=df["label"]

    )

    val_df, test_df = train_test_split(

        remaining_df,

        test_size=test_ratio / (
            val_ratio + test_ratio
        ),

        random_state=seed,

        stratify=remaining_df["label"]

    )

    splits = {

        "train": train_df,

        "val": val_df,

        "test": test_df

    }

    summary = {

        "seed": seed,

        "total_images": len(df),

        "ratios": {

            "train": train_ratio,

            "validation": val_ratio,

            "test": test_ratio

        },

        "classes": config["dataset"]["classes"],

        "splits": {}

    }

    for name, split_df in splits.items():

        split_df = split_df.sample(

            frac=1,

            random_state=seed

        ).reset_index(drop=True)

        split_df.to_csv(

            SPLIT_DIR / f"{name}.csv",

            index=False

        )

        class_counts = {

            class_name: int(
                (split_df["label"] == class_name).sum()
            )

            for class_name in config["dataset"]["classes"]

        }

        summary["splits"][name] = {

            "total": len(split_df),

            "class_counts": class_counts

        }

        splits[name] = split_df

        print(
            f"{name.upper()}: {len(split_df)} images"
        )

    with open(

        summary_file,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        "\nNew 80/10/10 split created successfully."
    )

    return (

        splits["train"],

        splits["val"],

        splits["test"]

    )


def build_transforms(config):

    size = config["dataset"]["image_size"]

    train_transform = transforms.Compose([

        transforms.Resize((size, size)),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(

            brightness=0.10,

            contrast=0.10

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )

    ])

    eval_transform = transforms.Compose([

        transforms.Resize((size, size)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )

    ])

    return train_transform, eval_transform


def get_dataloaders(

    force_new_split=False,

    batch_size=None

):

    config = load_config()

    seed = config["project"]["seed"]

    seed_everything(seed)

    train_df, val_df, test_df = create_or_load_splits(

        config,

        force_new=force_new_split

    )

    train_transform, eval_transform = build_transforms(
        config
    )

    train_dataset = BrainMRIDataset(

        SPLIT_DIR / "train.csv",

        transform=train_transform

    )

    val_dataset = BrainMRIDataset(

        SPLIT_DIR / "val.csv",

        transform=eval_transform

    )

    test_dataset = BrainMRIDataset(

        SPLIT_DIR / "test.csv",

        transform=eval_transform

    )

    settings = config["training"]

    if batch_size is None:

        batch_size = settings["effective_batch_size"]

    workers = settings["num_workers"]

    generator = torch.Generator()

    generator.manual_seed(seed)

    common = {

        "batch_size": batch_size,

        "num_workers": workers,

        "pin_memory": (

            settings["pin_memory"]

            and torch.cuda.is_available()

        ),

        "persistent_workers": (

            settings["persistent_workers"]

            and workers > 0

        ),

        "worker_init_fn": seed_worker

    }

    if workers > 0:

        common["prefetch_factor"] = (
            settings["prefetch_factor"]
        )

    train_loader = DataLoader(

        train_dataset,

        shuffle=True,

        generator=generator,

        drop_last=True,

        **common

    )

    val_loader = DataLoader(

        val_dataset,

        shuffle=False,

        drop_last=False,

        **common

    )

    test_loader = DataLoader(

        test_dataset,

        shuffle=False,

        drop_last=False,

        **common

    )

    print("\nDataLoaders created successfully!")

    print("Train images:", len(train_dataset))

    print("Validation images:", len(val_dataset))

    print("Test images:", len(test_dataset))

    print("Train batches:", len(train_loader))

    print("Validation batches:", len(val_loader))

    print("Test batches:", len(test_loader))

    print("Batch size:", batch_size)

    print("Workers:", workers)

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    return (

        train_loader,

        val_loader,

        test_loader

    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        get_dataloaders()
    )

    images, labels = next(iter(train_loader))

    print("\nBatch verification:")

    print("Image shape:", images.shape)

    print("Label shape:", labels.shape)

    print("Image dtype:", images.dtype)

    print("Label dtype:", labels.dtype)

    print("\nDataset pipeline verification successful!")
