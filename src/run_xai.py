import gc
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from Models.factory import build_model, MODEL_NAMES

from src.data_pipeline import (
    load_config,
    build_transforms,
    seed_everything,
    seed_worker
)

from src.dataset import BrainMRIDataset

from src.xai import generate_cam_maps

from src.paths import SPLIT_DIR, ensure_model_result_dirs


ROOT = Path(__file__).resolve().parents[1]

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def tensor_to_image(tensor):

    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)

    image = image * STD + MEAN

    image = np.clip(image, 0, 1)

    return image


def save_original(image, path):

    image_uint8 = (
        image * 255
    ).round().astype(np.uint8)

    Image.fromarray(
        image_uint8
    ).save(path)


def save_heatmap(cam, path):

    cmap = plt.get_cmap("jet")

    heatmap = cmap(cam)[..., :3]

    heatmap = (
        heatmap * 255
    ).round().astype(np.uint8)

    Image.fromarray(
        heatmap
    ).save(path)


def save_overlay(image, cam, path, alpha=0.40):

    cmap = plt.get_cmap("jet")

    heatmap = cmap(cam)[..., :3]

    overlay = (
        (1 - alpha) * image
        + alpha * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    overlay = (
        overlay * 255
    ).round().astype(np.uint8)

    Image.fromarray(
        overlay
    ).save(path)


def make_directories(base_dir, classes):

    paths = {

        "original": base_dir / "Original",

        "gradcam": base_dir / "GradCAM",

        "gradcam++": base_dir / "GradCAMPlusPlus"

    }

    for path in paths.values():

        for class_name in classes:

            (path / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

    return paths


def run_model_xai(model_name, test_loader, device, config):

    result_paths = ensure_model_result_dirs(model_name)

    checkpoint_path = (
        result_paths["checkpoints"] / "best_model.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    xai_root = result_paths["xai"]

    classes = config["dataset"]["classes"]

    num_classes = len(classes)

    samples_per_class = config["xai"]["samples_per_class"]

    paths = make_directories(
        xai_root,
        classes
    )

    model = None

    try:

        model = build_model(
            model_name,
            num_classes=num_classes,
            pretrained=False
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(device)

        model.eval()

        counts = {
            class_name: 0
            for class_name in classes
        }

        metadata = []

        print("\nGenerating XAI:", model_name)

        for images, labels in tqdm(
            test_loader,
            desc=f"XAI {model_name}"
        ):

            true_label = int(labels.item())

            true_class = classes[true_label]

            if counts[true_class] >= samples_per_class:

                continue

            image_tensor = images.to(device)

            image = tensor_to_image(images[0])

            sample_number = counts[true_class] + 1

            filename = (
                f"{true_class}_{sample_number:02d}"
            )

            original_path = (
                paths["original"]
                / true_class
                / f"{filename}.png"
            )

            save_original(
                image,
                original_path
            )

            sample_metadata = {

                "image": filename,

                "true_class": true_class,

                "true_label": true_label

            }

            for method in [
                "gradcam",
                "gradcam++"
            ]:

                logits, cam_maps = generate_cam_maps(

                    model,

                    image_tensor,

                    target_class=true_label,

                    method=method

                )

                probabilities = torch.softmax(
                    logits.float(),
                    dim=1
                )[0]

                predicted_label = int(
                    probabilities.argmax().item()
                )

                confidence = float(
                    probabilities[predicted_label].item()
                )

                combined_cam = np.mean(
                    np.stack(cam_maps, axis=0),
                    axis=0
                )

                combined_cam = np.clip(
                    combined_cam,
                    0,
                    1
                )

                method_dir = paths[method] / true_class

                save_heatmap(

                    combined_cam,

                    method_dir / f"{filename}_heatmap.png"

                )

                save_overlay(

                    image,

                    combined_cam,

                    method_dir / f"{filename}_overlay.png"

                )

                if len(cam_maps) > 1:

                    backbone_dir = (
                        method_dir / "Backbone_Maps"
                    )

                    backbone_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    backbone_names = [
                        "DenseNet169",
                        "ResNet50"
                    ]

                    for branch_index, branch_cam in enumerate(
                        cam_maps
                    ):

                        branch_name = backbone_names[
                            branch_index
                        ]

                        save_heatmap(

                            branch_cam,

                            backbone_dir / (
                                f"{filename}_{branch_name}.png"
                            )

                        )

                sample_metadata[method] = {

                    "predicted_class": classes[predicted_label],

                    "confidence": confidence,

                    "number_of_backbone_maps": len(cam_maps)

                }

            metadata.append(sample_metadata)

            counts[true_class] += 1

            if all(
                count >= samples_per_class
                for count in counts.values()
            ):

                break

        missing = {

            name: samples_per_class - count

            for name, count in counts.items()

            if count < samples_per_class

        }

        if missing:

            raise RuntimeError(
                f"Not enough test samples per class: {missing}"
            )

        summary = {

            "model_name": model_name,

            "method": [
                "Grad-CAM",
                "Grad-CAM++"
            ],

            "samples_per_class": samples_per_class,

            "class_sample_counts": counts,

            "total_samples": sum(counts.values()),

            "target": "Ground-truth class",

            "samples": metadata

        }

        with open(
            xai_root / "xai_summary.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

        print("\nXAI completed:", model_name)

        print("Sample counts:", counts)

        return summary

    finally:

        del model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


def run_all_xai(model_names=None):

    config = load_config()

    seed_everything(
        config["project"]["seed"]
    )

    if model_names is None:

        model_names = MODEL_NAMES

    device = torch.device(
        config["project"]["device"]
    )

    _, eval_transform = build_transforms(config)

    test_dataset = BrainMRIDataset(

        SPLIT_DIR / "test.csv",

        transform=eval_transform

    )

    settings = config["training"]

    generator = torch.Generator()

    generator.manual_seed(
        config["project"]["seed"]
    )

    loader_settings = {

        "batch_size": 1,

        "shuffle": False,

        "num_workers": settings["num_workers"],

        "pin_memory": (
            settings["pin_memory"]
            and device.type == "cuda"
        ),

        "persistent_workers": (
            settings["persistent_workers"]
            and settings["num_workers"] > 0
        ),

        "worker_init_fn": seed_worker,

        "generator": generator

    }

    if settings["num_workers"] > 0:

        loader_settings["prefetch_factor"] = (
            settings["prefetch_factor"]
        )

    test_loader = DataLoader(
        test_dataset,
        **loader_settings
    )

    results = []

    try:

        for index, model_name in enumerate(
            model_names,
            start=1
        ):

            result = run_model_xai(

                model_name,

                test_loader,

                device,

                config

            )

            results.append(result)

            print(
                f"XAI progress: {index}/{len(model_names)}"
            )

        return results

    finally:

        iterator = getattr(
            test_loader,
            "_iterator",
            None
        )

        if iterator is not None:

            iterator._shutdown_workers()

        del test_loader

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


if __name__ == "__main__":

    torch.set_num_threads(4)

    run_all_xai()
