r"""
Cross-dataset evaluation for NeuroFusionNet.

This script:
- Loads the already-trained best NeuroFusionNet checkpoint from Results/NeuroFusionNet.
- Evaluates every valid image under Dataset/External/<class_name>.
- Does NOT train, fine-tune, or update model weights.
- Saves evaluation, predictions, benchmark, metadata, and Grad-CAM/Grad-CAM++ outputs
  under Results/Cross_Dataset/NeuroFusionNet.

Run from D:\Thesis:
    python -m src.cross_dataset_test
"""

import gc
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from Models.factory import build_model
from src.metrics import calculate_classification_metrics
from src.benchmark import benchmark_model
from src.visualization import (
    plot_confusion_matrix,
    plot_roc_pr_curves,
)
from src.xai import generate_cam_maps


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

MODEL_NAME = "neurofusionnet"
CHECKPOINT_PATH = (
    ROOT / "Results" / "NeuroFusionNet" /
    "Checkpoints" / "best_model.pth"
)
EXTERNAL_ROOT = ROOT / "Dataset" / "External"
OUTPUT_ROOT = (
    ROOT / "Results" / "Cross_Dataset" / "NeuroFusionNet"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, allow_nan=False)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExternalBrainMRIDataset(Dataset):
    """Reads all images from four class folders without creating a split."""

    def __init__(self, external_root, class_names, transform):
        self.external_root = Path(external_root)
        self.class_names = list(class_names)
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }
        self.transform = transform
        self.samples = []

        if not self.external_root.is_dir():
            raise FileNotFoundError(
                f"External dataset folder not found: {self.external_root}"
            )

        for class_name in self.class_names:
            class_dir = self.external_root / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"External class folder not found: {class_dir}"
                )

            image_paths = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not image_paths:
                raise RuntimeError(
                    f"No supported images found in: {class_dir}"
                )

            for image_path in image_paths:
                relative_path = image_path.relative_to(ROOT).as_posix()
                self.samples.append(
                    (image_path, self.class_to_idx[class_name], relative_path)
                )

        if not self.samples:
            raise RuntimeError("No external images were found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, relative_path = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, relative_path


def build_eval_transform(config):
    """Must match the main dataset's validation/test preprocessing."""
    size = int(config["dataset"]["image_size"])

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def tensor_to_rgb_image(tensor):
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    return np.clip(image, 0.0, 1.0)


def save_rgb_image(image_array, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = (
        np.clip(image_array, 0.0, 1.0) * 255
    ).round().astype(np.uint8)

    Image.fromarray(image_uint8).save(output_path)


def save_cam_heatmap(cam, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    heatmap_uint8 = (heatmap * 255).round().astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(output_path)


def save_cam_overlay(image_array, cam, output_path, alpha=0.40):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = (
        (1.0 - alpha) * image_array
        + alpha * heatmap
    )
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_uint8 = (overlay * 255).round().astype(np.uint8)
    Image.fromarray(overlay_uint8).save(output_path)


def create_output_directories(class_names):
    directories = {
        "evaluation": OUTPUT_ROOT / "Evaluation",
        "benchmark": OUTPUT_ROOT / "Benchmark",
        "reports": OUTPUT_ROOT / "Reports",
        "xai_original": OUTPUT_ROOT / "XAI" / "Original",
        "xai_gradcam": OUTPUT_ROOT / "XAI" / "GradCAM",
        "xai_gradcam_pp": OUTPUT_ROOT / "XAI" / "GradCAMPlusPlus",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        for key in ("xai_original", "xai_gradcam", "xai_gradcam_pp"):
            (directories[key] / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

    (OUTPUT_ROOT / "Checkpoint_Info").mkdir(
        parents=True,
        exist_ok=True
    )

    return directories


def verify_checkpoint(checkpoint, class_names):
    if checkpoint.get("model_name") != MODEL_NAME:
        raise RuntimeError(
            "The selected checkpoint is not a NeuroFusionNet checkpoint. "
            f"Found: {checkpoint.get('model_name')}"
        )

    checkpoint_classes = checkpoint.get("class_names")

    if checkpoint_classes != list(class_names):
        raise RuntimeError(
            "Class order mismatch between checkpoint and current config.\n"
            f"Checkpoint: {checkpoint_classes}\n"
            f"Config: {list(class_names)}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")


def evaluate_external_dataset(model, loader, device, class_names, amp_enabled):
    y_true = []
    y_pred = []
    y_prob = []
    image_paths = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    model.eval()

    with torch.inference_mode():
        for images, labels, paths in tqdm(
            loader,
            desc="External dataset evaluation"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)

            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())
            image_paths.extend(list(paths))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_seconds = time.perf_counter() - start_time

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) != len(loader.dataset):
        raise RuntimeError(
            f"Evaluated {len(y_true)} images, "
            f"but external dataset contains {len(loader.dataset)}."
        )

    if len(image_paths) != len(loader.dataset):
        raise RuntimeError("Prediction path count does not match dataset size.")

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        y_prob,
        class_names,
    )

    metrics["model_name"] = MODEL_NAME
    metrics["evaluation_dataset"] = "External"
    metrics["test_samples"] = int(len(y_true))
    metrics["external_evaluation_time_seconds"] = float(elapsed_seconds)

    prediction_frame = pd.DataFrame({
        "image_path": image_paths,
        "true_label": [class_names[index] for index in y_true],
        "predicted_label": [class_names[index] for index in y_pred],
        "correct": (y_true == y_pred).tolist(),
    })

    for class_index, class_name in enumerate(class_names):
        prediction_frame[f"probability_{class_name}"] = y_prob[:, class_index]

    return metrics, y_true, y_pred, y_prob, prediction_frame


def generate_external_xai(
    model,
    dataset,
    device,
    class_names,
    samples_per_class=5,
    seed=42,
):
    xai_root = OUTPUT_ROOT / "XAI"
    original_root = xai_root / "Original"
    gradcam_root = xai_root / "GradCAM"
    gradcam_pp_root = xai_root / "GradCAMPlusPlus"

    rng = np.random.default_rng(seed)
    selected_indices = {}

    for class_index, class_name in enumerate(class_names):
        indices = [
            index
            for index, (_, label, _) in enumerate(dataset.samples)
            if label == class_index
        ]

        if len(indices) < samples_per_class:
            raise RuntimeError(
                f"Not enough {class_name} images for XAI: {len(indices)}"
            )

        selected_indices[class_name] = sorted(
            rng.choice(
                indices,
                size=samples_per_class,
                replace=False
            ).tolist()
        )

    model.eval()
    metadata = []
    sample_counts = {class_name: 0 for class_name in class_names}

    print("\nGenerating external Grad-CAM and Grad-CAM++ maps...")

    for class_name in class_names:
        for sample_number, dataset_index in enumerate(
            selected_indices[class_name],
            start=1
        ):
            image_tensor, true_label, relative_path = dataset[dataset_index]
            input_tensor = image_tensor.unsqueeze(0).to(device)

            original_image = tensor_to_rgb_image(image_tensor)

            filename = f"{class_name}_{sample_number:02d}"

            save_rgb_image(
                original_image,
                original_root / class_name / f"{filename}.png"
            )

            sample_record = {
                "sample": filename,
                "image_path": relative_path,
                "true_class": class_name,
                "true_label": int(true_label),
            }

            for method, method_root in (
                ("gradcam", gradcam_root),
                ("gradcam++", gradcam_pp_root),
            ):
                logits, cam_maps = generate_cam_maps(
                    model,
                    input_tensor,
                    target_class=int(true_label),
                    method=method,
                )

                probabilities = torch.softmax(
                    logits.float(),
                    dim=1
                )[0]

                predicted_index = int(probabilities.argmax().item())
                confidence = float(probabilities[predicted_index].item())

                combined_cam = np.mean(
                    np.stack(cam_maps, axis=0),
                    axis=0
                )
                combined_cam = np.clip(combined_cam, 0.0, 1.0)

                output_class_dir = method_root / class_name

                save_cam_heatmap(
                    combined_cam,
                    output_class_dir / f"{filename}_heatmap.png"
                )

                save_cam_overlay(
                    original_image,
                    combined_cam,
                    output_class_dir / f"{filename}_overlay.png"
                )

                if len(cam_maps) > 1:
                    backbone_dir = output_class_dir / "Backbone_Maps"
                    backbone_dir.mkdir(parents=True, exist_ok=True)

                    backbone_names = ["DenseNet169", "ResNet50"]

                    for branch_index, branch_cam in enumerate(cam_maps):
                        save_cam_heatmap(
                            branch_cam,
                            backbone_dir /
                            f"{filename}_{backbone_names[branch_index]}.png"
                        )

                sample_record[method] = {
                    "predicted_class": class_names[predicted_index],
                    "confidence": confidence,
                    "target_used": "ground_truth_class",
                    "backbone_maps": len(cam_maps),
                }

            metadata.append(sample_record)
            sample_counts[class_name] += 1

    summary = {
        "model_name": MODEL_NAME,
        "dataset": "External",
        "methods": ["Grad-CAM", "Grad-CAM++"],
        "target": "Ground-truth class",
        "samples_per_class": samples_per_class,
        "class_sample_counts": sample_counts,
        "total_samples": sum(sample_counts.values()),
        "samples": metadata,
    }

    save_json(
        json_safe(summary),
        xai_root / "xai_summary.json"
    )

    return summary


def main():
    torch.set_num_threads(4)

    config = load_config()
    class_names = list(config["dataset"]["classes"])
    num_classes = int(config["dataset"]["num_classes"])

    if len(class_names) != 4 or num_classes != 4:
        raise RuntimeError(
            f"Expected four classes; found {class_names}, num_classes={num_classes}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"NeuroFusionNet best checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(config["project"]["device"])

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is configured but is not available. "
            "Stopping rather than silently switching devices."
        )

    directories = create_output_directories(class_names)

    print("=" * 72)
    print("NEUROFUSIONNET — EXTERNAL DATASET EVALUATION")
    print("=" * 72)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("External dataset:", EXTERNAL_ROOT)
    print("Device:", device)
    print("Training/fine-tuning: DISABLED")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )
    verify_checkpoint(checkpoint, class_names)

    print("Checkpoint model:", checkpoint["model_name"])
    print("Checkpoint best epoch:", checkpoint.get("epoch"))
    print("Checkpoint validation Macro-F1:", checkpoint.get("best_val_macro_f1"))
    print("Checkpoint class order:", checkpoint["class_names"])

    transform = build_eval_transform(config)

    dataset = ExternalBrainMRIDataset(
        EXTERNAL_ROOT,
        class_names,
        transform
    )

    class_counts = {
        class_name: sum(
            label == class_index
            for _, label, _ in dataset.samples
        )
        for class_index, class_name in enumerate(class_names)
    }

    print("\nExternal image counts:")
    for class_name in class_names:
        print(f"  {class_name}: {class_counts[class_name]}")
    print("  TOTAL:", len(dataset))

    batch_size = 2
    workers = int(config["training"]["num_workers"])
    amp_enabled = (
        bool(config["training"]["mixed_precision"])
        and device.type == "cuda"
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }

    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            config["training"].get("prefetch_factor", 2)
        )

    loader = DataLoader(dataset, **loader_kwargs)

    model = None

    try:
        model = build_model(
            MODEL_NAME,
            num_classes=num_classes,
            pretrained=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        model = model.to(device)
        model.eval()

        # External evaluation only. No optimizer, backward pass, or weight updates.
        metrics, y_true, y_pred, y_prob, predictions = (
            evaluate_external_dataset(
                model,
                loader,
                device,
                class_names,
                amp_enabled
            )
        )

        predictions.to_csv(
            directories["evaluation"] / "test_predictions.csv",
            index=False
        )

        save_json(
            json_safe(metrics),
            directories["evaluation"] / "metrics.json"
        )

        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

        pd.DataFrame([scalar_metrics]).to_csv(
            directories["evaluation"] / "metrics_summary.csv",
            index=False
        )

        pd.DataFrame(
            metrics["classification_report"]
        ).T.to_csv(
            directories["evaluation"] / "classification_report.csv"
        )

        confusion = np.asarray(metrics["confusion_matrix"])

        pd.DataFrame(
            confusion,
            index=class_names,
            columns=class_names
        ).to_csv(
            directories["evaluation"] / "confusion_matrix.csv"
        )

        plot_confusion_matrix(
            confusion,
            class_names,
            directories["evaluation"] / "confusion_matrix.png"
        )

        plot_roc_pr_curves(
            y_true,
            y_prob,
            class_names,
            directories["evaluation"]
        )

        print("\nEXTERNAL TEST METRICS")
        for key in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "balanced_accuracy",
            "roc_auc_macro",
            "pr_auc_macro",
        ):
            print(f"{key}: {metrics[key]:.6f}")

        print("\nRunning inference benchmark...")
        benchmark = benchmark_model(
            model,
            CHECKPOINT_PATH,
            device,
            image_size=int(config["dataset"]["image_size"]),
        )

        benchmark["model_name"] = MODEL_NAME
        benchmark["dataset"] = "External"

        save_json(
            json_safe(benchmark),
            directories["benchmark"] / "efficiency_metrics.json"
        )

        print("\nGenerating external XAI samples...")
        xai_summary = generate_external_xai(
            model,
            dataset,
            device,
            class_names,
            samples_per_class=int(
                config.get("xai", {}).get("samples_per_class", 5)
            ),
            seed=int(config["project"]["seed"]),
        )

        manifest_frame = pd.DataFrame([
            {
                "image_path": relative_path,
                "label": class_names[label],
                "label_idx": label,
            }
            for _, label, relative_path in dataset.samples
        ])

        manifest_path = (
            directories["reports"] / "external_test_manifest.csv"
        )
        manifest_frame.to_csv(manifest_path, index=False)

        manifest_string = "\n".join(
            manifest_frame["image_path"].tolist()
        )

        metadata = {
            "model_name": MODEL_NAME,
            "evaluation_type": "external_cross_dataset_test",
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "training_performed": False,
            "fine_tuning_performed": False,
            "threshold_tuning_performed": False,
            "checkpoint_path": str(
                CHECKPOINT_PATH.relative_to(ROOT)
            ),
            "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_validation_macro_f1": checkpoint.get(
                "best_val_macro_f1"
            ),
            "external_dataset_root": str(
                EXTERNAL_ROOT.relative_to(ROOT)
            ),
            "external_image_count": len(dataset),
            "class_names_and_order": class_names,
            "class_counts": class_counts,
            "class_to_index": {
                name: index
                for index, name in enumerate(class_names)
            },
            "image_size": int(config["dataset"]["image_size"]),
            "preprocessing": (
                "Resize to configured image size; RGB; ToTensor; "
                "ImageNet mean/std normalization."
            ),
            "batch_size": batch_size,
            "num_workers": workers,
            "mixed_precision": amp_enabled,
            "manifest_path": str(
                manifest_path.relative_to(ROOT)
            ),
            "manifest_paths_sha256": sha256_text(manifest_string),
            "metrics": json_safe(metrics),
            "benchmark": json_safe(benchmark),
            "xai_total_samples": xai_summary["total_samples"],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        }

        save_json(
            json_safe(metadata),
            directories["reports"] / "cross_dataset_metadata.json"
        )

        save_json(
            {
                "model_name": MODEL_NAME,
                "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_best_validation_macro_f1": checkpoint.get(
                    "best_val_macro_f1"
                ),
                "class_names_and_order": class_names,
                "loaded_with_strict_state_dict": True,
                "training_or_fine_tuning_performed": False,
            },
            OUTPUT_ROOT / "Checkpoint_Info" / "checkpoint_metadata.json"
        )

        print("\n" + "=" * 72)
        print("CROSS-DATASET EVALUATION COMPLETED")
        print("=" * 72)
        print("External images evaluated:", len(dataset))
        print("Results saved to:", OUTPUT_ROOT)
        print("Training/fine-tuning: NOT PERFORMED")

    finally:
        if loader is not None:
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()

        del model
        del loader
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
