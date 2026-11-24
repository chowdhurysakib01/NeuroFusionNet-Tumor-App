import gc
import json

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from Models.factory import build_model, MODEL_NAMES

from src.data_pipeline import (
    get_dataloaders,
    load_config
)

from src.metrics import (
    calculate_classification_metrics
)

from src.benchmark import benchmark_model

from src.visualization import (
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_training_history
)

from src.paths import (
    COMPARISON_DIR,
    ensure_model_result_dirs
)


ROOT = Path(__file__).resolve().parents[1]



def make_json_safe(value):

    if isinstance(value, dict):

        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):

        return make_json_safe(value.tolist())

    if isinstance(value, np.integer):

        return int(value)

    if isinstance(value, np.floating):

        value = float(value)

    if isinstance(value, float):

        if not np.isfinite(value):
            return None

        return value

    return value


def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            make_json_safe(data),
            f,
            indent=4,
            allow_nan=False
        )


def evaluate_model(model_name):

    config = load_config()

    device = torch.device(
        config["project"]["device"]
    )

    if device.type == "cuda" and not torch.cuda.is_available():

        raise RuntimeError("CUDA is unavailable.")

    class_names = config["dataset"]["classes"]

    num_classes = config["dataset"]["num_classes"]

    batch_size = config["training"]["model_batch_sizes"][
        model_name
    ]

    result_paths = ensure_model_result_dirs(model_name)

    checkpoint_path = (
        result_paths["checkpoints"] / "best_model.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    evaluation_dir = result_paths["evaluation"]
    training_dir = result_paths["training"]
    benchmark_dir = result_paths["benchmark"]

    print("\n" + "=" * 70)

    print("EVALUATING:", model_name)

    print("=" * 70)

    train_loader = None

    val_loader = None

    test_loader = None

    model = None

    try:

        train_loader, val_loader, test_loader = (
            get_dataloaders(
                batch_size=batch_size
            )
        )

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

        amp_enabled = (

            config["training"]["mixed_precision"]

            and device.type == "cuda"

        )

        y_true = []

        y_pred = []

        y_prob = []

        print("\nRunning held-out test evaluation...")

        with torch.inference_mode():

            for images, labels in tqdm(
                test_loader,
                desc="Testing"
            ):

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(

                    device_type="cuda",

                    dtype=torch.float16,

                    enabled=amp_enabled

                ):

                    outputs = model(images)

                probabilities = torch.softmax(
                    outputs.float(),
                    dim=1
                )

                predictions = probabilities.argmax(
                    dim=1
                )

                y_true.extend(
                    labels.cpu().numpy().tolist()
                )

                y_pred.extend(
                    predictions.cpu().numpy().tolist()
                )

                y_prob.extend(
                    probabilities.cpu().numpy().tolist()
                )

        y_true = np.asarray(y_true)

        y_pred = np.asarray(y_pred)

        y_prob = np.asarray(y_prob)

        if len(y_true) != len(test_loader.dataset):

            raise RuntimeError(
                "Not all test images were evaluated."
            )

        metrics = calculate_classification_metrics(

            y_true,

            y_pred,

            y_prob,

            class_names

        )

        metrics["model_name"] = model_name

        metrics["test_samples"] = int(len(y_true))

        metrics["best_epoch"] = int(
            checkpoint["epoch"]
        )

        metrics["best_validation_macro_f1"] = float(
            checkpoint["best_val_macro_f1"]
        )

        save_json(
            metrics,
            evaluation_dir / "metrics.json"
        )

        # Scalar metrics CSV

        scalar_metrics = {

            k: v

            for k, v in metrics.items()

            if isinstance(v, (int, float))

        }

        pd.DataFrame(

            [scalar_metrics]

        ).to_csv(

            evaluation_dir / "metrics_summary.csv",

            index=False

        )

        # Classification report

        pd.DataFrame(

            metrics["classification_report"]

        ).T.to_csv(

            evaluation_dir / "classification_report.csv"

        )

        # Confusion matrix

        confusion = np.asarray(
            metrics["confusion_matrix"]
        )

        pd.DataFrame(

            confusion,

            index=class_names,

            columns=class_names

        ).to_csv(

            evaluation_dir / "confusion_matrix.csv"

        )

        # Predictions

        prediction_data = {

            "image_path": test_loader.dataset.data["path"].tolist(),

            "true_label": [
                class_names[i] for i in y_true
            ],

            "predicted_label": [
                class_names[i] for i in y_pred
            ],

            "correct": (
                y_true == y_pred
            ).tolist()

        }

        for i, class_name in enumerate(class_names):

            prediction_data[
                f"probability_{class_name}"
            ] = y_prob[:, i]

        pd.DataFrame(
            prediction_data
        ).to_csv(

            evaluation_dir / "test_predictions.csv",

            index=False

        )

        # Confusion matrix visualization

        plot_confusion_matrix(

            confusion,

            class_names,

            evaluation_dir / "confusion_matrix.png"

        )

        # ROC and PR curves

        plot_roc_pr_curves(

            y_true,

            y_prob,

            class_names,

            evaluation_dir

        )

        # Training history graphs

        history_file = (
            training_dir / "training_history.csv"
        )

        if history_file.exists():

            plot_training_history(

                history_file,

                training_dir

            )

        # Benchmarking

        print("\nRunning inference benchmark...")

        benchmark = benchmark_model(

            model,

            checkpoint_path,

            device,

            image_size=config["dataset"]["image_size"]

        )

        benchmark["model_name"] = model_name

        save_json(

            benchmark,

            benchmark_dir / "efficiency_metrics.json"

        )

        print("\nTEST RESULTS")

        print("-" * 50)

        for key in [

            "accuracy",

            "precision_macro",

            "recall_macro",

            "f1_macro",

            "balanced_accuracy",

            "roc_auc_macro",

            "pr_auc_macro"

        ]:

            print(
                f"{key}: {metrics[key]}"
            )

        print("\nBENCHMARK RESULTS")

        for key, value in benchmark.items():

            print(
                f"{key}: {value}"
            )

        print(
            "\nEvaluation completed:",
            model_name
        )

        return {

            "model_name": model_name,

            "metrics": metrics,

            "benchmark": benchmark

        }

    finally:

        for loader in [
            train_loader,
            val_loader,
            test_loader
        ]:

            if loader is not None:

                iterator = getattr(
                    loader,
                    "_iterator",
                    None
                )

                if iterator is not None:

                    iterator._shutdown_workers()

        del model

        del train_loader

        del val_loader

        del test_loader

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


def evaluate_all_models(model_names=None):

    if model_names is None:

        model_names = MODEL_NAMES

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for index, model_name in enumerate(
        model_names,
        start=1
    ):

        result = evaluate_model(model_name)

        all_results.append(result)

        print(
            f"\nEvaluation progress: {index}/{len(model_names)}"
        )

    # Overall comparison table

    comparison = []

    for result in all_results:

        row = {
            "model_name": result["model_name"]
        }

        row.update({

            k: v

            for k, v in result["metrics"].items()

            if isinstance(v, (int, float))

        })

        row.update({

            "inference_latency_ms":
                result["benchmark"]["inference_latency_ms"],

            "throughput_images_per_second":
                result["benchmark"][
                    "inference_throughput_images_per_second"
                ],

            "checkpoint_size_mb":
                result["benchmark"]["checkpoint_size_mb"],

            "gpu_peak_memory_mb":
                result["benchmark"]["gpu_peak_memory_mb"]

        })

        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(

        COMPARISON_DIR / "all_models_comparison.csv",

        index=False

    )

    save_json(

        all_results,

        COMPARISON_DIR / "all_models_evaluation.json"

    )

    print("\nAll model evaluations completed.")

    return all_results


if __name__ == "__main__":

    evaluate_all_models()
