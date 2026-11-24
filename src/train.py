import gc
import json
import time

from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from Models.factory import (
    build_model,
    MODEL_NAMES
)

from src.data_pipeline import (
    get_dataloaders,
    load_config,
    seed_everything
)

from src.paths import (
    COMPARISON_DIR,
    ensure_model_result_dirs
)

from src.visualization import plot_training_history


ROOT = Path(__file__).resolve().parents[1]



def get_device(config):

    requested = config["project"]["device"]

    if requested == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return torch.device(requested)


def get_autocast(device, enabled):

    if enabled and device.type == "cuda":

        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        )

    return nullcontext()


def shutdown_loader(loader):

    if loader is None:
        return

    iterator = getattr(
        loader,
        "_iterator",
        None
    )

    if iterator is not None:

        iterator._shutdown_workers()


def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            allow_nan=False
        )


def run_training_epoch(

    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled,
    accumulation_steps,
    gradient_clipping

):

    model.train()

    total_loss = 0.0

    total_samples = 0

    all_predictions = []

    all_labels = []

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Training",
        leave=False
    )

    for batch_idx, (images, labels) in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        group_start = (
            batch_idx // accumulation_steps
        ) * accumulation_steps

        group_size = min(
            accumulation_steps,
            len(loader) - group_start
        )

        with get_autocast(
            device,
            amp_enabled
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            scaled_loss = loss / group_size

        scaler.scale(
            scaled_loss
        ).backward()

        should_update = (

            (batch_idx + 1) % accumulation_steps == 0

            or batch_idx + 1 == len(loader)

        )

        if should_update:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clipping
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        batch_size = labels.size(0)

        total_loss += (
            loss.detach().item() * batch_size
        )

        total_samples += batch_size

        predictions = outputs.detach().argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = total_loss / max(
        total_samples,
        1
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return {

        "loss": float(epoch_loss),

        "accuracy": float(epoch_accuracy),

        "macro_f1": float(epoch_f1)

    }


@torch.no_grad()
def validate_epoch(

    model,
    loader,
    criterion,
    device,
    amp_enabled

):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_predictions = []

    all_labels = []

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with get_autocast(
            device,
            amp_enabled
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    return {

        "loss": float(
            total_loss / max(total_samples, 1)
        ),

        "accuracy": float(
            accuracy_score(
                all_labels,
                all_predictions
            )
        ),

        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            )
        )

    }


def train_one_model(model_name):

    config = load_config()

    seed = config["project"]["seed"]

    seed_everything(seed)

    device = get_device(config)

    settings = config["training"]

    num_classes = config["dataset"]["num_classes"]

    epochs = settings["epochs"]

    patience = settings["early_stopping_patience"]

    learning_rate = settings["learning_rate"]

    weight_decay = settings["weight_decay"]

    amp_enabled = (

        settings["mixed_precision"]

        and device.type == "cuda"

    )

    batch_size = settings["model_batch_sizes"][
        model_name
    ]

    accumulation_steps = settings[
        "gradient_accumulation"
    ][model_name]

    effective_batch_size = (
        batch_size * accumulation_steps
    )

    if effective_batch_size != settings["effective_batch_size"]:

        raise ValueError(
            f"Effective batch size mismatch for {model_name}"
        )

    result_paths = ensure_model_result_dirs(model_name)

    training_dir = result_paths["training"]
    checkpoint_dir = result_paths["checkpoints"]

    best_checkpoint = checkpoint_dir / "best_model.pth"
    last_checkpoint = checkpoint_dir / "last_model.pth"

    print("\n" + "=" * 70)

    print("STARTING MODEL:", model_name)

    print("=" * 70)

    print("Device:", device)

    print("Batch size:", batch_size)

    print("Gradient accumulation:", accumulation_steps)

    print("Effective batch size:", effective_batch_size)

    print("Mixed precision:", amp_enabled)

    print("Epochs:", epochs)

    print("Pretrained:", config["models"]["pretrained"])

    train_loader = None

    val_loader = None

    test_loader = None

    model = None

    optimizer = None

    scheduler = None

    scaler = None

    history = []

    best_f1 = -1.0

    best_epoch = 0

    patience_counter = 0

    start_time = time.perf_counter()

    try:

        train_loader, val_loader, test_loader = (
            get_dataloaders(
                batch_size=batch_size
            )
        )

        model = build_model(

            model_name,

            num_classes=num_classes,

            pretrained=config["models"]["pretrained"]

        )

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay

        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=epochs,

            eta_min=1e-6

        )

        scaler = torch.amp.GradScaler(

            "cuda",

            enabled=amp_enabled

        )

        model_start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):

            epoch_start = time.perf_counter()

            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            train_results = run_training_epoch(

                model,

                train_loader,

                criterion,

                optimizer,

                scaler,

                device,

                amp_enabled,

                accumulation_steps,

                settings["gradient_clipping"]

            )

            val_results = validate_epoch(

                model,

                val_loader,

                criterion,

                device,

                amp_enabled

            )

            scheduler.step()

            if device.type == "cuda":

                torch.cuda.synchronize()

            epoch_time = (
                time.perf_counter() - epoch_start
            )

            current_lr = optimizer.param_groups[0]["lr"]

            record = {

                "epoch": epoch,

                "train_loss": train_results["loss"],

                "val_loss": val_results["loss"],

                "train_accuracy": train_results["accuracy"],

                "val_accuracy": val_results["accuracy"],

                "train_macro_f1": train_results["macro_f1"],

                "val_macro_f1": val_results["macro_f1"],

                "learning_rate": current_lr,

                "epoch_time_seconds": epoch_time

            }

            history.append(record)

            print(
                f"Train Loss: {train_results['loss']:.4f}"
            )

            print(
                f"Validation Loss: {val_results['loss']:.4f}"
            )

            print(
                f"Train Accuracy: {train_results['accuracy']:.4f}"
            )

            print(
                f"Validation Accuracy: {val_results['accuracy']:.4f}"
            )

            print(
                f"Validation Macro-F1: {val_results['macro_f1']:.4f}"
            )

            print(
                f"Epoch Time: {epoch_time:.2f} seconds"
            )

            if val_results["macro_f1"] > best_f1:

                best_f1 = val_results["macro_f1"]

                best_epoch = epoch

                patience_counter = 0

                torch.save({

                    "model_name": model_name,

                    "epoch": epoch,

                    "model_state_dict": model.state_dict(),

                    "best_val_macro_f1": best_f1,

                    "class_names": config["dataset"]["classes"],

                    "config": config

                }, best_checkpoint)

                print(
                    "Best checkpoint saved."
                )

            else:

                patience_counter += 1

            if patience_counter >= patience:

                print(
                    "\nEarly stopping triggered."
                )

                break

        total_training_time = (
            time.perf_counter() - start_time
        )

        torch.save({

            "model_name": model_name,

            "epoch": len(history),

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "scaler_state_dict": scaler.state_dict(),

            "best_epoch": best_epoch,

            "best_val_macro_f1": best_f1,

            "class_names": config["dataset"]["classes"],

            "config": config

        }, last_checkpoint)

        history_df = pd.DataFrame(history)

        history_df.to_csv(

            training_dir / "training_history.csv",

            index=False

        )

        summary = {

            "model_name": model_name,

            "epochs_completed": len(history),

            "best_epoch": best_epoch,

            "best_validation_macro_f1": best_f1,

            "total_training_time_seconds": total_training_time,

            "batch_size": batch_size,

            "gradient_accumulation_steps": accumulation_steps,

            "effective_batch_size": effective_batch_size,

            "device": str(device),

            "mixed_precision": amp_enabled,

            "pretrained": config["models"]["pretrained"],

            "best_checkpoint": str(best_checkpoint),

            "last_checkpoint": str(last_checkpoint),

            "result_directory": str(result_paths["root"])

        }

        save_json(

            summary,

            training_dir / "training_summary.json"

        )

        plot_training_history(
            training_dir / "training_history.csv",
            training_dir
        )

        print("\n" + "=" * 70)

        print("TRAINING COMPLETED:", model_name)

        print("Best epoch:", best_epoch)

        print("Best validation Macro-F1:", best_f1)

        print(
            "Total training time:",
            round(total_training_time, 2),
            "seconds"
        )

        print("=" * 70)

        return summary

    finally:

        for loader in [
            train_loader,
            val_loader,
            test_loader
        ]:

            shutdown_loader(loader)

        del model

        del optimizer

        del scheduler

        del scaler

        del train_loader

        del val_loader

        del test_loader

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


def train_all_models(model_names=None):

    if model_names is None:

        model_names = MODEL_NAMES

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name in model_names:

        result = train_one_model(
            model_name
        )

        results.append(result)

        print(
            f"\nCompleted {len(results)}/{len(model_names)} models."
        )

    save_json(

        results,

        COMPARISON_DIR / "training_summary_all_models.json"

    )

    print("\nAll selected models trained successfully.")

    return results


if __name__ == "__main__":

    torch.set_num_threads(4)

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True

        torch.backends.cuda.matmul.allow_tf32 = True

    train_all_models()
