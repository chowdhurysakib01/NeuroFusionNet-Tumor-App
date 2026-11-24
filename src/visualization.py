import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


def plot_training_history(history_csv, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)

    plots = [
        (
            "train_loss",
            "val_loss",
            "Training and Validation Loss",
            "loss_curve.png"
        ),
        (
            "train_accuracy",
            "val_accuracy",
            "Training and Validation Accuracy",
            "accuracy_curve.png"
        ),
        (
            "train_macro_f1",
            "val_macro_f1",
            "Training and Validation Macro-F1",
            "macro_f1_curve.png"
        )
    ]

    for train_col, val_col, title, filename in plots:

        plt.figure(figsize=(9, 6))

        plt.plot(
            df["epoch"],
            df[train_col],
            label="Training",
            linewidth=2
        )

        plt.plot(
            df["epoch"],
            df[val_col],
            label="Validation",
            linewidth=2
        )

        plt.xlabel("Epoch")
        plt.ylabel(title.split(" and ")[-1])
        plt.title(title)

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["epoch"],
        df["learning_rate"],
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "learning_rate_curve.png",
        dpi=300
    )

    plt.close()


def plot_confusion_matrix(
    matrix,
    class_names,
    output_path
):

    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right"
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_roc_pr_curves(
    y_true,
    y_prob,
    class_names,
    output_dir
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)

    y_binary = label_binarize(
        y_true,
        classes=np.arange(num_classes)
    )

    # ROC curves

    plt.figure(figsize=(9, 7))

    roc_values = {}

    for i, class_name in enumerate(class_names):

        fpr, tpr, _ = roc_curve(
            y_binary[:, i],
            y_prob[:, i]
        )

        class_auc = auc(fpr, tpr)

        roc_values[class_name] = float(class_auc)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} (AUC={class_auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Multiclass ROC Curves")

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=300
    )

    plt.close()

    # Precision-Recall curves

    plt.figure(figsize=(9, 7))

    pr_values = {}

    for i, class_name in enumerate(class_names):

        precision, recall, _ = precision_recall_curve(
            y_binary[:, i],
            y_prob[:, i]
        )

        ap = average_precision_score(
            y_binary[:, i],
            y_prob[:, i]
        )

        pr_values[class_name] = float(ap)

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{class_name} (AP={ap:.4f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Multiclass Precision-Recall Curves")

    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()

    with open(
        output_dir / "curve_auc_values.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "roc_auc_per_class": roc_values,
                "pr_auc_per_class": pr_values
            },
            f,
            indent=4
        )
