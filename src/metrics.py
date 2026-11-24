import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import label_binarize


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_prob,
    class_names
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    num_classes = len(class_names)

    results = {

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        )

    }

    y_true_binary = label_binarize(
        y_true,
        classes=np.arange(num_classes)
    )

    try:

        results["roc_auc_macro"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
        )

    except ValueError:

        results["roc_auc_macro"] = float("nan")

    try:

        results["pr_auc_macro"] = float(
            average_precision_score(
                y_true_binary,
                y_prob,
                average="macro"
            )
        )

    except ValueError:

        results["pr_auc_macro"] = float("nan")

    results["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    ).tolist()

    results["classification_report"] = (
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
    )

    return results


def calculate_macro_f1(y_true, y_pred):

    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


def calculate_accuracy(y_true, y_pred):

    return float(
        accuracy_score(y_true, y_pred)
    )
