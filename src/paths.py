from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "Results"
COMPARISON_DIR = RESULTS_DIR / "Comparison"
SHARED_DIR = RESULTS_DIR / "Shared"
SPLIT_DIR = SHARED_DIR / "Splits"

MODEL_RESULT_FOLDERS = {
    "custom_cnn": "CustomCNN",
    "densenet169": "DenseNet169",
    "mobilenetv3": "MobileNetV3",
    "resnet50": "ResNet50",
    "vgg19": "VGG19",
    "densenet169_resnet50_fusion": "DenseNet169_ResNet50_Fusion",
    "neurofusionnet": "NeuroFusionNet",
}


def get_model_paths(model_name):
    if model_name not in MODEL_RESULT_FOLDERS:
        raise ValueError(f"Unknown model name: {model_name}")

    model_root = RESULTS_DIR / MODEL_RESULT_FOLDERS[model_name]

    return {
        "root": model_root,
        "checkpoints": model_root / "Checkpoints",
        "training": model_root / "Training",
        "evaluation": model_root / "Evaluation",
        "xai": model_root / "XAI",
        "benchmark": model_root / "Benchmark",
    }


def ensure_model_result_dirs(model_name):
    paths = get_model_paths(model_name)
    for key, path in paths.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_global_result_dirs():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "results": RESULTS_DIR,
        "comparison": COMPARISON_DIR,
        "shared": SHARED_DIR,
        "splits": SPLIT_DIR,
    }
