from pathlib import Path

from Models.factory import MODEL_NAMES
from src.data_pipeline import load_config
from src.paths import (
    RESULTS_DIR,
    COMPARISON_DIR,
    SPLIT_DIR,
    get_model_paths,
)


def main():
    config = load_config()

    batch_sizes = set(
        config["training"]["model_batch_sizes"].values()
    )
    accumulation_steps = set(
        config["training"]["gradient_accumulation"].values()
    )

    assert len(batch_sizes) == 1, (
        f"Model physical batch sizes differ: {batch_sizes}"
    )
    assert len(accumulation_steps) == 1, (
        f"Gradient accumulation values differ: {accumulation_steps}"
    )

    physical_batch = next(iter(batch_sizes))
    accumulation = next(iter(accumulation_steps))
    effective_batch = physical_batch * accumulation

    assert effective_batch == config["training"]["effective_batch_size"], (
        "Configured effective batch size does not match physical batch × accumulation."
    )

    assert config["split"]["train"] == 0.80
    assert config["split"]["validation"] == 0.10
    assert config["split"]["test"] == 0.10

    print("Common physical batch size:", physical_batch)
    print("Common gradient accumulation:", accumulation)
    print("Common effective batch size:", effective_batch)
    print("Split:", "80/10/10")

    print("\nExpected model result paths:")
    for model_name in MODEL_NAMES:
        paths = get_model_paths(model_name)
        for key in ["checkpoints", "training", "evaluation", "xai", "benchmark"]:
            path = paths[key]
            assert path.is_relative_to(RESULTS_DIR), (
                f"Output path is outside Results: {path}"
            )
            print(f"{model_name:35s} {key:12s} -> {path.relative_to(RESULTS_DIR)}")

    assert SPLIT_DIR.is_relative_to(RESULTS_DIR)
    assert COMPARISON_DIR.is_relative_to(RESULTS_DIR)

    print("\nAll model outputs, comparison reports and split manifests are inside Results.")
    print("Result structure verification: PASSED")


if __name__ == "__main__":
    main()
