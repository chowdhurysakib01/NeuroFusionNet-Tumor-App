import gc
import torch

from Models.factory import build_model, MODEL_NAMES


def main():

    torch.set_num_threads(4)

    dummy = torch.randn(1, 3, 224, 224)

    print("\nTesting all seven architectures...\n")

    for name in MODEL_NAMES:

        print("=" * 55)
        print("Model:", name)

        model = build_model(
            name,
            num_classes=4,
            pretrained=False
        )

        model.eval()

        with torch.inference_mode():
            output = model(dummy)

        params = sum(
            p.numel() for p in model.parameters()
        )

        print("Output shape:", tuple(output.shape))
        print("Parameters:", f"{params:,}")

        assert output.shape == (1, 4), (
            f"Incorrect output for {name}"
        )

        if hasattr(model, "get_cam_target_layers"):
            print("CAM target layers:",
                  len(model.get_cam_target_layers()))

        elif hasattr(model, "get_cam_target_layer"):
            print("CAM target layer: Available")

        print("STATUS: PASSED")

        del model, output
        gc.collect()

    print("\nAll seven architectures passed!")


if __name__ == "__main__":
    main()
