import gc
import torch

from Models.factory import build_model, MODEL_NAMES

from src.xai import generate_cam_maps


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.set_num_threads(4)

    print("XAI device:", device)

    for model_name in MODEL_NAMES:

        print("\n" + "=" * 60)

        print("Testing:", model_name)

        model = build_model(
            model_name,
            num_classes=4,
            pretrained=False
        )

        model = model.to(device)

        model.eval()

        image = torch.randn(
            1, 3, 224, 224,
            device=device
        )

        for method in ["gradcam", "gradcam++"]:

            logits, maps = generate_cam_maps(
                model,
                image,
                target_class=0,
                method=method
            )

            assert logits.shape == (1, 4)

            assert len(maps) in (1, 2)

            for cam in maps:

                assert cam.shape == (224, 224)

                assert torch.isfinite(
                    torch.tensor(cam)
                ).all()

            print(
                method,
                "PASSED",
                "| Backbone maps:",
                len(maps),
                "| Shape:",
                maps[0].shape
            )

        del model, image, logits, maps

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print("MODEL PASSED:", model_name)

    print("\nAll seven models passed both XAI tests!")


if __name__ == "__main__":

    main()
