import numpy as np
import torch
import torch.nn.functional as F


def normalize_cam(cam):

    cam = cam - cam.min()

    maximum = cam.max()

    if maximum > 1e-8:
        cam = cam / maximum

    return cam


def generate_cam_maps(
    model,
    image,
    target_class,
    method="gradcam"
):

    model.eval()

    if method not in ("gradcam", "gradcam++"):
        raise ValueError(f"Unknown XAI method: {method}")

    if hasattr(model, "get_cam_target_layers"):

        target_layers = model.get_cam_target_layers()

    elif hasattr(model, "get_cam_target_layer"):

        target_layers = [model.get_cam_target_layer()]

    else:

        raise AttributeError(
            "Model does not provide CAM target layers."
        )

    activations = []

    hooks = []

    def forward_hook(module, inputs, output):

        if not isinstance(output, torch.Tensor):
            raise TypeError("CAM target layer must return a tensor.")

        activations.append(output)

    for layer in target_layers:

        hooks.append(
            layer.register_forward_hook(forward_hook)
        )

    try:

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            image = image.detach().requires_grad_(True)

            logits = model(image)

            if len(activations) != len(target_layers):

                raise RuntimeError(
                    "Some CAM target layers were not activated."
                )

            score = logits[:, target_class].sum()

            gradients = torch.autograd.grad(
                outputs=score,
                inputs=tuple(activations),
                retain_graph=False,
                create_graph=False,
                allow_unused=False
            )

            cam_maps = []

            for activation, gradient in zip(
                activations,
                gradients
            ):

                activation = activation.detach().float()

                gradient = gradient.detach().float()

                if method == "gradcam":

                    weights = gradient.mean(
                        dim=(2, 3),
                        keepdim=True
                    )

                else:

                    gradient_squared = gradient.pow(2)

                    gradient_cubed = gradient.pow(3)

                    activation_sum = activation.sum(
                        dim=(2, 3),
                        keepdim=True
                    )

                    denominator = (
                        2.0 * gradient_squared
                        + activation_sum * gradient_cubed
                    )

                    alpha = gradient_squared / (
                        denominator + 1e-8
                    )

                    weights = (
                        alpha * F.relu(gradient)
                    ).sum(
                        dim=(2, 3),
                        keepdim=True
                    )

                cam = (
                    weights * activation
                ).sum(dim=1, keepdim=True)

                cam = F.relu(cam)

                cam = F.interpolate(
                    cam,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                cam = cam[0, 0]

                cam = normalize_cam(cam)

                cam_maps.append(
                    cam.detach().cpu().numpy()
                )

            return (
                logits.detach(),
                cam_maps
            )

    finally:

        for hook in hooks:
            hook.remove()

        model.zero_grad(set_to_none=True)
