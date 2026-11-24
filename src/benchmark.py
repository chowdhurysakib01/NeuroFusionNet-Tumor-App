import time
from pathlib import Path

import torch


def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_model(
    model,
    checkpoint_path,
    device,
    image_size=224,
    warmup=10,
    iterations=50
):

    model.eval()

    dummy = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device
    )

    amp_enabled = device.type == "cuda"

    if amp_enabled:
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup

    with torch.inference_mode():

        for _ in range(warmup):

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):

                _ = model(dummy)

    synchronize(device)

    # Timed inference

    start = time.perf_counter()

    with torch.inference_mode():

        for _ in range(iterations):

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):

                _ = model(dummy)

    synchronize(device)

    elapsed = time.perf_counter() - start

    latency_seconds = elapsed / iterations

    latency_ms = latency_seconds * 1000

    throughput = 1.0 / latency_seconds

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    parameter_size_mb = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)

    checkpoint_size_mb = (
        Path(checkpoint_path).stat().st_size
        / (1024 ** 2)
    )

    peak_memory_mb = None

    if device.type == "cuda":

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

    results = {

        "parameters_total": int(total_parameters),

        "parameters_trainable": int(trainable_parameters),

        "parameter_memory_mb": round(
            parameter_size_mb, 4
        ),

        "checkpoint_size_mb": round(
            checkpoint_size_mb, 4
        ),

        "inference_latency_ms": round(
            latency_ms, 4
        ),

        "inference_throughput_images_per_second": round(
            throughput, 4
        ),

        "gpu_peak_memory_mb": (
            round(peak_memory_mb, 4)
            if peak_memory_mb is not None
            else None
        ),

        "benchmark_batch_size": 1,

        "warmup_iterations": warmup,

        "measured_iterations": iterations

    }

    del dummy

    return results
