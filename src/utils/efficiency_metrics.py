"""
Computational efficiency metrics for model comparison

Measures inference time and memory consumption to demonstrate
practical deployment feasibility of proposed models vs baselines.
"""

import time
import torch
import numpy as np
from typing import Dict
from contextlib import contextmanager


@contextmanager
def cuda_timing():
    """Context manager for accurate CUDA timing"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))[2]
    else:
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000


def measure_model_efficiency(model: torch.nn.Module,
                            input_shape: tuple = (1, 4, 96, 96, 96),
                            num_warmup: int = 5,
                            num_runs: int = 20,
                            device: str = 'cuda') -> Dict[str, float]:
    """
    Measure inference time and memory for a model.

    Args:
        model: PyTorch model to benchmark
        input_shape: Shape of input tensor (B, C, D, H, W)
        num_warmup: Warmup iterations (not timed)
        num_runs: Number of timed iterations
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dict with timing and memory metrics
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    model = model.to(device)
    model.eval()

    # Create dummy input
    x = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # Reset memory stats
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    # Memory measurement
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        peak_memory = 0.0
        current_memory = 0.0

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'mean_inference_ms': np.mean(times),
        'std_inference_ms': np.std(times),
        'min_inference_ms': np.min(times),
        'max_inference_ms': np.max(times),
        'peak_memory_mb': peak_memory,
        'current_memory_mb': current_memory,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'throughput_samples_sec': 1000.0 / np.mean(times),
        'device': device,
        'input_shape': input_shape,
        'num_runs': num_runs
    }


def measure_sliding_window_efficiency(model: torch.nn.Module,
                                      full_volume_shape: tuple = (1, 4, 128, 128, 128),
                                      roi_size: tuple = (96, 96, 96),
                                      sw_batch_size: int = 1,
                                      overlap: float = 0.5,
                                      device: str = 'cuda') -> Dict[str, float]:
    """
    Measure efficiency during sliding window inference (realistic scenario).

    Args:
        model: PyTorch model
        full_volume_shape: Full input volume shape
        roi_size: Sliding window ROI size
        sw_batch_size: Sliding window batch size
        overlap: Overlap ratio
        device: Device to run on

    Returns:
        Dict with sliding window inference metrics
    """
    from monai.inferers import sliding_window_inference

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)
    model.eval()

    x = torch.randn(*full_volume_shape, device=device)

    # Reset memory
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Time sliding window inference
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = sliding_window_inference(
            x, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, overlap=overlap
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = 0.0

    return {
        'sw_inference_ms': elapsed_ms,
        'sw_peak_memory_mb': peak_memory,
        'full_volume_shape': full_volume_shape,
        'roi_size': roi_size,
        'overlap': overlap
    }


def compare_model_efficiency(models: Dict[str, torch.nn.Module],
                            input_shape: tuple = (1, 4, 96, 96, 96),
                            device: str = 'cuda') -> Dict[str, Dict]:
    """
    Compare efficiency metrics across multiple models.

    Args:
        models: Dict mapping model_name -> model
        input_shape: Input shape for benchmarking
        device: Device to run on

    Returns:
        Dict mapping model_name -> efficiency_metrics
    """
    results = {}
    for name, model in models.items():
        print(f"Benchmarking {name}...")
        results[name] = measure_model_efficiency(model, input_shape, device=device)
    return results


def format_efficiency_table(results: Dict[str, Dict]) -> str:
    """
    Format efficiency comparison results as a table.

    Args:
        results: Output from compare_model_efficiency

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 90)
    lines.append("Computational Efficiency Comparison")
    lines.append("=" * 90)
    lines.append(
        f"{'Model':<20} {'Time(ms)':<12} {'±Std':<10} {'Memory(MB)':<12} "
        f"{'Params(M)':<12} {'Throughput':<12}"
    )
    lines.append("-" * 90)

    for name, metrics in results.items():
        lines.append(
            f"{name:<20} {metrics['mean_inference_ms']:<12.2f} "
            f"±{metrics['std_inference_ms']:<8.2f} "
            f"{metrics['peak_memory_mb']:<12.1f} "
            f"{metrics['total_params']/1e6:<12.2f} "
            f"{metrics['throughput_samples_sec']:<12.2f}"
        )

    lines.append("=" * 90)
    return '\n'.join(lines)


def get_model_complexity(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get model complexity metrics.

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter counts and layer info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Count layers by type
    layer_counts = {}
    for module in model.modules():
        layer_type = type(module).__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'layer_counts': layer_counts
    }


if __name__ == "__main__":
    # Example: benchmark a simple model
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 64, 3, padding=1)
            self.norm = nn.BatchNorm3d(64)
            self.relu = nn.ReLU()
            self.out = nn.Conv3d(64, 3, 1)

        def forward(self, x):
            x = self.relu(self.norm(self.conv(x)))
            return self.out(x)

    model = DummyModel()
    print("Efficiency metrics:")
    results = measure_model_efficiency(model, device='cpu', num_runs=5)
    for k, v in results.items():
        print(f"  {k}: {v}")
