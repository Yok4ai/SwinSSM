"""
Benchmark inference time, peak GPU memory, and FLOPs for:
  - MambaUNETR  (MambaUNETR-Seed42)
  - SwinMamba   (SwinMamba-Seed42)
  - SwinUNETRPlus (SwinUNETRPlus-Seed42)

Input shape: (1, 4, 96, 96, 96)
"""

import gc
import os, sys, time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CKPT = Path("checkpoints")
INPUT_SHAPE = (4, 96, 96, 96)  # C, D, H, W (no batch)
N_WARMUP = 10
N_RUNS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BENCHMARKS = {
    "SwinUNETR":     CKPT / "SwinUNETRPlus-Seed42" / "best_model.pth",
    "SwinUNETRPlus": CKPT / "SwinUNETRPlus-Seed42" / "best_model.pth",
    "MambaUNETR":    CKPT / "MambaUNETR-Seed42"    / "best_model.pth",
    "SwinMamba":     CKPT / "SwinMamba-Seed42"      / "best_model.pth",
}

# Model type tag so we know which class to instantiate
MODEL_TYPES = {
    "SwinUNETR":     "swinunetr",
    "MambaUNETR":    "mambaunetr",
    "SwinMamba":     "swinmamba",
    "SwinUNETRPlus": "swinunetrplus",
}


def build_model(model_type: str):
    """Instantiate model with default training hyperparameters."""
    feat = [48, 96, 192, 384]
    depths = [2, 2, 2, 2]

    if model_type == "swinunetr":
        from src.models.swinunetrplus import SwinUNETR
        # Vanilla: all Plus features disabled
        return SwinUNETR(
            in_channels=4, out_channels=3,
            feature_size=48, depths=depths,
            norm_name="instance",
        )
    elif model_type == "mambaunetr":
        from src.models.mambaunetr import MambaUNETR
        return MambaUNETR(
            in_chans=4, out_chans=3,
            feat_size=feat, depths=depths,
            spatial_dims=3, norm_name="instance",
        )
    elif model_type == "swinmamba":
        from src.models.swinmamba import SwinMamba
        return SwinMamba(
            in_chans=4, out_chans=3,
            depths=depths, feat_size=feat,
            drop_path_rate=0.1, layer_scale_init_value=1e-6,
            spatial_dims=3, norm_name="instance",
            window_size=[4, 4, 4], mlp_ratio=4.0,
            num_heads=[3, 6, 12, 24],
            d_state=16, d_conv=3, expand=1,
        )
    elif model_type == "swinunetrplus":
        from src.models.swinunetrplus import SwinUNETR
        # Flags confirmed from state dict keys:
        # hierarchical_skip_* → use_hierarchical_skip=True
        # layers1c/2c/3c/4c  → use_v2=True
        return SwinUNETR(
            in_channels=4, out_channels=3,
            feature_size=48, depths=depths,
            use_v2=True,
            use_hierarchical_skip=True,
            norm_name="instance",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_model(name: str, pth_path: Path):
    model = build_model(MODEL_TYPES[name])
    state_dict = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model


def measure_latency(model, x: torch.Tensor) -> dict:
    """Warmup then time with CUDA events."""
    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Timed runs
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        start_evt.record()
        with torch.no_grad():
            _ = model(x)
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))

    times = np.array(times_ms)
    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "vols_per_sec": 1000.0 / times.mean(),
    }


def measure_memory(model, x: torch.Tensor) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    return {"peak_mb": peak_mb}


def measure_flops(model) -> dict:
    from ptflops import get_model_complexity_info
    # Mamba selective_scan requires CUDA — run on GPU
    model_cuda = model.to(DEVICE)
    try:
        macs, params = get_model_complexity_info(
            model_cuda,
            INPUT_SHAPE,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return {
            "gmacs": macs / 1e9,
            "params_m": params / 1e6,
        }
    except Exception as e:
        return {"gmacs": None, "params_m": None, "error": str(e)}
    finally:
        model.to(DEVICE)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def run_benchmark(name: str, ckpt_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    print("  Loading checkpoint...", end=" ", flush=True)
    model = load_model(name, ckpt_path)
    print("done")

    x = torch.randn(1, *INPUT_SHAPE, device=DEVICE, dtype=torch.float32)

    # FLOPs (must run on CUDA for Mamba models)
    print("  Computing FLOPs...", end=" ", flush=True)
    flops = measure_flops(model)
    print("done")

    # Memory
    print("  Measuring peak GPU memory...", end=" ", flush=True)
    mem = measure_memory(model, x)
    print("done")

    # Latency
    print(f"  Timing {N_RUNS} runs (after {N_WARMUP} warmup)...", end=" ", flush=True)
    lat = measure_latency(model, x)
    print("done")

    # Params (fallback if ptflops failed)
    params_m = flops.get("params_m") or count_params(model)

    print(f"\n  Parameters:       {params_m:.2f} M")
    if flops.get("gmacs") is not None:
        print(f"  GMACs:            {flops['gmacs']:.2f}")
        print(f"  GFLOPs:           {flops['gmacs'] * 2:.2f}")
    else:
        print(f"  GMACs:            N/A ({flops.get('error', '')})")
        print(f"  GFLOPs:           N/A")
    print(f"  Peak GPU memory:  {mem['peak_mb']:.1f} MB")
    print(f"  Latency (mean):   {lat['mean_ms']:.2f} ± {lat['std_ms']:.2f} ms")
    print(f"  Latency (min):    {lat['min_ms']:.2f} ms")
    print(f"  Throughput:       {lat['vols_per_sec']:.3f} vol/s")

    result = {
        "name": name,
        "params_m": params_m,
        **flops,
        **mem,
        **lat,
    }

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input shape: (1, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}, {INPUT_SHAPE[2]}, {INPUT_SHAPE[3]})")

    results = []
    for name, ckpt in BENCHMARKS.items():
        # Clear GPU memory from previous model before each benchmark
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        r = run_benchmark(name, ckpt)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params(M)':>10} {'GMACs':>8} {'GFLOPs':>8} {'PeakMem(MB)':>12} {'Lat(ms)':>10} {'Vol/s':>8}")
    print("-" * 82)
    for r in results:
        gmacs  = f"{r['gmacs']:.1f}"      if r.get("gmacs") else "N/A"
        gflops = f"{r['gmacs'] * 2:.1f}"  if r.get("gmacs") else "N/A"
        print(
            f"{r['name']:<20} {r['params_m']:>10.2f} {gmacs:>8} {gflops:>8} "
            f"{r['peak_mb']:>12.1f} {r['mean_ms']:>10.2f} {r['vols_per_sec']:>8.3f}"
        )


if __name__ == "__main__":
    main()
