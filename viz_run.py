"""
Visualization: best & worst case | GradCAM | Integrated Gradients |
Attention/activation maps | frame-by-frame slice GIFs.

Usage:
  !python viz_run.py              # full run (inference + viz)
  !python viz_run.py --skip_inference  # reload saved arrays, re-run viz only
"""
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--skip_inference", action="store_true")
_ap.add_argument("--gradcam_only", action="store_true",
                 help="Skip inference; load only best/worst cases for GradCAM/IG/activation maps.")
_ap.add_argument("--best_idx",  type=int, default=4,  help="Best case index (from previous run)")
_ap.add_argument("--worst_idx", type=int, default=3,  help="Worst case index (from previous run)")
_args = _ap.parse_args()
import sys, os
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Research-paper typography ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":           22,
    "font.weight":         "bold",
    "axes.titlesize":      26,
    "axes.titleweight":    "bold",
    "axes.titlepad":       6,
    "axes.labelsize":      22,
    "axes.labelweight":    "bold",
    "figure.titlesize":    30,
    "figure.titleweight":  "bold",
    "xtick.labelsize":     18,
    "ytick.labelsize":     18,
})
from pathlib import Path
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import GradCAM
from captum.attr import IntegratedGradients
import imageio

from dataset_setup import prepare_brats_data
from src.utils.validation import StandaloneValidationPipeline as BraTSValidator
from src.utils.visualizations import resize_cam, show_cam_overlay, normalize_attr

# ── Config ────────────────────────────────────────────────────────────────────
DATA     = "/path/to/BRATS2023-training"
CKPT     = Path("checkpoints")
OUT_DIR  = Path("visualizations"); OUT_DIR.mkdir(exist_ok=True)
MAX_CASES = 30

COMMON = dict(
    data_dir     = "dataset/dataset.json",
    dataset      = "brats2023",
    batch_size   = 1,
    num_workers  = 3,
    roi_size     = (96, 96, 96),
    sw_batch_size = 2,
    overlap      = 0.5,
    threshold    = 0.6,
    feature_size = 48,
    use_v2       = True,
    max_batches  = MAX_CASES,
    log_to_wandb = False,
    d_state      = 16,
    d_conv       = 3,
    expand       = 1,
    precision    = "bf16",
)

MODELS_CFG = {
    "SwinMamba": dict(
        checkpoint_path = str(CKPT / "SwinMamba-Seed42/best_model.pth"),
        use_mamba       = True,
        mamba_type      = "swinmamba",
    ),
    "MambaUNETR": dict(
        checkpoint_path = str(CKPT / "MambaUNETR-Seed42/best_model.pth"),
        use_mamba       = True,
        mamba_type      = "mambaunetr",
    ),
    "SwinUNETRPlus": dict(
        checkpoint_path       = str(CKPT / "SwinUNETRPlus-Seed42/best_model.pth"),
        use_mamba             = False,
        use_hierarchical_skip = True,
    ),
}

TUMOR_COLORS = ["#FF2D55", "#BF5FFF", "#00B4FF"]  # TC=crimson  WT=violet  ET=sky-blue
LEGEND_LABEL = "Crimson=TC  Violet=WT  Blue=ET"

TARGET_CLASS   = 1   # WT
TARGET_LAYER   = "encoder1"   # used in main comparison figures for all models
TARGET_LAYERS  = {n: TARGET_LAYER for n in ["SwinMamba", "MambaUNETR", "SwinUNETRPlus"]}
SWEEP_LAYERS   = ["encoder1", "encoder2", "encoder3", "encoder4", "encoder5"]
MODALITY_NAMES = ["T1", "T1c", "T2", "FLAIR"]
CHANNEL_IDX    = 0   # T1c for display
IG_STEPS       = 20

# ── 1. Dataset ────────────────────────────────────────────────────────────────
print("Preparing dataset.json...")
prepare_brats_data(DATA, "dataset")

# ── 2. Load validators ────────────────────────────────────────────────────────
validators = {}
for name, cfg in MODELS_CFG.items():
    print(f"\nLoading {name}...")
    v = BraTSValidator(**COMMON, **cfg)
    v.setup()
    validators[name] = v

CACHE = OUT_DIR / "inference_cache.npz"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _obj_array(lst):
    """Create a 1-D object array without broadcasting (avoids shape-mismatch on uniform shapes)."""
    arr = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        arr[i] = x
    return arr

def mid(vol, axis=2):
    return np.take(vol, vol.shape[axis]//2, axis=axis)

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-6)

def tumor_crop(img_np, lbl_np, roi=96):
    """96³ crop of img_np centered on WT tumor. img:[4,H,W,D] lbl:[3,H,W,D]"""
    wt = lbl_np[1]          # [H,W,D]
    r  = roi // 2
    H, W, D = wt.shape
    if wt.any():
        coords = np.argwhere(wt)
        cH, cW, cD = coords.mean(axis=0).astype(int)
    else:
        cH, cW, cD = H//2, W//2, D//2
    h0 = int(np.clip(cH - r, 0, H - roi))
    w0 = int(np.clip(cW - r, 0, W - roi))
    d0 = int(np.clip(cD - r, 0, D - roi))
    return img_np[:, h0:h0+roi, w0:w0+roi, d0:d0+roi]

def _inset_label(ax, text, loc="bottom"):
    """Draw bold text inside the axes so no space above/below is needed."""
    y, va = (0.03, "bottom") if loc == "bottom" else (0.97, "top")
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha="center", va=va, fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55))

def overlay_mask(ax, img_sl, mask_sl, title):
    """mask_sl: [3,H,W] binary — TC=crimson WT=violet ET=blue"""
    ax.imshow(img_sl, cmap="gray")
    for c, col in enumerate(TUMOR_COLORS):
        m = mask_sl[c].astype(float)
        if m.any():
            ax.imshow(np.ma.masked_where(m == 0, m),
                      cmap=plt.cm.colors.ListedColormap([col]),
                      alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

# ── 3. Inference on 30 cases ──────────────────────────────────────────────────
if _args.gradcam_only:
    best_idx  = _args.best_idx
    worst_idx = _args.worst_idx
    need = {best_idx, worst_idx}
    _case_buf = {}
    print(f"\nLoading & running inference on cases {sorted(need)}...")
    for i, batch in enumerate(validators["SwinMamba"].val_loader):
        if i in need:
            _case_buf[i] = batch
        if len(_case_buf) == len(need):
            break

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    all_imgs   = {}
    all_lbls   = {}
    all_preds  = {n: {} for n in MODELS_CFG}
    case_dices = {n: {} for n in MODELS_CFG}

    for idx, batch in _case_buf.items():
        img = batch["image"]
        lbl = batch["label"]
        all_imgs[idx] = img[0].numpy()
        all_lbls[idx] = lbl[0].numpy()
        for name, v in validators.items():
            with torch.no_grad():
                logits = sliding_window_inference(
                    img.to(v.device), roi_size=(96, 96, 96), sw_batch_size=2,
                    predictor=v.model, overlap=0.5, mode="gaussian",
                )
            pred = (torch.sigmoid(logits) > 0.6).cpu()
            all_preds[name][idx] = pred[0].numpy()
            dice_metric(y_pred=pred, y=lbl.cpu())
            case_dices[name][idx] = dice_metric.aggregate().item()
            dice_metric.reset()

    mean_scores = {idx: np.mean([case_dices[n][idx] for n in MODELS_CFG]) for idx in need}

elif _args.skip_inference:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE}. Run without --skip_inference first."
        )
    print(f"\nLoading cached inference from {CACHE}...")
    _d = np.load(str(CACHE), allow_pickle=True)
    all_imgs   = list(_d["all_imgs"])
    all_lbls   = list(_d["all_lbls"])
    all_preds  = {n: list(_d[f"pred_{n}"]) for n in MODELS_CFG}
    case_dices = {n: list(_d[f"dice_{n}"]) for n in MODELS_CFG}
    best_idx   = int(_d["best_idx"]); worst_idx = int(_d["worst_idx"])
    mean_scores = _d["mean_scores"]
    print(f"Best case {best_idx}  Worst case {worst_idx}")
else:
    print("\nRunning inference on 30 val cases...")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    all_imgs, all_lbls = [], []
    all_preds   = {n: [] for n in MODELS_CFG}
    case_dices  = {n: [] for n in MODELS_CFG}

    cases_dir = OUT_DIR / "per_case"; cases_dir.mkdir(exist_ok=True)

    ref_loader = validators["SwinMamba"].val_loader
    for i, batch in enumerate(tqdm(ref_loader, desc="cases", total=MAX_CASES)):
        if i >= MAX_CASES:
            break
        img = batch["image"]   # [1,4,H,W,D]
        lbl = batch["label"]   # [1,3,H,W,D]
        img_np = img[0].numpy()
        lbl_np = lbl[0].numpy()
        all_imgs.append(img_np)
        all_lbls.append(lbl_np)

        for name, v in validators.items():
            with torch.no_grad():
                logits = sliding_window_inference(
                    img.to(v.device), roi_size=(96,96,96), sw_batch_size=2,
                    predictor=v.model, overlap=0.5, mode="gaussian",
                )
            pred = (torch.sigmoid(logits) > 0.6).cpu()
            all_preds[name].append(pred[0].numpy())
            dice_metric(y_pred=pred, y=lbl.cpu())
            case_dices[name].append(dice_metric.aggregate().item())
            dice_metric.reset()

        # ── Save on-the-fly: prediction grid for this case ───────────────────
        mean_d  = np.mean([case_dices[n][-1] for n in MODELS_CFG])
        img_sl  = mid(img_np[CHANNEL_IDX])
        lbl_sl  = np.stack([mid(lbl_np[c]) for c in range(3)])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(f"Case {i:02d}  mean_dice={mean_d:.3f}  |  {LEGEND_LABEL}", fontsize=9)
        axes[0].imshow(img_sl, cmap="gray"); axes[0].axis("off"); axes[0].set_title("MRI T1c", fontsize=7)
        axes[1].imshow(img_sl, cmap="gray")
        for c, col in enumerate(TUMOR_COLORS):
            m = lbl_sl[c].astype(float)
            if m.any():
                axes[1].imshow(np.ma.masked_where(m==0,m),
                               cmap=plt.cm.colors.ListedColormap([col]), alpha=0.5, vmin=0, vmax=1)
        axes[1].axis("off"); axes[1].set_title("GT", fontsize=7)
        for ai, name in enumerate(["SwinMamba","MambaUNETR","SwinUNETRPlus"], start=2):
            pred_sl = np.stack([mid(all_preds[name][-1][c]) for c in range(3)])
            axes[ai].imshow(img_sl, cmap="gray")
            for c, col in enumerate(TUMOR_COLORS):
                m = pred_sl[c].astype(float)
                if m.any():
                    axes[ai].imshow(np.ma.masked_where(m==0,m),
                                   cmap=plt.cm.colors.ListedColormap([col]), alpha=0.5, vmin=0, vmax=1)
            axes[ai].axis("off")
            axes[ai].set_title(f"{name}\n{case_dices[name][-1]:.3f}", fontsize=7)
        plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
        plt.savefig(str(cases_dir / f"case_{i:02d}.png"), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  case {i:02d}  " + "  ".join(f"{n}={case_dices[n][-1]:.3f}" for n in MODELS_CFG))

    mean_scores = np.mean([case_dices[n] for n in MODELS_CFG], axis=0)
    best_idx    = int(np.argmax(mean_scores))
    worst_idx   = int(np.argmin(mean_scores))
    print(f"\nBest  case {best_idx}: dice={mean_scores[best_idx]:.4f}")
    print(f"Worst case {worst_idx}: dice={mean_scores[worst_idx]:.4f}")

    # Cache to disk so we can re-run viz without re-inference
    np.savez(str(CACHE),
             all_imgs=_obj_array(all_imgs),
             all_lbls=_obj_array(all_lbls),
             mean_scores=mean_scores, best_idx=best_idx, worst_idx=worst_idx,
             **{f"pred_{n}": _obj_array(all_preds[n]) for n in MODELS_CFG},
             **{f"dice_{n}": np.array(case_dices[n]) for n in MODELS_CFG})

# ── Free GPU after inference — load one model at a time for GradCAM/IG ───────
from contextlib import contextmanager

for v in validators.values():
    v.model.cpu()
torch.cuda.empty_cache()
print("Models moved to CPU. GPU memory freed.")

@contextmanager
def model_on_gpu(name):
    """Context manager: move named model to GPU, others stay on CPU."""
    v = validators[name]
    v.model.to(v.device)
    try:
        yield v
    finally:
        v.model.cpu()
        torch.cuda.empty_cache()

# ── 4. Main prediction grid ───────────────────────────────────────────────────
print("\nSaving prediction grid...")
fig = plt.figure(figsize=(10, 22))
gs  = gridspec.GridSpec(5, 2, figure=fig)
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.01,
                    hspace=0.22, wspace=0.04)

for ci, (idx, tag) in enumerate([(best_idx, "Best"), (worst_idx, "Worst")]):
    img_np = all_imgs[idx]
    lbl_np = all_lbls[idx]
    img_sl = mid(img_np[CHANNEL_IDX])
    lbl_sl = np.stack([mid(lbl_np[c]) for c in range(3)])

    ax = fig.add_subplot(gs[0, ci])
    ax.imshow(img_sl, cmap="gray"); ax.axis("off")
    ax.set_title(f"{tag} Case (Dice={mean_scores[idx]:.3f})\nMRI T1c")

    ax = fig.add_subplot(gs[1, ci])
    overlay_mask(ax, img_sl, lbl_sl, "Ground Truth")

    for ri, name in enumerate(["SwinMamba", "MambaUNETR", "SwinUNETRPlus"], start=2):
        ax = fig.add_subplot(gs[ri, ci])
        pred_sl = np.stack([mid(all_preds[name][idx][c]) for c in range(3)])
        overlay_mask(ax, img_sl, pred_sl, name)

plt.suptitle(f"Best vs. Worst Case Predictions  |  {LEGEND_LABEL}")
out = str(OUT_DIR / "predictions_best_worst.png")
plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved → {out}")

import gc as _gc

SWEEP_DIR = OUT_DIR / "sweeps"; SWEEP_DIR.mkdir(exist_ok=True)

# ── Sweep helpers ─────────────────────────────────────────────────────────────
def _gradcam_sweep(model_name, case_tag, case_idx):
    """GradCAM for every encoder layer — one figure."""
    img_np  = all_imgs[case_idx]
    lbl_np  = all_lbls[case_idx]
    crop_np = tumor_crop(img_np, lbl_np)
    mid_d   = crop_np.shape[3] // 2
    img_sl  = crop_np[CHANNEL_IDX, :, :, mid_d]
    nl      = len(SWEEP_LAYERS)
    fig, axes = plt.subplots(2, nl, figsize=(6 * nl, 10))
    fig.suptitle(f"{model_name} Grad-CAM Layer Sweep — {case_tag.capitalize()} Case  (Tumor ROI)", y=1.02)
    with model_on_gpu(model_name) as v:
        for col, layer in enumerate(SWEEP_LAYERS):
            try:
                img_t   = torch.from_numpy(crop_np).unsqueeze(0).to(v.device)
                gc_cam  = GradCAM(nn_module=v.model, target_layers=layer)
                cam_raw = gc_cam(x=img_t, class_idx=TARGET_CLASS)
                cam     = resize_cam(cam_raw, img_t.shape[2:])
                cam_np  = cam[0].detach().cpu().numpy()
                cam_sl  = cam_np[0, :, :, mid_d]
                del gc_cam, cam_raw, cam, img_t
                axes[0, col].imshow(img_sl, cmap="gray"); axes[0, col].axis("off")
                axes[0, col].set_title(layer)
                axes[1, col].imshow(img_sl, cmap="gray")
                axes[1, col].imshow(cam_sl, cmap="jet_r", alpha=0.6)
                axes[1, col].axis("off"); axes[1, col].set_title("Grad-CAM")
            except Exception as e:
                axes[0, col].axis("off"); axes[0, col].set_title(f"{layer}\n(error)")
                axes[1, col].axis("off")
                print(f"    {model_name}/{layer}: {e}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, hspace=0.20, wspace=0.04)
    path = str(SWEEP_DIR / f"gradcam_{model_name.lower()}_{case_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def _ig_modality_sweep(model_name, case_tag, case_idx):
    """IG attribution for each of the 4 MRI modalities — one figure."""
    img_np  = all_imgs[case_idx]
    lbl_np  = all_lbls[case_idx]
    crop_np = tumor_crop(img_np, lbl_np)
    mid_d   = crop_np.shape[3] // 2
    nm      = len(MODALITY_NAMES)
    fig, axes = plt.subplots(2, nm, figsize=(6 * nm, 10))
    fig.suptitle(f"{model_name} IG Modality Sweep — {case_tag.capitalize()} Case  (Tumor ROI)", y=1.02)
    with model_on_gpu(model_name) as v:
        img_t = torch.from_numpy(crop_np).unsqueeze(0).to(v.device).requires_grad_(True)
        def fwd(x, _model=v.model):
            return _model(x)[:, TARGET_CLASS].sum().unsqueeze(0)
        attr    = IntegratedGradients(fwd).attribute(
                      img_t, target=None, n_steps=IG_STEPS, internal_batch_size=1)
        attr_np = attr.detach().cpu().numpy()[0]   # [4,H,W,D]
        del img_t, attr
    for col, (ch, mname) in enumerate(zip(range(4), MODALITY_NAMES)):
        img_sl  = crop_np[ch, :, :, mid_d]
        attr_sl = normalize_attr(attr_np[ch, :, :, mid_d])
        axes[0, col].imshow(img_sl, cmap="gray"); axes[0, col].axis("off")
        axes[0, col].set_title(mname)
        axes[1, col].imshow(img_sl, cmap="gray")
        axes[1, col].imshow(attr_sl, cmap="magma", alpha=0.8)
        axes[1, col].axis("off"); axes[1, col].set_title("IG Attribution")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, hspace=0.20, wspace=0.04)
    path = str(SWEEP_DIR / f"ig_modality_{model_name.lower()}_{case_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def _activation_sweep(model_name, case_tag, case_idx):
    """Mean channel activation for every encoder layer — one figure."""
    img_np  = all_imgs[case_idx]
    lbl_np  = all_lbls[case_idx]
    crop_np = tumor_crop(img_np, lbl_np)
    mid_d   = crop_np.shape[3] // 2
    img_sl  = crop_np[CHANNEL_IDX, :, :, mid_d]
    spatial = (96, 96, 96)
    nl      = len(SWEEP_LAYERS)
    fig, axes = plt.subplots(2, nl, figsize=(6 * nl, 10))
    fig.suptitle(f"{model_name} Activation Layer Sweep — {case_tag.capitalize()} Case  (Tumor ROI)", y=1.02)
    with model_on_gpu(model_name) as v:
        named = dict(v.model.named_modules())
        for col, layer in enumerate(SWEEP_LAYERS):
            if layer not in named:
                axes[0, col].axis("off"); axes[0, col].set_title(f"{layer}\n(missing)")
                axes[1, col].axis("off"); continue
            acts = []
            def _hook(module, inp, out, _store=acts):
                o = out[0] if isinstance(out, (tuple, list)) else out
                _store.append(o.detach().cpu())
            handle = named[layer].register_forward_hook(_hook)
            img_t = torch.from_numpy(crop_np).unsqueeze(0).to(v.device)
            with torch.no_grad():
                _ = v.model(img_t)
            handle.remove(); del img_t
            if not acts:
                axes[0, col].axis("off"); axes[0, col].set_title(f"{layer}\n(no act)")
                axes[1, col].axis("off"); continue
            act = acts[0].mean(dim=1, keepdim=True).float()
            act = torch.nn.functional.interpolate(act, size=spatial, mode="trilinear", align_corners=False)
            act_np = norm01(act[0, 0].numpy())
            act_sl = act_np[:, :, mid_d]
            axes[0, col].imshow(img_sl, cmap="gray"); axes[0, col].axis("off")
            axes[0, col].set_title(layer)
            axes[1, col].imshow(img_sl, cmap="gray")
            axes[1, col].imshow(act_sl, cmap="hot", alpha=0.6)
            axes[1, col].axis("off"); axes[1, col].set_title("Activation")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, hspace=0.20, wspace=0.04)
    path = str(SWEEP_DIR / f"activation_{model_name.lower()}_{case_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


# ── Run all sweeps: every model × every case × every viz type ─────────────────
print("\nRunning comprehensive sweeps (GradCAM / IG Modality / Activation)...")
for _mname in ["SwinMamba", "MambaUNETR", "SwinUNETRPlus"]:
    for _ctag, _cidx in [("best", best_idx), ("worst", worst_idx)]:
        print(f"\n  [{_mname}  {_ctag}]")
        _gradcam_sweep(_mname, _ctag, _cidx)
        _gc.collect(); torch.cuda.empty_cache()
        _ig_modality_sweep(_mname, _ctag, _cidx)
        _gc.collect(); torch.cuda.empty_cache()
        _activation_sweep(_mname, _ctag, _cidx)
        _gc.collect(); torch.cuda.empty_cache()


# ── 5. GradCAM — all 3 models ─────────────────────────────────────────────────
print("\nGenerating GradCAM (all 3 models)...")
for case_tag, case_idx in [("best", best_idx), ("worst", worst_idx)]:
    img_np  = all_imgs[case_idx]
    lbl_np  = all_lbls[case_idx]
    crop_np = tumor_crop(img_np, lbl_np)           # [4,96,96,96]
    mid_d   = crop_np.shape[3] // 2
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Grad-CAM — {case_tag.capitalize()} Case  (Whole Tumor, Tumor ROI)",
        y=1.02,
    )

    for col, name in enumerate(["SwinMamba", "MambaUNETR", "SwinUNETRPlus"]):
        layer = TARGET_LAYERS[name]
        with model_on_gpu(name) as v:
            img_t   = torch.from_numpy(crop_np).unsqueeze(0).to(v.device)
            gc_cam  = GradCAM(nn_module=v.model, target_layers=layer)
            cam_raw = gc_cam(x=img_t, class_idx=TARGET_CLASS)
            cam     = resize_cam(cam_raw, img_t.shape[2:])
            cam_np  = cam[0].detach().cpu().numpy()
            del gc_cam, cam_raw, cam, img_t
        img_sl = crop_np[CHANNEL_IDX, :, :, mid_d]
        cam_sl = cam_np[0, :, :, mid_d]

        axes[0, col].imshow(img_sl, cmap="gray"); axes[0, col].axis("off")
        axes[0, col].set_title(f"{name}  [{layer}]\nMRI T1c")
        axes[1, col].imshow(img_sl, cmap="gray")
        axes[1, col].imshow(cam_sl, cmap="jet_r", alpha=0.6)
        axes[1, col].axis("off")
        axes[1, col].set_title("Grad-CAM Overlay")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.01,
                        hspace=0.20, wspace=0.04)
    save_path = str(OUT_DIR / f"gradcam_all_{case_tag}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

# ── 6. Integrated Gradients — all 3 models ───────────────────────────────────
_gc.collect(); torch.cuda.empty_cache()
print("\nGenerating Integrated Gradients (all 3 models)...")
for case_tag, case_idx in [("best", best_idx), ("worst", worst_idx)]:
    img_np  = all_imgs[case_idx]
    lbl_np  = all_lbls[case_idx]
    crop_np = tumor_crop(img_np, lbl_np)           # [4,96,96,96]
    mid_d   = crop_np.shape[3] // 2
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Integrated Gradients — {case_tag.capitalize()} Case  (Whole Tumor, Tumor ROI)",
        y=1.02,
    )

    for col, name in enumerate(["SwinMamba", "MambaUNETR", "SwinUNETRPlus"]):
        with model_on_gpu(name) as v:
            img_t = torch.from_numpy(crop_np).unsqueeze(0).to(v.device).requires_grad_(True)

            def fwd(x, _model=v.model):
                return _model(x)[:, TARGET_CLASS].sum().unsqueeze(0)

            attr    = IntegratedGradients(fwd).attribute(
                          img_t, target=None, n_steps=IG_STEPS, internal_batch_size=1)
            attr_np = attr.detach().cpu().numpy()[0]
            del img_t, attr
        img_sl  = crop_np[CHANNEL_IDX, :, :, mid_d]
        attr_sl = normalize_attr(attr_np[CHANNEL_IDX, :, :, mid_d])

        axes[0, col].imshow(img_sl, cmap="gray"); axes[0, col].axis("off")
        axes[0, col].set_title(f"{name}\nMRI T1c")
        axes[1, col].imshow(img_sl, cmap="gray")
        axes[1, col].imshow(attr_sl, cmap="magma", alpha=0.8)
        axes[1, col].axis("off")
        axes[1, col].set_title("Integrated Gradients")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.01,
                        hspace=0.20, wspace=0.04)
    save_path = str(OUT_DIR / f"ig_all_{case_tag}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

# ── 7. Activation maps — encoder4 output via hook (all 3 models) ─────────────
_gc.collect(); torch.cuda.empty_cache()
print("\nGenerating activation maps (encoder4)...")
for case_tag, case_idx in [("best", best_idx), ("worst", worst_idx)]:
    img_np  = all_imgs[case_idx]
    mid_d   = img_np.shape[2]//2
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Encoder4 Feature Activation Map — {case_tag.capitalize()} Case",
        y=1.02,
    )

    spatial_size = tuple(img_np.shape[1:])   # (H,W,D)
    for col, name in enumerate(["SwinMamba", "MambaUNETR", "SwinUNETRPlus"]):
        with model_on_gpu(name) as v:
            img_t   = torch.from_numpy(img_np).unsqueeze(0).to(v.device)
            acts    = []

            def _hook(module, inp, out, _store=acts):
                o = out[0] if isinstance(out, (tuple, list)) else out
                _store.append(o.detach().cpu())

            handle = dict(v.model.named_modules())["encoder4"].register_forward_hook(_hook)
            with torch.no_grad():
                _ = sliding_window_inference(
                    img_t, roi_size=(96,96,96), sw_batch_size=2,
                    predictor=v.model, overlap=0.5, mode="gaussian",
                )
            handle.remove()

        act = acts[0].mean(dim=1, keepdim=True).float()  # [1,1,H',W',D']
        act = torch.nn.functional.interpolate(
            act, size=spatial_size, mode="trilinear", align_corners=False
        )
        act_np = norm01(act[0, 0].numpy())
        img_sl = img_np[CHANNEL_IDX, :, mid_d, :]
        act_sl = act_np[:, mid_d, :]

        axes[0, col].imshow(img_sl, cmap="gray"); axes[0, col].axis("off")
        axes[0, col].set_title(f"{name}\nMRI T1c")
        axes[1, col].imshow(img_sl, cmap="gray")
        axes[1, col].imshow(act_sl, cmap="hot", alpha=0.6)
        axes[1, col].axis("off")
        axes[1, col].set_title("Feature Activation")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.01,
                        hspace=0.20, wspace=0.04)
    save_path = str(OUT_DIR / f"activations_all_{case_tag}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved → {save_path}")

# ── 8. Frame-by-frame GIFs — prediction overlay per model ────────────────────
print("\nGenerating slice-by-slice GIFs...")

def make_gif(case_tag, case_idx):
    img_np  = all_imgs[case_idx]
    lbl_np  = all_lbls[case_idx]
    n_slices = img_np.shape[2]
    frames  = []

    for s in range(n_slices):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(f"{case_tag} case — slice {s+1}/{n_slices}  "
                     f"{LEGEND_LABEL}", fontsize=9)
        img_sl = img_np[CHANNEL_IDX, :, s, :]
        lbl_sl = np.stack([lbl_np[c, :, s, :] for c in range(3)])

        axes[0].imshow(img_sl, cmap="gray"); axes[0].axis("off")
        axes[0].set_title("MRI T1c", fontsize=7)

        axes[1].imshow(img_sl, cmap="gray")
        for c, col in enumerate(TUMOR_COLORS):
            m = lbl_sl[c].astype(float)
            if m.any():
                axes[1].imshow(np.ma.masked_where(m==0, m),
                               cmap=plt.cm.colors.ListedColormap([col]),
                               alpha=0.5, vmin=0, vmax=1)
        axes[1].axis("off"); axes[1].set_title("GT", fontsize=7)

        for ai, name in enumerate(["SwinMamba", "MambaUNETR", "SwinUNETRPlus"], start=2):
            pred_sl = np.stack([all_preds[name][case_idx][c, :, s, :] for c in range(3)])
            axes[ai].imshow(img_sl, cmap="gray")
            for c, col in enumerate(TUMOR_COLORS):
                m = pred_sl[c].astype(float)
                if m.any():
                    axes[ai].imshow(np.ma.masked_where(m==0, m),
                                   cmap=plt.cm.colors.ListedColormap([col]),
                                   alpha=0.5, vmin=0, vmax=1)
            axes[ai].axis("off"); axes[ai].set_title(name, fontsize=7)

        plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)

        # Save PNG first (while fig is still open)
        slices_dir = OUT_DIR / f"slices_{case_tag}"; slices_dir.mkdir(exist_ok=True)
        png_path = str(slices_dir / f"slice_{s:03d}.png")
        plt.savefig(png_path, dpi=100, bbox_inches="tight")

        # Grab frame for GIF from the same figure
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    gif_path = str(OUT_DIR / f"slices_{case_tag}.gif")
    imageio.mimsave(gif_path, frames, fps=8, loop=0)
    print(f"Saved → {gif_path}  +  {n_slices} PNGs in slices_{case_tag}/")

if not _args.gradcam_only:
    make_gif("best",  best_idx)
    make_gif("worst", worst_idx)

print(f"\nAll outputs saved to {OUT_DIR}/")
print("  predictions_best_worst.png")
print("  gradcam_all_best.png / gradcam_all_worst.png")
print("  ig_all_best.png / ig_all_worst.png")
print("  activations_all_best.png / activations_all_worst.png")
print("  slices_best.gif / slices_worst.gif")
