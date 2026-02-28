# SwinSSM

A PyTorch implementation of hybrid Swin Transformer and Mamba architectures for 3D brain tumor segmentation, featuring 14 loss functions, adaptive scheduling, and local minima escape strategies.

## Installation

### On Kaggle

Clone the repository and install the package:

```python
!if [ -d "SwinSSM" ]; then \
    cd SwinSSM && \
    git fetch origin && \
    git checkout main && \
    git pull origin main && \
    cd ..; \
else \
    git clone --branch main https://github.com/Yok4ai/SwinSSM.git; \
fi

!pip install -q ./SwinSSM
```

### Local Installation

```bash
git clone https://github.com/Yok4ai/SwinSSM.git
cd SwinSSM
pip install -e .
```

## Project Structure

```
SwinSSM/
├── src/
│   ├── models/         # Model architectures and training pipeline
│   ├── data/           # Data loading and augmentation (BraTS 2021/2023)
│   └── utils/          # Visualization, validation, and statistical utilities
├── run.py        # Main CLI entry point (30+ parameters)
├── main.py              # Core training orchestration
├── setup.py             # Package setup
└── requirements.txt     # Dependencies
```

## Execution Flow

1. **`run.py`**: Parses CLI arguments, sets up the environment and data, then calls `main()`.
2. **`main.py`**: Coordinates the full training pipeline:
   1. Calls `get_transforms` from `src/data/augmentations.py`
   2. Calls `get_dataloaders` from `src/data/dataloader.py`
   3. Calls `setup_training` from `src/models/trainer.py` to initialize the model and Lightning Trainer
   4. Calls `train_model` from `src/models/trainer.py` to run training, validation, and checkpointing
3. **`src/models/pipeline.py`**: Wraps the model in a Lightning module with 14 loss functions and adaptive scheduling.

## Quick Start

### Choose a Loss Strategy

**Baseline:**
```bash
python run.py --loss_type dicece --use_class_weights
```

**Standard:**
```bash
python run.py --loss_type generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5
```

**Research / Competition:**
```bash
python run.py --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling --structure_epochs 40 --boundary_epochs 70 \
  --use_warm_restarts --restart_period 30
```

### Running on Kaggle

Pass your dataset path directly with `--data_dir`:

```python
!python ./SwinSSM/run.py --data_dir /kaggle/input/your-brats-dataset --dataset brats2023 --loss_type generalized_dice_focal --use_class_weights
```

### Running Locally

```bash
python run.py --data_dir /path/to/BRATS2023-training --dataset brats2023
```

## Key Features

### Model Architectures

- **SwinUNETR**: Baseline Swin Transformer UNETR
- **SwinUNETR++**: Enhanced with multi-scale attention, hierarchical skip connections, and V2 residual blocks
- **SwinMamba**: Swin Transformer encoder + multi-directional Mamba stages
- **SegMamba / MambaUNETR**: Mamba encoder with SwinUNETR-style decoder
- **VITMamba**: Axial ViT (stages 0-1) + Mamba (stages 2-3)
- **MambaFormer**: NNFormer Swin (stages 0-1) + Mamba (stages 2-3)

### Loss Functions (14 total)

- **Basic**: Dice, DiceCE, DiceFocal, Generalized Dice, Focal, Tversky, Hausdorff
- **Hybrid**: GDL+Focal+Tversky, Dice+Hausdorff combinations
- **Adaptive**: Structure-boundary scheduling, progressive hybrid, complexity cascade, dynamic performance-based

### Training Features

- Adaptive loss scheduling with linear, exponential, and cosine weight transitions
- Warm restarts and plateau detection for local minima escape
- Mixed precision training (fp16/bf16/fp32)
- Multi-GPU DDP support
- Smart sampling for class-imbalanced data
- WandB experiment logging

## CLI Reference

### Core Parameters

```bash
--dataset           brats2021 | brats2023 | combined
--epochs            Number of training epochs (default: 100)
--batch_size        Batch size (default: 2)
--learning_rate     Learning rate (default: 1e-4)
--feature_size      Model feature size (default: 48)
--loss_type         See loss function options below
--precision         16-mixed | bf16-mixed | 32 (default: 16-mixed)
--optimizer_type    adamw | adamw8bit | muon (default: adamw)
```

### Loss Function Options

```bash
--loss_type dice | dicece | dicefocal | generalized_dice | generalized_dice_focal |
           focal | tversky | hausdorff | hybrid_gdl_focal_tversky | hybrid_dice_hausdorff |
           adaptive_structure_boundary | adaptive_progressive_hybrid |
           adaptive_complexity_cascade | adaptive_dynamic_hybrid
```

### Mamba Architecture

```bash
--use_mamba                           Enable Mamba architecture
--mamba_type                          segmamba | swinmamba | mambaunetr |
                                      vitmamba | mambaformer | mambaswin
--d_state 16                          Mamba state dimension
--d_conv 3                            Mamba convolution width
--expand 1                            Mamba expansion factor
--window_size 4 4 4                   Window size for SwinMamba model
```

### SwinUNETR++ Enhancements

```bash
--use_v2                    V2 residual blocks
--patch_norm                Patch normalization
--use_multi_scale_attention Multi-scale window attention
--use_adaptive_window       Adaptive window sizing
--use_cross_layer_fusion    Cross-layer attention fusion
--use_hierarchical_skip     Hierarchical skip connections
--use_enhanced_v2_blocks    Enhanced V2 residual blocks
--use_modality_attention    Modality attention module
```

### Adaptive Scheduling

```bash
--use_adaptive_scheduling
--adaptive_schedule_type    linear | exponential | cosine (default: linear)
--structure_epochs 30       Epochs to focus on structure learning
--boundary_epochs 50        Epochs to focus on boundary refinement
--min_loss_weight 0.1       Minimum loss component weight
--max_loss_weight 2.0       Maximum loss component weight
```

### Local Minima Escape

```bash
--use_warm_restarts         Enable cosine annealing with warm restarts
--restart_period 20         Restart period in epochs
--restart_mult 1            Period multiplier
```

### Inference and Post-processing

```bash
--roi_size 96 96 96         Sliding window ROI size
--sw_batch_size 1           Sliding window batch size
--overlap 0.7               Sliding window overlap
--threshold 0.5             Output binarization threshold
--wt_threshold 250          WT volume post-processing threshold
--tc_threshold 150          TC volume post-processing threshold
--et_threshold 100          ET volume post-processing threshold
```

## Advanced Usage

### Package Import

```python
from src.models.swinunetrplus import SwinUNETR
from src.models.pipeline import BrainTumorSegmentation

model = BrainTumorSegmentation(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_type='adaptive_progressive_hybrid',
    use_adaptive_scheduling=True,
    use_warm_restarts=True
)
```

### Validation with Test-Time Augmentation

```bash
python validate_with_tta.py \
    --checkpoint_path checkpoints/best_model.ckpt \
    --data_dir dataset/dataset.json \
    --dataset brats2023 \
    --use_tta
```

### Statistical Analysis

```bash
python run_statistical_analysis.py \
    --baseline_csv baseline_dice.csv \
    --proposed_csv proposed_dice.csv
```

### Inference Benchmarking

```bash
python benchmark_inference.py
```

## License

MIT License
