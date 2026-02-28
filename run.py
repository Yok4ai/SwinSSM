# run.py
import sys
import os
from dataset_setup import setup_data
from main import main
import argparse
import torch
import warnings

# CLI argument parsing for key parameters
def parse_cli_args():
    parser = argparse.ArgumentParser()

### Basic configuration parameters

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Dataset and training parameters
    parser.add_argument('--dataset', type=str, default='brats2023', choices=['brats2021', 'brats2023', 'combined'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of data loader workers')
    parser.add_argument('--img_size', type=int, default=96, help='Input image size')
    parser.add_argument('--roi_size', type=int, nargs=3, default=[96, 96, 96], help='ROI size for sliding window inference (default: 96 96 96)')
    parser.add_argument('--feature_size', type=int, default=48, help='Model feature size')
    parser.add_argument('--loss_type', type=str, default='dice', 
                        choices=['dice', 'dicece', 'dicefocal', 'generalized_dice', 'generalized_dice_focal', 
                                'focal', 'tversky', 'hausdorff', 's3f', 'hybrid_gdl_focal_tversky', 'hybrid_dice_hausdorff',
                                'adaptive_structure_boundary', 'adaptive_progressive_hybrid', 
                                'adaptive_complexity_cascade', 'adaptive_dynamic_hybrid'], 
                        help='Loss function type')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--data_dir', nargs='+', default=None, help='Path(s) to raw BraTS data. If omitted, defaults to dataset/BRATS2023-training or dataset/BRATS2021-training depending on --dataset.')

### Basic configuration parameters

### Advanced training configuration

    # Warmup epochs for learning rate scheduler
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for LR scheduler')

    # Optimizer type
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'adamw8bit', 'muon'], help='Optimizer type: adamw, adamw8bit, or muon (default: adamw)')
    parser.add_argument('--optimizer_betas', type=float, nargs=2, default=[0.9, 0.999], help='Optimizer beta parameters (default: 0.9 0.999)')
    parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Optimizer epsilon parameter (default: 1e-8)')
    parser.add_argument('--muon_scheduler', action='store_true', help='Enable learning rate scheduling for Muon optimizer (disabled by default for optimal performance)')
    parser.add_argument('--save_interval', type=int, default=1, help='Checkpoint save interval in epochs (default: 1)')

    # Sliding window inference parameters
    parser.add_argument('--sw_batch_size', type=int, default=1, help='Sliding window batch size (default: 1)')
    parser.add_argument('--overlap', type=float, default=0.7, help='Sliding window inference overlap (default: 0.7)')

    # Early stopping and Callback parameters
    parser.add_argument('--early_stopping_patience', type=int, default=None, help='Early stopping patience epochs (disabled by default)')
    parser.add_argument('--limit_val_batches', type=int, default=5, help='Limit validation batches for faster validation (default: 5)')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval in epochs (default: 1)')
    parser.add_argument('--fast_validation', action='store_true', help='Enable fast validation mode (only loss and dice, no wandb logging)')

    # Gradient accumulation and clipping parameters
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients over (default: 1)')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value (default: 0)')
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm', choices=['value', 'norm'], help='Gradient clipping algorithm (default: norm)')
    
    # Model compilation parameter
    parser.add_argument('--compile_model', action='store_true', help='Enable torch.compile for faster inference (default: False)')
    
    # Checkpointing and resuming
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file (.ckpt)')
    parser.add_argument('--load_pretrained', type=str, default=None, help='Path to checkpoint file (.pth/.ckpt) - load pretrained weights, start training from epoch 0')

    # Data efficiency parameters
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='Fraction of training data to use (0.3 = 30%, default: 1.0)')
    
    # Smart sampling parameters
    parser.add_argument('--smart_sampling', action='store_true', help='Enable intelligent sample selection based on tumor characteristics')
    parser.add_argument('--sample_fraction', type=float, default=0.3, help='Fraction of data for smart sampling (default: 0.3)')
    parser.add_argument('--sampling_strategy', type=str, default='balanced', choices=['balanced', 'hard', 'diverse'], 
                        help='Smart sampling strategy (default: balanced)')
    
    # Precision parameter 
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['16-mixed', 'bf16-mixed', '32'], help='Training precision: 16-mixed, bf16-mixed, or 32 (default: 16-mixed)')
    
    # Strategy parameter
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'ddp_spawn', 'dp', 'single_device'], help='Training strategy: auto, ddp, ddp_spawn, dp, or single_device (default: auto)')

    # Post-processing parameters
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for post-processing discrete output (default: 0.5)')
    parser.add_argument('--wt_threshold', type=int, default=250, help='BraTS WT volume threshold for postprocessing (default: 250)')
    parser.add_argument('--tc_threshold', type=int, default=150, help='BraTS TC volume threshold for postprocessing (default: 150)')
    parser.add_argument('--et_threshold', type=int, default=100, help='BraTS ET volume threshold for postprocessing (default: 100)')

    
### Advanced training configuration

### Loss function parameters

    # Class Weights
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for loss (default: False)')
    parser.add_argument('--class_weights', type=float, nargs=3, default=[3.0, 1.0, 5.0], help='Class weights for TC, WT, ET (default: 3.0 1.0 5.0)')

    # Loss Weights
    parser.add_argument('--tversky_alpha', type=float, default=0.5, help='Tversky loss alpha parameter (default: 0.5)')
    parser.add_argument('--tversky_beta', type=float, default=0.5, help='Tversky loss beta parameter (default: 0.5)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None, help='Focal loss alpha parameter (default: None)')
    parser.add_argument('--gdl_weight_type', type=str, default='square', choices=['square', 'simple', 'uniform'], 
                        help='Generalized Dice Loss weight type (default: square)')
    parser.add_argument('--gdl_lambda', type=float, default=1.0, help='Generalized Dice Loss lambda parameter (default: 1.0)')
    parser.add_argument('--hausdorff_alpha', type=float, default=2.0, help='Hausdorff loss alpha parameter (default: 2.0)')
    parser.add_argument('--lambda_dice', type=float, default=1.0, help='Lambda weight for Dice loss component (default: 1.0)')
    parser.add_argument('--lambda_focal', type=float, default=1.0, help='Lambda weight for Focal loss component (default: 1.0)')
    parser.add_argument('--lambda_tversky', type=float, default=1.0, help='Lambda weight for Tversky loss component (default: 1.0)')
    parser.add_argument('--lambda_hausdorff', type=float, default=1.0, help='Lambda weight for Hausdorff loss component (default: 1.0)')

    # S3F loss parameters
    parser.add_argument('--s3f_alpha', type=float, default=1.0, help='S3F focal loss weight (default: 1.0)')
    parser.add_argument('--s3f_beta', type=float, default=0.5, help='S3F structure (IoU) loss weight (default: 0.5)')
    parser.add_argument('--s3f_delta', type=float, default=0.3, help='S3F boundary (Hausdorff) loss weight (default: 0.3)')
    parser.add_argument('--s3f_gamma', type=float, default=2.0, help='S3F focal gamma parameter (default: 2.0)')

    # Adaptive loss scheduling parameters
    parser.add_argument('--use_adaptive_scheduling', action='store_true', help='Enable adaptive loss scheduling (default: False)')
    parser.add_argument('--adaptive_schedule_type', type=str, default='linear', choices=['linear', 'exponential', 'cosine'],
                        help='Type of adaptive scheduling (default: linear)')
    parser.add_argument('--structure_epochs', type=int, default=30, help='Epochs to focus on structure learning (default: 30)')
    parser.add_argument('--boundary_epochs', type=int, default=50, help='Epochs to focus on boundary refinement (default: 50)')
    parser.add_argument('--schedule_start_epoch', type=int, default=10, help='Epoch to start adaptive scheduling (default: 10)')
    parser.add_argument('--min_loss_weight', type=float, default=0.1, help='Minimum weight for any loss component (default: 0.1)')
    parser.add_argument('--max_loss_weight', type=float, default=2.0, help='Maximum weight for any loss component (default: 2.0)')

### Loss function parameters

   
### Scheduling and optimizer parameters

    # Learning rate scheduling parameters
    parser.add_argument('--use_warm_restarts', action='store_true', help='Enable cosine annealing with warm restarts (default: False)')
    parser.add_argument('--restart_period', type=int, default=20, help='Restart period in epochs (default: 20)')
    parser.add_argument('--restart_mult', type=int, default=1, help='Restart period multiplier (default: 1)')
    parser.add_argument('--use_reduce_lr_on_plateau', action='store_true', help='Enable ReduceLROnPlateau scheduler (default: False)')
    parser.add_argument('--plateau_patience', type=int, default=5, help='Patience for ReduceLROnPlateau (default: 5)')
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate (default: 1e-7)')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='Use cosine annealing scheduler without warmup (default: False)')

### Scheduling and optimization parameters


### Model architecture arguments

    parser.add_argument('--patch_norm', action='store_true', help='Enable patch normalization (default: False)')
    parser.add_argument('--use_modality_attention', action='store_true', help='Enable Modality Attention module (default: False)')
    parser.add_argument('--use_v2', action='store_true', help='Enable SwinUNETR V2 residual blocks (default: False)')
    
    # SwinUNETRPlus enhancement arguments (disabled by default for vanilla behavior)
    parser.add_argument('--use_multi_scale_attention', action='store_true', help='Enable multi-scale window attention (default: False)')
    parser.add_argument('--use_adaptive_window', action='store_true', help='Enable adaptive window sizing (default: False)')
    parser.add_argument('--use_cross_layer_fusion', action='store_true', help='Enable cross-layer attention fusion (default: False)')
    parser.add_argument('--use_hierarchical_skip', action='store_true', help='Enable hierarchical skip connections (default: False)')
    parser.add_argument('--use_enhanced_v2_blocks', action='store_true', help='Enable enhanced V2 residual blocks (default: False)')
    parser.add_argument('--multi_scale_window_sizes', type=int, nargs='+', default=[7, 5, 3], help='Window sizes for multi-scale attention (default: 7 5 3)')
    
    # Mamba architecture parameters
    parser.add_argument('--use_mamba', action='store_true', help='Use Mamba architecture (default: False)')
    parser.add_argument('--mamba_type', type=str, default='segmamba',
                        choices=['segmamba', 'swinmamba', 'mambaunetr',
                                 'vitmamba', 'mambaformer', 'mambaswin'],
                        help='Type of Mamba architecture (default: segmamba)')
    
    # Mamba d_ parameters
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension (default: 16)')
    parser.add_argument('--d_conv', type=int, default=3, help='Mamba convolution width parameter (default: 3)')  
    parser.add_argument('--expand', type=int, default=1, help='Mamba expansion factor (default: 1)')
    
    parser.add_argument('--window_size', type=int, nargs=3, default=[4, 4, 4], help='Window size for SwinMamba model (default: 4 4 4)')
    
### Model architecture arguments

    return parser.parse_args()

cli_args = parse_cli_args()

"""
## COMPREHENSIVE EXAMPLES - All Toggleable Parameters

###  PURE VANILLA SWINUNETR (Original Architecture)
# All enhancements disabled - exactly like original SwinUNETR
python run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 48 \
  --loss_type dice \
  --precision 16-mixed \
  --optimizer_type adamw

###  VANILLA + V2 RESIDUAL BLOCKS  
# Just enable V2 blocks - the standard SwinUNETR V2
python run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 48 \
  --loss_type dice \
  --use_v2 \
  --precision bf16-mixed

###  ENHANCED WITH PATCH NORMALIZATION
# Vanilla + patch normalization (good for medical imaging)
python run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 48 \
  --loss_type dice \
  --patch_norm \
  --optimizer_type adamw8bit

###  MODERATE ENHANCEMENT
# V2 + Multi-scale attention + Patch norm
python run.py \
  --dataset brats2023 \
  --epochs 75 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 48 \
  --loss_type dicece \
  --use_v2 \
  --use_multi_scale_attention \
  --patch_norm \
  --use_class_weights \
  --class_weights 3.0 1.0 5.0 \
  --precision bf16-mixed \
  --optimizer_type adamw

###  ADVANCED ENHANCEMENT  
# V2 + Multi-scale + Hierarchical skips + Enhanced blocks
python run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --learning_rate 8e-5 \
  --feature_size 48 \
  --loss_type dicefocal \
  --use_v2 \
  --use_multi_scale_attention \
  --use_hierarchical_skip \
  --use_enhanced_v2_blocks \
  --patch_norm \
  --multi_scale_window_sizes 7 5 3 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0 \
  --use_modality_attention \
  --precision bf16-mixed \
  --optimizer_type adamw8bit

###  FULL SWINUNETR PLUS (All Enhancements)
# Maximum architectural improvements
python run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --feature_size 48 \
  --loss_type generalized_dice_focal \
  --use_v2 \
  --use_multi_scale_attention \
  --use_adaptive_window \
  --use_cross_layer_fusion \
  --use_hierarchical_skip \
  --use_enhanced_v2_blocks \
  --patch_norm \
  --multi_scale_window_sizes 9 7 5 3 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0 \
  --use_modality_attention \
  --overlap 0.75 \
  --threshold 0.4 \
  --precision bf16-mixed \
  --optimizer_type adamw8bit

###  MEMORY-OPTIMIZED ENHANCED
# Good balance of features for smaller GPUs
python run.py \
  --dataset brats2023 \
  --epochs 80 \
  --batch_size 1 \
  --img_size 96 \
  --roi_size 96 96 96 \
  --learning_rate 1e-4 \
  --feature_size 24 \
  --loss_type dice \
  --use_v2 \
  --use_multi_scale_attention \
  --patch_norm \
  --multi_scale_window_sizes 5 3 \
  --use_class_weights \
  --early_stopping_patience 12 \
  --limit_val_batches 3 \
  --precision 16-mixed \
  --optimizer_type adamw8bit

###  HIGH-PERFORMANCE TRAINING
# For powerful GPUs with lots of memory
python run.py \
  --dataset brats2023 \
  --epochs 150 \
  --batch_size 2 \
  --img_size 128 \
  --roi_size 128 128 128 \
  --learning_rate 8e-5 \
  --feature_size 64 \
  --loss_type generalized_dice_focal \
  --use_v2 \
  --use_multi_scale_attention \
  --use_hierarchical_skip \
  --use_enhanced_v2_blocks \
  --patch_norm \
  --multi_scale_window_sizes 9 7 5 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0 \
  --use_modality_attention \
  --overlap 0.8 \
  --warmup_epochs 15 \
  --early_stopping_patience 20 \
  --precision 32 \
  --optimizer_type adamw

###  SWINMAMBA HYBRID (NEW!) - Best of Both Worlds
# SwinUNETR hierarchical structure + Mamba O(N) efficiency
python run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 48 \
  --loss_type dicece \
  --use_mamba \
  --mamba_type swinmamba \
  --use_class_weights \
  --class_weights 3.0 1.0 5.0 \
  --precision 16-mixed \
  --optimizer_type adamw

###  PURE SEGMAMBA - Maximum Efficiency  
# Pure state-space model for ultimate memory savings
python run.py \
  --dataset brats2023 \
  --epochs 80 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --feature_size 48 \
  --loss_type dicece \
  --use_mamba \
  --mamba_type segmamba \
  --use_class_weights \
  --precision 16-mixed \
  --optimizer_type adamw8bit

###  CURRENT TOGGLEABLE PARAMETERS SUMMARY:
# Architecture Enhancements:
#   --use_v2                      # V2 residual blocks
#   --patch_norm                  # Patch normalization
#   --use_multi_scale_attention   # Multi-scale window attention
#   --use_adaptive_window         # Adaptive window sizing
#   --use_hierarchical_skip       # Hierarchical skip connections
#   --use_enhanced_v2_blocks      # Enhanced V2 residual blocks
#   --use_modality_attention      # Modality attention module
#
# Configuration:
#   --multi_scale_window_sizes 7 5 3  # Window sizes for multi-scale
#   --use_class_weights           # Enable class weighting
#   --class_weights 3.0 1.0 5.0   # Weights for TC, WT, ET
#   --overlap 0.7                 # Sliding window overlap
#   --threshold 0.5               # Output threshold
#   --optimizer_type {adamw,adamw8bit}  # Optimizer selection
#   --precision {16-mixed,bf16-mixed,32}  # Training precision

## Example usage - Multiple loss function options:

# Standard Dice Loss
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type dice \
  --learning_rate 5e-4 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# DiceFocal Loss
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type dicefocal \
  --learning_rate 5e-4 \
  --focal_gamma 2.0 \
  --lambda_dice 1.0 \
  --lambda_focal 1.0 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Hybrid: Generalized Dice + Focal + Tversky
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type hybrid_gdl_focal_tversky \
  --learning_rate 5e-4 \
  --gdl_lambda 1.0 \
  --lambda_focal 0.5 \
  --lambda_tversky 0.3 \
  --focal_gamma 2.0 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Hybrid: Dice + Hausdorff
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type hybrid_dice_hausdorff \
  --learning_rate 5e-4 \
  --lambda_dice 1.0 \
  --lambda_hausdorff 0.1 \
  --hausdorff_alpha 2.0 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Adaptive Structure-Boundary Scheduling
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type adaptive_structure_boundary \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --adaptive_schedule_type cosine \
  --schedule_start_epoch 15 \
  --min_loss_weight 0.2 \
  --max_loss_weight 1.5 \
  --use_class_weights

# Adaptive Progressive Hybrid
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --loss_type adaptive_progressive_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --schedule_start_epoch 10 \
  --use_class_weights

# Adaptive Complexity Cascade
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type adaptive_complexity_cascade \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --adaptive_schedule_type linear \
  --use_class_weights

# Adaptive Dynamic Hybrid (Performance-based)
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type adaptive_dynamic_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --schedule_start_epoch 20 \
  --use_class_weights

# Warm Restarts for Local Minima Escape
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type generalized_dice_focal \
  --learning_rate 1e-4 \
  --use_warm_restarts \
  --restart_period 25 \
  --restart_mult 1 \
  --use_class_weights

# Adaptive + Warm Restarts (SOTA Combination)
!python ./SwinMamba/run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --loss_type adaptive_progressive_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --use_warm_restarts \
  --restart_period 30 \
  --use_class_weights

# Alternative: Memory-optimized for smaller GPU
!python run.py \
  --dataset brats2023 \
  --epochs 75 \
  --batch_size 1 \
  --img_size 96 \
  --roi_size 96 96 96 \
  --learning_rate 3e-4 \
  --warmup_epochs 8 \
  --early_stopping_patience 12 \
  --limit_val_batches 5 \
  --use_class_weights \

# Conservative: Standard settings for longer training
!python run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 2 \
  --img_size 96 \
  --roi_size 96 96 96 \
  --learning_rate 1e-4 \
  --warmup_epochs 10 \
  --early_stopping_patience 15 \
  --use_class_weights

###  GRADIENT ACCUMULATION EXAMPLES
# Simulate larger batch sizes with gradient accumulation for better stability

# Effective batch size 8 (batch_size=2 * accumulate_grad_batches=4)
!python run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 2 \
  --accumulate_grad_batches 4 \
  --learning_rate 1e-4 \
  --loss_type dicece \
  --use_class_weights

# VRAM-limited: effective batch size 8 with minimal memory usage
!python run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --accumulate_grad_batches 8 \
  --learning_rate 8e-5 \
  --loss_type generalized_dice_focal \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Smart sampling + gradient accumulation for fast stable training
!python run.py \
  --dataset brats2023 \
  --smart_sampling \
  --sample_fraction 0.3 \
  --batch_size 1 \
  --accumulate_grad_batches 4 \
  --epochs 100 \
  --learning_rate 2e-4 \
  --use_class_weights

# Competition setup: large effective batch with gradient accumulation
!python run.py \
  --dataset brats2023 \
  --epochs 150 \
  --batch_size 1 \
  --accumulate_grad_batches 16 \
  --learning_rate 5e-5 \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --structure_epochs 50 \
  --boundary_epochs 90 \
  --use_class_weights \
  --class_weights 5.0 1.0 8.0

# Omit --use_class_weights, --use_modality_attention to disable them (they are store_true flags)
"""




#### Setup the environment and prepare data ####
_local_defaults = {
    "brats2023": "dataset/BRATS2023-training",
    "brats2021": "dataset/BRATS2021-training",
}

if cli_args.data_dir is not None:
    custom_input_dir = cli_args.data_dir[0] if len(cli_args.data_dir) == 1 else cli_args.data_dir
    if len(cli_args.data_dir) == 1:
        print(f"Processing dataset from: {custom_input_dir}")
    else:
        print(f"Processing multiple dataset directories from: {cli_args.data_dir}")
    output_dir = setup_data(custom_input_dir)
    if output_dir:
        print(f"Processed dataset saved to: {output_dir}")
    else:
        print(f"Failed to process dataset from: {custom_input_dir}")
        sys.exit(1)
elif cli_args.dataset in _local_defaults and os.path.exists(_local_defaults[cli_args.dataset]):
    default_local = _local_defaults[cli_args.dataset]
    print(f"Using default local dataset path: {default_local}")
    output_dir = setup_data(default_local)
    if output_dir:
        print(f"Processed dataset saved to: {output_dir}")
    else:
        print(f"Failed to process dataset from: {default_local}")
        sys.exit(1)
else:
    print(f"No dataset found. Pass --data_dir or place data at {_local_defaults.get(cli_args.dataset, 'dataset/<dataset-dir>')}.")
    sys.exit(1)


# Experimental Configuration
args = argparse.Namespace(
    # Data parameters
    input_dir=output_dir,
    batch_size=cli_args.batch_size,
    num_workers=cli_args.num_workers,
    pin_memory=True,
    persistent_workers=False,
    dataset=cli_args.dataset,  # Use the selected dataset
    
    # Model parameters
    img_size=cli_args.img_size,
    in_channels=4,
    out_channels=3,
    feature_size=cli_args.feature_size,

    # Training parameters
    learning_rate=cli_args.learning_rate,  # Now from CLI
    weight_decay=1e-5,
    warmup_epochs=cli_args.warmup_epochs,
    epochs=cli_args.epochs,
    accelerator='gpu',
    devices="auto",
    precision=cli_args.precision,  # Now configurable
    strategy=cli_args.strategy,
    log_every_n_steps=1,
    enable_checkpointing=True,
    benchmark=True,
    profiler="simple",
    use_amp=True,  # Enable mixed precision
    gradient_clip_val=cli_args.gradient_clip_val,
    gradient_clip_algorithm=cli_args.gradient_clip_algorithm,
    use_v2=cli_args.use_v2,  # Now toggleable from CLI
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    downsample="mergingv2",
    
    # Enhanced model options
    use_class_weights=cli_args.use_class_weights,
    use_modality_attention=cli_args.use_modality_attention,
    
    # Loss and training configuration
    class_weights=tuple(cli_args.class_weights),  # TC, WT, ET
    threshold=cli_args.threshold,
    wt_threshold=cli_args.wt_threshold,
    tc_threshold=cli_args.tc_threshold,
    et_threshold=cli_args.et_threshold,
    optimizer_betas=tuple(cli_args.optimizer_betas),
    optimizer_eps=cli_args.optimizer_eps,
    
    # Validation settings
    val_interval=cli_args.val_interval,
    save_interval=cli_args.save_interval,
    early_stopping_patience=cli_args.early_stopping_patience,
    limit_val_batches=cli_args.limit_val_batches,
    
    # Inference parameters
    roi_size=cli_args.roi_size,
    sw_batch_size=cli_args.sw_batch_size,
    overlap=cli_args.overlap,
    loss_type=cli_args.loss_type,
    
    # Loss function parameters
    tversky_alpha=cli_args.tversky_alpha,
    tversky_beta=cli_args.tversky_beta,
    focal_gamma=cli_args.focal_gamma,
    focal_alpha=cli_args.focal_alpha,
    gdl_weight_type=cli_args.gdl_weight_type,
    gdl_lambda=cli_args.gdl_lambda,
    hausdorff_alpha=cli_args.hausdorff_alpha,
    lambda_dice=cli_args.lambda_dice,
    lambda_focal=cli_args.lambda_focal,
    lambda_tversky=cli_args.lambda_tversky,
    lambda_hausdorff=cli_args.lambda_hausdorff,
    
    # S3F loss parameters
    s3f_alpha=cli_args.s3f_alpha,
    s3f_beta=cli_args.s3f_beta,
    s3f_delta=cli_args.s3f_delta,
    s3f_gamma=cli_args.s3f_gamma,
    
    # Adaptive loss scheduling parameters
    use_adaptive_scheduling=cli_args.use_adaptive_scheduling,
    adaptive_schedule_type=cli_args.adaptive_schedule_type,
    structure_epochs=cli_args.structure_epochs,
    boundary_epochs=cli_args.boundary_epochs,
    schedule_start_epoch=cli_args.schedule_start_epoch,
    min_loss_weight=cli_args.min_loss_weight,
    max_loss_weight=cli_args.max_loss_weight,
    
    # Learning rate scheduling parameters
    use_warm_restarts=cli_args.use_warm_restarts,
    restart_period=cli_args.restart_period,
    restart_mult=cli_args.restart_mult,
    use_reduce_lr_on_plateau=cli_args.use_reduce_lr_on_plateau,
    plateau_patience=cli_args.plateau_patience,
    plateau_factor=cli_args.plateau_factor,
    min_lr=cli_args.min_lr,
    
    # Checkpoint resuming
    resume_from_checkpoint=cli_args.resume_from_checkpoint,
    load_pretrained=cli_args.load_pretrained,
    
    # SwinUNETRPlus enhancement parameters (disabled by default for vanilla behavior)
    use_multi_scale_attention=cli_args.use_multi_scale_attention,
    use_adaptive_window=cli_args.use_adaptive_window,
    use_cross_layer_fusion=cli_args.use_cross_layer_fusion,
    use_hierarchical_skip=cli_args.use_hierarchical_skip,
    use_enhanced_v2_blocks=cli_args.use_enhanced_v2_blocks,
    multi_scale_window_sizes=cli_args.multi_scale_window_sizes,
    patch_norm=cli_args.patch_norm,
    optimizer_type=cli_args.optimizer_type,
    
    # Data efficiency parameters
    limit_train_batches=cli_args.limit_train_batches,
    # Smart sampling parameters
    smart_sampling=cli_args.smart_sampling,
    sample_fraction=cli_args.sample_fraction,
    sampling_strategy=cli_args.sampling_strategy,
    accumulate_grad_batches=cli_args.accumulate_grad_batches,
    fast_validation=cli_args.fast_validation,
    use_mamba=cli_args.use_mamba,
    mamba_type=cli_args.mamba_type,
    # Mamba d_ parameters
    d_state=cli_args.d_state,
    d_conv=cli_args.d_conv,
    expand=cli_args.expand,
    window_size=cli_args.window_size,

    # Additional parameters
    compile_model=cli_args.compile_model,
    muon_scheduler=cli_args.muon_scheduler,
    use_cosine_scheduler=cli_args.use_cosine_scheduler,
    # Reproducibility
    seed=cli_args.seed,
)

# Print final configuration summary (only on main process to avoid duplicates in DDP)
def print_config_summary():
    """Print configuration summary only on the main process"""
    # Check if we're in a distributed environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:  # Only print on main process
        print("\n===  ADAPTIVE SWINUNETR CONFIGURATION ===")
        print(f" Batch size: {args.batch_size}")
        print(f" Gradient accumulation: {args.accumulate_grad_batches} (effective batch: {args.batch_size * args.accumulate_grad_batches})")
        print(f" Image size: {args.img_size}")
        print(f" Learning rate: {args.learning_rate}")
        print(f" Warmup epochs: {args.warmup_epochs}")
        print(f" Loss type: {args.loss_type}")
        print(f" SW batch size: {args.sw_batch_size}")
        print(f" Total epochs: {args.epochs}")
        print(f" Dataset: {args.dataset}")
        print(f" Use class weights: {args.use_class_weights}")
        print(f" Use modality attention: {args.use_modality_attention}")
        print(f" Class weights: {args.class_weights}")
        # Removed dice_ce_weight and focal_weight - using lambda parameters instead
        print(f" Tversky : {args.tversky_alpha}, : {args.tversky_beta}")
        print(f" Focal : {args.focal_gamma}, : {args.focal_alpha}")
        print(f" GDL weight type: {args.gdl_weight_type}, : {args.gdl_lambda}")
        print(f" Hausdorff : {args.hausdorff_alpha}")
        print(f" Loss weights - Dice: {args.lambda_dice}, Focal: {args.lambda_focal}, Tversky: {args.lambda_tversky}, Hausdorff: {args.lambda_hausdorff}")
        print(f" Threshold: {args.threshold}")
        print(f" ROI size: {args.roi_size}")
        print(f" Early stop patience: {args.early_stopping_patience}")
        print(f" Limit val batches: {args.limit_val_batches}")
        print(f" Val interval: {args.val_interval}")
        print(f" Adaptive scheduling: {args.use_adaptive_scheduling}")
        if args.use_adaptive_scheduling:
            print(f" Schedule type: {args.adaptive_schedule_type}")
            print(f" Structure epochs: {args.structure_epochs}")
            print(f" Boundary epochs: {args.boundary_epochs}")
            print(f" Schedule start: {args.schedule_start_epoch}")
            print(f" Weight range: {args.min_loss_weight} - {args.max_loss_weight}")
        print(f" Warm restarts: {args.use_warm_restarts}")
        if args.use_warm_restarts:
            print(f" Restart period: {args.restart_period} epochs")
            print(f" Restart multiplier: {args.restart_mult}")
        print(f" ReduceLROnPlateau: {args.use_reduce_lr_on_plateau}")
        if args.use_reduce_lr_on_plateau:
            print(f" Plateau patience: {args.plateau_patience} epochs")
            print(f" Plateau factor: {args.plateau_factor}")
            print(f" Min LR: {args.min_lr}")
        
        # Print which scheduler will be used
        if args.use_warm_restarts:
            print(f" Scheduler: Warm Restarts (CosineAnnealingWarmRestarts)")
        elif args.use_reduce_lr_on_plateau:
            print(f" Scheduler: ReduceLROnPlateau")
        else:
            print(f" Scheduler: Warmup + Cosine Annealing")
        print(f"\n===  SWINUNETR PLUS ENHANCEMENTS ===")
        print(f" Multi-scale attention: {args.use_multi_scale_attention}")
        print(f" Adaptive window: {args.use_adaptive_window}")
        print(f" Cross-layer fusion: {args.use_cross_layer_fusion}")
        print(f" Hierarchical skip: {args.use_hierarchical_skip}")
        print(f" Enhanced V2 blocks: {args.use_enhanced_v2_blocks}")
        print(f" Multi-scale window sizes: {args.multi_scale_window_sizes}")
        print(f" Use V2 blocks: {args.use_v2}")
        print(f" Patch normalization: {args.patch_norm}")
        print(f" Optimizer: {args.optimizer_type}")
        print(f" Precision: {args.precision}")
        print(f" Use Mamba: {args.use_mamba}")
        print(f" Mamba Type: {args.mamba_type}")
        print(f" Mamba d_state: {args.d_state}")
        print(f" Mamba d_conv: {args.d_conv}")
        print(f" Mamba expand: {args.expand}")
        print(f" Window size: {args.window_size}")
        print(f" Compile model: {args.compile_model}")
        print(f" Input directory: {args.input_dir}")
        print(f" Output directory: {os.getcwd()}")
        print("========================================\n")

print_config_summary()


def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        main(args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nError: CUDA Out of Memory Error!")
        else:
            print(f"Error: Runtime error: {e}")
        raise e
    except ImportError as e:
        print(f"Error: Import error: {e}")
        raise e
    except Exception as e:
        print(f"Error: Unexpected error: {e}")
        raise e

# Start optimized training
if __name__ == "__main__":
    run_with_error_handling()