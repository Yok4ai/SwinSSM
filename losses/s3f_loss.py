"""
S3F (Structure + Spatial + Focal) Loss for medical image segmentation.

Implementation using MONAI's robust loss functions for optimal performance.
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss, HausdorffDTLoss, FocalLoss


class S3FLoss(nn.Module):
    """
    S3F (Structure + Spatial + Focal) U-Net Loss using MONAI components.
    
    Combines:
    1. Focal loss with uncertainty weighting
    2. Structure loss (IoU-based) 
    3. Boundary loss (Hausdorff distance)
    
    Args:
        alpha: Weight for focal loss component (default: 1.0)
        beta: Weight for structure (IoU) loss component (default: 0.5)
        delta: Weight for boundary (Hausdorff) loss component (default: 0.3)
        gamma: Focal loss gamma parameter (default: 2.0)
        class_weights: Optional class weights for loss functions
        include_background: Whether to include background class (default: True)
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.5, 
                 delta: float = 0.3,
                 gamma: float = 2.0,
                 class_weights=None,
                 include_background: bool = True):
        super().__init__()
        
        self.alpha = alpha  # Focal weight
        self.beta = beta    # Structure (IoU) weight  
        self.delta = delta  # Boundary (Hausdorff) weight
        self.gamma = gamma  # Focal gamma parameter
        
        # Structure loss using IoU (inverse Dice/Jaccard)
        self.iou_loss = DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, jaccard=True,
            to_onehot_y=False, sigmoid=True, weight=class_weights,
            include_background=include_background, reduction='mean',
        )
        
        # Boundary loss using Hausdorff distance
        self.hausdorff_loss = HausdorffDTLoss(
            alpha=2.0, include_background=include_background,
            to_onehot_y=False, sigmoid=True
        )
        
        # Focal loss for uncertainty weighting
        self.focal_loss = FocalLoss(
            gamma=gamma, alpha=None, weight=class_weights,
            reduction='none', to_onehot_y=False, 
            include_background=include_background
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute S3F loss.
        
        Args:
            pred: Predicted logits (B, C, H, W, D)
            target: Ground truth (B, C, H, W, D)
            
        Returns:
            Combined S3F loss (scalar)
        """
        epsilon = 1e-7
        
        # Get MONAI's focal loss (per-voxel, no reduction)
        focal_per_voxel = self.focal_loss(pred, target)  # Shape: (B, C, H, W, D)
        
        # Apply sigmoid for uncertainty weighting
        p = torch.sigmoid(pred).clamp(epsilon, 1 - epsilon)
        
        # Uncertainty weight (higher for uncertain predictions near 0.5)
        w_u = 1.0 - (p - 0.5).abs()
        
        # Apply uncertainty weighting to focal loss
        weighted_focal = focal_per_voxel * w_u
        focal_loss = weighted_focal.mean()

        # Structure loss using inverse IoU (MONAI's DiceLoss with jaccard=True computes 1-IoU)
        structure_loss = self.iou_loss(pred, target)  # Already inverse: higher loss for lower IoU

        # Boundary loss using Hausdorff distance
        boundary_loss = self.hausdorff_loss(pred, target)

        # Combined S3F loss
        total_loss = (
            self.alpha * focal_loss + 
            self.beta * structure_loss + 
            self.delta * boundary_loss
        )
        
        return total_loss


def s3f_unet_loss(pred: torch.Tensor, target: torch.Tensor,
                  alpha: float = 1.0, beta: float = 0.5, delta: float = 0.3,
                  gamma: float = 2.0, class_weights=None,
                  include_background: bool = True) -> torch.Tensor:
    """
    Functional interface for S3F loss.
    
    Args:
        pred: Predicted logits (B, C, H, W, D)
        target: Ground truth (B, C, H, W, D)
        alpha: Weight for focal loss component
        beta: Weight for structure (IoU) loss component  
        delta: Weight for boundary (Hausdorff) loss component
        gamma: Focal loss gamma parameter
        class_weights: Optional class weights for loss functions
        include_background: Whether to include background class
        
    Returns:
        Combined S3F loss (scalar)
    """
    loss_fn = S3FLoss(alpha=alpha, beta=beta, delta=delta, gamma=gamma,
                      class_weights=class_weights, include_background=include_background)
    return loss_fn(pred, target)