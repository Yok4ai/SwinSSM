import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, TverskyLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete, RandFlipd, RandRotate90d, Lambda, ToTensord
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms.spatial.array import Flip, Rotate90
from monai.transforms.utils import map_classes_to_indices
import json
import argparse
from pathlib import Path
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path for imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import existing components
try:
    from src.models.swinunetrplus import SwinUNETR
    from src.data.augmentations import get_transforms
    from src.data.convert_labels import ConvertLabels
except ImportError:
    # Fallback to relative imports
    from ..models.swinunetrplus import SwinUNETR
    from ..data.augmentations import get_transforms
    from ..data.convert_labels import ConvertLabels

# Attempt to import Mamba architectures
try:
    #  Import Mamba architectures
    try:
        from src.models.segmamba import SegMamba
        from src.models.swinmamba import SwinMamba
        from src.models.mambaunetr import MambaUNETR
    except ImportError:
        from ..models.segmamba import SegMamba
        from ..models.swinmamba import SwinMamba
        from ..models.mambaunetr import MambaUNETR
    MAMBA_AVAILABLE = True
    print("Mamba architectures available for validation")
except ImportError as e:
    MAMBA_AVAILABLE = False
    SegMamba = None
    SwinMamba = None
    MambaUNETR = None
    print(f"Mamba architectures not available: {e}")
    print("Install mamba-ssm: pip install mamba-ssm")

from dataset_setup import prepare_brats_data

class BraTSVolumeThresholding:
    """
    BraTS multi-region volume thresholding: Remove small predictions in WT, TC, and ET
    to reduce false positives and improve ranking metrics.
    
    Based on validation set optimization: WT=250, TC=150, ET=100 voxels
    Removes entire regions if volume < threshold to get perfect scores.
    """
    def __init__(self, wt_threshold=250, tc_threshold=150, et_threshold=100):
        self.wt_threshold = wt_threshold
        self.tc_threshold = tc_threshold  
        self.et_threshold = et_threshold
    
    def __call__(self, pred):
        # pred shape: (C, H, W, D) where C=3 for [TC, WT, ET]
        
        # Extract channels: TC=0, WT=1, ET=2
        tc_channel = pred[0].clone()  # Tumor Core
        wt_channel = pred[1].clone()  # Whole Tumor  
        et_channel = pred[2].clone()  # Enhancing Tumor
        
        # Count voxels for each region
        tc_volume = tc_channel.sum().item()
        wt_volume = wt_channel.sum().item() 
        et_volume = et_channel.sum().item()
        
        # Remove ET if volume < threshold (most aggressive)
        if et_volume < self.et_threshold:
            et_channel = torch.zeros_like(et_channel).float()
        
        # Remove TC if volume < threshold (affects TC and implicitly ET)
        if tc_volume < self.tc_threshold:
            tc_channel = torch.zeros_like(tc_channel).float()
            # If TC is removed, ET should also be removed (ET  TC)
            et_channel = torch.zeros_like(et_channel).float()
        
        # Remove WT if volume < threshold (affects all regions)
        if wt_volume < self.wt_threshold:
            wt_channel = torch.zeros_like(wt_channel).float()
            tc_channel = torch.zeros_like(tc_channel).float() 
            et_channel = torch.zeros_like(et_channel).float()
        
        return torch.stack([tc_channel, wt_channel, et_channel], dim=0)

class ModalityAttentionModule(nn.Module):
    """
    Modality Attention Module for better feature extraction across different MRI modalities.
    Learns importance weights for each modality channel.
    """
    def __init__(self, in_channels: int = 4, reduction_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        # Channel attention components
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        # Shared MLP
        hidden_channels = max(1, in_channels // reduction_ratio)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )
        # Spatial attention components
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, d, h, w = x.size()
        # Channel attention
        avg_pool = self.global_avg_pool(x).view(batch_size, channels)
        max_pool = self.global_max_pool(x).view(batch_size, channels)
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        channel_attention = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1, 1)
        x_channel = x * channel_attention
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_concat))
        x_refined = x_channel * spatial_attention
        return x_refined + x  # Residual connection

class TTATransforms:
    """Test Time Augmentation transforms for medical image segmentation."""
    
    def __init__(self, spatial_axes: List[int] = [0, 1, 2]):
        self.spatial_axes = spatial_axes
        
    def get_tta_transforms(self) -> List[Dict[str, Any]]:
        """
        Get list of TTA transforms for medical image segmentation.
        Each transform is a dict with 'forward' and 'inverse' functions.
        """
        transforms = []
        
        # Original (no augmentation)
        transforms.append({
            'name': 'original',
            'forward': lambda x: x,
            'inverse': lambda x: x
        })
        
        # Horizontal flip (axis 0)
        transforms.append({
            'name': 'flip_0',
            'forward': lambda x: torch.flip(x, dims=[2]),  # D dimension
            'inverse': lambda x: torch.flip(x, dims=[2])
        })
        
        # Vertical flip (axis 1) 
        transforms.append({
            'name': 'flip_1',
            'forward': lambda x: torch.flip(x, dims=[3]),  # H dimension
            'inverse': lambda x: torch.flip(x, dims=[3])
        })
        
        # Depth flip (axis 2)
        transforms.append({
            'name': 'flip_2', 
            'forward': lambda x: torch.flip(x, dims=[4]),  # W dimension
            'inverse': lambda x: torch.flip(x, dims=[4])
        })
        
        # 90-degree rotation in axial plane
        transforms.append({
            'name': 'rot90_01',
            'forward': lambda x: torch.rot90(x, k=1, dims=[3, 4]),  # H, W
            'inverse': lambda x: torch.rot90(x, k=-1, dims=[3, 4])
        })
        
        # 180-degree rotation in axial plane
        transforms.append({
            'name': 'rot180_01',
            'forward': lambda x: torch.rot90(x, k=2, dims=[3, 4]),  # H, W
            'inverse': lambda x: torch.rot90(x, k=-2, dims=[3, 4])
        })
        
        # 270-degree rotation in axial plane
        transforms.append({
            'name': 'rot270_01',
            'forward': lambda x: torch.rot90(x, k=3, dims=[3, 4]),  # H, W
            'inverse': lambda x: torch.rot90(x, k=-3, dims=[3, 4])
        })
        
        # Combined transforms
        transforms.append({
            'name': 'flip_0_rot90',
            'forward': lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[3, 4]),
            'inverse': lambda x: torch.flip(torch.rot90(x, k=-1, dims=[3, 4]), dims=[2])
        })
        
        return transforms

class StandaloneValidationPipeline:
    """Standalone validation pipeline with Test Time Augmentation support."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 data_dir: str = None,
                 input_dir: str = None,
                 dataset: str = "brats2023",
                 batch_size: int = 1,
                 num_workers: int = 4,
                 roi_size: Tuple[int, int, int] = (96, 96, 96),
                 sw_batch_size: int = 1,
                 overlap: float = 0.7,
                 threshold: float = 0.5,
                 use_tta: bool = False,
                 tta_merge_mode: str = "mean",  # "mean", "median", "max"
                 device: str = "cuda",
                 class_weights: Tuple[float, float, float] = (3.0, 1.0, 5.0),
                 # Model architecture parameters
                 feature_size: int = 48,
                 use_v2: bool = False,
                 depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
                 num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
                 downsample: str = "mergingv2",
                 use_modality_attention: bool = False,
                 # SwinUNETRPlus enhancement parameters (disabled by default for backwards compatibility)
                 use_multi_scale_attention: bool = False,
                 use_adaptive_window: bool = False,
                 use_cross_layer_fusion: bool = False,
                 use_hierarchical_skip: bool = False,
                 use_enhanced_v2_blocks: bool = False,
                 multi_scale_window_sizes: Tuple[int, ...] = (7, 5, 3),
                 patch_norm: bool = False,
                 # Validation settings
                 max_batches: int = None,  # Limit number of validation batches
                 save_predictions: bool = False,
                 output_dir: str = "./validation_results",
                 log_to_wandb: bool = False,
                 wandb_project: str = "validation",
                 outlier_threshold: float = 0.3,
                 use_test_split: bool = False,
                 # Mamba architecture parameters
                 use_mamba: bool = False,                     #  Use Mamba architecture with O(N) linear complexity!
                 mamba_type: str = 'segmamba',            # Type of Mamba: 'segmamba', 'swinmamba', 'mambaunetr', 'vitmamba', 'mambaformer', 'mambaswin'
                 d_state: int = 16,                           # Mamba state dimension
                 d_conv: int = 4,                             # Mamba convolution width
                 expand: int = 2,                             # Mamba expansion factor
                 # BraTS volume thresholding parameters
                 wt_threshold: int = 250,                 # BraTS WT volume threshold for postprocessing
                 tc_threshold: int = 150,                 # BraTS TC volume threshold for postprocessing  
                 et_threshold: int = 100,                 # BraTS ET volume threshold for postprocessing
                 # Model optimization parameters
                 compile_model: bool = False,              # Enable torch.compile for faster inference
                 precision: str = "fp32",                 # Precision for inference: "32", "16", "bf16", "mixed"
                 ):
        
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.input_dir = input_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.threshold = threshold
        self.use_tta = use_tta
        self.tta_merge_mode = tta_merge_mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights
        
        # Model parameters
        self.feature_size = feature_size
        self.use_v2 = use_v2
        self.depths = depths
        self.num_heads = num_heads
        self.downsample = downsample
        self.use_modality_attention = use_modality_attention
        
        # SwinUNETRPlus enhancement parameters
        self.use_multi_scale_attention = use_multi_scale_attention
        self.use_adaptive_window = use_adaptive_window
        self.use_cross_layer_fusion = use_cross_layer_fusion
        self.use_hierarchical_skip = use_hierarchical_skip
        self.use_enhanced_v2_blocks = use_enhanced_v2_blocks
        self.multi_scale_window_sizes = multi_scale_window_sizes
        self.patch_norm = patch_norm
        
        # Validation settings
        self.max_batches = max_batches
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir)
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.outlier_threshold = outlier_threshold
        self.use_test_split = use_test_split
        
        # Mamba parameters
        self.use_mamba = use_mamba
        self.mamba_type = mamba_type
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # BraTS volume thresholding parameters
        self.wt_threshold = wt_threshold
        self.tc_threshold = tc_threshold
        self.et_threshold = et_threshold
        
        # Model optimization parameters
        self.compile_model = compile_model
        self.precision = precision
        
        # Initialize modality attention module
        if self.use_modality_attention:
            self.modality_attention = ModalityAttentionModule(in_channels=4)
        else:
            self.modality_attention = None
        
        # Initialize components
        self.model = None
        self.val_loader = None
        self.tta_transforms = TTATransforms() if use_tta else None
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.jaccard_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
        self.hausdorff_metric_95 = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
        
        # Post-processing
        self.post_trans = Compose([
            Activations(sigmoid=True),      
            AsDiscrete(threshold=self.threshold),
            BraTSVolumeThresholding(
                wt_threshold=self.wt_threshold,
                tc_threshold=self.tc_threshold, 
                et_threshold=self.et_threshold
            ),     
            ])
        
        # Results storage
        self.results = {
            'mean_dice': [],      # Volume-weighted (MONAI aggregate)
            'avg_dice': [],       # Class-weighted (arithmetic mean of TC, WT, ET)
            'dice_tc': [],
            'dice_wt': [], 
            'dice_et': [],
            'mean_iou': [],
            'hausdorff': [],
            'hausdorff_95': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def setup(self):
        """Setup model, data loader, and logging."""
        print("Setting up validation pipeline...")
        
        # Create output directory
        if self.save_predictions:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize WandB if requested
        if self.log_to_wandb:
            wandb.init(
                project=self.wandb_project,
                name=f"validation_{self.dataset}_{int(time.time())}",
                config={
                    "checkpoint_path": self.checkpoint_path,
                    "dataset": self.dataset,
                    "use_tta": self.use_tta,
                    "tta_merge_mode": self.tta_merge_mode,
                    "roi_size": self.roi_size,
                    "overlap": self.overlap,
                    "threshold": self.threshold
                }
            )
        
        # Setup model
        self._setup_model()
        
        # Setup data loader
        self._setup_dataloader()
        
        print(f"Model loaded from: {self.checkpoint_path}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Total batches: {len(self.val_loader)}")
        if self.max_batches:
            print(f"Limited to: {self.max_batches} batches")
        print(f"Using TTA: {self.use_tta}")
        if self.use_tta:
            print(f"TTA merge mode: {self.tta_merge_mode}")
            print(f"TTA transforms: {len(self.tta_transforms.get_tta_transforms())}")
        
    def _setup_model(self):
        """Initialize and load the SwinUNETR or Mamba model."""
        
        #  Choose between SwinMamba Hybrid, Pure SegMamba, or SwinUNETRPlus
        if self.use_mamba:
            if not MAMBA_AVAILABLE:
                print("Mamba not available. Install with: pip install mamba-ssm")
                print("Falling back to SwinUNETRPlus.")
                ModelClass = SwinUNETR
            else:
                if self.mamba_type == 'swinmamba':
                    ModelClass = SwinMamba
                    print("Using SwinMamba for validation")
                elif self.mamba_type == 'mambaunetr':
                    ModelClass = MambaUNETR
                    print("Using MambaUNETR for validation")
                else:  # segmamba
                    ModelClass = SegMamba
                    print("Using SegMamba for validation")
        else:
            ModelClass = SwinUNETR
            
        # Model creation based on choice
        if self.use_mamba and MAMBA_AVAILABLE:
            if self.mamba_type == 'swinmamba':
                # SwinMamba: SwinUNETR encoder + multi-directional Mamba
                self.model = ModelClass(
                    in_chans=4,
                    out_chans=3,
                    depths=[2, 2, 2, 2],
                    feat_size=[self.feature_size, self.feature_size*2,
                              self.feature_size*4, self.feature_size*8],
                    drop_path_rate=0.1,
                    layer_scale_init_value=1e-6,
                    spatial_dims=3,
                    norm_name="instance",
                    window_size=[4, 4, 4],
                    mlp_ratio=4.0,
                    num_heads=[3, 6, 12, 24],
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                )
            elif self.mamba_type == 'mambaunetr':
                # MambaUNETR: Mamba encoder + SwinUNETR decoder
                self.model = ModelClass(
                    in_chans=4,
                    out_chans=3,
                    feat_size=[self.feature_size, self.feature_size*2,
                              self.feature_size*4, self.feature_size*8],
                    depths=list(self.depths),
                    spatial_dims=3,
                    norm_name="instance",
                )
            else:  # segmamba
                # SegMamba model
                self.model = ModelClass(
                    in_chans=4,
                    out_chans=3,
                    feat_size=[self.feature_size, self.feature_size*2,
                              self.feature_size*4, self.feature_size*8],
                    depths=list(self.depths),
                    spatial_dims=3,
                    norm_name="instance",
                )
        else:
            # SwinUNETRPlus with all enhancements
            self.model = ModelClass(
                in_channels=4,
                out_channels=3,
                feature_size=self.feature_size,
                use_checkpoint=True,
                use_v2=self.use_v2,
                spatial_dims=3,
                depths=self.depths,
                num_heads=self.num_heads,
                norm_name="instance",
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                downsample=self.downsample,
                # SwinUNETRPlus enhancement parameters
                use_multi_scale_attention=self.use_multi_scale_attention,
                use_adaptive_window=self.use_adaptive_window,
                use_cross_layer_fusion=self.use_cross_layer_fusion,
                use_hierarchical_skip=self.use_hierarchical_skip,
                use_enhanced_v2_blocks=self.use_enhanced_v2_blocks,
                multi_scale_window_sizes=self.multi_scale_window_sizes,
                patch_norm=self.patch_norm,
            )
        
        # Apply torch.compile if enabled
        if self.compile_model:
            self.model = torch.compile(self.model)
            print(" Model compiled with torch.compile for faster inference!")
        
        # Load checkpoint
        if self.checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint - extract only the SwinUNETR model weights
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    # Remove the 'model.' prefix to match direct SwinUNETR
                    new_key = key[6:]
                    model_state_dict[new_key] = value
            self.model.load_state_dict(model_state_dict, strict=False)
        else:
            # PyTorch state dict
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Move modality attention to device if it exists
        if self.modality_attention is not None:
            self.modality_attention.to(self.device)
            self.modality_attention.eval()
        
    def _setup_dataloader(self):
        """Setup validation data loader."""
        # Get transforms
        _, val_transforms = get_transforms(img_size=self.roi_size[0], dataset=self.dataset)
        
        # Handle data directory - create dataset.json if needed
        if self.input_dir and not self.data_dir:
            # Create dataset.json from input directory
            print(f"Creating dataset.json from input directory: {self.input_dir}")
            # Use current working directory or temporary directory instead of hardcoded kaggle path
            output_dir = os.path.join(os.getcwd(), "json")
            os.makedirs(output_dir, exist_ok=True)
            prepare_brats_data(self.input_dir, output_dir)
            dataset_path = os.path.join(output_dir, "dataset.json")
            print(f"Created dataset.json at: {dataset_path}")
        else:
            # Use existing data_dir
            dataset_path = self.data_dir if self.data_dir.endswith('dataset.json') else os.path.join(self.data_dir, "dataset.json")
        
        # Load dataset
        with open(dataset_path) as f:
            datalist = json.load(f)["training"]
        
        # Use same split logic as dataloader: 60/20/20 train/val/test
        # First split: 80% train+val, 20% test
        train_val_files, test_files = train_test_split(datalist, test_size=0.2, random_state=42)
        
        if self.use_test_split:
            # Use test split (20% of total data)
            val_files = test_files
            print(f"Using test split for evaluation: {len(val_files)} samples")
        else:
            # Use validation split (20% of total data, different from test)
            _, val_files = train_test_split(train_val_files, test_size=0.25, random_state=43)
            print(f"Using validation split for evaluation: {len(val_files)} samples")
        
        # Create validation dataset and loader
        val_ds = Dataset(data=val_files, transform=val_transforms)
        self.val_loader = DataLoader(
            val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def forward(self, x):
        """Forward pass with optional modality attention and precision support."""
        if self.precision == "bf16":
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                return self._forward_impl(x)
        elif self.precision == "fp16":
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                return self._forward_impl(x)
        elif self.precision == "fp32":
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        """Implementation of forward pass."""
        if self.modality_attention is not None:
            x = self.modality_attention(x)
        return self.model(x)
        
    def predict_with_tta(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform prediction with Test Time Augmentation.
        
        Args:
            inputs: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Ensemble prediction tensor
        """
        if not self.use_tta:
            return sliding_window_inference(
                inputs, 
                roi_size=self.roi_size, 
                sw_batch_size=self.sw_batch_size,
                predictor=self.forward, 
                overlap=self.overlap
            )
        
        tta_transforms = self.tta_transforms.get_tta_transforms()
        predictions = []
        
        for transform_dict in tta_transforms:
            # Apply forward transform
            augmented_inputs = transform_dict['forward'](inputs)
            
            # Predict
            with torch.no_grad():
                pred = sliding_window_inference(
                    augmented_inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.forward,
                    overlap=self.overlap
                )
            
            # Apply inverse transform to prediction
            pred = transform_dict['inverse'](pred)
            predictions.append(pred)
        
        # Ensemble predictions
        predictions = torch.stack(predictions, dim=0)  # (n_transforms, B, C, D, H, W)
        
        if self.tta_merge_mode == "mean":
            ensemble_pred = torch.mean(predictions, dim=0)
        elif self.tta_merge_mode == "median":
            ensemble_pred = torch.median(predictions, dim=0)[0]
        elif self.tta_merge_mode == "max":
            ensemble_pred = torch.max(predictions, dim=0)[0]
        else:
            raise ValueError(f"Unknown TTA merge mode: {self.tta_merge_mode}")
            
        return ensemble_pred
        
    def compute_metrics(self, outputs: List[torch.Tensor], labels: List[torch.Tensor]) -> Dict[str, float]:
        """Compute comprehensive metrics for the batch."""
        # Stack outputs and labels if they're lists
        outputs = torch.stack(outputs) if isinstance(outputs, list) else outputs
        labels = torch.stack(labels) if isinstance(labels, list) else labels
        
        outputs = outputs.float()
        labels = labels.float()
        
        # Flatten all but batch dimension for precision/recall/F1
        outputs_flat = outputs.view(outputs.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        
        # True Positives, False Positives, False Negatives
        tp = (outputs_flat * labels_flat).sum(dim=1)
        fp = (outputs_flat * (1 - labels_flat)).sum(dim=1)
        fn = ((1 - outputs_flat) * labels_flat).sum(dim=1)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item()
        }
        
    def validate(self) -> Dict[str, float]:
        """Run validation with TTA and return comprehensive metrics."""
        print("\nStarting validation...")
        
        self.model.eval()
        
        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.jaccard_metric.reset()
        self.hausdorff_metric.reset()
        self.hausdorff_metric_95.reset()
        
        total_time = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Break if we've reached the maximum number of batches
                if self.max_batches and batch_idx >= self.max_batches:
                    print(f"\nReached maximum batch limit ({self.max_batches}), stopping validation.")
                    break
                    
                start_time = time.time()
                
                val_inputs = batch["image"].to(self.device)
                val_labels = batch["label"].to(self.device)
                
                # Predict with or without TTA
                val_outputs = self.predict_with_tta(val_inputs)
                
                # Post-process outputs
                val_outputs_processed = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels_processed = decollate_batch(val_labels)
                
                # Compute metrics
                self.dice_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                self.dice_metric_batch(y_pred=val_outputs_processed, y=val_labels_processed)
                self.jaccard_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                self.hausdorff_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                self.hausdorff_metric_95(y_pred=val_outputs_processed, y=val_labels_processed)
                
                # Compute additional metrics
                additional_metrics = self.compute_metrics(val_outputs_processed, val_labels_processed)
                
                # Store batch results
                dice_batch = self.dice_metric_batch.aggregate()
                mean_iou = self.jaccard_metric.aggregate().item()
                
                # Calculate both dice metrics
                dice_tc = dice_batch[0].item()
                dice_wt = dice_batch[1].item()
                dice_et = dice_batch[2].item()
                
                # Volume-weighted dice (MONAI aggregate method)
                mean_dice = self.dice_metric.aggregate().item()
                
                # Class-weighted dice (arithmetic mean of TC, WT, ET)
                avg_dice = (dice_tc + dice_wt + dice_et) / 3.0
                
                # Hausdorff distance (standard) - overall and per-class
                hausdorff_values = self.hausdorff_metric.aggregate(reduction='none')
                if not isinstance(hausdorff_values, torch.Tensor):
                    hausdorff_values = torch.tensor(hausdorff_values)
                
                
                # Overall mean
                valid = torch.isfinite(hausdorff_values)
                hausdorff = hausdorff_values[valid].mean().item() if valid.any() else float('nan')
                
                # Hausdorff distance 95th percentile - overall and per-class
                hausdorff_95_values = self.hausdorff_metric_95.aggregate(reduction='none')
                if not isinstance(hausdorff_95_values, torch.Tensor):
                    hausdorff_95_values = torch.tensor(hausdorff_95_values)
                
                
                # Overall mean
                valid_95 = torch.isfinite(hausdorff_95_values)
                hausdorff_95 = hausdorff_95_values[valid_95].mean().item() if valid_95.any() else float('nan')
                
                # Store results
                self.results['mean_dice'].append(mean_dice)
                self.results['avg_dice'].append(avg_dice)
                self.results['dice_tc'].append(dice_tc)
                self.results['dice_wt'].append(dice_wt)
                self.results['dice_et'].append(dice_et)
                self.results['mean_iou'].append(mean_iou)
                self.results['hausdorff'].append(hausdorff)
                self.results['hausdorff_95'].append(hausdorff_95)
                self.results['precision'].append(additional_metrics['precision'])
                self.results['recall'].append(additional_metrics['recall'])
                self.results['f1'].append(additional_metrics['f1'])
                
                
                # Log sample visualizations to wandb
                if self.log_to_wandb and batch_idx % 1 == 0:
                    self._log_sample_visualizations(val_inputs, val_outputs_processed, val_labels_processed, batch_idx)
                
                # Save predictions if requested
                if self.save_predictions:
                    for i, (pred, label) in enumerate(zip(val_outputs_processed, val_labels_processed)):
                        pred_path = self.output_dir / f"pred_batch{batch_idx}_sample{i}.npy"
                        label_path = self.output_dir / f"label_batch{batch_idx}_sample{i}.npy"
                        np.save(pred_path, pred.cpu().numpy())
                        np.save(label_path, label.cpu().numpy())
                
                # Reset metrics for next batch
                self.dice_metric.reset()
                self.dice_metric_batch.reset()
                self.jaccard_metric.reset()
                self.hausdorff_metric.reset()
                self.hausdorff_metric_95.reset()
                
                # Track timing
                batch_time = time.time() - start_time
                total_time += batch_time
                num_samples += val_inputs.size(0)
                
                # Log progress for every batch
                total_batches = self.max_batches if self.max_batches else len(self.val_loader)
                print(f"Batch {batch_idx}/{total_batches}: "
                      f"Mean={mean_dice:.4f}, Avg={avg_dice:.4f}, IoU={mean_iou:.4f}, "
                      f"Time={batch_time:.2f}s")
        
        # Compute final statistics
        final_results = {}
        
        # Count outliers based on mean_dice threshold
        mean_dice_values = [v for v in self.results['mean_dice'] if not np.isnan(v)]
        outlier_indices = [i for i, v in enumerate(mean_dice_values) if v < self.outlier_threshold]
        num_outliers = len(outlier_indices)
        num_total_cases = len(mean_dice_values)
        outlier_percentage = (num_outliers / num_total_cases * 100) if num_total_cases > 0 else 0
        
        final_results['total_cases'] = num_total_cases
        final_results['outlier_cases'] = num_outliers
        final_results['outlier_percentage'] = outlier_percentage
        final_results['outlier_threshold'] = self.outlier_threshold
        
        for metric_name, values in self.results.items():
            if values:
                # Filter out NaN values for statistics
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    # All cases statistics
                    final_results[f"{metric_name}_mean_all"] = np.mean(valid_values)
                    final_results[f"{metric_name}_std_all"] = np.std(valid_values)
                    final_results[f"{metric_name}_median_all"] = np.median(valid_values)
                    
                    # Outlier-filtered statistics (only for mean_dice-based filtering)
                    if metric_name == 'mean_dice':
                        # Filter based on dice threshold
                        filtered_values = [v for v in valid_values if v >= self.outlier_threshold]
                    else:
                        # For other metrics, filter using the same indices as mean_dice outliers
                        filtered_values = [valid_values[i] for i in range(len(valid_values)) 
                                         if i not in outlier_indices]
                    
                    if filtered_values:
                        final_results[f"{metric_name}_mean_filtered"] = np.mean(filtered_values)
                        final_results[f"{metric_name}_std_filtered"] = np.std(filtered_values)
                        final_results[f"{metric_name}_median_filtered"] = np.median(filtered_values)
                    else:
                        final_results[f"{metric_name}_mean_filtered"] = float('nan')
                        final_results[f"{metric_name}_std_filtered"] = float('nan')
                        final_results[f"{metric_name}_median_filtered"] = float('nan')
                else:
                    # No valid values
                    final_results[f"{metric_name}_mean_all"] = float('nan')
                    final_results[f"{metric_name}_std_all"] = float('nan')
                    final_results[f"{metric_name}_median_all"] = float('nan')
                    final_results[f"{metric_name}_mean_filtered"] = float('nan')
                    final_results[f"{metric_name}_std_filtered"] = float('nan')
                    final_results[f"{metric_name}_median_filtered"] = float('nan')
        
        
        # Add timing information
        final_results['avg_time_per_sample'] = total_time / num_samples if num_samples > 0 else 0
        final_results['total_validation_time'] = total_time
        final_results['num_samples'] = num_samples
        
        return final_results
    
    
    def _log_sample_visualizations(self, val_inputs: torch.Tensor, val_outputs_processed: List[torch.Tensor], 
                                   val_labels_processed: List[torch.Tensor], batch_idx: int):
        """Log sample visualizations to wandb with proper batch/sample grouping."""
        try:
            # Process first sample in batch only for consistent comparison
            sample_idx = 0
            pred = val_outputs_processed[sample_idx].detach().cpu().numpy()  # All prediction channels
            label = val_labels_processed[sample_idx].detach().cpu().numpy()  # All label channels
            
            # Pick a middle slice for 3D volumes
            slice_idx = pred.shape[-1] // 2
            
            # Create a single log entry for this batch with all comparisons
            log_dict = {
                "validation_step": batch_idx,
                "batch_idx": batch_idx,
                "sample_idx": sample_idx
            }
            
            # Channel mapping: TC=0, WT=1, ET=2 -> Display order: WT, TC, ET
            channel_names = ["WT", "TC", "ET"]
            channel_indices = [1, 0, 2]  # WT=1, TC=0, ET=2
            
            # Create one large visualization with all 3 tumor regions
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create 2x3 subplot: 2 rows (pred/label), 3 columns (WT/TC/ET)
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
            
            for col_idx, (ch_name, ch_idx) in enumerate(zip(channel_names, channel_indices)):
                if ch_idx < pred.shape[0]:  # Ensure channel exists
                    # Get statistics for this channel
                    pred_max = pred[ch_idx].max().item()
                    label_max = label[ch_idx].max().item()
                    pred_sum = pred[ch_idx].sum().item()
                    label_sum = label[ch_idx].sum().item()
                    
                    # Get the actual image data
                    pred_img = pred[ch_idx, ..., slice_idx]
                    label_img = label[ch_idx, ..., slice_idx]
                    
                    # Prediction (top row)
                    axes[0, col_idx].imshow(pred_img, cmap='hot', vmin=0, vmax=1)
                    axes[0, col_idx].set_title(f'Batch {batch_idx} - Prediction {ch_name}\nmax:{pred_max:.3f}, sum:{pred_sum:.1f}', fontsize=12)
                    axes[0, col_idx].axis('off')
                    
                    # Ground truth (bottom row)
                    axes[1, col_idx].imshow(label_img, cmap='hot', vmin=0, vmax=1)
                    axes[1, col_idx].set_title(f'Batch {batch_idx} - Ground Truth {ch_name}\nmax:{label_max:.3f}, sum:{label_sum:.1f}', fontsize=12)
                    axes[1, col_idx].axis('off')
            
            plt.tight_layout(pad=2.0)  # More padding for readability with larger image
            
            # Log the combined comparison with higher DPI for better WandB preview
            log_dict[f"all_regions_comparison"] = wandb.Image(plt, caption=f"Batch {batch_idx} - All Regions Comparison")
            plt.close()  # Clean up
            
            # Log all comparisons for this batch at once
            wandb.log(log_dict, step=batch_idx)
            
        except Exception as e:
            print(f"Warning: Failed to log visualizations for batch {batch_idx}: {e}")
        
    def print_results(self, results: Dict[str, float]):
        """Print validation results."""
        print("\n" + "="*100)
        print("COMPREHENSIVE VALIDATION RESULTS")
        print("="*100)
        
        # Configuration
        print(f"Dataset: {self.dataset}")
        print(f"Checkpoint: {os.path.basename(self.checkpoint_path)}")
        print(f"TTA enabled: {self.use_tta}")
        if self.use_tta:
            print(f"TTA merge mode: {self.tta_merge_mode}")
        print(f"ROI size: {self.roi_size}")
        print(f"Overlap: {self.overlap}")
        print(f"Threshold: {self.threshold}")
        
        # Sample information and outlier analysis
        total_cases = results.get('total_cases', 0)
        outlier_cases = results.get('outlier_cases', 0)
        outlier_pct = results.get('outlier_percentage', 0)
        outlier_thresh = results.get('outlier_threshold', 0.3)
        
        print(f"\nSAMPLE ANALYSIS:")
        print("-" * 50)
        print(f"Total cases: {total_cases}")
        print(f"Outlier threshold (Dice): {outlier_thresh:.3f}")
        print(f"Outlier cases: {outlier_cases} ({outlier_pct:.1f}%)")
        print(f"Valid cases after filtering: {total_cases - outlier_cases}")
        
        # Main performance metrics - ALL CASES
        print(f"\nPERFORMANCE METRICS - ALL CASES (n={total_cases}):")
        print("-" * 50)
        
        dice_mean_all = results.get('mean_dice_mean_all', float('nan'))
        dice_std_all = results.get('mean_dice_std_all', float('nan'))
        dice_median_all = results.get('mean_dice_median_all', float('nan'))
        print(f"Mean Dice (volume): {dice_mean_all:.4f}  {dice_std_all:.4f} (median: {dice_median_all:.4f})")
        
        avg_dice_mean_all = results.get('avg_dice_mean_all', float('nan'))
        avg_dice_std_all = results.get('avg_dice_std_all', float('nan'))
        avg_dice_median_all = results.get('avg_dice_median_all', float('nan'))
        print(f"Avg Dice (class):  {avg_dice_mean_all:.4f}  {avg_dice_std_all:.4f} (median: {avg_dice_median_all:.4f})")
        
        tc_mean_all = results.get('dice_tc_mean_all', float('nan'))
        tc_std_all = results.get('dice_tc_std_all', float('nan'))
        tc_median_all = results.get('dice_tc_median_all', float('nan'))
        print(f"TC Dice Score:     {tc_mean_all:.4f}  {tc_std_all:.4f} (median: {tc_median_all:.4f})")
        
        wt_mean_all = results.get('dice_wt_mean_all', float('nan'))
        wt_std_all = results.get('dice_wt_std_all', float('nan'))
        wt_median_all = results.get('dice_wt_median_all', float('nan'))
        print(f"WT Dice Score:     {wt_mean_all:.4f}  {wt_std_all:.4f} (median: {wt_median_all:.4f})")
        
        et_mean_all = results.get('dice_et_mean_all', float('nan'))
        et_std_all = results.get('dice_et_std_all', float('nan'))
        et_median_all = results.get('dice_et_median_all', float('nan'))
        print(f"ET Dice Score:     {et_mean_all:.4f}  {et_std_all:.4f} (median: {et_median_all:.4f})")
        
        iou_mean_all = results.get('mean_iou_mean_all', float('nan'))
        iou_std_all = results.get('mean_iou_std_all', float('nan'))
        iou_median_all = results.get('mean_iou_median_all', float('nan'))
        print(f"Mean IoU:          {iou_mean_all:.4f}  {iou_std_all:.4f} (median: {iou_median_all:.4f})")
        
        # Hausdorff metrics - overall and per-class
        hd_mean_all = results.get('hausdorff_mean_all', float('nan'))
        hd_std_all = results.get('hausdorff_std_all', float('nan'))
        hd_median_all = results.get('hausdorff_median_all', float('nan'))
        print(f"Hausdorff Dist.:   {hd_mean_all:.4f}  {hd_std_all:.4f} (median: {hd_median_all:.4f})")
        
        
        hd95_mean_all = results.get('hausdorff_95_mean_all', float('nan'))
        hd95_std_all = results.get('hausdorff_95_std_all', float('nan'))
        hd95_median_all = results.get('hausdorff_95_median_all', float('nan'))
        print(f"Hausdorff 95:      {hd95_mean_all:.4f}  {hd95_std_all:.4f} (median: {hd95_median_all:.4f})")
        
        
        # Additional metrics
        precision_mean_all = results.get('precision_mean_all', float('nan'))
        precision_std_all = results.get('precision_std_all', float('nan'))
        print(f"Precision:         {precision_mean_all:.4f}  {precision_std_all:.4f}")
        
        recall_mean_all = results.get('recall_mean_all', float('nan'))
        recall_std_all = results.get('recall_std_all', float('nan'))
        print(f"Recall:            {recall_mean_all:.4f}  {recall_std_all:.4f}")
        
        f1_mean_all = results.get('f1_mean_all', float('nan'))
        f1_std_all = results.get('f1_std_all', float('nan'))
        print(f"F1 Score:          {f1_mean_all:.4f}  {f1_std_all:.4f}")
        
        # Main performance metrics - FILTERED CASES (outliers removed)
        filtered_cases = total_cases - outlier_cases
        if filtered_cases > 0:
            print(f"\nPERFORMANCE METRICS - OUTLIERS REMOVED (n={filtered_cases}):")
            print("-" * 50)
            
            dice_mean_filt = results.get('mean_dice_mean_filtered', float('nan'))
            dice_std_filt = results.get('mean_dice_std_filtered', float('nan'))
            dice_median_filt = results.get('mean_dice_median_filtered', float('nan'))
            print(f"Mean Dice (volume): {dice_mean_filt:.4f}  {dice_std_filt:.4f} (median: {dice_median_filt:.4f})")
            
            avg_dice_mean_filt = results.get('avg_dice_mean_filtered', float('nan'))
            avg_dice_std_filt = results.get('avg_dice_std_filtered', float('nan'))
            avg_dice_median_filt = results.get('avg_dice_median_filtered', float('nan'))
            print(f"Avg Dice (class):  {avg_dice_mean_filt:.4f}  {avg_dice_std_filt:.4f} (median: {avg_dice_median_filt:.4f})")
            
            tc_mean_filt = results.get('dice_tc_mean_filtered', float('nan'))
            tc_std_filt = results.get('dice_tc_std_filtered', float('nan'))
            tc_median_filt = results.get('dice_tc_median_filtered', float('nan'))
            print(f"TC Dice Score:     {tc_mean_filt:.4f}  {tc_std_filt:.4f} (median: {tc_median_filt:.4f})")
            
            wt_mean_filt = results.get('dice_wt_mean_filtered', float('nan'))
            wt_std_filt = results.get('dice_wt_std_filtered', float('nan'))
            wt_median_filt = results.get('dice_wt_median_filtered', float('nan'))
            print(f"WT Dice Score:     {wt_mean_filt:.4f}  {wt_std_filt:.4f} (median: {wt_median_filt:.4f})")
            
            et_mean_filt = results.get('dice_et_mean_filtered', float('nan'))
            et_std_filt = results.get('dice_et_std_filtered', float('nan'))
            et_median_filt = results.get('dice_et_median_filtered', float('nan'))
            print(f"ET Dice Score:     {et_mean_filt:.4f}  {et_std_filt:.4f} (median: {et_median_filt:.4f})")
            
            iou_mean_filt = results.get('mean_iou_mean_filtered', float('nan'))
            iou_std_filt = results.get('mean_iou_std_filtered', float('nan'))
            iou_median_filt = results.get('mean_iou_median_filtered', float('nan'))
            print(f"Mean IoU:          {iou_mean_filt:.4f}  {iou_std_filt:.4f} (median: {iou_median_filt:.4f})")
            
            # Hausdorff metrics filtered - overall and per-class
            hd_mean_filt = results.get('hausdorff_mean_filtered', float('nan'))
            hd_std_filt = results.get('hausdorff_std_filtered', float('nan'))
            hd_median_filt = results.get('hausdorff_median_filtered', float('nan'))
            print(f"Hausdorff Dist.:   {hd_mean_filt:.4f}  {hd_std_filt:.4f} (median: {hd_median_filt:.4f})")
            
            
            hd95_mean_filt = results.get('hausdorff_95_mean_filtered', float('nan'))
            hd95_std_filt = results.get('hausdorff_95_std_filtered', float('nan'))
            hd95_median_filt = results.get('hausdorff_95_median_filtered', float('nan'))
            print(f"Hausdorff 95:      {hd95_mean_filt:.4f}  {hd95_std_filt:.4f} (median: {hd95_median_filt:.4f})")
            
            
            # Additional metrics filtered
            precision_mean_filt = results.get('precision_mean_filtered', float('nan'))
            precision_std_filt = results.get('precision_std_filtered', float('nan'))
            print(f"Precision:         {precision_mean_filt:.4f}  {precision_std_filt:.4f}")
            
            recall_mean_filt = results.get('recall_mean_filtered', float('nan'))
            recall_std_filt = results.get('recall_std_filtered', float('nan'))
            print(f"Recall:            {recall_mean_filt:.4f}  {recall_std_filt:.4f}")
            
            f1_mean_filt = results.get('f1_mean_filtered', float('nan'))
            f1_std_filt = results.get('f1_std_filtered', float('nan'))
            print(f"F1 Score:          {f1_mean_filt:.4f}  {f1_std_filt:.4f}")
        
        # Timing
        avg_time = results.get('avg_time_per_sample', 0)
        total_time = results.get('total_validation_time', 0)
        print(f"\nTIMING:")
        print("-" * 50)
        print(f"Avg time per sample: {avg_time:.2f}s")
        print(f"Total validation time: {total_time:.2f}s")
        
        
        
        # Validation summary
        print(f"\nVALIDATION SUMMARY:")
        print("-" * 50)
        print(f" Dataset: {self.dataset}")
        print(f" Total cases: {total_cases}")
        print(f" Outlier exclusion: {outlier_cases} cases with Dice < {outlier_thresh:.3f} ({outlier_pct:.1f}%)")
        print(f" Mean Dice (volume-weighted): {dice_mean_all:.4f}  {dice_std_all:.4f}")
        print(f" Avg Dice (class-weighted): {avg_dice_mean_all:.4f}  {avg_dice_std_all:.4f}")
        if filtered_cases > 0:
            print(f" Mean Dice (filtered): {dice_mean_filt:.4f}  {dice_std_filt:.4f}")
            print(f" Avg Dice (filtered): {avg_dice_mean_filt:.4f}  {avg_dice_std_filt:.4f}")
        print(f" Median Dice: {dice_median_all:.4f}")
        print(f" TTA: {'Enabled' if self.use_tta else 'Disabled'}")
        
        print("="*100)
        
        # Log to WandB
        if self.log_to_wandb:
            wandb.log(results)
            
    def save_results(self, results: Dict[str, float], filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"validation_results_{self.dataset}_{timestamp}.json"
            
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_path = self.output_dir / filename
        
        # Include configuration in results
        config = {
            'checkpoint_path': self.checkpoint_path,
            'dataset': self.dataset,
            'use_tta': self.use_tta,
            'tta_merge_mode': self.tta_merge_mode,
            'roi_size': self.roi_size,
            'overlap': self.overlap,
            'threshold': self.threshold,
            'batch_size': self.batch_size
        }
        
        full_results = {
            'config': config,
            'results': results,
            'raw_data': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
            
        print(f"\nResults saved to: {results_path}")
        
    def run(self):
        """Run the complete validation pipeline."""
        self.setup()
        results = self.validate()
        self.print_results(results)
        
        # Always save results summary (JSON), optionally save prediction arrays
        self.save_results(results)
            
        if self.log_to_wandb:
            wandb.finish()
            
        return results

def main():
    """Main function for standalone validation."""
    parser = argparse.ArgumentParser(description="Standalone SwinUNETR Validation with TTA")
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint (.ckpt or .pth)')
    
    # Data source arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data_dir', type=str,
                           help='Path to data directory containing dataset.json')
    data_group.add_argument('--input_dir', type=str,
                           help='Path to input directory with BraTS data (will create dataset.json in ./json)')
    
    # Dataset and basic settings
    parser.add_argument('--dataset', type=str, default='brats2023', 
                        choices=['brats2021', 'brats2023'],
                        help='Dataset format (default: brats2023)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for validation (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=48,
                        help='Model feature size (default: 48)')
    parser.add_argument('--use_v2', action='store_true', default=False,
                        help='Use SwinUNETR V2 (default: False)')
    parser.add_argument('--downsample', type=str, default='mergingv2',
                        help='Downsample method (default: mergingv2)')
    parser.add_argument('--use_modality_attention', action='store_true',
                        help='Use modality attention module (default: False)')
    
    # SwinUNETRPlus enhancement parameters (disabled by default for backwards compatibility)
    parser.add_argument('--use_multi_scale_attention', action='store_true',
                        help='Enable multi-scale window attention (default: False)')
    parser.add_argument('--use_adaptive_window', action='store_true',
                        help='Enable adaptive window sizing (default: False)')
    parser.add_argument('--use_cross_layer_fusion', action='store_true',
                        help='Enable cross-layer attention fusion (default: False)')
    parser.add_argument('--use_hierarchical_skip', action='store_true',
                        help='Enable hierarchical skip connections (default: False)')
    parser.add_argument('--use_enhanced_v2_blocks', action='store_true',
                        help='Enable enhanced V2 residual blocks (default: False)')
    parser.add_argument('--multi_scale_window_sizes', type=int, nargs='+', default=[7, 5, 3],
                        help='Window sizes for multi-scale attention (default: 7 5 3)')
    parser.add_argument('--patch_norm', action='store_true',
                        help='Enable patch normalization (default: False)')
    
    # Inference parameters
    parser.add_argument('--roi_size', type=int, nargs=3, default=[96, 96, 96],
                        help='ROI size for sliding window inference (default: 96 96 96)')
    parser.add_argument('--sw_batch_size', type=int, default=1,
                        help='Sliding window batch size (default: 1)')
    parser.add_argument('--overlap', type=float, default=0.7,
                        help='Sliding window overlap (default: 0.7)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for post-processing (default: 0.5)')
    
    # TTA parameters
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Enable Test Time Augmentation (default: False)')
    parser.add_argument('--tta_merge_mode', type=str, default='mean',
                        choices=['mean', 'median', 'max'],
                        help='TTA ensemble method (default: mean)')
    
    # Output and logging
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to validate (default: None - all batches)')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Output directory for results (default: ./validation_results)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction arrays (default: False)')
    parser.add_argument('--log_to_wandb', action='store_true',
                        help='Log results to Weights & Biases (default: False)')
    parser.add_argument('--wandb_project', type=str, default='validation',
                        help='WandB project name (default: validation)')
    parser.add_argument('--outlier_threshold', type=float, default=0.3,
                        help='Dice threshold for outlier detection (default: 0.3)')
    parser.add_argument('--use_test_split', action='store_true',
                        help='Use test split instead of validation split (default: False)')
    
    # Mamba architecture parameters
    parser.add_argument('--use_mamba', action='store_true',
                        help=' Use Mamba architecture with O(N) linear complexity - MASSIVE VRAM savings! (default: False)')
    parser.add_argument('--mamba_type', type=str, default='segmamba',
                        choices=['segmamba', 'swinmamba', 'mambaunetr', 'vitmamba', 'mambaformer', 'mambaswin'],
                        help='Type of Mamba architecture (default: segmamba)')
    parser.add_argument('--d_state', type=int, default=16,
                        help='Mamba state dimension (default: 16)')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Mamba convolution width (default: 4)')  
    parser.add_argument('--expand', type=int, default=2,
                        help='Mamba expansion factor (default: 2)')
    
    # BraTS volume thresholding parameters
    parser.add_argument('--wt_threshold', type=int, default=250, 
                        help='BraTS WT volume threshold for postprocessing (default: 250)')
    parser.add_argument('--tc_threshold', type=int, default=150, 
                        help='BraTS TC volume threshold for postprocessing (default: 150)')
    parser.add_argument('--et_threshold', type=int, default=100, 
                        help='BraTS ET volume threshold for postprocessing (default: 100)')
    
    # Model optimization parameters
    parser.add_argument('--compile_model', action='store_true',
                        help='Enable torch.compile for faster inference (default: False)')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['bf16', 'fp16', 'fp32'],
                        help='Precision mode: mixed (auto), bf16, fp16, fp32 (default: mixed)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Create validation pipeline
    pipeline = StandaloneValidationPipeline(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        use_tta=args.use_tta,
        tta_merge_mode=args.tta_merge_mode,
        device=args.device,
        feature_size=args.feature_size,
        use_v2=args.use_v2,
        downsample=args.downsample,
        use_modality_attention=args.use_modality_attention,
        # SwinUNETRPlus enhancement parameters
        use_multi_scale_attention=args.use_multi_scale_attention,
        use_adaptive_window=args.use_adaptive_window,
        use_cross_layer_fusion=args.use_cross_layer_fusion,
        use_hierarchical_skip=args.use_hierarchical_skip,
        use_enhanced_v2_blocks=args.use_enhanced_v2_blocks,
        multi_scale_window_sizes=tuple(args.multi_scale_window_sizes),
        patch_norm=args.patch_norm,
        max_batches=args.max_batches,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        outlier_threshold=args.outlier_threshold,
        use_test_split=args.use_test_split,
        # Mamba parameters
        use_mamba=args.use_mamba,
        mamba_type=args.mamba_type,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        # BraTS volume thresholding parameters
        wt_threshold=args.wt_threshold,
        tc_threshold=args.tc_threshold,
        et_threshold=args.et_threshold,
        # Model optimization parameters
        compile_model=args.compile_model,
        precision=args.precision
    )
    
    # Run validation
    results = pipeline.run()
    
    return results

if __name__ == "__main__":
    main()