#trainer.py
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.models.pipeline import BrainTumorSegmentation

def setup_training(train_loader, val_loader, args, fold_idx=None):
    """
    Setup training with direct parameter passing from args
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        args: Argument namespace containing all configuration parameters (use_class_weights should be set in the namespace)
        fold_idx: Cross-validation fold index (optional)
    Returns:
        model: BrainTumorSegmentation model
        trainer: PyTorch Lightning trainer
    """
    # Early stopping callback using parameters from args - only if patience is specified
    callbacks = []
    if args.early_stopping_patience is not None:
        early_stop_callback = EarlyStopping(
            monitor="val_mean_dice",
            min_delta=0.001,
            patience=args.early_stopping_patience,
            verbose=True,
            mode='max',
            check_finite=True,
            strict=False  # Allow missing metrics during first epochs
        )
        callbacks.append(early_stop_callback)

    # Model checkpoint callback - include fold info if cross-validation
    checkpoint_dir = 'checkpoints' if fold_idx is None else f'checkpoints/fold_{fold_idx}'
    filename_template = 'swinunetr-{epoch:02d}-{val_mean_dice:.4f}' if fold_idx is None else f'swinunetr-fold{fold_idx}-{{epoch:02d}}-{{val_mean_dice:.4f}}'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_template,
        monitor='val_mean_dice',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    # Learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.extend([checkpoint_callback, lr_monitor])

    # Generate dynamic wandb run name based on key parameters
    run_name_parts = []
    run_name_parts.append(getattr(args, 'loss_type', 'dice'))
    run_name_parts.append(f"e{args.epochs}")
    run_name_parts.append(f"lr{args.learning_rate:.0e}")
    
    # Add fold info if cross-validation
    if fold_idx is not None:
        cv_folds = getattr(args, 'cv_folds', 5)
        run_name_parts.append(f"cv{cv_folds}fold{fold_idx}")
    
    if getattr(args, 'use_warm_restarts', False):
        run_name_parts.append(f"warm-{getattr(args, 'restart_period', 20)}")
    
    if getattr(args, 'use_adaptive_scheduling', False):
        run_name_parts.append("adaptive")
    
    if getattr(args, 'use_class_weights', False):
        run_name_parts.append("weighted")
    
    # Add cross-validation prefix if applicable
    cv_prefix = "CV-" if getattr(args, 'use_cross_validation', False) else ""
    dynamic_name = cv_prefix + "-".join(run_name_parts)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="brain-tumor-segmentation",
        name=dynamic_name,
        log_model=False
    )

    # Initialize model
    model = BrainTumorSegmentation(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        learning_rate=args.learning_rate,
        feature_size=args.feature_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        use_v2=args.use_v2,
        depths=args.depths,
        num_heads=args.num_heads,
        downsample=args.downsample,
        use_class_weights=getattr(args, 'use_class_weights', True),
        loss_type=getattr(args, 'loss_type', 'dice'),
        use_modality_attention=getattr(args, 'use_modality_attention', False),
        overlap=getattr(args, 'overlap', 0.7),
        class_weights=getattr(args, 'class_weights', (3.0, 1.0, 5.0)),
        threshold=getattr(args, 'threshold', 0.5),
        optimizer_betas=getattr(args, 'optimizer_betas', (0.9, 0.999)),
        optimizer_eps=getattr(args, 'optimizer_eps', 1e-8),
        # New loss parameters
        tversky_alpha=getattr(args, 'tversky_alpha', 0.5),
        tversky_beta=getattr(args, 'tversky_beta', 0.5),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
        focal_alpha=getattr(args, 'focal_alpha', None),
        gdl_weight_type=getattr(args, 'gdl_weight_type', 'square'),
        gdl_lambda=getattr(args, 'gdl_lambda', 1.0),
        hausdorff_alpha=getattr(args, 'hausdorff_alpha', 2.0),
        lambda_dice=getattr(args, 'lambda_dice', 1.0),
        lambda_focal=getattr(args, 'lambda_focal', 1.0),
        lambda_tversky=getattr(args, 'lambda_tversky', 1.0),
        lambda_hausdorff=getattr(args, 'lambda_hausdorff', 1.0),
        # Adaptive loss scheduling parameters
        use_adaptive_scheduling=getattr(args, 'use_adaptive_scheduling', False),
        adaptive_schedule_type=getattr(args, 'adaptive_schedule_type', 'linear'),
        structure_epochs=getattr(args, 'structure_epochs', 30),
        boundary_epochs=getattr(args, 'boundary_epochs', 50),
        schedule_start_epoch=getattr(args, 'schedule_start_epoch', 10),
        min_loss_weight=getattr(args, 'min_loss_weight', 0.1),
        max_loss_weight=getattr(args, 'max_loss_weight', 2.0),
        # Learning rate scheduling parameters
        use_warm_restarts=getattr(args, 'use_warm_restarts', False),
        restart_period=getattr(args, 'restart_period', 20),
        restart_mult=getattr(args, 'restart_mult', 1),
        # ReduceLROnPlateau parameters
        use_reduce_lr_on_plateau=getattr(args, 'use_reduce_lr_on_plateau', False),
        plateau_patience=getattr(args, 'plateau_patience', 5),
        plateau_factor=getattr(args, 'plateau_factor', 0.5),
        min_lr=getattr(args, 'min_lr', 1e-7),
        # SwinUNETRPlus enhancement parameters (disabled by default for vanilla behavior)
        use_multi_scale_attention=getattr(args, 'use_multi_scale_attention', False),
        use_adaptive_window=getattr(args, 'use_adaptive_window', False),
        use_cross_layer_fusion=getattr(args, 'use_cross_layer_fusion', False),
        use_hierarchical_skip=getattr(args, 'use_hierarchical_skip', False),
        use_enhanced_v2_blocks=getattr(args, 'use_enhanced_v2_blocks', False),
        multi_scale_window_sizes=getattr(args, 'multi_scale_window_sizes', [7, 5, 3]),
        patch_norm=getattr(args, 'patch_norm', False),
        optimizer_type=getattr(args, 'optimizer_type', 'adamw'),
        fast_validation=getattr(args, 'fast_validation', False),
        use_mamba=getattr(args, 'use_mamba', False),
        mamba_type=getattr(args, 'mamba_type', 'segmamba'),
        # Mamba d_ parameters
        d_state=getattr(args, 'd_state', 16),
        d_conv=getattr(args, 'd_conv', 3),
        expand=getattr(args, 'expand', 1),
        window_size=getattr(args, 'window_size', [4, 4, 4]),
        # Post-processing parameters
        wt_threshold=getattr(args, 'wt_threshold', 250),
        tc_threshold=getattr(args, 'tc_threshold', 150),
        et_threshold=getattr(args, 'et_threshold', 100),
        # S3F loss parameters
        s3f_alpha=getattr(args, 's3f_alpha', 1.0),
        s3f_beta=getattr(args, 's3f_beta', 0.5),
        s3f_delta=getattr(args, 's3f_delta', 0.3),
        s3f_gamma=getattr(args, 's3f_gamma', 2.0),
        # Compile model if specified
        compile_model=getattr(args, 'compile_model', False),
        # Muon scheduler parameter
        muon_scheduler=getattr(args, 'muon_scheduler', False),
        # Cosine annealing scheduler parameter
        use_cosine_scheduler=getattr(args, 'use_cosine_scheduler', False),
        # Reproducibility
        seed=getattr(args, 'seed', 42),
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        strategy=args.strategy,
        gradient_clip_val=getattr(args, 'gradient_clip_val', 0),
        gradient_clip_algorithm=getattr(args, 'gradient_clip_algorithm', 'norm'),
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        enable_checkpointing=args.enable_checkpointing,
        benchmark=args.benchmark,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        accumulate_grad_batches=getattr(args, 'accumulate_grad_batches', 1),
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        profiler=args.profiler
    )
    
    # Enable non-strict loading for checkpoint compatibility
    trainer.strategy.strict_loading = False

    return model, trainer

def train_model(model, trainer, train_loader, val_loader, resume_from_checkpoint=None, load_pretrained=None):
    """
    Train the model using the provided trainer and data loaders
    
    Args:
        model: BrainTumorSegmentation model
        trainer: PyTorch Lightning trainer
        train_loader: Training data loader
        val_loader: Validation data loader
        resume_from_checkpoint: Path to checkpoint file to resume from
        load_pretrained: Path to checkpoint file - load pretrained weights, start from epoch 0
    """
    try:
        ckpt_path = resume_from_checkpoint
        
        # Handle pretrained weight loading
        if load_pretrained:
            import torch
            import os
            print(f"Loading pretrained weights from: {load_pretrained}")
            if os.path.exists(load_pretrained):
                # Handle PyTorch 2.6+ security changes - use weights_only=False for trusted checkpoints
                try:
                    # First try with weights_only=True (secure)
                    checkpoint = torch.load(load_pretrained, map_location='cpu', weights_only=True)
                except Exception as e:
                    if "weights_only" in str(e) or "MONAI" in str(e):
                        print("Warning:  Checkpoint contains MONAI classes, loading with weights_only=False (trusted source)")
                        # For trusted checkpoints with MONAI classes
                        checkpoint = torch.load(load_pretrained, map_location='cpu', weights_only=False)
                    else:
                        raise e
                
                # Extract model state dict from Lightning checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("OK: Loaded state_dict from Lightning checkpoint")
                else:
                    state_dict = checkpoint
                    print("OK: Loaded raw state_dict")
                
                # Load the weights
                model.load_state_dict(state_dict, strict=False)
                print("OK: Pretrained weights loaded successfully, starting training from epoch 0")
                ckpt_path = None  # Don't resume training state
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {load_pretrained}")
        
        # Start training
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path
        )
        
        # Log final metrics
        print("\n=== Training Complete ===")
        print(f"Best validation dice score: {model.best_metric:.4f}")
        print(f"Best epoch: {model.best_metric_epoch}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e