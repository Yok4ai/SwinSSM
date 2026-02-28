from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandRotate90d,
    RandAffined,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Rand3DElasticd,
    RandCoarseDropoutd,
    SpatialPadd,
)
from data.convert_labels import ConvertLabels

def get_transforms(img_size, dataset="brats2023"):
    """Get training and validation transforms for BraTS data."""
    train_transforms = Compose(
        [
        # Essential loading and preprocessing
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertLabels(keys="label", dataset=dataset),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # nnUNet-style preprocessing additions
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        
        # Ensure minimum size for cropping - pad if needed to avoid dimension errors
        SpatialPadd(keys=["image", "label"], spatial_size=[img_size, img_size, img_size], mode="constant"),
        
        ScaleIntensityRangePercentilesd(
            keys="image", 
            lower=0.5, 
            upper=99.5, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),

        # Spatial cropping - now guaranteed to work since we padded above
        RandSpatialCropd(keys=["image", "label"], roi_size=[img_size, img_size, img_size], random_size=False),

        # Geometric augmentations - proven to boost dice
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
        
        # Elastic deformations - realistic brain tissue deformation
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(7, 13),            # Smooth deformations for brain tissue
            magnitude_range=(50, 150),       # Gentle deformation - brain is delicate
            prob=0.1,                      # Conservative - computationally expensive
            rotate_range=(0.05, 0.05, 0.05), # Very subtle rotations (~3 degrees max)
            shear_range=(0.02, 0.02, 0.02),  # Minimal shear for brain anatomy
            translate_range=(2, 2, 2),       # Small translations in voxels
            scale_range=(0.05, 0.05, 0.05),  # Minimal scaling changes
            mode="bilinear",                 # Interpolation mode
            padding_mode="reflection",       # Handle boundaries naturally
        ),
        
        # Additional affine transformations for more geometric variety
        RandAffined(
            keys=["image", "label"],
            prob=0.15,                        # Slightly higher probability than elastic
            rotate_range=(0.1, 0.1, 0.1),   # Small rotations (~6 degrees)
            shear_range=(0.05, 0.05, 0.05), # Small shear deformations
            translate_range=(3, 3, 3),       # Small translations in voxels
            scale_range=(0.1, 0.1, 0.1),    # Small scaling variations (±10%)
            mode=("bilinear", "nearest"),    # Interpolation for image, label
            padding_mode="reflection",       # Natural boundary handling
        ),

        # Intensity augmentations
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

        ## Light Gaussian noise for robustness
        RandGaussianNoised(keys="image", prob=0.2, std=0.01),

        # Coarse dropout - forces model to use multiple regions, prevents overfitting
        RandCoarseDropoutd(
            keys=["image"],
            holes=5,                    # Number of rectangular regions to drop
            spatial_size=(8, 8, 8),    # Size of each dropout region  
            dropout_holes=True,         # Actually drop the regions (vs fill)
            fill_value=0,               # Fill dropped regions with 0
            max_holes=10,               # Maximum number of holes
            prob=0.1,                   # Good probability for robustness
        ),

        # Contrast adjustment - helps with tumor boundary detection         
        RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.8, 1.3)),  # Conservative range
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertLabels(keys="label", dataset=dataset),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        
        # nnUNet-style preprocessing for validation too
        CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        ScaleIntensityRangePercentilesd(
            keys="image", 
            lower=0.5, 
            upper=99.5, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    return train_transforms, val_transforms