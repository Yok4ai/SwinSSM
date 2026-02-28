import json
import os
import numpy as np
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .smart_sampling import get_smart_sample_indices, print_sampling_stats


def get_dataloaders(data_dir, batch_size, num_workers, train_transforms, val_transforms, 
                   smart_sampling=False, sample_fraction=1.0, sampling_strategy='balanced', dataset='brats2023'):
    """
    Create training and validation DataLoaders.
    
    Args:
        data_dir (str): Directory containing the dataset.json file
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for DataLoader
        train_transforms: Transformations for training data
        val_transforms: Transformations for validation data
        smart_sampling (bool): Enable smart sampling
        sample_fraction (float): Fraction of data for smart sampling
        sampling_strategy (str): Smart sampling strategy
        dataset (str): Dataset type ('brats2021', 'brats2023', or 'combined')
    
    Returns:
        tuple: (train_loader, val_loader, test_files)
    """
    # Load dataset
    dataset_path = data_dir if data_dir.endswith('dataset.json') else os.path.join(data_dir, "dataset.json")
    with open(dataset_path) as f:
        datalist = json.load(f)["training"]
    
    print(f"Found {len(datalist)} samples in dataset")
    
    # Train/val/test split: 60% train, 20% val, 20% test
    # First split: 80% train+val, 20% test
    train_val_files, test_files = train_test_split(datalist, test_size=0.2, random_state=42)
    
    # Second split: 75% train, 25% val (of the remaining 80%)
    # This gives us 60% train, 20% val of total dataset
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, random_state=43)
    
    # Apply smart sampling if enabled
    if smart_sampling and sample_fraction < 1.0:
        print(f" Applying smart sampling: {sampling_strategy} strategy, {sample_fraction:.1%} of data")
        selected_indices = get_smart_sample_indices(train_files, sample_fraction, sampling_strategy, dataset)
        train_files = [train_files[i] for i in selected_indices]
        print_sampling_stats(train_val_files, selected_indices, dataset)
    
    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_files 