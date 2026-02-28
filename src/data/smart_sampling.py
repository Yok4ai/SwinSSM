import numpy as np
import nibabel as nib

def analyze_sample_difficulty(data_files, dataset="brats2023"):
    """
    Analyze training samples to categorize by difficulty/importance.
    Returns indices sorted by training value.
    """
    difficulties = []
    
    for idx, sample in enumerate(data_files):
        try:
            # Load label to analyze tumor characteristics
            label_path = sample['label']
            label_img = nib.load(label_path)
            label_data = label_img.get_fdata()
            
            # Calculate tumor characteristics
            total_voxels = np.prod(label_data.shape)
            
            # Count each tumor class (raw labels)
            ncr_voxels = np.sum(label_data == 1)  # Necrotic Core
            ed_voxels = np.sum(label_data == 2)   # Edema
            
            # Handle ET label for combined datasets - check both possible values
            if dataset == "combined":
                et_voxels = np.sum(label_data == 3) + np.sum(label_data == 4)  # BraTS 2023 + 2021
            elif dataset == "brats2021":
                et_voxels = np.sum(label_data == 4)  # Enhancing Tumor
            else:  # brats2023
                et_voxels = np.sum(label_data == 3)  # Enhancing Tumor
            
            # Calculate combined classes (as used in training)
            tc_voxels = ncr_voxels + et_voxels  # Tumor Core = NCR + ET
            wt_voxels = ncr_voxels + ed_voxels + et_voxels  # Whole Tumor = NCR + ED + ET
            
            # Use whole tumor for overall size analysis
            tumor_ratio = wt_voxels / total_voxels
            
            # Difficulty scoring (higher = more valuable for training)
            difficulty_score = 0
            
            # Define thresholds
            SMALL_MIN = 0.005
            MODERATE_LOW = 0.10
            MODERATE_HIGH = 0.15
            LARGE_MAX = 0.30

            if MODERATE_LOW <= tumor_ratio <= MODERATE_HIGH:
                difficulty_score += 1  # Most informative (moderate size)
            elif SMALL_MIN < tumor_ratio < MODERATE_LOW or MODERATE_HIGH < tumor_ratio <= LARGE_MAX:
                difficulty_score += 2  # Harder (small/moderately large)
            else:
                difficulty_score += 3  # Most difficult (very small or very large)
                

            # 2. Multi-class cases are more valuable (check raw components)
            classes_present = sum([ncr_voxels > 0, ed_voxels > 0, et_voxels > 0])
            difficulty_score += classes_present
            
            # 3. Balanced class distribution is valuable (use training classes)
            if tc_voxels > 0 and et_voxels > 0:  # Has both TC and ET components
                # Check if classes are reasonably balanced
                class_sizes = [tc_voxels, wt_voxels, et_voxels]
                class_sizes = [s for s in class_sizes if s > 0]  # Remove zeros
                if len(class_sizes) > 1:
                    size_ratio = max(class_sizes) / (min(class_sizes) + 1)
                    if size_ratio < 10:  # Not too imbalanced
                        difficulty_score += 2
                    
            # 4. Edge cases (very small ET) are important for learning boundaries
            if 0 < et_voxels < 100:  # Small but present enhancing tumor
                difficulty_score += 2
                
            difficulties.append({
                'idx': idx,
                'score': difficulty_score,
                'tumor_ratio': tumor_ratio,
                'classes': classes_present,
                'sample': sample
            })
            
        except Exception as e:
            print(f"Error analyzing {sample.get('label', 'unknown')}: {e}")
            # Assign neutral score for problematic samples
            difficulties.append({
                'idx': idx,
                'score': 1,
                'tumor_ratio': 0,
                'classes': 0,
                'sample': sample
            })
    
    # Sort by difficulty score (highest first - most valuable samples)
    difficulties.sort(key=lambda x: x['score'], reverse=True)
    
    return difficulties

def get_smart_sample_indices(data_files, fraction=0.3, strategy='balanced', dataset="brats2023"):
    """
    Get indices of most valuable samples for training.
    
    Args:
        data_files: List of training samples
        fraction: Fraction of data to select
        strategy: 'balanced', 'hard', or 'diverse'
    """
    difficulties = analyze_sample_difficulty(data_files, dataset)
    n_samples = int(len(data_files) * fraction)
    
    if strategy == 'balanced':
        # Take top scoring samples (most informative)
        selected = difficulties[:n_samples]
        
    elif strategy == 'hard':
        # Focus on challenging cases
        high_difficulty = [d for d in difficulties if d['score'] >= 6]
        if len(high_difficulty) >= n_samples:
            selected = high_difficulty[:n_samples]
        else:
            # Fill remaining with next best
            remaining = n_samples - len(high_difficulty)
            other_samples = [d for d in difficulties if d['score'] < 6]
            selected = high_difficulty + other_samples[:remaining]
            
    elif strategy == 'diverse':
        # Ensure diversity across tumor sizes and classes
        selected = []
        
        # Get samples from different tumor ratio ranges
        ranges = [
            (0, 0.005),      # Very small tumors
            (0.005, 0.02),   # Small tumors  
            (0.02, 0.1),     # Medium tumors
            (0.1, 0.5)       # Large tumors
        ]
        
        samples_per_range = n_samples // len(ranges)
        
        for min_ratio, max_ratio in ranges:
            range_samples = [d for d in difficulties 
                           if min_ratio <= d['tumor_ratio'] < max_ratio]
            selected.extend(range_samples[:samples_per_range])
            
        # Fill remaining slots with top scores
        remaining = n_samples - len(selected)
        if remaining > 0:
            unused = [d for d in difficulties if d not in selected]
            selected.extend(unused[:remaining])
    
    # Return original indices
    return [s['idx'] for s in selected]

def print_sampling_stats(data_files, selected_indices, dataset="brats2023"):
    """Print statistics about selected samples"""
    print(f"\n=== Smart Sampling Statistics ===")
    print(f"Total samples: {len(data_files)}")
    print(f"Selected: {len(selected_indices)} ({len(selected_indices)/len(data_files)*100:.1f}%)")
    
    difficulties = analyze_sample_difficulty(data_files, dataset)
    selected_difficulties = [difficulties[i] for i in selected_indices]
    
    print(f"Average difficulty score: {np.mean([d['score'] for d in selected_difficulties]):.2f}")
    print(f"Tumor ratio range: {np.min([d['tumor_ratio'] for d in selected_difficulties]):.4f} - {np.max([d['tumor_ratio'] for d in selected_difficulties]):.4f}")
    print("=====================================\n")