import torch
from monai.transforms import MapTransform

class ConvertLabels(MapTransform):
    """
    Convert labels to multi channels based on BRATS classes:
    For all datasets (unified mapping):
        label 1 is Necrotic Tumor Core (NCR)
        label 2 is Edema (ED)  
        label 3 is Enhancing Tumor (ET) (mapped uniformly from BRATS 2021 label 4)
        label 0 is everything else (background)
    """
    def __init__(self, keys, dataset="brats2023"):
        super().__init__(keys)
        self.dataset = dataset

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Normalize BRATS 2021 labels to match BRATS 2023: map label 4 -> 3
            if self.dataset == "brats2021" or self.dataset == "combined":
                d[key] = torch.where(d[key] == 4, 3, d[key])
            
            result = []
            # Tumor Core (TC) = NCR + Enhancing Tumor (ET)
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # Whole Tumor (WT) = NCR + Edema + Enhancing Tumor
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            # Enhancing Tumor (ET)
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d