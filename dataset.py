import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import numpy as np
# from utils import remap_patch


class SICAPFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading SICAP patch features and corresponding masks for a single slide.
    """
    
    def __init__(self, slide_id, features_dir, masks_dir, transform=None):
        """
        Args:
            slide_id: Slide ID to load (e.g., '16B0001851')
            features_dir: Base directory containing feature folders
            masks_dir: Base directory containing mask folders
            transform: Optional transform to apply to masks
        """
        self.slide_id = slide_id
        self.features_dir = Path(features_dir) / slide_id
        self.masks_dir = Path(masks_dir) / slide_id
        self.transform = transform
        
        # Get all .pt feature files
        self.feature_files = sorted(list(self.features_dir.glob('*.pt')))
        
        if not self.feature_files:
            raise ValueError(f"No .pt files found in {self.features_dir}")
        
        # Verify corresponding masks exist
        self._verify_masks()
        
    def _verify_masks(self):
        """Verify that all feature files have corresponding mask files."""
        for feat_path in self.feature_files:
            mask_path = self.masks_dir / f"{feat_path.stem}.jpg"
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'features': torch.Tensor of shape [512] (squeezed from [1, 512])
                - 'mask': torch.Tensor of shape [512, 512]
                - 'filename': str, name of the patch
                - 'slide_id': str, slide ID for stitching later
        """
        # Load feature
        feat_path = self.feature_files[idx]
        features = torch.load(feat_path)
        
        # Squeeze the batch dimension [1, 512] -> [512]
        if features.dim() == 2 and features.shape[0] == 1:
            features = features.squeeze(0)
        
        # Load mask
        mask_path = self.masks_dir / f"{feat_path.stem}.jpg"
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)  # Convert to numpy array
        mask = remap_patch(mask) # Remap into [0, 1, 2, 3]
        mask = torch.from_numpy(mask)  # Convert to tensor
        
        # Apply transform if provided
        if self.transform:
            mask = self.transform(mask)
        
        return {
            'features': features,
            'mask': mask,
            'filename': feat_path.stem,
            'slide_id': self.slide_id
        }


class SICAPMultiSlideDataset(Dataset):
    """
    Dataset that combines multiple slides.
    """
    
    def __init__(self, slide_ids, features_dir, masks_dir, transform=None):
        """
        Args:
            slide_ids: List of slide IDs to load
            features_dir: Base directory containing feature folders
            masks_dir: Base directory containing mask folders
            transform: Optional transform to apply to masks
        """
        self.datasets = []
        self.cumulative_sizes = [0]
        
        for slide_id in slide_ids:
            dataset = SICAPFeatureDataset(slide_id, features_dir, masks_dir, transform)
            self.datasets.append(dataset)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))
        
        self.total_size = self.cumulative_sizes[-1]
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """Get item from the appropriate dataset."""
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cum_size:
                dataset_idx = i
                break
        
        # Get local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        
        return self.datasets[dataset_idx][local_idx]


class CAMELYON16_Slide_Dataset(Dataset):
    """
    Patch-level dataset for a single CAMELYON16 slide stored in a dual-feature h5 file.

    Each __getitem__ returns one patch, matching the SICAPFeatureDataset contract:
        'features'  : torch.Tensor [512]           pooled 1D embedding (float32)
        'tokens'    : torch.Tensor [768, 16, 16]   2D visual tokens    (float32)
        'mask'      : torch.Tensor [16, 16]        placeholder (all -1) until masks are ready
        'filename'  : str                          "<slide_id>_patch_<idx>"
        'slide_id'  : str                          slide name (no .h5 extension)
        'coords'    : torch.Tensor [2]             (x, y) level-0 coordinates (int64)
    """

    def __init__(self, h5_path, transform=None):
        """
        Args:
            h5_path: Path to the slide's .h5 feature file.
            transform: Optional transform applied to the mask tensor.
        """
        self.h5_path   = Path(h5_path)
        self.slide_id  = self.h5_path.stem
        self.transform = transform

        # Read patch count without loading data
        with h5py.File(self.h5_path, 'r') as f:
            self.n_patches = f['embeddings'].shape[0]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Single-row read — avoids loading the full slide into RAM
        with h5py.File(self.h5_path, 'r') as f:
            features = torch.from_numpy(f['embeddings'][idx].astype(np.float32))  # [512]
            tokens   = torch.from_numpy(f['tokens'][idx].astype(np.float32))      # [768, 16, 16]
            coords   = torch.from_numpy(f['coords'][idx])                          # [2]

        # Placeholder mask — replace once ground-truth masks are available
        mask = torch.full((tokens.shape[-2], tokens.shape[-1]), -1, dtype=torch.long)

        if self.transform:
            mask = self.transform(mask)

        return {
            'features': features,                            # [512]         float32
            'tokens':   tokens,                              # [768, 16, 16] float32
            'mask':     mask,                                # [16, 16]      int64
            'filename': f'{self.slide_id}_patch_{idx}',     # str
            'slide_id': self.slide_id,                       # str
            'coords':   coords,                              # [2]           int64
        }


class CAMELYON16MultiSlideDataset(Dataset):
    """
    Aggregates all per-slide CAMELYON16_Slide_Dataset instances into one flat
    patch-level dataset, mirroring SICAPMultiSlideDataset.
    """

    def __init__(self, feature_dir, transform=None):
        """
        Args:
            feature_dir: Directory containing per-slide .h5 files.
            transform: Optional transform forwarded to each slide dataset.
        """
        h5_files = sorted(Path(feature_dir).glob('*.h5'))
        if not h5_files:
            raise ValueError(f"No .h5 files found in {feature_dir}")

        self.datasets         = []
        self.cumulative_sizes = [0]

        for h5_path in h5_files:
            ds = CAMELYON16_Slide_Dataset(h5_path, transform)
            self.datasets.append(ds)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))

        self.total_size = self.cumulative_sizes[-1]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")

        # Binary search for the right slide
        lo, hi = 0, len(self.datasets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cumulative_sizes[mid + 1]:
                hi = mid
            else:
                lo = mid + 1

        local_idx = idx - self.cumulative_sizes[lo]
        return self.datasets[lo][local_idx]


def get_slide_ids(features_dir):
    """
    Get all available slide IDs from the features directory.
    """
    features_path = Path(features_dir)
    slide_ids = [d.name for d in features_path.iterdir() if d.is_dir()]
    return sorted(slide_ids)


if __name__ == '__main__':
    feature_dir = "/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual"

    # Test single-slide dataset
    import glob
    h5_files = sorted(glob.glob(f"{feature_dir}/*.h5"))
    print(f"Found {len(h5_files)} slides")

    slide_ds = CAMELYON16_Slide_Dataset(h5_files[0])
    print(f"\nSingle slide: {slide_ds.slide_id}  ({len(slide_ds)} patches)")
    sample = slide_ds[0]
    print(f"  features : {sample['features'].shape}  {sample['features'].dtype}")
    print(f"  tokens   : {sample['tokens'].shape}   {sample['tokens'].dtype}")
    print(f"  mask     : {sample['mask'].shape}     {sample['mask'].dtype}")
    print(f"  filename : {sample['filename']}")
    print(f"  slide_id : {sample['slide_id']}")
    print(f"  coords   : {sample['coords']}")

    # Test multi-slide dataset
    multi_ds = CAMELYON16MultiSlideDataset(feature_dir)
    print(f"\nMulti-slide total patches: {len(multi_ds)}")
    sample2 = multi_ds[0]
    print(f"  features : {sample2['features'].shape}")
    print(f"  slide_id : {sample2['slide_id']}")
