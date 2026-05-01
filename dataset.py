import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils import remap_patch


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

class CAMELYON16Dataset(Dataset):
    def __init__(self, dataset_dir, transform=None, feature_dir="/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual"):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "masks"
        self.feature_dir = feature_dir
        self.transform = transform
        self.feature_files = sorted([f for f in os.listdir(self.feature_dir) if f[-3:] == ".h5"])

    def _verify_masks(self):
        """Verify that all feature files have corresponding mask files."""
        for feat_path in self.feature_files:
            pass
            # mask_path = self.masks_dir / f"{feat_path.stem}.jpg"
            # if not mask_path.exists():
            #     raise FileNotFoundError(f"Mask file not found: {mask_path}")

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
        feat_slide_name = self.feature_files[idx]
        feat_path = os.path.join(self.feature_dir, feat_slide_name)
        
        with h5py.File(feat_path, 'r') as wsi:
            patch_coords = wsi['coords'][:]  # [num_patches, 2]
            patch_embeddings = wsi['embeddings'][:]  # [num_patches, 512]
            patch_tokens = wsi['tokens'][:]  # [num_patches, 768, 16, 16]

        # Load masks
        # TO be comopleted

        if self.transform:
            patch_embeddings = self.transform(patch_embeddings)
            patch_tokens = self.transform(patch_tokens)

        return {
            'features': patch_embeddings,
            'mask': patch_tokens,
            'slide_id': feat_slide_name.replace(".h5", ""),
            'coords': patch_coords
        }


def get_slide_ids(features_dir):
    """
    Get all available slide IDs from the features directory.
    """
    features_path = Path(features_dir)
    slide_ids = [d.name for d in features_path.iterdir() if d.is_dir()]
    return sorted(slide_ids)


if __name__ == '__main__':

    dataset = CAMELYON16Dataset('/home/nadun/wd/datasets/camelyon16/test')
    print("#"*30)
    print(len(dataset))
    sample = dataset[0]
    print(sample['features'].shape)
    print(sample['mask'].shape)
    print(sample['slide_id'])
    print(sample['coords'])

    # # Test the dataset
    # features_dir = '/home/nadun/wd/datasets/SICAP-test/features'
    # masks_dir = '/home/nadun/wd/datasets/SICAP-test/masks'
    
    # # Get all available slides
    # slide_ids = get_available_slide_ids(features_dir)
    # print(f"Total slides: {len(slide_ids)}")
    
    # # Test multi-slide dataset (all slides)
    # dataset = SICAPMultiSlideDataset(slide_ids, features_dir, masks_dir)
    # print(f"Total dataset size: {len(dataset)} patches")
    
    # # Test loading a sample
    # sample = dataset[5995]
    # print(f"\nSample:")
    # print(f"  Features shape: {sample['features'].shape}")
    # print(f"  Mask shape: {sample['mask'].shape}")
    # print(f"  Filename: {sample['filename']}")
    # print(f"  Slide ID: {sample['slide_id']}")
