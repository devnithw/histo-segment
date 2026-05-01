import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import h5py

test_data_path = "/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual"

test_imgs = os.listdir(test_data_path)
print(f"length of test images: {len(test_imgs)}")
test_imgs = [img for img in test_imgs if img[-3:] == ".h5"]
print(f"length of complete test images: {len(test_imgs)}")

count = 0
for img in test_imgs:
    img_path = os.path.join(test_data_path, img)
    with h5py.File(img_path, 'r') as f:
        print("keys:", f.keys())
        print(f"coords shape: {f['coords'].shape}")
        print(f"embeddings shape: {f['embeddings'].shape}")
        print(f"tokens shape: {f['tokens'].shape}")
        print(f"coords dtype: {f['coords'].dtype}")
        print(f"embeddings dtype: {f['embeddings'].dtype}")
        print(f"tokens dtype: {f['tokens'].dtype}")
        print(f"coords: {f['coords']}")
    
    break

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


if __name__ == '__main__':
    dataset = CAMELYON16Dataset('/home/nadun/wd/datasets/camelyon16/test')
    print("#"*30)
    print(len(dataset))
    sample = dataset[0]
    print(sample['features'].shape)
    print(sample['mask'].shape)
    print(sample['slide_id'])
    print(sample['coords'])