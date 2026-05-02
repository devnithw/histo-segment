import time
import torch
from torch.utils.data import DataLoader
from dataset import CAMELYON16MultiSlideDataset

test_feature_dir = '/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual'
test_mask_dir = '/home/nadun/wd/datasets/camelyon16/test/patched_masks'

print("Loading dataset...")
dataset = CAMELYON16MultiSlideDataset(feature_dir=test_feature_dir, mask_dir=test_mask_dir)
print(f"Total size: {len(dataset)}")

test_indices = torch.randperm(len(dataset))[:50]
subset = torch.utils.data.Subset(dataset, test_indices)

loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=8)

print("Starting read...")
start = time.time()
for i, batch in enumerate(loader):
    print(f"Batch {i} loaded in {time.time() - start:.2f}s")
    start = time.time()
