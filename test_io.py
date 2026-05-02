import time
import os
import h5py
from PIL import Image

h5_path = "/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual/test_046.h5"
mask_dir = "/home/nadun/wd/datasets/camelyon16/test/patched_masks/test_046"

print("Starting IO Test...")

start = time.time()
with h5py.File(h5_path, 'r') as f:
    for i in range(16):
        features = f['embeddings'][i]
        tokens = f['tokens'][i]
        coords = f['coords'][i]
print(f"HDF5 reading 16 items took: {time.time() - start:.4f}s")

start = time.time()
with h5py.File(h5_path, 'r') as f:
    coords = f['coords'][:16]
    
for i in range(16):
    img_path = f"{mask_dir}/{i}_{coords[i][0]}_{coords[i][1]}.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img.load()
print(f"PNG reading 16 items took: {time.time() - start:.4f}s")

