import argparse
import os
import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import concurrent.futures

from model import SingleScaleDecoder

def colorize_mask(mask):
    """
    Map class indices to grayscale intensities for visibility.
    Assuming classes:
      0 = background -> 0 (black)
      1 = normal tissue -> 127 (gray)
      2 = tumor -> 255 (white)
    """
    colored = np.zeros_like(mask, dtype=np.uint8)
    colored[mask == 1] = 127
    colored[mask == 2] = 255
    return colored

class H5InferenceDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.num_patches = f['tokens'].shape[0]

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Open in each worker to avoid hdf5 multiprocessing issues
        with h5py.File(self.h5_path, 'r') as f:
            tokens = torch.from_numpy(f['tokens'][idx].astype(np.float32))
            coords = f['coords'][idx]
        return tokens, coords, idx

def save_patch(mask_colored, img_path):
    img = Image.fromarray(mask_colored, mode='L')
    img.save(img_path)

def main(input_wsi, model_checkpoint, output_dir, batch_size, num_classes, num_workers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Num classes: {num_classes}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint: {model_checkpoint}")
    print(f"HDF5 path: {input_wsi}")

    # Initialize model
    print(f"Loading model checkpoint: {model_checkpoint}")
    model = SingleScaleDecoder(
        in_channels=768,
        num_classes=num_classes,
        input_size=(512, 512)
    ).to(device)

    # Force lazy initialization of upsample blocks
    dummy_input = torch.zeros(1, 768, 16, 16, device=device)
    with torch.no_grad():
        model(dummy_input)

    # Load checkpoint
    ckpt = torch.load(model_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")

    # Process slide
    slide_id = os.path.basename(input_wsi).replace('.h5', '')
    output_dir = os.path.join(output_dir, slide_id)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = H5InferenceDataset(input_wsi)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Total patches to process: {len(dataset)}")
    
    # Thread pool for saving images without blocking GPU
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    save_futures = []
    
    # Process in batches
    for tokens, coords, indices in tqdm(dataloader, desc="Inferencing"):
        tokens = tokens.to(device)
        
        # Forward pass
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(tokens) # [B, num_classes, 512, 512]
        
        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy() # [B, 512, 512]
        
        # Dispatch save tasks
        for i in range(len(preds)):
            patch_idx = indices[i].item()
            x, y = coords[i].numpy()
            
            mask = preds[i]
            mask_colored = colorize_mask(mask)
            
            img_path = os.path.join(output_dir, f"{patch_idx}_{int(x)}_{int(y)}.png")
            save_futures.append(executor.submit(save_patch, mask_colored, img_path))
            
            # Prevent extreme memory buildup of pending tasks
            if len(save_futures) > 1000:
                concurrent.futures.wait(save_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                save_futures = [fut for fut in save_futures if not fut.done()]

    # Wait for remaining saves
    if save_futures:
        concurrent.futures.wait(save_futures)
    executor.shutdown()

    print(f"\nInference complete! Saved {len(dataset)} patches to {output_dir}/")

if __name__ == '__main__':
    h5_path="/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual/test_001.h5"
    checkpoint="/home/nadun/wd/segmentation/checkpoints/camelyon16/run_05-03-02-47_bs16_ep7_lr0.001/best_model.pth"
    output_dir="/home/nadun/wd/segmentation/inference_results/patched_masks"
    batch_size=16 
    num_workers=4 
    num_classes=3
    
    main(input_wsi = h5_path,
        model_checkpoint = checkpoint,
        output_dir = output_dir,
        batch_size = batch_size,
        num_classes = num_classes,
        num_workers = num_workers
    )
