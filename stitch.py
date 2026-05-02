import argparse
import os
import glob
import numpy as np
import h5py
import multiresolutionimageinterface as mir
from PIL import Image
from tqdm import tqdm

def main(patch_dir, input_wsi_path, output_mask, gt_mask):
    print(f"Reading HDF5 coordinates to determine dimensions: {input_wsi_path}")
    with h5py.File(input_wsi_path, 'r') as f:
        coords = f['coords'][:]
    
    # Calculate bounding box dimensions (assuming 512x512 patches)
    width = int(np.max(coords[:, 0])) + 512
    height = int(np.max(coords[:, 1])) + 512
    print(f"Inferred WSI Dimensions: {width} x {height}")

    img_gt = None
    if gt_mask:
        print(f"Loading Ground Truth mask for metrics: {gt_mask}")
        reader = mir.MultiResolutionImageReader()
        img_gt = reader.open(gt_mask)
        if img_gt is None:
            raise ValueError(f"Could not open {gt_mask}")

    print("Mapping predicted PNG patches...")
    png_files = glob.glob(os.path.join(patch_dir, "*.png"))
    patch_map = {}
    for p in png_files:
        basename = os.path.basename(p).replace(".png", "")
        # Format is idx_x_y.png
        parts = basename.split("_")
        if len(parts) >= 3:
            x, y = int(parts[-2]), int(parts[-1])
            patch_map[(x, y)] = p
            
    print(f"Found {len(patch_map)} predicted patches.")

    print(f"Initializing writer for: {output_mask}")
    os.makedirs(os.path.dirname(output_mask), exist_ok=True)
    writer = mir.MultiResolutionImageWriter()
    writer.openFile(output_mask)
    writer.setTileSize(512)
    writer.setCompression(mir.Compression_LZW)
    writer.setDataType(mir.DataType_UChar)
    writer.setColorType(mir.ColorType_Monochrome)
    writer.writeImageInformation(width, height)

    write_tile_size = 4096
    patch_size = 512
    
    # Metrics tracking for Dice
    intersection_tumor = 0
    pred_tumor_sum = 0
    gt_tumor_sum = 0

    print("Stitching patches and calculating metrics...")
    # Iterate over larger blocks to minimize slow I/O calls
    for y in tqdm(range(0, height, write_tile_size)):
        for x in range(0, width, write_tile_size):
            w = min(write_tile_size, width - x)
            h = min(write_tile_size, height - y)
            
            block = np.zeros((h, w), dtype=np.uint8)
            
            # Load GT block once if provided
            if img_gt is not None:
                gt_block = img_gt.getUCharPatch(x, y, w, h, 0)
                gt_block = np.array(gt_block).reshape(h, w)
            
            # Fill the block with any 512x512 patches that belong here
            for py in range(y, y + h, patch_size):
                for px in range(x, x + w, patch_size):
                    if (px, py) in patch_map:
                        img = Image.open(patch_map[(px, py)])
                        pred_patch = np.array(img)
                        
                        pw = min(patch_size, width - px)
                        ph = min(patch_size, height - py)
                        pred_patch = pred_patch[:ph, :pw]
                        
                        # Place it in our memory block
                        block[py - y : py - y + ph, px - x : px - x + pw] = pred_patch
                        
                        # Calculate Dice metrics ONLY for extracted patches
                        if img_gt is not None:
                            gt_patch = gt_block[py - y : py - y + ph, px - x : px - x + pw]
                            
                            is_tumor_pred = (pred_patch == 255)
                            is_tumor_gt = (gt_patch == 2)
                            
                            intersection_tumor += np.logical_and(is_tumor_pred, is_tumor_gt).sum()
                            pred_tumor_sum += is_tumor_pred.sum()
                            gt_tumor_sum += is_tumor_gt.sum()
                            
            # Write the entire 4096x4096 block to disk in a single call
            writer.writeBaseImagePartToLocation(block.flatten(), x, y)
            
    writer.finishImage()
    print(f"Stitched TIF saved to: {output_mask}")

    if img_gt is not None:
        # Calculate final Dice score
        smooth = 1e-6
        tumor_dice = (2.0 * intersection_tumor + smooth) / (pred_tumor_sum + gt_tumor_sum + smooth)
        
        print("\n" + "="*40)
        print("PATCHWISE METRICS (WSI Level)")
        print("="*40)
        print(f"Tumor Dice Score : {tumor_dice:.4f}")
        print(f"Tumor Intersection Pixels: {intersection_tumor}")
        print(f"Tumor Predicted Pixels   : {pred_tumor_sum}")
        print(f"Tumor GT Pixels          : {gt_tumor_sum}")
        print("="*40)

if __name__ == '__main__':

    patch_dir = "/home/nadun/wd/segmentation/inference_results/patched_masks/test_046"
    h5_path = "/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual/test_046.h5"
    output_tif = "/home/nadun/wd/segmentation/inference_results/masks/test_046_pred_mask.tif"
    gt_mask = "/home/nadun/wd/datasets/camelyon16/test/masks/test_046_mask.tif"

    main(patch_dir = patch_dir, input_wsi_path=h5_path, output_mask=output_tif, gt_mask=gt_mask)
