import os
import glob
import h5py
import numpy as np
import multiresolutionimageinterface as mir
from PIL import Image
from tqdm import tqdm

def main(patch_dir, h5_path, output_tif, gt_mask_path=None):
    # 1. Determine Dimensions and Scale Factor from H5
    print(f"Analyzing H5 coordinates: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
    
    # Calculate Stride from H5. 
    # Your log showed stride [0, 1024], but patches are 512px.
    # Scale factor = 1024 / 512 = 2.
    stride_val = np.max(np.abs(coords[1] - coords[0])) if len(coords) > 1 else 1024
    patch_size = 512
    scale_factor = int(stride_val // patch_size)
    
    if scale_factor < 1: scale_factor = 1
    print(f"Detected scale factor: {scale_factor} (Stride: {stride_val} -> Target: {patch_size})")

    # Calculate shrunken canvas dimensions so patches become contiguous
    width = (int(np.max(coords[:, 0])) // scale_factor) + patch_size
    height = (int(np.max(coords[:, 1])) // scale_factor) + patch_size
    print(f"Target Mask Dimensions: {width} x {height}")

    # 2. Map existing PNG files using the SAME scaling logic
    print("Mapping predicted patches...")
    png_files = glob.glob(os.path.join(patch_dir, "*.png"))
    patch_map = {}
    for p in png_files:
        basename = os.path.basename(p).replace(".png", "")
        parts = basename.split("_")
        if len(parts) >= 3:
            # Scale coordinates down to match contiguous 512-grid
            x_scaled = int(parts[-2]) // scale_factor
            y_scaled = int(parts[-1]) // scale_factor
            patch_map[(x_scaled, y_scaled)] = p
    print(f"Found {len(patch_map)} valid patches.")

    # 3. Setup GT Reader
    img_gt = None
    if gt_mask_path:
        reader = mir.MultiResolutionImageReader()
        img_gt = reader.open(gt_mask_path)

    # 4. Setup WSI Writer
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    writer = mir.MultiResolutionImageWriter()
    writer.openFile(output_tif)
    writer.setTileSize(512)
    writer.setCompression(mir.Compression_LZW)
    writer.setDataType(mir.DataType_UChar)
    writer.setColorType(mir.ColorType_Monochrome)
    writer.writeImageInformation(width, height)

    write_tile_size = 512 # MUST match writer.setTileSize(512)
    intersection, pred_sum, gt_sum = 0, 0, 0

    # 5. Stitching Loop
    print("Stitching WSI...")
    for y in tqdm(range(0, height, write_tile_size)):
        for x in range(0, width, write_tile_size):
            w = min(write_tile_size, width - x)
            h = min(write_tile_size, height - y)
            block = np.zeros((h, w), dtype=np.uint8)

            # Pull GT at full resolution and downsample to match prediction
            if img_gt is not None:
                gt_patch_raw = img_gt.getUCharPatch(
                    x * scale_factor, y * scale_factor, 
                    w * scale_factor, h * scale_factor, 0
                )
                gt_patch_2d = np.array(gt_patch_raw).reshape(h * scale_factor, w * scale_factor)
                gt_block = np.array(Image.fromarray(gt_patch_2d).resize((w, h), Image.NEAREST))

            # Fill the block with 512x512 patches
            for py in range(y, y + h, patch_size):
                for px in range(x, x + w, patch_size):
                    if (px, py) in patch_map:
                        pred_patch = np.array(Image.open(patch_map[(px, py)]))
                        
                        pw = min(patch_size, width - px)
                        ph = min(patch_size, height - py)
                        pred_patch = pred_patch[:ph, :pw]
                        
                        block[py-y : py-y+ph, px-x : px-x+pw] = pred_patch
                        
                        if img_gt is not None:
                            gt_patch = gt_block[py-y : py-y+ph, px-x : px-x+pw]
                            is_tumor_pred = (pred_patch == 255)
                            is_tumor_gt = (gt_patch == 2) 
                            
                            intersection += np.logical_and(is_tumor_pred, is_tumor_gt).sum()
                            pred_sum += is_tumor_pred.sum()
                            gt_sum += is_tumor_gt.sum()

            writer.writeBaseImagePartToLocation(block.flatten(), x, y)

    writer.finishImage()
    print(f"Success! Saved to: {output_tif}")

    if img_gt is not None:
        dice = (2. * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
        print(f"Final WSI Dice Score: {dice:.4f}")

if __name__ == '__main__':
    PATCH_DIR = "/home/nadun/wd/segmentation/inference_results/patched_masks/test_001"
    H5_PATH = "/home/nadun/wd/datasets/camelyon16/test/trident/20x_512px_0px_overlap/features_conch_v1_dual/test_001.h5"
    OUTPUT_TIF = "/home/nadun/wd/segmentation/inference_results/masks/test_001_pred_mask.tif"
    GT_MASK = "/home/nadun/wd/datasets/camelyon16/test/masks/test_001_mask.tif"

    main(PATCH_DIR, H5_PATH, OUTPUT_TIF, GT_MASK)