"""
extract_conch_dual.py

Extract both the 1-D pooled embedding AND the 2-D visual tokens from CONCH v1
(conch_ViT-B-16) for every patch in a Trident-patchified WSI dataset.

Input layout (Trident output):
  <coords_dir>/patches/<slide_name>_patches.h5   ← patch coords (x, y at level-0)
  <wsi_source>/<slide_name>.tif                  ← raw whole-slide images

Output layout:
  <output_dir>/<slide_name>.h5
    /embeddings   (N, 512)         – pooled CLS embedding  [float16]
    /tokens       (N, 768, 16, 16) – 2-D spatial tokens    [float16]
    /coords       (N, 2)           – (x, y) level-0 coords [int64]

Shapes for conch_ViT-B-16 (256-px model input, patch_size=16):
  pooled_embedding : (B, 512)
  raw_tokens       : (B, 256, 768)  →  reshaped to (B, 768, 16, 16)

Usage:
  python extract_conch_dual.py [--slide test_001]
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# CONCH
sys.path.insert(0, '/home/nadun/wd/CONCH')
from conch.open_clip_custom import create_model_from_pretrained

# Configuration
TRIDENT_JOB_DIR = '/home/nadun/wd/datasets/camelyon16/camelyon16_test/trident'
COORDS_SUBDIR   = '20x_512px_0px_overlap'
WSI_SOURCE      = '/home/nadun/wd/datasets/camelyon16/camelyon16_test/images'
WSI_EXT         = '.tif'

CONCH_MODEL_CFG = 'conch_ViT-B-16'
CONCH_CKPT      = '/home/nadun/hdd/CONCH/checkpoints/conch/pytorch_model.bin'

OUTPUT_DIR      = os.path.join(
    TRIDENT_JOB_DIR, COORDS_SUBDIR, 'features_conch_v1_dual'
)

BATCH_SIZE  = 128
NUM_WORKERS = 8
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# ViT-B/16 at 256-px input → 256/16 = 16×16 = 256 patch tokens (768-dim each)
GRID_H = GRID_W = 16
EMBED_DIM       = 768

def read_trident_coords(h5_path: str):
    """Read (attrs, coords) from a trident *_patches.h5 coord file."""
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        attrs  = dict(f['coords'].attrs)
    return attrs, coords


class WSITileDataset(Dataset):
    """
    Reads patches on-the-fly from a WSI via openslide, using pre-computed
    level-0 (x, y) coordinates.  Handles magnification scaling internally.
    """
    def __init__(self, slide, coords, patch_size_target, downsample, transform=None):
        self.slide             = slide
        self.coords            = coords
        self.patch_size_target = patch_size_target
        self.transform         = transform

        # Best openslide level at the requested downsample
        self.level            = slide.get_best_level_for_downsample(downsample)
        self.level_downsample = slide.level_downsamples[self.level]
        # Number of pixels to read at that level
        self.read_size = round(patch_size_target * downsample / self.level_downsample)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = int(self.coords[idx, 0]), int(self.coords[idx, 1])
        tile  = self.slide.read_region(
            (x, y), self.level, (self.read_size, self.read_size)
        ).convert('RGB')
        if tile.size != (self.patch_size_target, self.patch_size_target):
            tile = tile.resize(
                (self.patch_size_target, self.patch_size_target), Image.BILINEAR
            )
        if self.transform:
            tile = self.transform(tile)
        return tile, torch.tensor([x, y], dtype=torch.int64)

# Model
def load_conch(device):
    print(f"Loading CONCH from: {CONCH_CKPT}")
    model, preprocess = create_model_from_pretrained(CONCH_MODEL_CFG, CONCH_CKPT)
    model = model.to(device).eval()
    print("CONCH loaded ✓")
    return model, preprocess

# Per-slide extraction
@torch.inference_mode()
def extract_for_slide(slide_name, model, preprocess, device):
    coords_dir  = os.path.join(TRIDENT_JOB_DIR, COORDS_SUBDIR)
    coords_path = os.path.join(coords_dir, 'patches', f'{slide_name}_patches.h5')
    wsi_path    = os.path.join(WSI_SOURCE, f'{slide_name}{WSI_EXT}')
    out_path    = os.path.join(OUTPUT_DIR, f'{slide_name}.h5')
    tmp_path    = out_path + '.tmp'

    if os.path.exists(out_path):
        print(f"  [SKIP] {slide_name} — already done.")
        return
    if not os.path.exists(coords_path):
        print(f"  [SKIP] {slide_name} — coords not found.")
        return
    if not os.path.exists(wsi_path):
        print(f"  [SKIP] {slide_name} — WSI not found.")
        return

    slide = openslide.OpenSlide(wsi_path)
    coords_attrs, coords = read_trident_coords(coords_path)
    patch_size  = int(coords_attrs['patch_size'])
    level0_mag  = float(coords_attrs['level0_magnification'])
    target_mag  = float(coords_attrs['target_magnification'])
    downsample  = level0_mag / target_mag
    n_patches   = len(coords)

    print(f"\n  {slide_name}: {n_patches} patches  "
          f"(patch_size={patch_size}, {level0_mag:.0f}x→{target_mag:.0f}x, "
          f"ds={downsample:.1f})")

    dataset = WSITileDataset(
        slide=slide,
        coords=coords,
        patch_size_target=patch_size,
        downsample=downsample,
        transform=preprocess,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    written = 0
    grid_h = grid_w = embed_d = None

    with h5py.File(tmp_path, 'w') as f:
        ds_emb = ds_tok = ds_coords = None

        for imgs, batch_coords in tqdm(dataloader, desc=f"  {slide_name}", leave=False):
            imgs = imgs.to(device, non_blocking=True)

            # fp16 autocast — faster on Ampere+ and halves VRAM usage
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(device != 'cpu')):
                pooled, tokens = model.visual(imgs)

            # pooled : (B, 512)
            emb = pooled.half().cpu().numpy()

            # tokens : (B, N, D) → (B, D, H, W)
            B, N, D = tokens.shape
            H = W = int(N ** 0.5)
            assert H * W == N, f"Token count {N} is not a perfect square."
            tok = (tokens.half()
                         .permute(0, 2, 1)
                         .reshape(B, D, H, W)
                         .cpu().numpy())

            bc = batch_coords.numpy()  # (B, 2)

            # Create datasets on the first batch
            if ds_emb is None:
                grid_h, grid_w, embed_d = H, W, D
                cb = min(BATCH_SIZE, n_patches)
                ds_emb = f.create_dataset(
                    'embeddings',
                    shape=(0, 512), maxshape=(None, 512),
                    dtype='float16', chunks=(cb, 512), compression='lzf',
                )
                ds_tok = f.create_dataset(
                    'tokens',
                    shape=(0, D, H, W), maxshape=(None, D, H, W),
                    dtype='float16', chunks=(cb, D, H, W),
                )
                ds_coords = f.create_dataset(
                    'coords',
                    shape=(0, 2), maxshape=(None, 2),
                    dtype='int64', chunks=(cb, 2),
                )
                for k, v in coords_attrs.items():
                    try:
                        ds_coords.attrs[k] = v
                    except Exception:
                        pass

            # Append batch
            end = written + len(emb)
            ds_emb.resize(   (end, 512)     )
            ds_tok.resize(   (end, D, H, W) )
            ds_coords.resize((end, 2)       )
            ds_emb[written:end]    = emb
            ds_tok[written:end]    = tok
            ds_coords[written:end] = bc
            written = end

        f.attrs['encoder']    = 'conch_ViT-B-16'
        f.attrs['slide_name'] = slide_name
        f.attrs['n_patches']  = written
        f.attrs['grid_h']     = grid_h or GRID_H
        f.attrs['grid_w']     = grid_w or GRID_W

    os.replace(tmp_path, out_path)
    slide.close()

    print(f"  embeddings : ({written}, 512) fp16")
    print(f"  tokens     : ({written}, {embed_d}, {grid_h}, {grid_w}) fp16")
    print(f"  Saved → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract CONCH pooled embeddings + 2-D visual tokens.'
    )
    parser.add_argument(
        '--slide', type=str, default=None,
        help='Single slide name (e.g. test_001). Omit to process all slides.'
    )
    args = parser.parse_args()

    patches_dir = os.path.join(TRIDENT_JOB_DIR, COORDS_SUBDIR, 'patches')
    all_h5      = sorted(Path(patches_dir).glob('*_patches.h5'))
    if not all_h5:
        print(f"No *_patches.h5 found in {patches_dir}. Exiting.")
        sys.exit(1)

    all_slides = [p.stem.replace('_patches', '') for p in all_h5]

    if args.slide:
        if args.slide not in all_slides:
            print(f"Slide '{args.slide}' not found.")
            sys.exit(1)
        slide_list = [args.slide]
    else:
        slide_list = all_slides

    print(f"Device     : {DEVICE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Workers    : {NUM_WORKERS}")
    print(f"Output     : {OUTPUT_DIR}")
    print(f"Slides     : {len(slide_list)}")
    print("=" * 60)

    model, preprocess = load_conch(DEVICE)

    for i, slide_name in enumerate(slide_list, 1):
        print(f"\n[{i}/{len(slide_list)}] {slide_name}")
        try:
            extract_for_slide(slide_name, model, preprocess, DEVICE)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {slide_name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
