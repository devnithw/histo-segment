# CAMELYON16 WSI Segmentation Pipeline

## Overview
This pipeline trains a segmentation model using pre-extracted CONCH visual tokens from CAMELYON16 Whole Slide Images (WSIs). The model learns to upsample `768`-dimensional 2D CONCH token grids (`16x16`) into high-resolution `512x512` segmentation masks. It includes a complete flow for training, patch-level inference, and stitching patches back into contiguous WSI masks for evaluation.

## Architecture

### Model Components

1. **SingleScaleDecoder** (`model.py`)
   - Minimal single-scale decoder for foundational model features.
   - Progressive upsampling path with skip-like logic.
   - Upsamples `[B, 768, 16, 16]` tokens to `[B, num_classes, 512, 512]`.
   - Pipeline:
     - Input: `[B, 768, 16, 16]` CONCH tokens.
     - Width projection (1x1 conv) to `head_dim` (e.g., 256).
     - Successive upsampling blocks (Bilinear Interpolation + 3x3 Conv + ReLU).
     - Final 1x1 conv for class prediction.

### Dataset

**CAMELYON16_Slide_Dataset & CAMELYON16MultiSlideDataset** (`dataset.py`)
- Designed to handle massive `.h5` files containing WSI patches.
- Solves severe mechanical HDD I/O bottlenecks:
  - Uses `128MB` chunk caches (`rdcc_nbytes=1024*1024*128`) to mitigate HDF5 chunking inefficiencies.
  - Sorts indices in training/validation to ensure sequential disk reads, speeding up I/O by orders of magnitude.
- Returns:
  - `features`: `[512]` tensor (pooled embeddings).
  - `tokens`: `[768, 16, 16]` tensor (visual tokens).
  - `mask`: `[512, 512]` tensor (ground truth from PNGs).
  - `coords`: `[2]` tensor (Level 0 x,y coordinates).
  - `slide_id`: Slide identifier for tracking.

## File Structure

```
/home/nadun/wd/segmentation/
├── model.py              # SingleScaleDecoder architecture
├── dataset.py            # HDF5-optimized PyTorch dataset classes
├── train.py              # Training script with sequential I/O optimization
├── inference.py          # Fast patch-level inference script
├── stitch.py             # Stitches patch masks into full WSI TIFFs
├── loss.py               # Combined CrossEntropy + Dice Loss (CEDiceLoss)
├── engine.py             # Training and validation loops
├── utils.py              # Metrics (F1, Dice, IoU, etc.) and plotting
└── checkpoints/          # Model checkpoints
```

## Configuration

### Training Parameters (in `train.py`)
- **Batch size**: 16
- **Epochs**: 8
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Loss function**: CEDiceLoss
- **LR Scheduler**: CosineAnnealingLR (eta_min=7.5e-5)
- **Train/Val subset**: Dynamic subset ratios (e.g., 0.05 / 0.001) with sequential sorting.
- **Num classes**: 3 (Background, Normal Tissue, Tumor)

### Model Parameters
- **Input Tokens**: `[768, 16, 16]` (CONCH token output)
- **Output size**: `512x512`

## Usage

### 1. Train Model
```bash
conda activate vlm
python train.py
```
*Note: Ensure `HDF5_USE_FILE_LOCKING=FALSE` is set to prevent multi-processing deadlocks (handled automatically in `train.py`).*

### 2. Run Inference
```bash
conda activate vlm
python inference.py
```
Generates patch-level predictions as `.png` files in `inference_results/patched_masks/`. Uses ThreadPoolExecutor to prevent disk writes from blocking the GPU.

### 3. Stitch WSI Masks
```bash
conda activate vlm
python stitch.py
```
Takes the predicted `.png` patches and reconstructs them into a seamless WSI `.tif` mask using ASAP's `MultiResolutionImageWriter`. 
- Automatically calculates coordinate scale factors (e.g., 20x vs 40x).
- Performs full WSI-level Dice score evaluation against ground truth masks.

## Evaluation Metrics
The pipeline computes and tracks:
- **Dice Score** (Patch-level & WSI-level)
- **IoU (Intersection over Union)**
- **Precision, Recall, Specificity**
- **F1-Score** (Per-class and Macro-averaged)

## Important Optimizations
- **Sequential Disk Reads**: Mechanical HDD thrashing was eliminated by disabling `shuffle=True` and sorting subset indices.
- **HDF5 Chunk Caching**: HDF5 cache was increased from 1MB to 128MB per file to handle massive non-contiguous gzip chunks without repeatedly decompressing the same data.
