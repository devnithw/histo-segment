# SICAP Segmentation Pipeline

## Overview
This pipeline trains a segmentation model using pre-extracted CONCH features from SICAP patch images. The model learns to upsample 512-dimensional CONCH embeddings to 512x512 segmentation masks.

## Architecture

### Model Components

1. **UNetDecoder** (`model.py`)
   - Standalone decoder class for modular design
   - Progressive upsampling with skip connections
   - 4 upsampling blocks: 32x32 → 64x64 → 128x128 → 256x256 → 512x512
   - Each block: Upsample → Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
   - Final 1x1 conv for class prediction

2. **ModaSegNet** (`model.py`)
   - Complete segmentation model
   - Pipeline:
     - Input: [B, 512] CONCH features
     - Linear projection: [B, 512]
     - Reshape to spatial: [B, 512, 1, 1]
     - Initial conv: [B, 512, 1, 1]
     - Interpolate to: [B, 512, 32, 32]
     - UNetDecoder: [B, 512, 32, 32] → [B, 4, 512, 512]
     - Output: [B, num_classes, 512, 512]
   - Modular design allows easy decoder replacement

### Dataset

**SICAPFeatureDataset** (`dataset.py`)
- Loads pre-extracted `.pt` feature files
- Loads corresponding `.jpg` mask files
- Returns:
  - `features`: [512] tensor (CONCH embeddings)
  - `mask`: [512, 512] tensor (ground truth segmentation)
  - `filename`: patch name
  - `slide_id`: slide identifier for stitching

**SICAPMultiSlideDataset** (`dataset.py`)
- Combines multiple slides into single dataset
- Handles all slides in the features directory
- Maintains slide_id for each sample

## File Structure

```
/home/nadun/wd/segmentation/
├── model.py              # Model definitions (UNetDecoder, ModaSegNet)
├── dataset.py            # PyTorch dataset classes
├── train.py              # Training script
├── extract_features.py   # CONCH feature extraction
└── checkpoints/          # Model checkpoints (created during training)
```

## Configuration

### Training Parameters (in `train.py`)
- **Batch size**: 16
- **Epochs**: 50
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Train/Val split**: 80/20
- **Num classes**: 4

### Model Parameters
- **Feature dimension**: 512 (CONCH output)
- **Output size**: 512x512
- **Number of classes**: 4
- **Decoder channels**: (256, 128, 64, 32)

## Usage

### 1. Extract Features (Already Done)
```bash
conda activate conch
cd /home/nadun/wd/segmentation
python extract_features.py
```

### 2. Test Dataset
```bash
conda activate conch
python dataset.py
```

### 3. Test Model
```bash
conda activate conch
python model.py
```

### 4. Train Model
```bash
conda activate conch
python train.py
```

## Training Process

The training script will:
1. Load all available slides from the features directory
2. Split data into 80% train, 20% validation
3. Train for 50 epochs with progress bars
4. Save best model based on validation loss
5. Save checkpoints every 10 epochs
6. Apply learning rate scheduling based on validation loss

### Checkpoints
- **best_model.pth**: Best model based on validation loss
- **checkpoint_epoch_N.pth**: Checkpoints every 10 epochs

Each checkpoint contains:
- `epoch`: Current epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `train_loss`: Training loss
- `val_loss`: Validation loss

## Dataset Statistics

- **Current dataset**: 1 slide (16B0001851)
- **Total patches**: 55
- **Feature shape**: [512]
- **Mask shape**: [512, 512]
- **Mask dtype**: int64
- **Unique classes per mask**: ~100-111 values

## Model Statistics

- **Total parameters**: ~3-5M (approximate)
- **Input**: [B, 512] CONCH features
- **Output**: [B, 4, 512, 512] segmentation logits

## Notes

1. **Slide ID tracking**: Each sample includes `slide_id` for later patch stitching
2. **Modular design**: UNetDecoder is separate, allowing easy replacement with other decoders
3. **Memory efficient**: Features are pre-extracted, avoiding repeated CONCH inference
4. **Scalable**: Dataset automatically handles all slides in the features directory

## Future Extensions

1. Extract features for all slides (currently only 16B0001851)
2. Implement patch stitching to reconstruct full WSI segmentation
3. Add evaluation metrics (IoU, Dice, etc.)
4. Experiment with different decoder architectures
5. Add data augmentation
6. Implement inference script for new slides
