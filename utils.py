import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from tqdm import tqdm

def remap_patch(mask):
    mask_remapped = np.zeros_like(mask, dtype=np.int64)
    
    # Define intensity ranges
    mask_remapped[mask < 25] = 0      # Black
    mask_remapped[(mask >= 25) & (mask < 75)] = 1   # Dark gray
    mask_remapped[(mask >= 75) & (mask < 125)] = 2  # Light gray
    mask_remapped[mask >= 125] = 3    # White
    
    return mask_remapped

def plot_losses(train_losses, val_losses, run_name, results_dir):
    """
    Plot and save training and validation losses.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Losses', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(results_dir, f'{run_name}_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Loss plot saved to: {plot_path}")


def calculate_dice_score(pred, target, num_classes, smooth=1e-6):
    """
    Calculate Dice score for multi-class segmentation.
    """
    dice_per_class = []
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        # Calculate intersection and union
        intersection = (pred_class * target_class).sum()
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        
        # Dice coefficient: 2 * |X ∩ Y| / (|X| + |Y|)
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_per_class.append(dice.item())
    
    dice_per_class = np.array(dice_per_class)
    mean_dice = dice_per_class.mean()
    
    return dice_per_class, mean_dice


def calculate_iou(pred, target, num_classes, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for multi-class segmentation.
    """
    iou_per_class = []
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        # Calculate intersection and union
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        
        # IoU: |X ∩ Y| / |X ∪ Y|
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou.item())
    
    iou_per_class = np.array(iou_per_class)
    mean_iou = iou_per_class.mean()
    
    return iou_per_class, mean_iou


def evaluate_metrics(model, dataloader, num_classes, device='cuda'):
    """
    Evaluate segmentation metrics (Dice and IoU) on a dataset.
    Memory-efficient version that computes metrics incrementally.
    """
    model.eval()
    
    # Clear GPU cache before evaluation
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("GPU cache cleared before evaluation")
    
    # Initialize accumulators for each class
    intersection_sum = np.zeros(num_classes)
    pred_sum = np.zeros(num_classes)
    target_sum = np.zeros(num_classes)
    union_sum = np.zeros(num_classes)
    
    print("\nEvaluating segmentation metrics...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing metrics")):
            features = batch['features'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Get predictions (argmax over class dimension)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)
            
            # Calculate per-class metrics for this batch
            for class_idx in range(num_classes):
                # Create binary masks for current class
                pred_class = (preds == class_idx).float()
                target_class = (masks == class_idx).float()
                
                # Accumulate statistics
                intersection = (pred_class * target_class).sum().item()
                pred_count = pred_class.sum().item()
                target_count = target_class.sum().item()
                union = pred_count + target_count - intersection
                
                intersection_sum[class_idx] += intersection
                pred_sum[class_idx] += pred_count
                target_sum[class_idx] += target_count
                union_sum[class_idx] += union
            
            # Free memory after each batch
            del features, masks, outputs, preds
            if device == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Calculate final Dice scores from accumulated statistics
    smooth = 1e-6
    dice_per_class = (2. * intersection_sum + smooth) / (pred_sum + target_sum + smooth)
    mean_dice = dice_per_class.mean()
    
    # Calculate final IoU scores from accumulated statistics
    iou_per_class = (intersection_sum + smooth) / (union_sum + smooth)
    mean_iou = iou_per_class.mean()
    
    # Store in dictionary
    metrics = {
        'dice_per_class': dice_per_class,
        'mean_dice': mean_dice,
        'iou_per_class': iou_per_class,
        'mean_iou': mean_iou
    }
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print segmentation metrics in a formatted way.
    """
    num_classes = len(metrics['dice_per_class'])
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    print("\n" + "="*60)
    print("SEGMENTATION METRICS")
    print("="*60)
    
    print("\nPer-Class Dice Scores:")
    print("-" * 60)
    for i, (name, dice) in enumerate(zip(class_names, metrics['dice_per_class'])):
        print(f"  {name:15s}: {dice:.4f}")
    print(f"\n  {'Mean Dice':15s}: {metrics['mean_dice']:.4f}")
    
    print("\nPer-Class IoU Scores:")
    print("-" * 60)
    for i, (name, iou) in enumerate(zip(class_names, metrics['iou_per_class'])):
        print(f"  {name:15s}: {iou:.4f}")
    print(f"\n  {'Mean IoU':15s}: {metrics['mean_iou']:.4f}")
    
    print("="*60)
