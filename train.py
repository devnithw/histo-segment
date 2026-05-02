import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from dataset import CAMELYON16MultiSlideDataset, get_slide_ids
from model import SingleScaleDecoder
from loss import CEDiceLoss
from engine import train, validate
from utils import plot_losses, evaluate_metrics, print_metrics
from sklearn.model_selection import train_test_split

# Set start time and base run name
dt = datetime.now().strftime("%m-%d-%H-%M")
base_run_name = f"run_{dt}"

def main():
    # Configuration
    train_dir = '/home/nadun/wd/datasets/camelyon16/train'
    test_dir = '/home/nadun/wd/datasets/camelyon16/test'
    train_feature_dir = f'{train_dir}/trident/20x_512px_0px_overlap/features_conch_v1_dual'
    test_feature_dir = f'{test_dir}/trident/20x_512px_0px_overlap/features_conch_v1_dual'
    train_mask_dir = f'{train_dir}/patched_masks'
    test_mask_dir = f'{test_dir}/patched_masks'
    base_checkpoint_dir = '/home/nadun/wd/segmentation/checkpoints/camelyon16'
    results_dir = '/home/nadun/wd/segmentation/results/camelyon16'
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 8
    TRAIN_SUBSET_RATIO = 0.0001 #0.05
    TEST_SUBSET_RATIO = 0.01
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 3
    NUM_WORKERS = 8
    TOKEN_DIM = 768
    OUTPUT_SIZE = (512, 512)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all available slide IDs
    val_ids = sorted(glob.glob(os.path.join(test_feature_dir, '*.h5')))
    print(f"Total val slides: {len(val_ids)}")
    train_ids = sorted(glob.glob(os.path.join(train_feature_dir, '*.h5')))
    print(f"Total train slides: {len(train_ids)}")
    
    # # Create datasets
    # print("\nCreating dataset...")
    # dataset = CAMELYON16MultiSlideDataset(
    #     feature_dir=features_dir,
    #     train=False
    # )
    # print(f"Total dataset size: {len(dataset)} patches")
    # return "Dataset created successfully!"

    # # Split into train/val (80/20 split)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, 
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    # print(f"Training set size: {len(train_dataset)} patches")
    # print(f"Validation set size: {len(val_dataset)} patches")

    train_dataset_full = CAMELYON16MultiSlideDataset(feature_dir=train_feature_dir, mask_dir=train_mask_dir)
    print(f"Train set full size: {len(train_dataset_full)} patches")

    test_dataset_full = CAMELYON16MultiSlideDataset(feature_dir=test_feature_dir, mask_dir=test_mask_dir)
    print(f"Test set full size: {len(test_dataset_full)} patches")

    # Define subset size
    train_subset_count = int(TRAIN_SUBSET_RATIO * len(train_dataset_full))
    test_subset_count = int(TEST_SUBSET_RATIO * len(test_dataset_full))

    # Resample subsets for this epoch
    train_indices = torch.randperm(len(train_dataset_full))[:train_subset_count]
    test_indices = torch.randperm(len(test_dataset_full))[:test_subset_count]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

    print(f"Training subset size: {len(train_dataset)} patches")
    print(f"Validation subset size: {len(val_dataset)} patches")
    
    # Create model
    print("\nInitializing model...")
    model = SingleScaleDecoder(
        in_channels=TOKEN_DIM,
        num_classes=NUM_CLASSES,
        input_size=OUTPUT_SIZE
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")
    
    # Loss function and optimizer
    criterion = CEDiceLoss(num_classes=NUM_CLASSES)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    
    # Create run-specific checkpoint directory with args in name
    run_name = f"{base_run_name}_bs{BATCH_SIZE}_ep{EPOCHS}_lr{LEARNING_RATE}"
    checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Resample subsets for this epoch
        train_indices = torch.randperm(len(train_dataset_full))[:train_subset_count]
        test_indices = torch.randperm(len(test_dataset_full))[:test_subset_count]
        
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
        val_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS 
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_checkpoint_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
            
        # Save model after each epoch
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, epoch_checkpoint_path)
        print(f"Saved epoch {epoch+1} checkpoint.")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Plot and save losses
    plot_losses(train_losses, val_losses, run_name, results_dir)
    
    # Evaluate segmentation metrics on validation set
    class_names = ['background', 'normal_tissue', 'tumor']
    metrics = evaluate_metrics(
        model=model,
        dataloader=val_loader,
        num_classes=NUM_CLASSES,
        device=device,
        ignore_index=0
    )
    
    # Print metrics
    print_metrics(metrics, class_names=class_names)

if __name__ == '__main__':
    main()