import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from dataset import SICAPMultiSlideDataset, get_available_slide_ids
from model import ModaSegNet
from engine import train, validate
from utils import plot_losses, evaluate_metrics, print_metrics

# Set start time and run name
dt = datetime.now().strftime("%m-%d-%H-%M")
run_name = f"run_{dt}"

def main():
    # Configuration
    features_dir = '/home/nadun/wd/datasets/SICAP-test/features'
    masks_dir = '/home/nadun/wd/datasets/SICAP-test/masks'
    checkpoint_dir = '/home/nadun/wd/segmentation/checkpoints'
    results_dir = '/home/nadun/wd/segmentation/results'
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 1
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 4
    FEATURE_DIM = 512
    OUTPUT_SIZE = (512, 512)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all available slide IDs
    slide_ids = get_available_slide_ids(features_dir)
    print(f"Total slides: {len(slide_ids)}")
    
    # Create datasets
    print("\nCreating dataset...")
    dataset = SICAPMultiSlideDataset(
        slide_ids=slide_ids,
        features_dir=features_dir,
        masks_dir=masks_dir
    )
    print(f"Total dataset size: {len(dataset)} patches")
    
    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set size: {len(train_dataset)} patches")
    print(f"Validation set size: {len(val_dataset)} patches")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1
    )
    
    # Create model
    print("\nInitializing model...")
    model = ModaSegNet(
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        output_size=OUTPUT_SIZE
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
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
            checkpoint_path = os.path.join(checkpoint_dir, f'{run_name}_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Plot and save losses
    plot_losses(train_losses, val_losses, run_name, results_dir)
    
    # Evaluate segmentation metrics on validation set
    class_names = ['NC (Class 0)', 'G3 (Class 1)', 'G4 (Class 2)', 'G5 (Class 3)']
    metrics = evaluate_metrics(
        model=model,
        dataloader=val_loader,
        num_classes=NUM_CLASSES,
        device=device
    )
    
    # Print metrics
    print_metrics(metrics, class_names=class_names)


if __name__ == '__main__':
    main()
