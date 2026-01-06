import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        features = batch['features'].to(device)  # [B, 512]
        masks = batch['mask'].to(device)  # [B, 512, 512]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)  # [B, num_classes, 512, 512]
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for batch in pbar:
            features = batch['features'].to(device)  # [B, 512]
            masks = batch['mask'].to(device)  # [B, 512, 512]
            
            # Forward pass
            outputs = model(features)  # [B, num_classes, 512, 512]
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss