import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = GradScaler('cuda')

    pbar = tqdm(dataloader, desc='Training', leave=False, dynamic_ncols=True, mininterval=1.0)
    for batch in pbar:
        tokens = batch['tokens'].to(device)   # [B, 768, 16, 16]
        masks = batch['mask'].to(device)       # [B, 512, 512]

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=(device.type == 'cuda')):
            outputs = model(tokens)            # [B, num_classes, 512, 512]
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False, dynamic_ncols=True, mininterval=1.0)
        for batch in pbar:
            tokens = batch['tokens'].to(device)   # [B, 768, 16, 16]
            masks = batch['mask'].to(device)       # [B, 512, 512]

            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                outputs = model(tokens)            # [B, num_classes, 512, 512]
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches