import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0, ignore_index: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C, H, W] - raw logits from model
            targets: [B, H, W] - class indices (0, 1, 2, 3)

        Returns:
            dice loss (scalar)
        """
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Apply softmax to logits to get probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Compute Dice coefficient per class
        dims = (0, 2, 3)  # Sum over batch, height, width
        
        intersection = torch.sum(probs * targets_one_hot, dims)  # [C]
        union = torch.sum(probs, dims) + torch.sum(targets_one_hot, dims)  # [C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [C]

        # Return 1 - mean dice as loss
        loss = 1.0 - dice

        if self.ignore_index is not None:
            # only averaging classes that are not background
            valid_indices = [i for i in range(self.num_classes) if i != self.ignore_index]
            return loss[valid_indices].mean()
        return loss.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, ce_weight=1.0, dice_weight=1.0, ignore_index: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Set default weights based on num_classes (can be tuned later)
        if num_classes == 2:
            weights = torch.tensor([0.1, 1.0])
        elif num_classes == 3:
            weights = torch.tensor([0.1, 1.0, 5.0])
        elif num_classes == 4:
            weights = torch.tensor([0.1, 1.0, 3.0, 5.0])
        else:
            weights = torch.ones(num_classes)
            
        self.register_buffer('weights', weights)
        self.ce = nn.CrossEntropyLoss(weight=self.weights)

        self.dice = DiceLoss(num_classes=num_classes, ignore_index=self.ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C, H, W] - raw logits from model
            targets: [B, H, W] - class indices (0, 1, 2, 3)

        Returns:
            combined loss
        """
        ce_loss = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


if __name__ == '__main__':
    # Test parameters
    batch_size = 8
    num_classes = 3
    height, width = 64, 64
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes, height, width) 
    # Targets: [B, H, W] with class indices 0-3
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    print(f"\nInput shapes:")
    print(f"  Logits:  {logits.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Target value range: [{targets.min().item()}, {targets.max().item()}]")
    
    # Test DiceLoss
    print("\n" + "-" * 60)
    print("Testing DiceLoss")
    print("-" * 60)
    dice_loss_fn = DiceLoss(num_classes=num_classes, ignore_index=0)
    dice_loss = dice_loss_fn(logits, targets)
    print(f"Dice Loss: {dice_loss.item():.4f}")
    print(f"Loss requires grad: {dice_loss.requires_grad}")
    
    # Test CEDiceLoss
    print("\n" + "-" * 60)
    print("Testing CEDiceLoss (CrossEntropy + Dice)")
    print("-" * 60)
    combined_loss_fn = CEDiceLoss(num_classes=num_classes, ce_weight=1.0, dice_weight=1.0, ignore_index=0)
    combined_loss = combined_loss_fn(logits, targets)
    print(f"Combined Loss: {combined_loss.item():.4f}")
    
