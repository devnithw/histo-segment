import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    """
    UNet-style decoder that upsamples features to generate segmentation masks.
    Takes spatial feature maps and progressively upsamples them.
    """
    def __init__(self, in_channels, num_classes, up_channels=(256, 128, 64, 32)):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for ch in up_channels:
            layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(prev_channels, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = ch

        self.decoder = nn.Sequential(*layers)

        self.segmentation_head = nn.Conv2d(
            prev_channels, num_classes, kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Segmentation logits [B, num_classes, H_out, W_out]
        """
        x = self.decoder(x)
        x = self.segmentation_head(x)
        return x


class ModaSegNet(nn.Module):
    """
    Complete segmentation model that takes CONCH features and outputs segmentation masks.
    
    Pipeline:
    1. Reshape 512-dim feature vector to spatial format (512, 1, 1)
    2. Upsample through decoder to target resolution
    3. Apply final interpolation to ensure exact output size
    """
    def __init__(self, feature_dim=512, num_classes=4, output_size=(512, 512)):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.output_size = output_size
        
        # Initial projection to create spatial features
        # We'll start with a 1x1 spatial map and upsample from there
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
        )
        
        # Initial conv to expand spatial dimensions
        # Start from 1x1 and upsample to 32x32 (5 upsample steps: 1->2->4->8->16->32)
        # Then decoder will upsample 32x32 to 512x512 (4 more steps: 32->64->128->256->512)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: 4 upsampling blocks (32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512)
        self.decoder = UNetDecoder(
            in_channels=512,
            num_classes=num_classes,
            up_channels=(256, 128, 64, 32)
        )

    def forward(self, x):
        """
        Args:
            x: CONCH features [B, 512]
        Returns:
            Segmentation logits [B, num_classes, 512, 512]
        """
        batch_size = x.shape[0]
        
        # Project features
        x = self.feature_projection(x)  # [B, 512]
        
        # Reshape to spatial format [B, 512, 1, 1]
        x = x.view(batch_size, self.feature_dim, 1, 1)
        
        # Apply initial conv
        x = self.initial_conv(x)  # [B, 512, 1, 1]
        
        # Upsample to 32x32 before decoder
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)  # [B, 512, 32, 32]
        
        # Pass through decoder (32x32 -> 512x512)
        seg_logits = self.decoder(x)  # [B, num_classes, 512, 512]
        
        # Ensure output matches exact target resolution
        if seg_logits.shape[2:] != self.output_size:
            seg_logits = F.interpolate(
                seg_logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )

        return seg_logits


if __name__ == '__main__':
    # Test the model
    model = ModaSegNet(feature_dim=512, num_classes=4, output_size=(512, 512))
    
    # Test with dummy input
    dummy_features = torch.randn(2, 512)  # Batch of 2
    output = model(dummy_features)
    
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [2, 4, 512, 512]")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


