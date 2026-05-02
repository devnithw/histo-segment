import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleDecoder(nn.Module):
    """
    Minimal single-scale decoder from:
    'A Benchmark of Foundation Model Encoders for Histopathological Image Segmentation'
    """

    def __init__(
        self,
        in_channels: int,     # d
        num_classes: int,     # C
        input_size: tuple,    # (H, W)
        head_dim: int = 256,  # D
    ):
        super().__init__()

        self.input_size = input_size
        self.head_dim = head_dim

        # Width projection
        self.proj = nn.Conv2d(in_channels, head_dim, kernel_size=1)

        # Upsample blocks
        self.upsample_blocks = nn.ModuleList()

        # we build blocks lazily after seeing first input
        self._built = False

        # Class logits
        self.classifier = nn.Conv2d(head_dim, num_classes, kernel_size=1)

    def _build_upsample_blocks(self, h, w, device=None):
        H, W = self.input_size

        s_h = math.ceil(math.log2(H / h))
        s_w = math.ceil(math.log2(W / w))
        s = max(s_h, s_w)

        for _ in range(s):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            if device is not None:
                block = block.to(device)
            self.upsample_blocks.append(block)

        self._built = True

    def forward(self, x):
        """
        Args:
            x: feature map tensor [B, d, h, w] loaded from .pt

        Returns:
            logits: [B, C, H, W]
        """
        B, d, h, w = x.shape

        # Build upsampling path once (based on encoder feature size)
        if not self._built:
            self._build_upsample_blocks(h, w, device=x.device)

        # Width projection
        x = self.proj(x)  # [B, D, h, w]

        # Progressive upsampling
        for block in self.upsample_blocks:
            x = block(x)

        # Class logits
        x = self.classifier(x)  # [B, C, h', w']

        # Final resize if needed
        if x.shape[2:] != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        return x


if __name__ == '__main__':
    # Test the model
    model = SingleScaleDecoder(in_channels=768, num_classes=2, input_size=(512, 512))
    
    # Test with dummy input
    dummy = torch.randn(2, 768, 16, 16)  # Batch of 2, 768 channels, 16x16 spatial
    output = model(dummy)
    
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [2, 2, 512, 512]")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")