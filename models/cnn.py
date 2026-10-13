"""
cnn.py
======

Teaching notes
--------------
RFI detection, as set up in this workshop, is a **pixel-level binary
segmentation** problem: given an amplitude patch of shape
(H, W), predict a mask of the same shape (H, W) where each pixel is
"clean" (0) or "RFI" (1).

A natural, well-known architecture for this kind of "image in, mask out"
task is a small **U-Net**: an encoder that progressively downsamples the
image while extracting features, and a decoder that progressively
upsamples back to the original resolution, with "skip connections" that
copy fine-grained detail from the encoder directly to the decoder (this
is what lets U-Nets produce sharp, precise masks instead of blurry ones).

This implementation is intentionally SMALL and SIMPLE:
- Only two downsampling / upsampling stages (real U-Nets often have 4-5).
- Plain Conv2d + BatchNorm + ReLU blocks, no fancy tricks.
- Meant to train quickly on a laptop/Colab GPU (or even CPU, slowly) so
  students can focus on understanding the pipeline rather than waiting
  hours for training.

If you want to extend this later (e.g. for the "real" second workshop
project), obvious next steps include: more encoder/decoder stages,
residual connections, attention gates, or trying a different backbone
entirely.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two 3x3 convolutions, each followed by BatchNorm and ReLU.

    This is the basic repeating building block of the U-Net: it lets the
    network learn local patterns (like "a sudden bright vertical stripe")
    while keeping the spatial resolution unchanged.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (feature maps) produced by this block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleRFIUNet(nn.Module):
    """
    A small U-Net for pixel-wise RFI segmentation.

    Input
    -----
    A batch of amplitude patches, shape (batch_size, in_channels, H, W).
    ``in_channels`` is normally 1 (amplitude only), but the architecture
    supports more (e.g. if a student wants to stack amplitude + phase as
    a 2-channel input - a nice extension exercise!).

    Output
    ------
    A batch of raw (pre-sigmoid) logits, shape (batch_size, 1, H, W).
    Apply ``torch.sigmoid`` to get RFI probabilities in [0, 1], and
    threshold (e.g. at 0.5) to get a binary predicted mask.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    base_channels : int, default 16
        Number of feature maps produced by the first conv block. Deeper
        stages double this (base_channels -> 2x -> 4x), which is the
        standard U-Net pattern: as spatial resolution decreases, channel
        depth increases to compensate.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # --- Encoder (downsampling path) ---
        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)  # H, W -> H/2, W/2

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)  # H/2, W/2 -> H/4, W/4

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(c2, c3)

        # --- Decoder (upsampling path) ---
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)  # H/4 -> H/2
        # After upsampling, we concatenate with the encoder's skip
        # connection (enc2 output), so the decoder block's input channels
        # are c2 (upsampled) + c2 (skip) = 2 * c2.
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)  # H/2 -> H
        self.dec1 = ConvBlock(c1 * 2, c1)

        # Final 1x1 convolution maps features -> a single-channel logit
        # map, one value per pixel.
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(
                f"SimpleRFIUNet requires both spatial dimensions to be divisible "
                f"by 4 (two 2x2 max-pooling stages), but got input of shape "
                f"(H={height}, W={width}). This usually means the patch_size / "
                f"patch_size_freq used to build the dataset (see "
                f"configs/dataset.yaml) isn't divisible by 4 - e.g. 16, 32, 64, "
                f"128, 256 all work; 22 or 100 do not."
            )

        # Encoder, saving outputs for skip connections.
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        # Decoder, concatenating skip connections along the channel axis.
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.out_conv(d1)
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model (for curiosity)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
