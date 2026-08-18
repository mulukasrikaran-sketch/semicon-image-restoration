"""
model.py
--------
This file defines our restoration network: RestoreNet.

WHAT IT DOES (in plain words):
- Takes in a small, noisy image (the "degraded" image).
- Passes it through an encoder (shrinks the image, learns what's noise vs. real detail).
- Passes it through a decoder (rebuilds the image, removing noise).
- Finally, upsamples the image 2x (makes it bigger) while adding back sharp detail.
- Outputs a clean, full-size image.

Think of it like: squint at a blurry photo (encoder), understand what's really
there (bottleneck), then redraw it cleanly and bigger (decoder + upsample).
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    A 'residual block': it looks at an image, tries to compute a small
    improvement, and adds that improvement back onto the original.
    This is easier for the network to learn than redrawing the whole
    image from scratch every time.
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(x + out)  # add the improvement back onto the input


class RestoreNet(nn.Module):
    """
    Full restoration network.

    Input:  degraded grayscale image, shape (Batch, 1, H, W)
    Output: restored grayscale image, shape (Batch, 1, 2H, 2W)
            (2x bigger, because the challenge always asks for 2x super-resolution,
             e.g. 128x128 -> 256x256, or 256x256 -> 512x512)
    """
    def __init__(self, base_ch=48):
        super().__init__()

        # STEM: first layer, just converts the 1-channel image into
        # "base_ch" feature channels the network can work with.
        self.stem = nn.Conv2d(1, base_ch, 3, padding=1)

        # ENCODER: shrink the image twice, doubling channels each time.
        # Shrinking forces the network to summarize "what's really there"
        # instead of just copying noise.
        self.enc1 = nn.Sequential(ResBlock(base_ch), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)   # H -> H/2

        self.enc2 = nn.Sequential(ResBlock(base_ch * 2), ResBlock(base_ch * 2))
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)  # H/2 -> H/4

        self.bottleneck = nn.Sequential(ResBlock(base_ch * 4), ResBlock(base_ch * 4))

        # DECODER: grow the image back up, using skip connections
        # (adding back the encoder's features) so fine details aren't lost.
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)  # H/4 -> H/2
        self.dec2 = nn.Sequential(ResBlock(base_ch * 2), ResBlock(base_ch * 2))

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)  # H/2 -> H
        self.dec1 = nn.Sequential(ResBlock(base_ch), ResBlock(base_ch))

        # SUPER-RESOLUTION HEAD: this is what makes the output image 2x BIGGER
        # than the input. PixelShuffle is a fast, artifact-light way to upsample.
        self.sr_conv = nn.Conv2d(base_ch, base_ch * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)  # turns (base_ch*4, H, W) into (base_ch, 2H, 2W)
        self.refine = nn.Sequential(ResBlock(base_ch), nn.Conv2d(base_ch, 1, 3, padding=1))

    def forward(self, x):
        s = self.stem(x)

        e1 = self.enc1(s)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b) + e2      # skip connection: add back encoder detail
        de2 = self.dec2(u2)

        u1 = self.up1(de2) + e1    # skip connection: add back encoder detail
        de1 = self.dec1(u1)

        sr = self.pixel_shuffle(self.sr_conv(de1))   # now image is 2x bigger
        out = self.refine(sr)

        # sigmoid squashes output to [0, 1], matching our normalized image range
        return torch.sigmoid(out)


if __name__ == "__main__":
    # Quick sanity check: run a fake image through the model and print shapes.
    model = RestoreNet()
    fake_input = torch.randn(2, 1, 128, 128)   # batch of 2, 1 channel, 128x128
    out = model(fake_input)
    print("Input shape: ", fake_input.shape)
    print("Output shape:", out.shape)   # should be [2, 1, 256, 256]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
