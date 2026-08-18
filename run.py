
import os
import sys
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# MODEL ARCHITECTURE
# Exact architecture used during training
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(x + out)


class RestoreNet(nn.Module):

    def __init__(self, base_ch=48):
        super().__init__()

        self.stem = nn.Conv2d(1, base_ch, 3, padding=1)

        self.enc1 = nn.Sequential(
            ResBlock(base_ch),
            ResBlock(base_ch)
        )

        self.down1 = nn.Conv2d(
            base_ch,
            base_ch * 2,
            4,
            stride=2,
            padding=1
        )

        self.enc2 = nn.Sequential(
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2)
        )

        self.down2 = nn.Conv2d(
            base_ch * 2,
            base_ch * 4,
            4,
            stride=2,
            padding=1
        )

        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 4),
            ResBlock(base_ch * 4)
        )

        self.up2 = nn.ConvTranspose2d(
            base_ch * 4,
            base_ch * 2,
            4,
            stride=2,
            padding=1
        )

        self.dec2 = nn.Sequential(
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2)
        )

        self.up1 = nn.ConvTranspose2d(
            base_ch * 2,
            base_ch,
            4,
            stride=2,
            padding=1
        )

        self.dec1 = nn.Sequential(
            ResBlock(base_ch),
            ResBlock(base_ch)
        )

        self.sr_conv = nn.Conv2d(
            base_ch,
            base_ch * 4,
            3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.refine = nn.Sequential(
            ResBlock(base_ch),
            nn.Conv2d(base_ch, 1, 3, padding=1)
        )

    def forward(self, x):

        s = self.stem(x)

        e1 = self.enc1(s)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b) + e2
        de2 = self.dec2(u2)

        u1 = self.up1(de2) + e1
        de1 = self.dec1(u1)

        sr = self.pixel_shuffle(
            self.sr_conv(de1)
        )

        out = self.refine(sr)

        return torch.sigmoid(out)


# ============================================================
# PREPROCESSING
# Same normalization used during training
# ============================================================

def percentile_clip_normalize(img_array):

    lo, hi = np.percentile(
        img_array,
        [1, 99]
    )

    if hi <= lo:
        lo = img_array.min()
        hi = img_array.max() + 1e-6

    img_clipped = np.clip(
        img_array,
        lo,
        hi
    )

    img_norm = (
        img_clipped - lo
    ) / (
        hi - lo + 1e-6
    )

    return img_norm.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    # Required KLA command:
    # python run.py <input-dir> <output-dir>

    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(
            f"ERROR: Input directory does not exist: {input_dir}"
        )
        sys.exit(1)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # Model path is relative to run.py
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        base_dir,
        "models",
        "restore_net.pt"
    )

    if not os.path.isfile(model_path):
        print(
            f"ERROR: Model weights not found: {model_path}"
        )
        sys.exit(1)

    # Load exact trained architecture
    model = RestoreNet().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    input_files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    )

    print(
        "Found",
        len(input_files),
        ".npy files."
    )

    with torch.no_grad():

        for i, fname in enumerate(input_files):

            input_path = os.path.join(
                input_dir,
                fname
            )

            output_path = os.path.join(
                output_dir,
                fname
            )

            # Load input
            img = np.load(
                input_path
            ).astype(np.float32)

            # Require grayscale 2D input
            if img.ndim != 2:
                raise RuntimeError(
                    f"Input {fname} has invalid shape "
                    f"{img.shape}; expected (H, W)."
                )

            # Same preprocessing as training
            img_norm = percentile_clip_normalize(
                img
            )

            # (H,W) -> (1,1,H,W)
            tensor = (
                torch.from_numpy(img_norm)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # Inference
            output = model(tensor)

            # Remove batch and channel dimensions
            restored = (
                output
                .squeeze()
                .cpu()
                .numpy()
            )

            # Required range
            restored = np.clip(
                restored,
                0.0,
                1.0
            ).astype(np.float32)

            # Validate output
            if restored.ndim != 2:
                raise RuntimeError(
                    f"Output {fname} has invalid shape "
                    f"{restored.shape}."
                )

            if not np.isfinite(restored).all():
                raise RuntimeError(
                    f"NaN or Inf detected in output {fname}."
                )

            if restored.min() < 0.0 or restored.max() > 1.0:
                raise RuntimeError(
                    f"Output {fname} is outside [0,1]."
                )

            # Save same filename
            np.save(
                output_path,
                restored
            )

            if (
                (i + 1) % 50 == 0
                or i == 0
            ):
                print(
                    f"{i + 1}/{len(input_files)} restored"
                )

    print("Restoration complete.")
    print(
        "Outputs written to:",
        output_dir
    )


if __name__ == "__main__":
    main()
