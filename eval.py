"""
eval.py
-------
!!! THIS IS THE MOST IMPORTANT FILE IN YOUR SUBMISSION !!!

KLA's benchmarking team will run this EXACT script, AS-IS, on their own
test images, on an H100 GPU, to score your model. It must run with ZERO
manual edits by them.

WHAT IT DOES:
  1. Takes a folder of degraded test images.
  2. Loads your trained model.
  3. Runs each image through the model to restore it.
  4. Saves every restored image into an output folder.
  5. Prints how long inference took (they care about speed).

HOW TO RUN IT (exactly how KLA will run it):
  python eval.py --input_dir /path/to/test_images --output_dir /path/to/restored_output

TEST THIS BEFORE SUBMITTING:
  Run this exact command on a fresh machine (or a fresh Colab runtime) to
  make sure it works with no hidden dependencies on your personal setup.
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from model import RestoreNet

CHECKPOINT_PATH = "./checkpoints/restore_net.pt"


def percentile_clip_normalize(img_array):
    """Same preprocessing used during training - must match exactly."""
    lo, hi = np.percentile(img_array, [1, 99])
    if hi <= lo:
        lo, hi = img_array.min(), img_array.max() + 1e-6
    img_clipped = np.clip(img_array, lo, hi)
    img_norm = (img_clipped - lo) / (hi - lo + 1e-6)
    return img_norm.astype(np.float32), lo, hi


def denormalize(img_norm, lo, hi):
    """Convert model output (0-1 range) back to a normal 0-255 image to save."""
    img = img_norm * (hi - lo) + lo
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Folder containing degraded test images")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Folder where restored images will be saved")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained model
    model = RestoreNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])
    print(f"Found {len(image_files)} images to restore.")

    total_time = 0.0

    with torch.no_grad():
        for fname in image_files:
            img = Image.open(os.path.join(args.input_dir, fname)).convert("L")
            img_arr = np.array(img, dtype=np.float32)
            norm_arr, lo, hi = percentile_clip_normalize(img_arr)

            tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

            start = time.time()
            output = model(tensor)
            if device == "cuda":
                torch.cuda.synchronize()  # make sure GPU finished before we stop the timer
            elapsed = time.time() - start
            total_time += elapsed

            output_arr = output.squeeze().cpu().numpy()
            restored_img = denormalize(output_arr, lo, hi)

            Image.fromarray(restored_img).save(os.path.join(args.output_dir, fname))

    if len(image_files) > 0:
        avg_time = total_time / len(image_files)
        print(f"Restored {len(image_files)} images.")
        print(f"Average inference time per image: {avg_time*1000:.1f} ms")
    else:
        print("No images found in input_dir - check the path.")


if __name__ == "__main__":
    main()
