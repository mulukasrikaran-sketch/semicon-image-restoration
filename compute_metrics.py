"""
compute_metrics.py
-------------------
After running eval.py, use THIS script to compute the SSIM and PSNR scores
you need for Slide 6 (Results) of your PPT. It compares your restored
images against the real ground truth images.

HOW TO RUN:
  pip install scikit-image --break-system-packages
  python compute_metrics.py --restored_dir ./outputs --gt_dir ./dataset/ground_truth
"""

import argparse
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restored_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    args = parser.parse_args()

    restored_files = set(os.listdir(args.restored_dir))
    gt_files = set(os.listdir(args.gt_dir))
    matched = sorted(restored_files & gt_files)

    if not matched:
        print("No matching filenames found between restored and ground truth folders.")
        return

    ssim_scores, psnr_scores = [], []

    for fname in matched:
        restored = np.array(Image.open(os.path.join(args.restored_dir, fname)).convert("L"))
        gt = np.array(Image.open(os.path.join(args.gt_dir, fname)).convert("L"))

        # If sizes don't match exactly, resize restored to match GT for comparison.
        if restored.shape != gt.shape:
            restored = np.array(Image.fromarray(restored).resize((gt.shape[1], gt.shape[0])))

        s = ssim(gt, restored, data_range=255)
        p = psnr(gt, restored, data_range=255)
        ssim_scores.append(s)
        psnr_scores.append(p)

    print(f"Evaluated {len(matched)} images.")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print("\n(Put these two numbers directly on Slide 6 of your PPT.)")


if __name__ == "__main__":
    main()
