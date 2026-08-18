
import os
import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset folders
GT_DIR = "./dataset/train/GT"
DEG_DIR = "./dataset/train/NoisyLR"


class RestorationDataset(Dataset):

    def __init__(self, gt_dir=GT_DIR, deg_dir=DEG_DIR, augment=False):
        self.gt_dir = gt_dir
        self.deg_dir = deg_dir
        self.augment = augment

        # Find files present in both folders
        gt_files = set(os.listdir(gt_dir))
        deg_files = set(os.listdir(deg_dir))

        self.filenames = sorted(gt_files & deg_files)

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No matching .npy files found between "
                f"{gt_dir} and {deg_dir}"
            )

        print(f"Loaded {len(self.filenames)} matched image pairs.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        fname = self.filenames[idx]

        # Load ground truth .npy
        gt_array = np.load(
            os.path.join(self.gt_dir, fname)
        ).astype(np.float32)

        # Load degraded/noisy .npy
        deg_array = np.load(
            os.path.join(self.deg_dir, fname)
        ).astype(np.float32)

        # Convert to PyTorch tensors
        # (H,W) -> (1,H,W)
        gt_tensor = torch.from_numpy(gt_array).unsqueeze(0)
        deg_tensor = torch.from_numpy(deg_array).unsqueeze(0)

        # No augmentation
        return deg_tensor, gt_tensor


if __name__ == "__main__":

    ds = RestorationDataset(augment=False)

    deg, gt = ds[0]

    print("Degraded shape:", deg.shape)
    print("Ground truth shape:", gt.shape)

    print("Degraded dtype:", deg.dtype)
    print("Ground truth dtype:", gt.dtype)

    print("Degraded min:", deg.min().item())
    print("Degraded max:", deg.max().item())

    print("GT min:", gt.min().item())
    print("GT max:", gt.max().item())
