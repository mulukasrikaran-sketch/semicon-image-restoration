# AI-Based Restoration of Degraded Images

## Overview

This solution restores degraded grayscale images using a PyTorch-based
deep learning model trained on paired degraded and ground-truth images.

The model accepts degraded `.npy` image arrays and produces restored
`.npy` grayscale arrays at the target resolution.

## Requirements

- Python 3.10+
- NVIDIA GPU recommended
- PyTorch
- NumPy

Install dependencies:

    pip install -r requirements.txt

## Submission Structure

    team_name/
    ├── run.py
    ├── requirements.txt
    ├── README.md
    └── models/
        └── restore_net.pt

## Input Format

The input directory must contain degraded grayscale images stored as
NumPy `.npy` files.

Example:

    input/
    ├── 000000.npy
    ├── 000001.npy
    └── ...

Each input is expected to be a 2D grayscale array.

## Output Format

For every input `.npy` file, one restored `.npy` file is generated with
exactly the same filename.

The restored output:

- is a 2D grayscale NumPy array
- has target resolution (256, 256)
- uses float32
- contains values in the range [0, 1]
- contains no NaN or Inf values

## Execution

Run the solution using:

    python run.py <input-dir> <output-dir>

Example:

    python run.py ./input ./output

The output directory is created automatically if it does not exist.

## Model

The trained model weights are included locally at:

    models/restore_net.pt

No internet connection, API key, additional model download, or manual
configuration is required during inference.

## Inference Pipeline

The inference process:

1. Loads each `.npy` degraded image.
2. Applies percentile-based normalization.
3. Runs the trained restoration network.
4. Produces the restored target-resolution image.
5. Clips output values to [0, 1].
6. Validates the output for NaN and Inf values.
7. Saves the result using the original filename.

## Offline Execution

All required model weights are included in the `models/` directory.
The inference script does not download external models or require
internet access.

## Hardware

The solution automatically uses an NVIDIA CUDA GPU when available.
If CUDA is unavailable, it falls back to CPU execution.
