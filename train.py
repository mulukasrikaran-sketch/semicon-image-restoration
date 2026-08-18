"""
train.py
--------
This is the script that actually TRAINS the model.

WHAT TRAINING MEANS (in plain words):
  1. Show the model a degraded image.
  2. Model guesses a restored image.
  3. Compare the guess to the real ground truth using our loss function.
  4. Nudge the model's internal numbers (weights) slightly so next time
     its guess is a little closer to correct.
  5. Repeat this thousands of times (each full pass over the dataset = 1 "epoch").

HOW TO RUN:
  python train.py

WHERE TO RUN IT:
  Use Google Colab or Kaggle with a free GPU (T4). On CPU this will be very
  slow. In Colab: Runtime -> Change runtime type -> GPU.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from model import RestoreNet
from losses import RestorationLoss
from dataset import RestorationDataset

# ---------------- SETTINGS YOU CAN TWEAK ----------------
BATCH_SIZE = 8
EPOCHS = 30            # increase if you have time left, decrease if you don't
LEARNING_RATE = 2e-4
CHECKPOINT_PATH = "./checkpoints/restore_net.pt"
LAST_CHECKPOINT_PATH = "./checkpoints/last_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------


def main():
    print(f"Using device: {DEVICE}")

    # 1. Load data and split into train/validation (90% train, 10% to check progress)
    full_dataset = RestorationDataset(augment=False)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Build the model, loss, and optimizer
    model = RestoreNet().to(DEVICE)
    criterion = RestorationLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    start_epoch = 1

    # Resume automatically if a previous checkpoint exists
    if os.path.exists(LAST_CHECKPOINT_PATH):
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming from epoch {start_epoch}")

    # 3. Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0

        for deg, gt in train_loader:
            deg, gt = deg.to(DEVICE), gt.to(DEVICE)

            optimizer.zero_grad()
            pred = model(deg)

            # Safety check: if model output size doesn't match GT (rounding issues
            # with odd input sizes), crop to the smaller common size.
            if pred.shape != gt.shape:
                h = min(pred.shape[-2], gt.shape[-2])
                w = min(pred.shape[-1], gt.shape[-1])
                pred = pred[..., :h, :w]
                gt = gt[..., :h, :w]

            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        # 4. Check performance on validation set (data the model hasn't trained on)
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for deg, gt in val_loader:
                deg, gt = deg.to(DEVICE), gt.to(DEVICE)
                pred = model(deg)
                if pred.shape != gt.shape:
                    h = min(pred.shape[-2], gt.shape[-2])
                    w = min(pred.shape[-1], gt.shape[-1])
                    pred = pred[..., :h, :w]
                    gt = gt[..., :h, :w]
                val_loss_total += criterion(pred, gt).item()
        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch {epoch}/{EPOCHS} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        # 5. Save the model every time it improves (so you always have your best version)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved new best model (val_loss={avg_val_loss:.4f}) to {CHECKPOINT_PATH}")

        # Save latest training state after EVERY epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, LAST_CHECKPOINT_PATH)

        print(f"  -> Saved latest checkpoint to {LAST_CHECKPOINT_PATH}")

    print("Training complete. Best model saved at:", CHECKPOINT_PATH)


if __name__ == "__main__":
    import os
    os.makedirs("./checkpoints", exist_ok=True)
    main()
