"""Training and fine-tuning routine for the In Silico Virtual Staining Generator."""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.virtual_staining_cgan import VirtualStainingGenerator, VirtualStainingLoss
from src.data.dataset import get_ooc_dataloaders
from src.utils.metrics import compute_translation_metrics


def train_virtual_staining_model(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    save_path: str = "synaptwin-ooc/weights/best_virtual_staining_model.pt",
    device: str = "cpu",
) -> None:
    """Trains the virtual staining generator and saves model checkpoint."""
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[*] Training Virtual Staining Generator on {device}...")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    train_loader, val_loader = get_ooc_dataloaders(
        train_size=60,
        val_size=15,
        batch_size=batch_size,
    )

    model = VirtualStainingGenerator(in_channels=1, out_channels=3, base_channels=32).to(device)
    criterion = VirtualStainingLoss(lambda_l1=1.0, lambda_ssim=0.6, lambda_grad=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_t = time.time()

        for batch in train_loader:
            bf = batch["brightfield"].to(device)
            fl_target = batch["fluorescence"].to(device)

            optimizer.zero_grad()
            pred = model(bf)
            loss, breakdown = criterion(pred, fl_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bf.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bf = batch["brightfield"].to(device)
                fl_target = batch["fluorescence"].to(device)
                pred = model(bf)
                v_loss, _ = criterion(pred, fl_target)
                val_running_loss += v_loss.item() * bf.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        elapsed = time.time() - start_t

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Checkpoint saved to {save_path} (Val Loss: {val_loss:.4f})")

    print("[✓] Model training completed successfully.")


if __name__ == "__main__":
    train_virtual_staining_model(epochs=3, batch_size=4)
