"""PyTorch Dataset and DataLoader wrappers for Organ-on-a-Chip datasets."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.ooc_synthesizer import OoCMicrographSynthesizer
from src.config import REFERENCE_COMPOUNDS


class OoCDataset(Dataset):
    """Dataset for paired Bright-Field and Virtual Fluorescence OoC images."""

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (512, 512),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.synthesizer = OoCMicrographSynthesizer(size=image_size, seed=seed)
        self.compounds = list(REFERENCE_COMPOUNDS.keys())
        
        # Pre-generate sample specifications
        self.samples_meta: List[Tuple[str, float]] = []
        rng = np.random.default_rng(seed)
        for _ in range(num_samples):
            comp = rng.choice(self.compounds)
            if comp in ["Vehicle (DMSO 0.1%)", "BDNF (10 ng/mL)"]:
                conc = 0.0
            else:
                conc = float(rng.uniform(0.1, 15.0))
            self.samples_meta.append((comp, conc))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        comp, conc = self.samples_meta[idx]
        sample = self.synthesizer.generate_microfluidic_chip_sample(
            compound_name=comp,
            concentration_um=conc,
        )

        bf = sample["brightfield"]  # (H, W)
        fl = sample["fluorescence"]  # (3, H, W)

        # Convert to torch tensor
        bf_tensor = torch.from_numpy(bf).unsqueeze(0).float()  # (1, H, W)
        fl_tensor = torch.from_numpy(fl).float()  # (3, H, W)

        deg_factor = torch.tensor(sample["metadata"]["degradation_factor"], dtype=torch.float32)

        return {
            "brightfield": bf_tensor,
            "fluorescence": fl_tensor,
            "degradation_factor": deg_factor,
            "compound_name": comp,
            "concentration_um": torch.tensor(conc, dtype=torch.float32),
        }


def get_ooc_dataloaders(
    train_size: int = 80,
    val_size: int = 20,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (512, 512),
) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation dataloaders."""
    train_ds = OoCDataset(num_samples=train_size, image_size=image_size, seed=101)
    val_ds = OoCDataset(num_samples=val_size, image_size=image_size, seed=202)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
