"""Configuration constants and biomedical parameters for SynapTwin-OoC."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MicrofluidicSpecs:
    """Microfluidic neural organ-on-a-chip physical specifications."""
    CHIP_TYPE: str = "Asymmetric Microfluidic Axon-Diode Chip"
    SOMA_CHAMBER_WIDTH_UM: float = 200.0
    MICROCHANNEL_LENGTH_UM: float = 500.0
    MICROCHANNEL_WIDTH_UM: float = 5.0
    MICROCHANNEL_HEIGHT_UM: float = 3.0
    AXONAL_CHAMBER_WIDTH_UM: float = 200.0
    CHANNEL_PITCH_UM: float = 25.0
    OPTICAL_MAGNIFICATION: float = 20.0
    PIXEL_SIZE_UM: float = 0.65  # 20x objective on standard sCMOS sensor


@dataclass(frozen=True)
class OpticalChannels:
    """Spectral imaging channel configurations."""
    INPUT_NAME: str = "Bright-field Phase Contrast (Label-Free)"
    INPUT_WAVELENGTH_NM: int = 650
    
    # Target in silico virtual staining channels
    CHANNELS: Tuple[str, ...] = (
        "DAPI (Nuclei / Soma Chromatin)",
        "Beta-III Tubulin (Microtubule Cytoskeleton)",
        "Synaptophysin (Presynaptic Puncta & Axon Terminals)",
    )
    WAVELENGTHS_NM: Tuple[int, ...] = (461, 509, 594)
    COLOR_HEX: Tuple[str, ...] = ("#1E90FF", "#00FF7F", "#FF4500")


# Standard reference pharmaceutical compound library for neurotoxicity profiling
REFERENCE_COMPOUNDS: Dict[str, Dict] = {
    "Vehicle (DMSO 0.1%)": {
        "class": "Negative Control",
        "expected_toxicity": "Safe",
        "ic50_um": float("inf"),
        "mechanism": "Non-toxic baseline medium",
    },
    "BDNF (10 ng/mL)": {
        "class": "Neurotrophic Factor",
        "expected_toxicity": "Neuroprotective / Growth Promoting",
        "ic50_um": float("inf"),
        "mechanism": "TrkB agonist, enhances axon guidance & arborization",
    },
    "Paclitaxel (Taxol)": {
        "class": "Chemotherapy / Microtubule Stabilizer",
        "expected_toxicity": "Moderate Neurotoxicity",
        "ic50_um": 2.4,
        "mechanism": "Axonal microtubule hyperstabilization and distal die-back",
    },
    "Cisplatin": {
        "class": "Chemotherapy / Platinum Agent",
        "expected_toxicity": "Severe Neurotoxicity",
        "ic50_um": 5.8,
        "mechanism": "DNA cross-linking, soma apoptosis, extensive neurite beading",
    },
    "Rotenone": {
        "class": "Mitochondrial Complex I Inhibitor",
        "expected_toxicity": "Severe Neurotoxicity",
        "ic50_um": 0.35,
        "mechanism": "Oxidative stress, dopaminergic axonal fragmentation",
    },
    "Glutamate (High Dose)": {
        "class": "Excitotoxin",
        "expected_toxicity": "Moderate-to-Severe Excitotoxicity",
        "ic50_um": 45.0,
        "mechanism": "NMDA receptor hyperactivation, calcium overload, soma swelling",
    },
}

IMAGE_SIZE = (512, 512)
DEVICE_PREFERENCE = "cuda"  # falls back to cpu automatically
