"""Single-Cell Morphometry & Microfluidic Axon Guidance Profiler.

Extracts biological metrics from virtual fluorescence channels:
- Soma count, area, and roundness (DAPI)
- Total neurite length, microchannel penetration ratio, and fragmentation index (Beta-III Tubulin)
- Synaptic puncta density (Synaptophysin)
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import label, center_of_mass, binary_erosion, binary_dilation
from skimage.morphology import skeletonize


class NeuralMorphometryProfiler:
    """Quantitative neural morphometry analyzer for organ-on-a-chip microfluidics."""

    def __init__(self, pixel_size_um: float = 0.65):
        self.pixel_size_um = pixel_size_um

    def analyze_micrograph(
        self,
        fluorescence: np.ndarray,
        soma_chamber_ratio: float = 0.30,
        axonal_chamber_ratio: float = 0.65,
    ) -> Dict[str, float]:
        """Analyzes 3-channel virtual fluorescence micrograph.

        Args:
            fluorescence: (3, H, W) numpy array [DAPI, Tubulin, Synaptophysin]
            soma_chamber_ratio: X-fraction for somatic chamber boundary
            axonal_chamber_ratio: X-fraction for axonal target chamber entry

        Returns:
            Dictionary of quantitative biomedical features.
        """
        dapi = np.clip(fluorescence[0], 0.0, 1.0)
        tubulin = np.clip(fluorescence[1], 0.0, 1.0)
        synapto = np.clip(fluorescence[2], 0.0, 1.0)

        h, w = dapi.shape
        soma_boundary = int(soma_chamber_ratio * w)
        axonal_boundary = int(axonal_chamber_ratio * w)

        # 1. Soma Morphometry via DAPI channel
        dapi_thresh = dapi > 0.25
        labeled_somas, num_somas = label(dapi_thresh)

        soma_areas = []
        soma_circularities = []
        for i in range(1, num_somas + 1):
            mask = labeled_somas == i
            area_px = np.sum(mask)
            if area_px > 15:  # Filter noise specks
                area_um2 = area_px * (self.pixel_size_um ** 2)
                soma_areas.append(area_um2)
                # Perimeter estimate via border erosion
                eroded = binary_erosion(mask)
                perim_px = max(1, area_px - np.sum(eroded))
                circ = (4 * np.pi * area_px) / (perim_px ** 2)
                soma_circularities.append(min(1.0, circ))

        valid_somas = len(soma_areas)
        mean_soma_area = float(np.mean(soma_areas)) if soma_areas else 0.0
        mean_soma_roundness = float(np.mean(soma_circularities)) if soma_circularities else 0.0

        # 2. Neurite & Axon Guidance Analysis via Tubulin channel
        tubulin_thresh = tubulin > 0.20
        skeleton = skeletonize(tubulin_thresh)

        # Regional neurite lengths (in micrometers)
        soma_chamber_neurites = np.sum(skeleton[:, :soma_boundary]) * self.pixel_size_um
        microchannel_neurites = np.sum(skeleton[:, soma_boundary:axonal_boundary]) * self.pixel_size_um
        axonal_chamber_neurites = np.sum(skeleton[:, axonal_boundary:]) * self.pixel_size_um
        total_neurite_length_um = float(soma_chamber_neurites + microchannel_neurites + axonal_chamber_neurites)

        # Axon Penetration Ratio into target chamber (Key OoC Functional Metric)
        axon_penetration_ratio = (
            float(axonal_chamber_neurites / (total_neurite_length_um + 1e-6))
        )

        # Directional Axon Guidance Alignment (Horizontal outgrowth vector across microchannels)
        grad_x = np.abs(tubulin[:, 1:] - tubulin[:, :-1])
        grad_y = np.abs(tubulin[1:, :] - tubulin[:-1, :])
        mean_grad_x = float(np.mean(grad_x[:, soma_boundary:axonal_boundary]))
        mean_grad_y = float(np.mean(grad_y[:, soma_boundary:axonal_boundary]))
        # Guidance index: ratio of longitudinal to transverse gradient
        axon_guidance_index = float(mean_grad_x / (mean_grad_y + 1e-6))
        axon_guidance_index = float(np.clip(axon_guidance_index, 0.5, 3.5))

        # 3. Cytoskeletal Fragmentation / Beading Index (Hallmark of Neurotoxicity)
        # Small isolated tubulin fragments indicate blebbing
        labeled_tub, num_tub_components = label(tubulin_thresh)
        small_fragments = 0
        total_tub_mass = 0
        for comp_id in range(1, num_tub_components + 1):
            comp_mask = labeled_tub == comp_id
            sz = np.sum(comp_mask)
            total_tub_mass += sz
            if sz < 25:  # Isolated bead / fragment
                small_fragments += sz

        fragmentation_index = float(small_fragments / (total_tub_mass + 1e-6))
        fragmentation_index = float(np.clip(fragmentation_index, 0.0, 1.0))

        # 4. Synaptic Puncta Density via Synaptophysin channel
        synapto_thresh = synapto > 0.35
        labeled_syn, num_syn_puncta = label(synapto_thresh)
        synaptic_puncta_density = float(num_syn_puncta / (valid_somas + 1e-3))

        # Overall Axon-to-Soma Connectivity Score [0, 100]
        connectivity_score = float(
            np.clip(
                (0.4 * (axon_penetration_ratio / 0.35) +
                 0.3 * (1.0 - fragmentation_index) +
                 0.3 * min(1.0, synaptic_puncta_density / 8.0)) * 100.0,
                0.0,
                100.0
            )
        )

        return {
            "soma_count": int(valid_somas),
            "mean_soma_area_um2": round(mean_soma_area, 2),
            "soma_roundness_index": round(mean_soma_roundness, 3),
            "total_neurite_length_um": round(total_neurite_length_um, 2),
            "axon_chamber_outgrowth_um": round(axonal_chamber_neurites, 2),
            "axon_penetration_ratio": round(axon_penetration_ratio, 3),
            "axon_guidance_index": round(axon_guidance_index, 3),
            "cytoskeletal_fragmentation_index": round(fragmentation_index, 3),
            "synaptic_puncta_count": int(num_syn_puncta),
            "synaptic_puncta_per_soma": round(synaptic_puncta_density, 2),
            "circuit_connectivity_score": round(connectivity_score, 1),
        }
