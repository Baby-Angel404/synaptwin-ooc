"""Synthetic neural organ-on-a-chip micrograph simulator.

Generates biologically faithful paired datasets of label-free Bright-Field
micrographs and 3-channel ground-truth fluorescence for training, testing,
and benchmark verification of in silico staining models.
"""

import math
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


class OoCMicrographSynthesizer:
    """Simulates multi-chamber microfluidic organ-on-a-chip optical microscopy."""

    def __init__(self, size: Tuple[int, int] = (512, 512), seed: int | None = None):
        self.height, self.width = size
        self.rng = np.random.default_rng(seed)

    def generate_microfluidic_chip_sample(
        self,
        compound_name: str = "Vehicle (DMSO 0.1%)",
        concentration_um: float = 0.0,
        num_somas: int = 35,
    ) -> Dict[str, np.ndarray]:
        """Generates a paired brightfield and 3-channel fluorescence micrograph.

        Returns:
            dict containing:
                'brightfield': np.ndarray of shape (H, W) in range [0, 1]
                'fluorescence': np.ndarray of shape (3, H, W) in range [0, 1]
                    Ch0: DAPI (Blue)
                    Ch1: Beta-III Tubulin (Green)
                    Ch2: Synaptophysin (Red)
                'masks': segmentation masks for somas, microchannels, axons
                'metadata': experimental parameters
        """
        h, w = self.height, self.width
        
        # Microfluidic spatial layout:
        # 0 <= x < 0.3 * w: Somatic Chamber (Cell seeding zone)
        # 0.3 * w <= x < 0.65 * w: Microchannel Barrier (Parallel micro-grooves)
        # 0.65 * w <= x < w: Axonal Chamber (Target destination)
        soma_boundary = int(0.30 * w)
        barrier_end = int(0.65 * w)
        
        # Initialize canvas
        dapi = np.zeros((h, w), dtype=np.float32)
        tubulin = np.zeros((h, w), dtype=np.float32)
        synaptophysin = np.zeros((h, w), dtype=np.float32)
        brightfield = np.ones((h, w), dtype=np.float32) * 0.55  # Neutral phase background
        
        # Microchannel geometry
        microchannel_ys = np.arange(25, h - 25, 24)
        channel_mask = np.zeros((h, w), dtype=np.float32)
        for y_ch in microchannel_ys:
            channel_mask[y_ch - 2 : y_ch + 3, soma_boundary:barrier_end] = 1.0

        # Toxicity degradation parameters based on compound & dose
        deg_factor = self._calculate_degradation(compound_name, concentration_um)
        axon_reach_modifier = max(0.05, 1.0 - 0.9 * deg_factor)
        beading_prob = min(0.95, 0.05 + 0.9 * deg_factor)
        soma_viability = max(0.1, 1.0 - 0.85 * deg_factor)

        # 1. Generate Cell Somas in the somatic chamber
        soma_positions: List[Tuple[float, float, float]] = []
        for _ in range(num_somas):
            sx = self.rng.uniform(15, soma_boundary - 15)
            sy = self.rng.uniform(20, h - 20)
            radius = self.rng.uniform(6.0, 11.0)
            soma_positions.append((sx, sy, radius))

            # DAPI nucleus signal
            yy, xx = np.ogrid[:h, :w]
            dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
            nuc_mask = dist_sq <= (radius * 0.7) ** 2
            chromatin_texture = self.rng.uniform(0.8, 1.2, size=(h, w)).astype(np.float32)
            dapi[nuc_mask] = np.maximum(
                dapi[nuc_mask],
                (1.0 - np.sqrt(dist_sq[nuc_mask]) / (radius * 0.7)) * chromatin_texture[nuc_mask] * soma_viability
            )

            # Tubulin in soma cytoplasm
            soma_mask = dist_sq <= radius ** 2
            tubulin[soma_mask] = np.maximum(
                tubulin[soma_mask],
                (1.0 - np.sqrt(dist_sq[soma_mask]) / radius) * 0.95 * soma_viability
            )

            # Brightfield phase-contrast optical signature (refractive halo + dark cytoplasm)
            halo_mask = (dist_sq <= (radius * 1.35) ** 2) & (dist_sq > (radius * 0.8) ** 2)
            core_mask = dist_sq <= (radius * 0.8) ** 2
            brightfield[halo_mask] += 0.22 * (1.0 - deg_factor * 0.3)
            brightfield[core_mask] -= 0.28 * soma_viability

        # 2. Generate Axon outgrowth and guidance through microchannels
        for sx, sy, radius in soma_positions:
            # Find closest microchannel
            closest_y = min(microchannel_ys, key=lambda y: abs(y - sy))
            
            # Draw neurite trajectory from soma toward microchannel entry
            curr_x, curr_y = sx, sy
            steps_to_ch = int(abs(soma_boundary - curr_x) / 1.5)
            for _ in range(steps_to_ch):
                curr_x += 1.5
                curr_y += (closest_y - curr_y) * 0.08 + self.rng.normal(0, 0.5)
                ix, iy = int(round(curr_x)), int(round(curr_y))
                if 0 <= ix < w and 0 <= iy < h:
                    tubulin[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2)] = np.maximum(
                        tubulin[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2)],
                        0.75 * soma_viability
                    )
                    brightfield[iy, ix] -= 0.15 * soma_viability

            # Axon enters microchannel and travels across barrier
            max_axon_x = soma_boundary + int((w - soma_boundary) * axon_reach_modifier * self.rng.uniform(0.7, 1.1))
            max_axon_x = min(w - 5, max_axon_x)

            for ax_x in range(soma_boundary, max_axon_x, 2):
                ax_y = int(round(closest_y + self.rng.normal(0, 0.4)))
                if 0 <= ax_x < w and 0 <= ax_y < h:
                    is_beaded = self.rng.random() < beading_prob
                    if not is_beaded or (ax_x % 6 < 2):
                        tub_val = 0.85 * (1.0 - deg_factor * 0.4)
                        tubulin[max(0, ax_y - 1) : min(h, ax_y + 2), max(0, ax_x - 1) : min(w, ax_x + 2)] = np.maximum(
                            tubulin[max(0, ax_y - 1) : min(h, ax_y + 2), max(0, ax_x - 1) : min(w, ax_x + 2)],
                            tub_val
                        )
                        brightfield[ax_y, ax_x] -= 0.12

                    # Synaptophysin puncta (synaptic vesicles / growth cone)
                    if ax_x >= barrier_end and not is_beaded and self.rng.random() < 0.25:
                        synaptophysin[max(0, ax_y - 2) : min(h, ax_y + 3), max(0, ax_x - 2) : min(w, ax_x + 3)] = np.maximum(
                            synaptophysin[max(0, ax_y - 2) : min(h, ax_y + 3), max(0, ax_x - 2) : min(w, ax_x + 3)],
                            self.rng.uniform(0.6, 1.0) * (1.0 - deg_factor)
                        )

            # Growth cone terminal in axonal chamber
            if max_axon_x >= barrier_end:
                gc_x = max_axon_x
                gc_y = closest_y
                if 0 <= gc_x < w and 0 <= gc_y < h:
                    synaptophysin[max(0, gc_y - 3) : min(h, gc_y + 4), max(0, gc_x - 3) : min(w, gc_x + 4)] = np.maximum(
                        synaptophysin[max(0, gc_y - 3) : min(h, gc_y + 4), max(0, gc_x - 3) : min(w, gc_x + 4)],
                        0.9 * (1.0 - deg_factor)
                    )

        # 3. Add PDMS microchannel barrier optical features to brightfield
        # PDMS microfluidic walls have strong phase boundaries
        brightfield[:, soma_boundary - 2 : soma_boundary + 3] -= 0.35
        brightfield[:, barrier_end - 2 : barrier_end + 3] -= 0.35
        for y_ch in microchannel_ys:
            brightfield[y_ch - 3, soma_boundary:barrier_end] += 0.15
            brightfield[y_ch + 3, soma_boundary:barrier_end] += 0.15
            brightfield[y_ch - 2 : y_ch + 3, soma_boundary:barrier_end] -= 0.08

        # 4. Realistic optical noise, blur, and contrast normalization
        # Phase-contrast low-frequency illumination gradient
        illumination_grad = np.linspace(0.95, 1.05, w)[np.newaxis, :]
        brightfield *= illumination_grad
        brightfield += self.rng.normal(0, 0.02, size=(h, w))
        brightfield = np.clip(brightfield, 0.0, 1.0)

        # Fluorescence optical point spread function (PSF) simulation
        dapi = gaussian_filter(dapi, sigma=1.0)
        tubulin = gaussian_filter(tubulin, sigma=0.8)
        synaptophysin = gaussian_filter(synaptophysin, sigma=0.6)

        # Background autofluorescence & sensor shot noise
        dapi += self.rng.normal(0, 0.015, size=(h, w))
        tubulin += self.rng.normal(0, 0.02, size=(h, w))
        synaptophysin += self.rng.normal(0, 0.015, size=(h, w))

        dapi = np.clip(dapi, 0.0, 1.0)
        tubulin = np.clip(tubulin, 0.0, 1.0)
        synaptophysin = np.clip(synaptophysin, 0.0, 1.0)

        fluorescence = np.stack([dapi, tubulin, synaptophysin], axis=0)  # Shape (3, H, W)

        metadata = {
            "compound_name": compound_name,
            "concentration_um": float(concentration_um),
            "degradation_factor": float(deg_factor),
            "num_somas": len(soma_positions),
            "microchannel_count": len(microchannel_ys),
            "dimensions": (h, w),
        }

        return {
            "brightfield": brightfield,
            "fluorescence": fluorescence,
            "channel_mask": channel_mask,
            "metadata": metadata,
        }

    def _calculate_degradation(self, compound_name: str, conc_um: float) -> float:
        """Calculates biological degradation index in [0, 1] using Hill equation."""
        if compound_name in ["Vehicle (DMSO 0.1%)", "BDNF (10 ng/mL)"] or conc_um <= 0:
            return 0.0
        
        # Hill coefficients
        hill_params = {
            "Paclitaxel (Taxol)": (2.4, 2.0),
            "Cisplatin": (5.8, 1.8),
            "Rotenone": (0.35, 2.5),
            "Glutamate (High Dose)": (45.0, 1.5),
        }
        ic50, hill_n = hill_params.get(compound_name, (5.0, 1.5))
        
        # Emax Hill equation: E = conc^n / (IC50^n + conc^n)
        deg = (conc_um ** hill_n) / (ic50 ** hill_n + conc_um ** hill_n)
        return float(np.clip(deg, 0.0, 1.0))
