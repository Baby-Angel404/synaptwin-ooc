"""Unified End-to-End Neural Organ-on-a-Chip AI Digital Twin Pipeline.

Orchestrates:
1. Label-free Bright-Field preprocessing
2. In Silico Virtual Fluorescence synthesis (cGAN / ResUNet)
3. Quantitative microfluidic morphometry & axon guidance extraction
4. Deep neurotoxicity & dose-response prediction
5. Clinical diagnostic telemetry & multi-panel reporting
"""

import json
import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from src.models.virtual_staining_cgan import VirtualStainingGenerator
from src.models.morphometry_profiler import NeuralMorphometryProfiler
from src.models.neurotoxicity_engine import NeurotoxicityScreeningEngine
from src.utils.visualization import plot_pipeline_comparison, create_false_color_composite
from src.utils.metrics import compute_translation_metrics


class SynapTwinPipeline:
    """End-to-End Organ-on-a-Chip AI Screening Suite."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Initialize Generative Model
        self.generator = VirtualStainingGenerator(
            in_channels=1,
            out_channels=3,
            base_channels=32,
        ).to(self.device)

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.generator.load_state_dict(state_dict)
        self.generator.eval()

        # 2. Initialize Analyzers
        self.profiler = NeuralMorphometryProfiler(pixel_size_um=0.65)
        self.toxicity_engine = NeurotoxicityScreeningEngine()

    def process_micrograph(
        self,
        brightfield: np.ndarray,
        compound_name: str = "Tested Compound",
        concentration_um: float = 1.0,
        ground_truth_fl: Optional[np.ndarray] = None,
        save_plot_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """Runs the complete 5-stage automated screening workflow on a micrograph.

        Args:
            brightfield: (H, W) or (1, H, W) numpy array in range [0, 1]
            compound_name: Drug/chemical name under investigation
            concentration_um: Tested dosage in micromolar (µM)
            ground_truth_fl: Optional (3, H, W) physical fluorescence ground truth
            save_plot_path: Optional path to save diagnostic figure

        Returns:
            Structured dictionary with virtual images, morphometry, toxicity, and timing.
        """
        start_time = time.time()

        # Preprocessing
        if brightfield.ndim == 3 and brightfield.shape[0] == 1:
            bf_2d = brightfield[0]
        else:
            bf_2d = brightfield
        bf_2d = np.clip(bf_2d, 0.0, 1.0).astype(np.float32)

        # Stage 1: In Silico Virtual Fluorescence Synthesis
        bf_tensor = torch.from_numpy(bf_2d).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_tensor = self.generator(bf_tensor)
        predicted_fl = pred_tensor.squeeze(0).cpu().numpy()  # Shape: (3, H, W)

        # Stage 2: Quantitative Microfluidic Morphometry
        morphometry = self.profiler.analyze_micrograph(predicted_fl)

        # Stage 3: Deep Neurotoxicity & Clinical Risk Scoring
        toxicity_report = self.toxicity_engine.predict_toxicity(
            morphometry=morphometry,
            compound_name=compound_name,
            tested_conc_um=concentration_um,
        )

        # Stage 4: Quantitative Metrics (if ground truth physical stain provided)
        translation_metrics = None
        if ground_truth_fl is not None:
            translation_metrics = compute_translation_metrics(predicted_fl, ground_truth_fl)

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Stage 5: Optional Multi-panel Diagnostic Render
        if save_plot_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
            plot_pipeline_comparison(
                brightfield=bf_2d,
                ground_truth_fl=ground_truth_fl,
                predicted_fl=predicted_fl,
                morphometry=morphometry,
                toxicity_report=toxicity_report,
                output_path=save_plot_path,
            )

        report = {
            "metadata": {
                "system": "SynapTwin-OoC AI BioLab Suite v1.0",
                "compound_name": compound_name,
                "concentration_um": float(concentration_um),
                "device": str(self.device),
                "latency_ms": round(elapsed_ms, 2),
            },
            "morphometry": morphometry,
            "toxicity_screening": toxicity_report,
            "translation_metrics": translation_metrics,
            "virtual_fluorescence_channels": {
                "Ch0_DAPI_mean": float(np.mean(predicted_fl[0])),
                "Ch1_Tubulin_mean": float(np.mean(predicted_fl[1])),
                "Ch2_Synaptophysin_mean": float(np.mean(predicted_fl[2])),
            },
        }

        return {
            "report": report,
            "predicted_fluorescence": predicted_fl,
            "brightfield": bf_2d,
        }
