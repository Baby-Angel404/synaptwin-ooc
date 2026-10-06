"""Visualization utilities for multi-channel fluorescence and morphometry overlays."""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Confocal fluorescence colormaps (Black background -> Spectral emission)
CMAP_DAPI = LinearSegmentedColormap.from_list("dapi", ["#050811", "#1E90FF", "#E0F7FA"])
CMAP_TUBULIN = LinearSegmentedColormap.from_list("tubulin", ["#051108", "#00FF7F", "#E8F8F5"])
CMAP_SYNAPTO = LinearSegmentedColormap.from_list("synapto", ["#140505", "#FF4500", "#FDEDEC"])


def create_false_color_composite(fluorescence: np.ndarray) -> np.ndarray:
    """Blends 3 fluorescence channels into an RGB false-color image.

    Args:
        fluorescence: (3, H, W) numpy array [DAPI, Tubulin, Synaptophysin] in [0, 1]

    Returns:
        rgb: (H, W, 3) numpy array in [0, 1]
    """
    dapi = np.clip(fluorescence[0], 0.0, 1.0)
    tubulin = np.clip(fluorescence[1], 0.0, 1.0)
    synapto = np.clip(fluorescence[2], 0.0, 1.0)

    h, w = dapi.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Additive spectral blending
    rgb[:, :, 0] = synapto * 1.0 + dapi * 0.12
    rgb[:, :, 1] = tubulin * 1.0 + synapto * 0.20 + dapi * 0.40
    rgb[:, :, 2] = dapi * 1.0 + tubulin * 0.25

    return np.clip(rgb, 0.0, 1.0)


def plot_pipeline_comparison(
    brightfield: np.ndarray,
    ground_truth_fl: Optional[np.ndarray],
    predicted_fl: np.ndarray,
    morphometry: Dict[str, float],
    toxicity_report: Dict[str, any],
    output_path: str = "pipeline_result.png",
) -> None:
    """Generates a publication-quality multi-panel visualization of the pipeline."""
    fig = plt.figure(figsize=(18, 10), facecolor="#0e1117")

    # Panel 1: Input Bright-Field
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(brightfield, cmap="gray")
    ax1.set_title("Input: Label-Free Bright-Field", color="white", fontsize=12, pad=10)
    ax1.axis("off")

    # Panel 2: Predicted Multi-Channel Composite
    pred_rgb = create_false_color_composite(predicted_fl)
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(pred_rgb)
    ax2.set_title("In Silico Virtual Staining (RGB)", color="#00FFCC", fontsize=12, pad=10)
    ax2.axis("off")

    # Panel 3: Virtual Channel 0 - DAPI
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(predicted_fl[0], cmap=CMAP_DAPI, vmin=0.0, vmax=1.0)
    ax3.set_title("Ch 0: DAPI (Nuclei/Soma)", color="#4DA6FF", fontsize=11, pad=10)
    ax3.axis("off")

    # Panel 4: Virtual Channel 1 - Beta-III Tubulin
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(predicted_fl[1], cmap=CMAP_TUBULIN, vmin=0.0, vmax=1.0)
    ax4.set_title("Ch 1: Beta-III Tubulin (Axons)", color="#66FF66", fontsize=11, pad=10)
    ax4.axis("off")

    # Panel 5: Virtual Channel 2 - Synaptophysin
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(predicted_fl[2], cmap=CMAP_SYNAPTO, vmin=0.0, vmax=1.0)
    ax5.set_title("Ch 2: Synaptophysin (Puncta)", color="#FF6666", fontsize=11, pad=10)
    ax5.axis("off")

    # Panel 6: Ground Truth Comparison (if available)
    ax6 = fig.add_subplot(2, 4, 6)
    if ground_truth_fl is not None:
        gt_rgb = create_false_color_composite(ground_truth_fl)
        ax6.imshow(gt_rgb)
        ax6.set_title("Physical Fluorescence GT", color="#FFCC00", fontsize=11, pad=10)
    else:
        ax6.text(0.5, 0.5, "No Physical Stain (100% In Silico)", color="gray", ha="center", va="center")
    ax6.axis("off")

    # Panel 7: Morphometry Radar / Bar Summary
    ax7 = fig.add_subplot(2, 4, 7)
    metrics_keys = [
        "Axon Penetration",
        "Axon Guidance",
        "Connectivity",
        "Integrity (1-Frag)",
    ]
    vals = [
        morphometry.get("axon_penetration_ratio", 0.0) / 0.5,
        morphometry.get("axon_guidance_index", 1.0) / 3.0,
        morphometry.get("circuit_connectivity_score", 0.0) / 100.0,
        1.0 - morphometry.get("cytoskeletal_fragmentation_index", 0.0),
    ]
    ax7.set_facecolor("#161b22")
    bars = ax7.barh(metrics_keys, np.clip(vals, 0.0, 1.0), color=["#1E90FF", "#00FF7F", "#9370DB", "#FF8C00"])
    ax7.set_xlim(0, 1.2)
    ax7.set_title("Morphometry Telemetry", color="white", fontsize=11, pad=10)
    ax7.tick_params(colors="white")
    ax7.grid(True, linestyle="--", alpha=0.3, color="gray")

    # Panel 8: Toxicity Screening Scorecard
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.set_facecolor("#161b22")
    risk_color = toxicity_report.get("risk_color", "#28A745")
    pred_cls = toxicity_report.get("predicted_class", "Safe")
    viab = toxicity_report.get("viability_index", 1.0) * 100.0
    comp_name = toxicity_report.get("compound_name", "Tested Agent")
    conc = toxicity_report.get("tested_concentration_um", 0.0)

    scorecard_text = (
        f"Compound: {comp_name}\n"
        f"Tested Dose: {conc:.2f} µM\n\n"
        f"Toxicity Tier:\n"
        f"  ► {pred_cls}\n\n"
        f"Viability: {viab:.1f}%\n"
        f"Estimated IC50: {toxicity_report.get('estimated_ic50', 'N/A')}\n"
        f"Confidence: {toxicity_report.get('confidence_score', 0.0)*100:.1f}%\n\n"
        f"Mechanisms:\n"
    )
    for m in toxicity_report.get("identified_mechanisms", [])[:2]:
        scorecard_text += f"• {m}\n"

    ax8.text(
        0.05, 0.95, scorecard_text,
        color="white", fontsize=10, verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor=risk_color, linewidth=2)
    )
    ax8.set_title("AI Screening Verdict", color="white", fontsize=11, pad=10)
    ax8.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
