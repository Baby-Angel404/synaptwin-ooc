"""Quantitative evaluation metrics for in silico fluorescence translation."""

import math
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_psnr(
    pred: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    return float(psnr_fn(target, pred, data_range=data_range))


def compute_ssim(
    pred: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> float:
    """Computes Structural Similarity Index (SSIM)."""
    if pred.ndim == 3 and pred.shape[0] == 3:
        # Multi-channel image (3, H, W)
        ssims = [
            ssim_fn(target[c], pred[c], data_range=data_range)
            for c in range(3)
        ]
        return float(np.mean(ssims))
    return float(ssim_fn(target, pred, data_range=data_range))


def compute_pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes Pearson Correlation Coefficient between flattened arrays."""
    p_flat = pred.flatten()
    t_flat = target.flatten()
    if np.std(p_flat) < 1e-7 or np.std(t_flat) < 1e-7:
        return 0.0
    corr = np.corrcoef(p_flat, t_flat)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


def compute_translation_metrics(
    pred: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    """Computes full suite of image translation metrics across all channels."""
    # Ensure inputs are normalized in [0, 1]
    p_norm = np.clip(pred, 0.0, 1.0)
    t_norm = np.clip(target, 0.0, 1.0)

    channel_names = ["DAPI", "Tubulin", "Synaptophysin"]
    metrics: Dict[str, float] = {}

    psnr_vals = []
    ssim_vals = []
    pcc_vals = []

    for c in range(3):
        p_c = p_norm[c]
        t_c = t_norm[c]

        psnr_c = compute_psnr(p_c, t_c)
        ssim_c = compute_ssim(p_c, t_c)
        pcc_c = compute_pearson_correlation(p_c, t_c)
        mae_c = float(np.mean(np.abs(p_c - t_c)))

        c_name = channel_names[c]
        metrics[f"{c_name}_PSNR"] = psnr_c
        metrics[f"{c_name}_SSIM"] = ssim_c
        metrics[f"{c_name}_PCC"] = pcc_c
        metrics[f"{c_name}_MAE"] = mae_c

        psnr_vals.append(psnr_c)
        ssim_vals.append(ssim_c)
        pcc_vals.append(pcc_c)

    metrics["Mean_PSNR"] = float(np.mean(psnr_vals))
    metrics["Mean_SSIM"] = float(np.mean(ssim_vals))
    metrics["Mean_PCC"] = float(np.mean(pcc_vals))

    return metrics
