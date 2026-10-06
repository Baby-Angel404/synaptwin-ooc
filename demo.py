"""SynapTwin-OoC: Standalone Single-Command Execution Script (demo.py).

Performs end-to-end inference on neural organ-on-a-chip bright-field micrographs:
1. Virtual Multi-Channel Staining (DAPI, Beta-III Tubulin, Synaptophysin)
2. Asymmetric Axon Guidance Morphometry
3. Pharmacological Neurotoxicity Screening
4. Generates multi-panel diagnostic figure and JSON report.
"""

import argparse
import json
import os
import sys
import numpy as np

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import SynapTwinPipeline
from src.data.ooc_synthesizer import OoCMicrographSynthesizer
from src.config import REFERENCE_COMPOUNDS


def main():
    parser = argparse.ArgumentParser(description="SynapTwin-OoC Single-Command Demo")
    parser.add_argument(
        "--compound",
        type=str,
        default="Paclitaxel (Taxol)",
        choices=list(REFERENCE_COMPOUNDS.keys()),
        help="Pharmaceutical compound to simulate",
    )
    parser.add_argument(
        "--dose",
        type=float,
        default=3.5,
        help="Dose concentration in micromolar (µM)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save diagnostic figure and JSON",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("==========================================================================")
    print("      SYNAPTWIN-OOC: NEURAL ORGAN-ON-A-CHIP AI DIGITAL TWIN SUITE         ")
    print("==========================================================================")
    print(f"[*] Testing Compound : {args.compound}")
    print(f"[*] Applied Dose     : {args.dose:.2f} µM")

    # 1. Acquire Microfluidic Micrograph
    print("\n[Stage 1] Synthesizing Microfluidic Micrograph Frame...")
    synthesizer = OoCMicrographSynthesizer(size=(512, 512), seed=777)
    sample = synthesizer.generate_microfluidic_chip_sample(
        compound_name=args.compound,
        concentration_um=args.dose,
    )
    bf = sample["brightfield"]
    gt_fl = sample["fluorescence"]
    print("  ✓ Acquired 512x512 label-free bright-field frame with microchannel barrier.")

    # 2. Run Unified Pipeline
    print("\n[Stage 2] Running In Silico Virtual Staining & Phenotyping Pipeline...")
    pipeline = SynapTwinPipeline(device="cpu")
    plot_file = os.path.join(args.output_dir, "demo_pipeline_diagnostic.png")
    result = pipeline.process_micrograph(
        brightfield=bf,
        compound_name=args.compound,
        concentration_um=args.dose,
        ground_truth_fl=gt_fl,
        save_plot_path=plot_file,
    )
    report = result["report"]

    # 3. Print Results
    morph = report["morphometry"]
    tox = report["toxicity_screening"]
    t_met = report["translation_metrics"]

    print("\n[Stage 3] Quantitative In Silico Translation Fidelity:")
    print(f"  • Mean SSIM                      : {t_met['Mean_SSIM']:.4f}")
    print(f"  • Mean PSNR                      : {t_met['Mean_PSNR']:.2f} dB")
    print(f"  • Pearson Correlation (PCC)      : {t_met['Mean_PCC']:.4f}")

    print("\n[Stage 4] Microfluidic Morphometry & Axon Guidance:")
    print(f"  • Somatic Chamber Cell Count     : {morph['soma_count']} somas")
    print(f"  • Total Neurite Length           : {morph['total_neurite_length_um']:.1f} µm")
    print(f"  • Axon Chamber Outgrowth         : {morph['axon_chamber_outgrowth_um']:.1f} µm")
    print(f"  • Axon Penetration Ratio         : {morph['axon_penetration_ratio']:.3f}")
    print(f"  • Axon Guidance Alignment Index  : {morph['axon_guidance_index']:.3f}")
    print(f"  • Cytoskeletal Fragmentation     : {morph['cytoskeletal_fragmentation_index']:.3f}")
    print(f"  • Circuit Connectivity Score     : {morph['circuit_connectivity_score']:.1f} / 100")

    print("\n[Stage 5] Pharmacological Neurotoxicity Verdict:")
    print(f"  • Safety Classification          : {tox['predicted_class']} (Grade {tox['toxicity_grade']})")
    print(f"  • Estimated Cellular Viability   : {tox['viability_index']*100:.1f}%")
    print(f"  • Screening Confidence           : {tox['confidence_score']*100:.1f}%")
    print(f"  • Estimated IC50                 : {tox['estimated_ic50']}")
    print(f"  • Identified Pathological Signs  : {', '.join(tox['identified_mechanisms'])}")

    # Save JSON report
    json_path = os.path.join(args.output_dir, "demo_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n==========================================================================")
    print(f"  ✓ Diagnostic Plot saved to : {plot_file}")
    print(f"  ✓ JSON Telemetry saved to  : {json_path}")
    print(f"  ✓ Total Execution Time     : {report['metadata']['latency_ms']:.1f} ms")
    print("==========================================================================")


if __name__ == "__main__":
    main()
