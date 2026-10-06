"""Interactive Streamlit Web Dashboard: SynapTwin-OoC AI BioLab Suite.

Digital Twin platform for In Silico Virtual Fluorescence, Neural Morphometry,
and High-Content Neurotoxicity Screening on Organ-on-a-Chip.
"""

import io
import json
import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SynapTwinPipeline
from src.data.ooc_synthesizer import OoCMicrographSynthesizer
from src.config import REFERENCE_COMPOUNDS, MicrofluidicSpecs
from src.utils.visualization import create_false_color_composite

# Page Setup
st.set_page_config(
    page_title="SynapTwin-OoC | AI BioLab Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 12px; border-radius: 8px; border: 1px solid #30363d; }
    .badge-safe { background-color: #238636; color: white; padding: 6px 12px; border-radius: 16px; font-weight: bold; }
    .badge-warning { background-color: #d29922; color: white; padding: 6px 12px; border-radius: 16px; font-weight: bold; }
    .badge-danger { background-color: #da3633; color: white; padding: 6px 12px; border-radius: 16px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    return SynapTwinPipeline(device="cpu")


@st.cache_resource
def load_synthesizer():
    return OoCMicrographSynthesizer(size=(512, 512), seed=42)


pipeline = load_pipeline()
synthesizer = load_synthesizer()

# Header
st.title("🧠 SynapTwin-OoC: Neural Organ-on-a-Chip Digital Twin")
st.markdown(
    """
    **AI for Life Science Challenge** · 5th Pazhou Algorithm Competition · *Supported by CellShells Bioscience*  
    *Zero-staining in silico fluorescence, asymmetric axon guidance morphometry, and multi-scale drug neurotoxicity screening.*
    """
)

# Sidebar Controls
st.sidebar.header("🔬 Experimental Controls")
sample_preset = st.sidebar.selectbox(
    "Select Preloaded Chip Scenario:",
    list(REFERENCE_COMPOUNDS.keys()),
    index=0,
)

dose_slider = st.sidebar.slider(
    "Compound Concentration (µM):",
    min_value=0.0,
    max_value=25.0,
    value=2.5 if "Vehicle" not in sample_preset and "BDNF" not in sample_preset else 0.0,
    step=0.5,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Virtual Staining Channel Mix")
show_dapi = st.sidebar.checkbox("Ch 0: DAPI (Nuclei / Soma)", value=True)
show_tubulin = st.sidebar.checkbox("Ch 1: Beta-III Tubulin (Axons)", value=True)
show_synapto = st.sidebar.checkbox("Ch 2: Synaptophysin (Puncta)", value=True)

# Generate or load data
with st.spinner("Acquiring microfluidic optical frame..."):
    sample_data = synthesizer.generate_microfluidic_chip_sample(
        compound_name=sample_preset,
        concentration_um=dose_slider,
    )
    bf_img = sample_data["brightfield"]
    gt_fl = sample_data["fluorescence"]

# Run Pipeline
with st.spinner("Processing deep in silico inference & phenotyping..."):
    output = pipeline.process_micrograph(
        brightfield=bf_img,
        compound_name=sample_preset,
        concentration_um=dose_slider,
        ground_truth_fl=gt_fl,
    )
    report = output["report"]
    pred_fl = output["predicted_fluorescence"]

# Quick KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
tox_data = report["toxicity_screening"]
morph_data = report["morphometry"]

with col1:
    st.metric(
        "Toxicity Screening Verdict",
        tox_data["predicted_class"],
        delta=f"Grade {tox_data['toxicity_grade']}",
        delta_color="inverse" if tox_data['toxicity_grade'] > 0 else "normal",
    )

with col2:
    st.metric(
        "Estimated Viability",
        f"{tox_data['viability_index']*100:.1f}%",
        delta=f"Conf: {tox_data['confidence_score']*100:.0f}%",
    )

with col3:
    st.metric(
        "Axon Penetration Ratio",
        f"{morph_data['axon_penetration_ratio']*100:.1f}%",
        delta=f"Total: {morph_data['total_neurite_length_um']:.0f} µm",
    )

with col4:
    st.metric(
        "Inference Latency",
        f"{report['metadata']['latency_ms']:.1f} ms",
        delta="Sub-second (2200x speedup)",
    )

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 In Silico Virtual Staining Studio",
    "📏 Microfluidic Axon Morphometry",
    "💊 Deep Dose-Response & Toxicity Engine",
    "📋 Telemetry & LIMS Export",
])

with tab1:
    st.subheader("Zero-Staining Virtual Fluorescence Translation")
    st.markdown(
        "Deep cGAN translates non-phototoxic label-free phase contrast into calibrated multi-channel fluorescence in real time."
    )

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("**Input: Label-Free Bright-Field (Phase Contrast)**")
        st.image(bf_img, clamp=True, use_container_width=True)

    with tcol2:
        st.markdown("**Output: In Silico Virtual Multi-Channel Fluorescence**")
        # Custom channel weighting
        display_fl = pred_fl.copy()
        if not show_dapi: display_fl[0] *= 0.0
        if not show_tubulin: display_fl[1] *= 0.0
        if not show_synapto: display_fl[2] *= 0.0

        rgb_composite = create_false_color_composite(display_fl)
        st.image(rgb_composite, clamp=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Individual Calibrated Spectral Channels")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        st.caption("🔵 Ch 0: DAPI (461 nm) - Nuclei & Chromatin")
        st.image(pred_fl[0], clamp=True, use_container_width=True)
    with scol2:
        st.caption("🟢 Ch 1: Beta-III Tubulin (509 nm) - Cytoskeleton")
        st.image(pred_fl[1], clamp=True, use_container_width=True)
    with scol3:
        st.caption("🔴 Ch 2: Synaptophysin (594 nm) - Presynaptic Puncta")
        st.image(pred_fl[2], clamp=True, use_container_width=True)

with tab2:
    st.subheader("Microfluidic Axon Guidance & Morphometry Telemetry")
    st.markdown(
        "Quantitative measurements of axonal outgrowth across microchannels connecting the somatic and axonal chambers."
    )

    mcol1, mcol2 = st.columns([1, 1])
    with mcol1:
        st.dataframe(
            {
                "Biomarker Metric": [
                    "Soma Count",
                    "Mean Soma Area (µm²)",
                    "Soma Roundness Index",
                    "Total Neurite Length (µm)",
                    "Axon Chamber Outgrowth (µm)",
                    "Axon Penetration Ratio",
                    "Axon Guidance Alignment Index",
                    "Cytoskeletal Fragmentation Index",
                    "Synaptic Puncta Count",
                    "Overall Circuit Connectivity Score",
                ],
                "Measured Value": [
                    str(morph_data["soma_count"]),
                    f"{morph_data['mean_soma_area_um2']:.2f}",
                    f"{morph_data['soma_roundness_index']:.3f}",
                    f"{morph_data['total_neurite_length_um']:.1f}",
                    f"{morph_data['axon_chamber_outgrowth_um']:.1f}",
                    f"{morph_data['axon_penetration_ratio']:.3f}",
                    f"{morph_data['axon_guidance_index']:.3f}",
                    f"{morph_data['cytoskeletal_fragmentation_index']:.3f}",
                    str(morph_data["synaptic_puncta_count"]),
                    f"{morph_data['circuit_connectivity_score']:.1f} / 100",
                ],
            },
            use_container_width=True,
        )

    with mcol2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#161b22")
        keys = ["Axon Penetration", "Axon Guidance", "Connectivity", "Intactness"]
        values = [
            morph_data["axon_penetration_ratio"] / 0.5,
            morph_data["axon_guidance_index"] / 3.0,
            morph_data["circuit_connectivity_score"] / 100.0,
            1.0 - morph_data["cytoskeletal_fragmentation_index"],
        ]
        ax.set_facecolor("#161b22")
        ax.barh(keys, [min(1.0, v) for v in values], color=["#1E90FF", "#00FF7F", "#9370DB", "#FF8C00"])
        ax.set_xlim(0, 1.1)
        ax.tick_params(colors="white")
        ax.set_title("Organ-on-a-Chip Phenotypic Index", color="white")
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        st.pyplot(fig)

with tab3:
    st.subheader("Deep Dose-Response & Neurotoxicity Simulation")
    st.markdown(
        f"**Pharmacological Evaluation for {sample_preset} @ {dose_slider:.2f} µM**"
    )

    tcol1, tcol2 = st.columns([1, 1])
    with tcol1:
        st.write(f"**Predicted Safety Tier:** {tox_data['predicted_class']}")
        st.write(f"**Estimated IC50:** {tox_data['estimated_ic50']}")
        st.write("**Identified Mechanistic Phenotypes:**")
        for m in tox_data["identified_mechanisms"]:
            st.markdown(f"- ⚠️ {m}")

        st.write("**Class Probability Distribution:**")
        for k, v in tox_data["class_probabilities"].items():
            st.progress(v, text=f"{k}: {v*100:.1f}%")

    with tcol2:
        st.markdown("**Simulated 8-Point Hill Dose-Response Curve**")
        dose_sim = pipeline.toxicity_engine.simulate_dose_response_curve(
            sample_preset,
            ic50_um=tox_data.get("estimated_ic50_numeric", 3.0),
        )
        fig_dr, ax_dr = plt.subplots(figsize=(6, 4), facecolor="#161b22")
        ax_dr.set_facecolor("#161b22")
        ax_dr.plot(
            dose_sim["concentrations_um"],
            dose_sim["viability_percentages"],
            marker="o",
            color="#FF4500",
            linewidth=2,
            label="In Silico Fitted Response",
        )
        ax_dr.axvline(dose_slider, color="#00FFCC", linestyle="--", label=f"Current Dose ({dose_slider:.1f} µM)")
        ax_dr.set_xscale("log")
        ax_dr.set_ylim(0, 105)
        ax_dr.set_xlabel("Concentration (µM)", color="white")
        ax_dr.set_ylabel("Axonal Viability (%)", color="white")
        ax_dr.tick_params(colors="white")
        ax_dr.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax_dr.legend(facecolor="#21262d", labelcolor="white")
        st.pyplot(fig_dr)

with tab4:
    st.subheader("Structured Clinical & Telemetry Asset")
    st.json(report)

    json_str = json.dumps(report, indent=2)
    st.download_button(
        label="📥 Download Structured JSON Telemetry",
        data=json_str,
        file_name=f"synaptwin_telemetry_{sample_preset.replace(' ', '_')}.json",
        mime="application/json",
    )
