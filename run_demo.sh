#!/usr/bin/env bash
set -e

echo "========================================================================"
echo "    Launching SynapTwin-OoC: AI Neural Organ-on-a-Chip Digital Twin     "
echo "========================================================================"

# Determine Python binary
if [ -f "./.venv/bin/python" ]; then
    PY_BIN="./.venv/bin/python"
    STREAMLIT_BIN="./.venv/bin/streamlit"
else
    PY_BIN="python3"
    STREAMLIT_BIN="streamlit"
fi

export PYTHONPATH="synaptwin-ooc:$PYTHONPATH"

# 1. Run Unit Tests
echo -e "\n[*] Running Unit Test Suite..."
$PY_BIN -m unittest discover -s synaptwin-ooc/tests

# 2. Run Single-Command Demo
echo -e "\n[*] Running Single-Command Inference Demo (demo.py)..."
$PY_BIN synaptwin-ooc/demo.py --compound "Paclitaxel (Taxol)" --dose 3.5 --output_dir synaptwin-ooc/outputs

# 3. Inform about Web Application
echo -e "\n========================================================================"
echo "  [✓] Inference & Diagnostic Report Generated Successfully!"
echo "  [✓] Diagnostic Image: synaptwin-ooc/outputs/demo_pipeline_diagnostic.png"
echo "  [✓] JSON Telemetry  : synaptwin-ooc/outputs/demo_report.json"
echo "------------------------------------------------------------------------"
echo "  To launch the Interactive Web Dashboard, run:"
echo "    $STREAMLIT_BIN run synaptwin-ooc/web_app/app.py"
echo "========================================================================"
