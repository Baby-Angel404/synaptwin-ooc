"""Integration test for end-to-end SynapTwin-OoC pipeline."""

import unittest
import numpy as np
from src.pipeline import SynapTwinPipeline
from src.data.ooc_synthesizer import OoCMicrographSynthesizer


class TestSynapTwinPipeline(unittest.TestCase):
    """End-to-end integration test."""

    def setUp(self):
        self.pipeline = SynapTwinPipeline(device="cpu")
        self.synthesizer = OoCMicrographSynthesizer(size=(256, 256), seed=123)

    def test_full_pipeline_run(self):
        sample = self.synthesizer.generate_microfluidic_chip_sample(
            compound_name="Vehicle (DMSO 0.1%)",
            concentration_um=0.0,
        )
        bf = sample["brightfield"]
        gt_fl = sample["fluorescence"]

        res = self.pipeline.process_micrograph(
            brightfield=bf,
            compound_name="Vehicle (DMSO 0.1%)",
            concentration_um=0.0,
            ground_truth_fl=gt_fl,
        )

        self.assertIn("report", res)
        self.assertIn("predicted_fluorescence", res)
        self.assertEqual(res["predicted_fluorescence"].shape, (3, 256, 256))

        rep = res["report"]
        self.assertIn("morphometry", rep)
        self.assertIn("toxicity_screening", rep)
        self.assertIn("translation_metrics", rep)

        # Vehicle should be classified as Safe
        self.assertEqual(rep["toxicity_screening"]["predicted_class"], "Safe / Non-Toxic")
        self.assertEqual(rep["toxicity_screening"]["toxicity_grade"], 0)


if __name__ == "__main__":
    unittest.main()
