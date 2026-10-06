"""Unit tests for SynapTwin-OoC deep learning architectures and modules."""

import unittest
import torch
import numpy as np

from src.models.virtual_staining_cgan import (
    VirtualStainingGenerator,
    PhysicsGuidedOpticalPrior,
    ResBlock,
    ChannelAttention,
)
from src.models.morphometry_profiler import NeuralMorphometryProfiler
from src.models.neurotoxicity_engine import NeurotoxicityScreeningEngine
from src.data.ooc_synthesizer import OoCMicrographSynthesizer


class TestSynapTwinModels(unittest.TestCase):
    """Verifies architectural integrity, tensor shapes, and functional validity."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.img_size = (128, 128)

    def test_channel_attention(self):
        ca = ChannelAttention(channels=16)
        x = torch.randn(self.batch_size, 16, 32, 32)
        out = ca(x)
        self.assertEqual(out.shape, x.shape)

    def test_residual_block(self):
        block = ResBlock(in_ch=16, out_ch=32)
        x = torch.randn(self.batch_size, 16, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (self.batch_size, 32, 32, 32))

    def test_physics_optical_prior(self):
        prior = PhysicsGuidedOpticalPrior()
        x = torch.rand(self.batch_size, 1, 128, 128)
        out = prior(x)
        self.assertEqual(out.shape, (self.batch_size, 3, 128, 128))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_virtual_staining_generator_forward(self):
        gen = VirtualStainingGenerator(in_channels=1, out_channels=3, base_channels=16)
        x = torch.rand(self.batch_size, 1, 128, 128)
        out = gen(x)
        self.assertEqual(out.shape, (self.batch_size, 3, 128, 128))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_morphometry_profiler(self):
        profiler = NeuralMorphometryProfiler()
        dummy_fl = np.zeros((3, 256, 256), dtype=np.float32)
        # Put dummy soma
        dummy_fl[0, 50:70, 50:70] = 0.8
        dummy_fl[1, 50:70, 50:70] = 0.8
        # Put dummy axon
        dummy_fl[1, 60, 70:200] = 0.9
        dummy_fl[2, 60, 180:200] = 0.7

        metrics = profiler.analyze_micrograph(dummy_fl)
        self.assertIn("soma_count", metrics)
        self.assertIn("axon_penetration_ratio", metrics)
        self.assertIn("circuit_connectivity_score", metrics)
        self.assertGreaterEqual(metrics["circuit_connectivity_score"], 0.0)

    def test_neurotoxicity_engine(self):
        engine = NeurotoxicityScreeningEngine()
        sample_morph = {
            "circuit_connectivity_score": 85.0,
            "cytoskeletal_fragmentation_index": 0.05,
            "axon_penetration_ratio": 0.38,
            "soma_count": 35,
        }
        res = engine.predict_toxicity(sample_morph, compound_name="Control", tested_conc_um=0.0)
        self.assertEqual(res["predicted_class"], "Safe / Non-Toxic")
        self.assertEqual(res["toxicity_grade"], 0)
        self.assertGreater(res["viability_index"], 0.70)


if __name__ == "__main__":
    unittest.main()
