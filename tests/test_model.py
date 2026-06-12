import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.transformer import ModelConfig, TransformerLM


class TransformerLMTest(unittest.TestCase):
    def small_model(self) -> TransformerLM:
        cfg = ModelConfig(vocab_size=260, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_seq_len=12, seed=5)
        return TransformerLM(cfg)

    def test_output_shape(self) -> None:
        model = self.small_model()
        logits = model.forward([[1, 10, 11, 2]])
        self.assertEqual(logits.shape, (1, 4, 260))

    def test_causal_mask_blocks_future_tokens(self) -> None:
        model = self.small_model()
        first = model.forward([[1, 10, 11, 12]])
        second = model.forward([[1, 10, 11, 13]])
        np.testing.assert_allclose(first[:, :3, :], second[:, :3, :], atol=1e-6)

    def test_save_load_exactness(self) -> None:
        model = self.small_model()
        ids = [[1, 10, 11, 2]]
        before = model.forward(ids)
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights.npz"
            config = Path(tmp) / "config.json"
            model.save(weights, config)
            loaded = TransformerLM.load(weights, config)
        after = loaded.forward(ids)
        np.testing.assert_allclose(before, after, atol=0.0)


if __name__ == "__main__":
    unittest.main()

