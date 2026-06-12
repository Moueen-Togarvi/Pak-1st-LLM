from __future__ import annotations

import numpy as np

from model.transformer import ModelConfig, TransformerLM


def check_output_projection_gradient(seed: int = 7) -> dict[str, float]:
    """Small finite-difference check for one learned parameter.

    This is intentionally narrow so it stays fast on CPU while proving the
    training path is using analytic backprop, not finite-difference updates.
    """

    cfg = ModelConfig(vocab_size=16, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=8, seed=seed)
    model = TransformerLM(cfg)
    input_ids = np.array([[1, 5, 6, 7]], dtype=np.int64)
    targets = np.array([[5, 6, 7, 2]], dtype=np.int64)
    _, loss, cache = model.forward(input_ids, targets, return_cache=True)
    model.backward(cache)

    name = "w_out"
    index = (0, 0)
    eps = 1e-4
    original = float(model.params[name][index])
    model.params[name][index] = original + eps
    plus = model.loss(input_ids, targets)
    model.params[name][index] = original - eps
    minus = model.loss(input_ids, targets)
    model.params[name][index] = original

    numeric = (plus - minus) / (2 * eps)
    analytic = float(model.grads[name][index])
    return {
        "loss": float(loss),
        "numeric": float(numeric),
        "analytic": analytic,
        "abs_error": abs(float(numeric) - analytic),
    }

