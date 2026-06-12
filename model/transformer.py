from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    seed: int = 42
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(**payload)


def softmax(x: Array, axis: int = -1) -> Array:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gelu(x: Array) -> Array:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def gelu_backward(dout: Array, x: Array) -> Array:
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x**3)
    tanh_u = np.tanh(u)
    du = c * (1.0 + 3.0 * 0.044715 * x**2)
    grad = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u**2) * du
    return dout * grad


class TransformerLM:
    """Small decoder-only transformer with explicit NumPy backpropagation."""

    def __init__(self, config: ModelConfig | dict[str, Any] | None = None):
        if config is None:
            config = ModelConfig()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.params: dict[str, Array] = {}
        self.grads: dict[str, Array] = {}
        self._init_params()

    @property
    def head_dim(self) -> int:
        return self.config.d_model // self.config.n_heads

    def _init_params(self) -> None:
        cfg = self.config

        def randn(shape: tuple[int, ...], scale: float = 0.02) -> Array:
            return (self.rng.standard_normal(shape) * scale).astype(np.float32)

        self.params["token_embedding"] = randn((cfg.vocab_size, cfg.d_model))
        self.params["position_embedding"] = randn((cfg.max_seq_len, cfg.d_model))

        for layer in range(cfg.n_layers):
            p = f"layers.{layer}"
            attn_scale = 1.0 / math.sqrt(cfg.d_model)
            self.params[f"{p}.wq"] = randn((cfg.d_model, cfg.d_model), attn_scale)
            self.params[f"{p}.wk"] = randn((cfg.d_model, cfg.d_model), attn_scale)
            self.params[f"{p}.wv"] = randn((cfg.d_model, cfg.d_model), attn_scale)
            self.params[f"{p}.wo"] = randn((cfg.d_model, cfg.d_model), attn_scale)
            self.params[f"{p}.ln1_gamma"] = np.ones((cfg.d_model,), dtype=np.float32)
            self.params[f"{p}.ln1_beta"] = np.zeros((cfg.d_model,), dtype=np.float32)
            self.params[f"{p}.ln2_gamma"] = np.ones((cfg.d_model,), dtype=np.float32)
            self.params[f"{p}.ln2_beta"] = np.zeros((cfg.d_model,), dtype=np.float32)
            self.params[f"{p}.w1"] = randn((cfg.d_model, cfg.d_ff), 1.0 / math.sqrt(cfg.d_model))
            self.params[f"{p}.b1"] = np.zeros((cfg.d_ff,), dtype=np.float32)
            self.params[f"{p}.w2"] = randn((cfg.d_ff, cfg.d_model), 1.0 / math.sqrt(cfg.d_ff))
            self.params[f"{p}.b2"] = np.zeros((cfg.d_model,), dtype=np.float32)

        self.params["ln_f_gamma"] = np.ones((cfg.d_model,), dtype=np.float32)
        self.params["ln_f_beta"] = np.zeros((cfg.d_model,), dtype=np.float32)
        self.params["w_out"] = randn((cfg.d_model, cfg.vocab_size), 1.0 / math.sqrt(cfg.d_model))
        self.params["b_out"] = np.zeros((cfg.vocab_size,), dtype=np.float32)
        self.zero_grad()

    def zero_grad(self) -> None:
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}

    def parameter_count(self) -> int:
        return int(sum(np.prod(param.shape) for param in self.params.values()))

    def _ids_array(self, input_ids: Array | list[int] | list[list[int]]) -> Array:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError("input_ids must have shape (T,) or (B, T)")
        if ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {ids.shape[1]} exceeds max_seq_len {self.config.max_seq_len}"
            )
        return ids

    def forward(
        self,
        input_ids: Array | list[int] | list[list[int]],
        targets: Array | list[int] | list[list[int]] | None = None,
        *,
        return_cache: bool = False,
    ) -> Array | tuple[Array, float, dict[str, Any]] | tuple[Array, dict[str, Any]]:
        ids = self._ids_array(input_ids)
        batch, seq_len = ids.shape
        positions = np.arange(seq_len, dtype=np.int64)
        x = self.params["token_embedding"][ids] + self.params["position_embedding"][positions][None, :, :]

        cache: dict[str, Any] = {"ids": ids, "positions": positions, "layers": []}
        for layer in range(self.config.n_layers):
            x, layer_cache = self._block_forward(x, layer)
            cache["layers"].append(layer_cache)

        h, ln_f_cache = self._layer_norm_forward(
            x, self.params["ln_f_gamma"], self.params["ln_f_beta"]
        )
        logits = h @ self.params["w_out"] + self.params["b_out"]
        cache["final_x"] = x
        cache["ln_f"] = ln_f_cache
        cache["h"] = h

        if targets is None:
            if return_cache:
                return logits, cache
            return logits

        loss, dlogits = self._cross_entropy(logits, targets)
        cache["dlogits"] = dlogits
        if return_cache:
            return logits, loss, cache
        return logits, loss, cache

    def loss(self, input_ids: Array | list[list[int]], targets: Array | list[list[int]]) -> float:
        _, loss, _ = self.forward(input_ids, targets, return_cache=True)
        return float(loss)

    def train_batch(self, input_ids: Array, targets: Array) -> float:
        _, loss, cache = self.forward(input_ids, targets, return_cache=True)
        self.backward(cache)
        return float(loss)

    def backward(self, cache: dict[str, Any]) -> None:
        self.zero_grad()
        dlogits = cache["dlogits"]
        h = cache["h"]

        h2 = h.reshape(-1, self.config.d_model)
        dlogits2 = dlogits.reshape(-1, self.config.vocab_size)
        self.grads["w_out"] += h2.T @ dlogits2
        self.grads["b_out"] += dlogits2.sum(axis=0)
        dh = dlogits @ self.params["w_out"].T

        dx = self._layer_norm_backward(dh, cache["ln_f"], "ln_f_gamma", "ln_f_beta")
        for layer in reversed(range(self.config.n_layers)):
            dx = self._block_backward(dx, cache["layers"][layer], layer)

        ids = cache["ids"]
        np.add.at(self.grads["token_embedding"], ids, dx)
        self.grads["position_embedding"][cache["positions"]] += dx.sum(axis=0)

    def _block_forward(self, x: Array, layer: int) -> tuple[Array, dict[str, Any]]:
        p = f"layers.{layer}"
        ln1, ln1_cache = self._layer_norm_forward(
            x, self.params[f"{p}.ln1_gamma"], self.params[f"{p}.ln1_beta"]
        )
        attn, attn_cache = self._attention_forward(ln1, layer)
        x_attn = x + attn
        ln2, ln2_cache = self._layer_norm_forward(
            x_attn, self.params[f"{p}.ln2_gamma"], self.params[f"{p}.ln2_beta"]
        )
        ff, ff_cache = self._ffn_forward(ln2, layer)
        out = x_attn + ff
        return out, {
            "x": x,
            "ln1": ln1_cache,
            "attn": attn_cache,
            "x_attn": x_attn,
            "ln2": ln2_cache,
            "ff": ff_cache,
        }

    def _block_backward(self, dout: Array, cache: dict[str, Any], layer: int) -> Array:
        p = f"layers.{layer}"

        dx_attn = dout.copy()
        dff = dout
        dln2 = self._ffn_backward(dff, cache["ff"], layer)
        dx_attn += self._layer_norm_backward(
            dln2, cache["ln2"], f"{p}.ln2_gamma", f"{p}.ln2_beta"
        )

        dx = dx_attn.copy()
        dattn = dx_attn
        dln1 = self._attention_backward(dattn, cache["attn"], layer)
        dx += self._layer_norm_backward(
            dln1, cache["ln1"], f"{p}.ln1_gamma", f"{p}.ln1_beta"
        )
        return dx

    def _layer_norm_forward(self, x: Array, gamma: Array, beta: Array) -> tuple[Array, tuple[Array, Array, Array]]:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = np.mean(centered**2, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.config.layer_norm_eps)
        x_hat = centered * inv_std
        out = x_hat * gamma + beta
        return out, (x_hat, inv_std, gamma)

    def _layer_norm_backward(
        self,
        dout: Array,
        cache: tuple[Array, Array, Array],
        gamma_name: str,
        beta_name: str,
    ) -> Array:
        x_hat, inv_std, gamma = cache
        self.grads[gamma_name] += np.sum(dout * x_hat, axis=(0, 1))
        self.grads[beta_name] += np.sum(dout, axis=(0, 1))
        dx_hat = dout * gamma
        dim = dx_hat.shape[-1]
        dx = (
            inv_std
            / dim
            * (
                dim * dx_hat
                - np.sum(dx_hat, axis=-1, keepdims=True)
                - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
            )
        )
        return dx

    def _attention_forward(self, x: Array, layer: int) -> tuple[Array, dict[str, Array]]:
        p = f"layers.{layer}"
        batch, seq_len, dim = x.shape
        heads = self.config.n_heads
        head_dim = self.head_dim

        q = x @ self.params[f"{p}.wq"]
        k = x @ self.params[f"{p}.wk"]
        v = x @ self.params[f"{p}.wv"]

        qh = q.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
        kh = k.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(qh, np.swapaxes(kh, -1, -2)) / math.sqrt(head_dim)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, None, :, :], -1e9, scores)
        attn = softmax(scores, axis=-1)
        ctx = np.matmul(attn, vh)
        merged = ctx.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
        out = merged @ self.params[f"{p}.wo"]
        return out, {
            "x": x,
            "q": qh,
            "k": kh,
            "v": vh,
            "attn": attn,
            "merged": merged,
        }

    def _attention_backward(self, dout: Array, cache: dict[str, Array], layer: int) -> Array:
        p = f"layers.{layer}"
        x = cache["x"]
        qh = cache["q"]
        kh = cache["k"]
        vh = cache["v"]
        attn = cache["attn"]
        merged = cache["merged"]
        batch, seq_len, dim = x.shape
        heads = self.config.n_heads
        head_dim = self.head_dim

        self.grads[f"{p}.wo"] += merged.reshape(-1, dim).T @ dout.reshape(-1, dim)
        dmerged = dout @ self.params[f"{p}.wo"].T
        dctx = dmerged.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

        dattn = np.matmul(dctx, np.swapaxes(vh, -1, -2))
        dvh = np.matmul(np.swapaxes(attn, -1, -2), dctx)
        dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
        scale = 1.0 / math.sqrt(head_dim)
        dqh = np.matmul(dscores, kh) * scale
        dkh = np.matmul(np.swapaxes(dscores, -1, -2), qh) * scale

        dq = dqh.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
        dk = dkh.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
        dv = dvh.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)

        x2 = x.reshape(-1, dim)
        self.grads[f"{p}.wq"] += x2.T @ dq.reshape(-1, dim)
        self.grads[f"{p}.wk"] += x2.T @ dk.reshape(-1, dim)
        self.grads[f"{p}.wv"] += x2.T @ dv.reshape(-1, dim)

        dx = dq @ self.params[f"{p}.wq"].T
        dx += dk @ self.params[f"{p}.wk"].T
        dx += dv @ self.params[f"{p}.wv"].T
        return dx

    def _ffn_forward(self, x: Array, layer: int) -> tuple[Array, dict[str, Array]]:
        p = f"layers.{layer}"
        z1 = x @ self.params[f"{p}.w1"] + self.params[f"{p}.b1"]
        a1 = gelu(z1)
        out = a1 @ self.params[f"{p}.w2"] + self.params[f"{p}.b2"]
        return out, {"x": x, "z1": z1, "a1": a1}

    def _ffn_backward(self, dout: Array, cache: dict[str, Array], layer: int) -> Array:
        p = f"layers.{layer}"
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        dim = self.config.d_model
        d_ff = self.config.d_ff

        self.grads[f"{p}.w2"] += a1.reshape(-1, d_ff).T @ dout.reshape(-1, dim)
        self.grads[f"{p}.b2"] += dout.sum(axis=(0, 1))
        da1 = dout @ self.params[f"{p}.w2"].T
        dz1 = gelu_backward(da1, z1)
        self.grads[f"{p}.w1"] += x.reshape(-1, dim).T @ dz1.reshape(-1, d_ff)
        self.grads[f"{p}.b1"] += dz1.sum(axis=(0, 1))
        return dz1 @ self.params[f"{p}.w1"].T

    def _cross_entropy(self, logits: Array, targets: Array | list[list[int]]) -> tuple[float, Array]:
        target_array = np.asarray(targets, dtype=np.int64)
        if target_array.ndim == 1:
            target_array = target_array[None, :]
        if target_array.shape != logits.shape[:2]:
            raise ValueError("targets must match logits batch and sequence dimensions")

        probs = softmax(logits, axis=-1)
        valid = target_array >= 0
        count = int(valid.sum())
        if count == 0:
            raise ValueError("targets contain no valid tokens")

        flat_probs = probs.reshape(-1, self.config.vocab_size)
        flat_targets = target_array.reshape(-1)
        flat_valid = valid.reshape(-1)
        selected = flat_probs[np.arange(flat_targets.size)[flat_valid], flat_targets[flat_valid]]
        loss = -np.log(selected + 1e-9).mean()

        dlogits = probs
        flat_dlogits = dlogits.reshape(-1, self.config.vocab_size)
        valid_indices = np.arange(flat_targets.size)[flat_valid]
        flat_dlogits[valid_indices, flat_targets[flat_valid]] -= 1.0
        flat_dlogits[~flat_valid] = 0.0
        dlogits /= count
        return float(loss), dlogits.astype(np.float32)

    def clip_grad_norm(self, max_norm: float) -> float:
        total_sq = 0.0
        for grad in self.grads.values():
            total_sq += float(np.sum(grad.astype(np.float64) ** 2))
        total_norm = math.sqrt(total_sq)
        if total_norm > max_norm > 0:
            scale = max_norm / (total_norm + 1e-12)
            for name in self.grads:
                self.grads[name] *= scale
        return total_norm

    def generate(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int | None = 40,
        eos_id: int | None = 2,
    ) -> list[int]:
        ids = list(input_ids)
        for _ in range(max_new_tokens):
            context = ids[-self.config.max_seq_len :]
            logits = self.forward(context)
            next_logits = logits[0, -1].astype(np.float64)
            if temperature <= 0:
                next_id = int(np.argmax(next_logits))
            else:
                next_logits /= temperature
                if top_k is not None and 0 < top_k < next_logits.size:
                    keep = np.argpartition(next_logits, -top_k)[-top_k:]
                    masked = np.full_like(next_logits, -np.inf)
                    masked[keep] = next_logits[keep]
                    next_logits = masked
                probs = softmax(next_logits)
                next_id = int(self.rng.choice(np.arange(next_logits.size), p=probs))
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return ids

    def save(self, weights_path: str | Path, config_path: str | Path | None = None) -> None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(weights_path, **self.params)
        if config_path is not None:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

    def load_weights(self, weights_path: str | Path) -> None:
        data = np.load(weights_path)
        missing = set(self.params) - set(data.files)
        extra = set(data.files) - set(self.params)
        if missing or extra:
            raise ValueError(f"weight key mismatch; missing={missing}, extra={extra}")
        for name in self.params:
            if self.params[name].shape != data[name].shape:
                raise ValueError(f"shape mismatch for {name}: {data[name].shape}")
            self.params[name] = data[name].astype(np.float32)
        self.zero_grad()

    @classmethod
    def load(cls, weights_path: str | Path, config_path: str | Path) -> "TransformerLM":
        config = ModelConfig.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
        model = cls(config)
        model.load_weights(weights_path)
        return model

