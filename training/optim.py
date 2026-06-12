from __future__ import annotations

import math

import numpy as np


class AdamOptimizer:
    """Adam optimizer over a dictionary of NumPy parameters."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, param in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1**self.t)
            v_hat = self.v[name] / (1.0 - self.beta2**self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def learning_rate(self) -> float:
        return self.lr

    def set_learning_rate(self, lr: float) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("learning rate must be positive and finite")
        self.lr = lr

