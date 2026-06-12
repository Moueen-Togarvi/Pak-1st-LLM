from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data.data_prep import build_training_sequences, ensure_sample_data, windows_for_lm
from model.transformer import TransformerLM
from tokenizer.byte_tokenizer import ByteTokenizer
from training.optim import AdamOptimizer


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    save_every: int = 5
    seed: int = 123


class Trainer:
    def __init__(
        self,
        model: TransformerLM,
        tokenizer: ByteTokenizer,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.optimizer = AdamOptimizer(model.params, lr=self.config.learning_rate)
        self.losses: list[float] = []

    def make_batch(self, examples: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        if not examples:
            raise ValueError("no examples provided")
        max_input = max(len(example) - 1 for example in examples)
        input_ids = np.full((len(examples), max_input), self.tokenizer.pad_id, dtype=np.int64)
        targets = np.full((len(examples), max_input), -100, dtype=np.int64)
        for row, example in enumerate(examples):
            inp = example[:-1]
            tgt = example[1:]
            input_ids[row, : len(inp)] = inp
            targets[row, : len(tgt)] = tgt
        return input_ids, targets

    def train_examples(self, examples: list[list[int]], *, epochs: int | None = None) -> list[float]:
        if not examples:
            raise ValueError("training examples are empty")
        epochs = epochs or self.config.epochs
        examples = windows_for_lm(examples, self.model.config.max_seq_len)
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(examples))
            epoch_losses: list[float] = []
            for start in range(0, len(order), self.config.batch_size):
                batch = [examples[i] for i in order[start : start + self.config.batch_size]]
                input_ids, targets = self.make_batch(batch)
                loss = self.model.train_batch(input_ids, targets)
                self.model.clip_grad_norm(self.config.grad_clip)
                self.optimizer.step(self.model.params, self.model.grads)
                epoch_losses.append(loss)
            avg_loss = float(np.mean(epoch_losses))
            self.losses.append(avg_loss)
            elapsed = time.time() - start_time
            print(f"epoch {epoch:03d}/{epochs:03d} loss={avg_loss:.4f} time={elapsed:.1f}s")
        return self.losses

    def evaluate_examples(self, examples: list[list[int]]) -> float:
        examples = windows_for_lm(examples, self.model.config.max_seq_len)
        losses: list[float] = []
        for start in range(0, len(examples), self.config.batch_size):
            batch = examples[start : start + self.config.batch_size]
            input_ids, targets = self.make_batch(batch)
            losses.append(self.model.loss(input_ids, targets))
        return float(np.mean(losses))

    def train_from_data(
        self,
        *,
        data_path: str | Path = "data/raw/conversations.txt",
        raw_folder: str | Path = "data/raw",
        weights_path: str | Path = "model/weights.npz",
        config_path: str | Path = "model/weights_config.json",
    ) -> list[float]:
        ensure_sample_data(data_path)
        sequences = build_training_sequences(
            self.tokenizer,
            conversations_path=data_path,
            raw_folder=raw_folder,
            max_length=self.model.config.max_seq_len + 1,
        )
        print(f"training examples: {len(sequences)}")
        losses = self.train_examples(sequences)
        self.model.save(weights_path, config_path)
        self._save_training_log(Path("training/loss_log.json"))
        return losses

    def _save_training_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.config), "losses": self.losses}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

