import contextlib
import io
import unittest

from model.transformer import ModelConfig, TransformerLM
from tokenizer.byte_tokenizer import ByteTokenizer
from training.gradient_check import check_output_projection_gradient
from training.trainer import Trainer, TrainingConfig


class TrainingTest(unittest.TestCase):
    def test_loss_decreases_on_50_examples(self) -> None:
        tokenizer = ByteTokenizer()
        cfg = ModelConfig(vocab_size=260, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_seq_len=12, seed=9)
        model = TransformerLM(cfg)
        trainer = Trainer(model, tokenizer, TrainingConfig(epochs=15, batch_size=10, learning_rate=0.01, grad_clip=1.0))
        examples = [[1, 20 + (i % 5), 40, 41, 2] for i in range(50)]
        initial = trainer.evaluate_examples(examples)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train_examples(examples, epochs=15)
        final = trainer.evaluate_examples(examples)
        self.assertLess(final, initial)

    def test_gradient_check_is_close(self) -> None:
        result = check_output_projection_gradient()
        self.assertLess(result["abs_error"], 2e-2)


if __name__ == "__main__":
    unittest.main()

