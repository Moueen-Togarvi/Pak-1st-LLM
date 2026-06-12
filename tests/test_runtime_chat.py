import unittest
import tempfile
from pathlib import Path

from data.data_prep import build_training_sequences
from main import should_capture_pending_answer
from myai.memory_store import MemoryStore, extract_teaching_pair
from model.transformer import ModelConfig, TransformerLM
from myai.runtime import generate_reply, lookup_known_reply
from tokenizer.byte_tokenizer import ByteTokenizer


class RuntimeChatTest(unittest.TestCase):
    def test_known_starter_replies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(lookup_known_reply("hi", repo_root=Path(tmp)))
            self.assertIsNone(lookup_known_reply("what is your name", repo_root=Path(tmp)))

    def test_memory_stores_facts_and_training_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            store.remember_turn("my name is Moueen", "Nice to meet you, Moueen.", source="test")

            self.assertTrue((root / ".myai" / "chat_history.jsonl").exists())
            self.assertEqual(store.facts()["user_name"], "Moueen")
            self.assertIn("Moueen", store.answer_from_memory("what is my name"))
            self.assertIn("Moueen", store.answer_from_memory("mera name kia hn??"))
            self.assertTrue((root / "data" / "raw" / "user_messages.txt").exists())

    def test_mixed_roman_urdu_name_fact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            store.remember_turn("mera name moon", "Nice to meet you, moon.", source="test")

            self.assertEqual(store.facts()["user_name"], "moon")
            self.assertEqual(store.answer_from_memory("mera name kia hn??"), "Tumhara naam moon hai.")

    def test_memory_persists_across_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = MemoryStore(root)
            first.remember_turn("my name is Moon", "Nice to meet you, Moon.", source="test")

            second = MemoryStore(root)
            self.assertEqual(second.answer_from_memory("what is my name"), "Tumhara naam Moon hai.")

            sequences = build_training_sequences(
                ByteTokenizer(),
                conversations_path=root / "data" / "raw" / "conversations.txt",
                raw_folder=root / "data" / "raw",
                max_length=64,
            )
            self.assertEqual(len(sequences), 1)

    def test_teach_pair_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            self.assertTrue(store.teach_pair("hi", "hello from user"))
            self.assertFalse(store.teach_pair("hi", "hello from user"))
            self.assertEqual(lookup_known_reply("hi", repo_root=root), "hello from user")

    def test_auto_teach_phrases_create_pairs(self) -> None:
        self.assertEqual(extract_teaching_pair("hi => hello"), ("hi", "hello"))
        self.assertEqual(extract_teaching_pair("Q: hi A: hello"), ("hi", "hello"))
        self.assertEqual(extract_teaching_pair("hi ka jawab hello hai"), ("hi", "hello"))
        self.assertEqual(extract_teaching_pair("jab main hi kaho to hello bolna"), ("hi", "hello"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            added, message = store.auto_teach_from_text("hi ka jawab hello hai")
            self.assertTrue(added)
            self.assertIn("Auto-taught", message)
            self.assertEqual(lookup_known_reply("hi", repo_root=root), "hello")

    def test_unknown_chat_uses_fallback_not_raw_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TransformerLM(ModelConfig(vocab_size=260, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_seq_len=12))
            reply = generate_reply(model, ByteTokenizer(), "random open question", repo_root=root)
            self.assertIn("Agle line", reply)

    def test_pending_answer_capture_rules(self) -> None:
        self.assertTrue(should_capture_pending_answer("hello moueen"))
        self.assertFalse(should_capture_pending_answer("what is this?"))
        self.assertFalse(should_capture_pending_answer("/memory"))
        self.assertFalse(should_capture_pending_answer("skip"))


if __name__ == "__main__":
    unittest.main()
