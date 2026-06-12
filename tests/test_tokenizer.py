import unittest

from tokenizer.byte_tokenizer import ByteTokenizer


class ByteTokenizerTest(unittest.TestCase):
    def test_roundtrip_many_scripts_and_code(self) -> None:
        tokenizer = ByteTokenizer()
        text = "hello Roman Urdu: kya haal? اردو\nprint('ok') # 123"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(tokenizer.vocab_size, 260)

    def test_special_tokens_are_visible_when_requested(self) -> None:
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode("hi")
        decoded = tokenizer.decode(ids, skip_special=False)
        self.assertTrue(decoded.startswith("<BOS>"))
        self.assertTrue(decoded.endswith("<EOS>"))


if __name__ == "__main__":
    unittest.main()

