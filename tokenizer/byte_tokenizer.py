from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SpecialTokens:
    pad: int = 0
    bos: int = 1
    eos: int = 2
    sep: int = 3


class ByteTokenizer:
    """UTF-8 byte tokenizer with fixed vocabulary and no unknown token.

    IDs 0-3 are special tokens. IDs 4-259 map directly to byte values 0-255.
    This keeps the tokenizer owned, deterministic, and language/script agnostic.
    """

    byte_offset = 4
    vocab_size = 260

    def __init__(self) -> None:
        self.special = SpecialTokens()
        self.special_names = {
            self.special.pad: "<PAD>",
            self.special.bos: "<BOS>",
            self.special.eos: "<EOS>",
            self.special.sep: "<SEP>",
        }

    @property
    def pad_id(self) -> int:
        return self.special.pad

    @property
    def bos_id(self) -> int:
        return self.special.bos

    @property
    def eos_id(self) -> int:
        return self.special.eos

    @property
    def sep_id(self) -> int:
        return self.special.sep

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
            if add_eos and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        chunks: list[str] = []
        raw = bytearray()

        def flush_raw() -> None:
            if raw:
                chunks.append(raw.decode("utf-8", errors="replace"))
                raw.clear()

        for token_id in ids:
            token_id = int(token_id)
            if self.byte_offset <= token_id < self.vocab_size:
                raw.append(token_id - self.byte_offset)
                continue

            flush_raw()
            if skip_special:
                continue
            chunks.append(self.special_names.get(token_id, f"<ID:{token_id}>"))

        flush_raw()
        return "".join(chunks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "utf8_byte",
            "vocab_size": self.vocab_size,
            "byte_offset": self.byte_offset,
            "special_tokens": {
                "pad": self.pad_id,
                "bos": self.bos_id,
                "eos": self.eos_id,
                "sep": self.sep_id,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        path = Path(path)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("type") != "utf8_byte":
                raise ValueError(f"Unsupported tokenizer type in {path}")
            if int(payload.get("vocab_size", cls.vocab_size)) != cls.vocab_size:
                raise ValueError("Tokenizer vocab size mismatch")
        return cls()

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def iter_windows(self, ids: Sequence[int], block_size: int) -> Iterable[list[int]]:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        for start in range(0, max(1, len(ids) - 1), block_size - 1):
            window = list(ids[start : start + block_size])
            if len(window) >= 2:
                yield window

