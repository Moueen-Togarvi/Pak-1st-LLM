from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizer.byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class ConversationPair:
    user: str
    assistant: str


SAMPLE_PAIRS = [
    ConversationPair("hello", "hello kya haal hai"),
    ConversationPair("tum kaun ho", "main tumhara local AI agent hoon"),
    ConversationPair("tum kya kar sakte ho", "main files read kar sakta hoon aur commands plan kar sakta hoon"),
    ConversationPair("python kya hai", "python ek programming language hai"),
    ConversationPair("repo status dikhao", "git status tool use karo"),
    ConversationPair("file read karo", "read_file tool se file ka content dekho"),
    ConversationPair("shukriya", "koi baat nahi khushi hui"),
    ConversationPair("bye", "khuda hafiz phir milte hain"),
    ConversationPair("plan banao", "pehle context read karo phir clear plan do"),
    ConversationPair("safe rehna", "destructive command permission ke bina mat chalao"),
]


def ensure_sample_data(path: str | Path = "data/raw/conversations.txt") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return path
    lines: list[str] = []
    for pair in SAMPLE_PAIRS:
        lines.extend(["---", f"INPUT: {pair.user}", f"OUTPUT: {pair.assistant}"])
    lines.append("---")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_conversations(path: str | Path) -> list[ConversationPair]:
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    pairs: list[ConversationPair] = []
    for block in content.split("---"):
        user = None
        assistant = None
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("INPUT:"):
                user = line[len("INPUT:") :].strip()
            elif line.startswith("OUTPUT:"):
                assistant = line[len("OUTPUT:") :].strip()
        if user and assistant:
            pairs.append(ConversationPair(user=user, assistant=assistant))
    return pairs


def conversation_text(pair: ConversationPair) -> str:
    return f"User: {pair.user}\nAI: {pair.assistant}"


def plain_texts_from_folder(path: str | Path) -> list[str]:
    path = Path(path)
    texts: list[str] = []
    if not path.exists():
        return texts
    for file_path in sorted(path.rglob("*.txt")):
        if file_path.name == "conversations.txt":
            continue
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            texts.append(text)
    return texts


def build_training_sequences(
    tokenizer: ByteTokenizer,
    *,
    conversations_path: str | Path = "data/raw/conversations.txt",
    raw_folder: str | Path = "data/raw",
    max_length: int | None = None,
) -> list[list[int]]:
    pairs = load_conversations(conversations_path)
    texts = [conversation_text(pair) for pair in pairs]
    texts.extend(plain_texts_from_folder(raw_folder))
    sequences = [
        tokenizer.encode(text, add_bos=True, add_eos=True, max_length=max_length)
        for text in texts
        if text.strip()
    ]
    return [seq for seq in sequences if len(seq) >= 2]


def windows_for_lm(sequences: Iterable[list[int]], max_seq_len: int) -> list[list[int]]:
    examples: list[list[int]] = []
    window_size = max_seq_len + 1
    stride = max(1, max_seq_len)
    for seq in sequences:
        if len(seq) <= window_size:
            examples.append(seq)
            continue
        for start in range(0, len(seq) - 1, stride):
            window = seq[start : start + window_size]
            if len(window) >= 2:
                examples.append(window)
    return examples

