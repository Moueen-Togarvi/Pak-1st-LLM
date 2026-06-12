from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizer.byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class ConversationPair:
    user: str
    assistant: str


def ensure_sample_data(path: str | Path = "data/raw/conversations.txt") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return path
    path.write_text(
        "# Add only your own taught pairs here.\n"
        "# Format:\n"
        "# ---\n"
        "# INPUT: your question\n"
        "# OUTPUT: your answer\n",
        encoding="utf-8",
    )
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


def load_conversation_folder(path: str | Path) -> list[ConversationPair]:
    path = Path(path)
    if not path.exists():
        return []
    pairs: list[ConversationPair] = []
    for file_path in sorted(path.glob("*conversations*.txt")):
        pairs.extend(load_conversations(file_path))
    return pairs


def append_conversation(path: str | Path, user: str, assistant: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    user = _single_line(user)
    assistant = _single_line(assistant)
    if not user or not assistant:
        return
    existing = {(pair.user, pair.assistant) for pair in load_conversations(path)}
    if (user, assistant) in existing:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"---\nINPUT: {user}\nOUTPUT: {assistant}\n")


def append_plain_text(path: str | Path, text: str) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _single_line(text)
    if not text:
        return False
    existing_lines = set()
    if path.exists():
        existing_lines = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    if text in existing_lines:
        return False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return True


def conversation_text(pair: ConversationPair) -> str:
    return f"User: {pair.user}\nAI: {pair.assistant}"


def plain_texts_from_folder(path: str | Path) -> list[str]:
    path = Path(path)
    texts: list[str] = []
    if not path.exists():
        return texts
    for file_path in sorted(path.rglob("*.txt")):
        if "conversations" in file_path.name:
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
    pairs = load_conversation_folder(raw_folder)
    if not pairs:
        pairs = load_conversations(conversations_path)
    texts = [conversation_text(pair) for pair in pairs]
    texts.extend(plain_texts_from_folder(raw_folder))
    sequences = [
        tokenizer.encode(text, add_bos=True, add_eos=True, max_length=max_length)
        for text in texts
        if text.strip()
    ]
    return [seq for seq in sequences if len(seq) >= 2]


def _single_line(text: str) -> str:
    return " ".join(text.strip().split())


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
