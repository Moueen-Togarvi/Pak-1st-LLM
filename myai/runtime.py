from __future__ import annotations

from pathlib import Path

from data.data_prep import build_training_sequences, ensure_sample_data, load_conversation_folder
from myai.memory_store import MemoryStore
from model.transformer import ModelConfig, TransformerLM
from tokenizer.byte_tokenizer import ByteTokenizer


DEFAULT_MODEL_CONFIG = ModelConfig()
FALLBACK_PREFIX = "Mujhe is ka exact jawab abhi user data se nahi mila."


def ensure_project_setup(repo_root: str | Path = ".") -> None:
    root = Path(repo_root)
    for relative in (
        "tokenizer",
        "model",
        "training",
        "data/raw",
        "data/processed",
        "agent",
        "tools",
        "interface",
        ".myai/skills",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    tokenizer = ByteTokenizer()
    tokenizer_path = root / "tokenizer" / "vocab.json"
    if not tokenizer_path.exists():
        tokenizer.save(tokenizer_path)

    ensure_sample_data(root / "data" / "raw" / "conversations.txt")

    memory_path = root / ".myai" / "memory.md"
    if not memory_path.exists():
        memory_path.write_text(
            "# Pak-1st-LLM Memory\n\n"
            "- Use Claude-Code-style capabilities and workflows for this project.\n",
            encoding="utf-8",
        )

    model_path = root / "model" / "weights.npz"
    config_path = root / "model" / "weights_config.json"
    if not model_path.exists() or not config_path.exists():
        model = TransformerLM(DEFAULT_MODEL_CONFIG)
        model.save(model_path, config_path)


def load_tokenizer(repo_root: str | Path = ".") -> ByteTokenizer:
    root = Path(repo_root)
    path = root / "tokenizer" / "vocab.json"
    if path.exists():
        return ByteTokenizer.load(path)
    tokenizer = ByteTokenizer()
    tokenizer.save(path)
    return tokenizer


def load_model(repo_root: str | Path = ".") -> TransformerLM:
    root = Path(repo_root)
    weights_path = root / "model" / "weights.npz"
    config_path = root / "model" / "weights_config.json"
    if weights_path.exists() and config_path.exists():
        return TransformerLM.load(weights_path, config_path)
    model = TransformerLM(DEFAULT_MODEL_CONFIG)
    model.save(weights_path, config_path)
    return model


def project_info(repo_root: str | Path = ".") -> dict[str, object]:
    root = Path(repo_root)
    tokenizer = load_tokenizer(root)
    model = load_model(root)
    sequences = build_training_sequences(
        tokenizer,
        conversations_path=root / "data" / "raw" / "conversations.txt",
        raw_folder=root / "data" / "raw",
        max_length=model.config.max_seq_len + 1,
    )
    return {
        "repo_root": str(root.resolve()),
        "vocab_size": tokenizer.vocab_size,
        "parameters": model.parameter_count(),
        "config": model.config.to_dict(),
        "training_examples": len(sequences),
        "weights_exists": (root / "model" / "weights.npz").exists(),
    }


def generate_reply(
    model: TransformerLM,
    tokenizer: ByteTokenizer,
    user_text: str,
    *,
    max_new_tokens: int = 80,
    repo_root: str | Path = ".",
) -> str:
    known_reply = lookup_known_reply(user_text, repo_root=repo_root)
    if known_reply is not None:
        return known_reply

    if not should_use_raw_model(repo_root):
        return fallback_reply(user_text, repo_root=repo_root)

    prompt = f"User: {user_text}\nAI:"
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False, max_length=model.config.max_seq_len)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        eos_id=tokenizer.eos_id,
    )
    new_ids = generated[len(input_ids) :]
    text = tokenizer.decode(new_ids)
    cleaned = text.split("User:", 1)[0].strip()
    if not _looks_readable(cleaned):
        return fallback_reply(user_text, repo_root=repo_root)
    return cleaned


def should_use_raw_model(repo_root: str | Path = ".") -> bool:
    flag_path = Path(repo_root) / ".myai" / "use_raw_model"
    return flag_path.exists()


def fallback_reply(user_text: str, *, repo_root: str | Path = ".") -> str:
    return (
        f"{FALLBACK_PREFIX} "
        "Agle line mein sahi jawab likh do; main automatically seekh lunga."
    )


def is_fallback_reply(text: str) -> bool:
    return text.startswith(FALLBACK_PREFIX)


def lookup_known_reply(user_text: str, *, repo_root: str | Path = ".") -> str | None:
    normalized = _normalize_text(user_text)
    if not normalized:
        return None

    memory_reply = MemoryStore(repo_root).answer_from_memory(user_text)
    if memory_reply is not None:
        return memory_reply

    conversations_folder = Path(repo_root) / "data" / "raw"
    for pair in load_conversation_folder(conversations_folder):
        if _normalize_text(pair.user) == normalized:
            return pair.assistant
    return None


def _looks_readable(text: str) -> bool:
    if not text or "\ufffd" in text:
        return False
    visible = [char for char in text if not char.isspace()]
    if len(visible) < 2:
        return False
    printable = sum(1 for char in visible if char.isprintable())
    return printable / max(1, len(visible)) > 0.9


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in lowered)
    return " ".join(cleaned.split())
