from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from data.data_prep import append_conversation, append_plain_text


UNDERTRAINED_MESSAGE = "(model is still undertrained; add data and run train for more epochs)"


class MemoryStore:
    """Tiny local memory for chat history, facts, and learnable pairs."""

    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.memory_dir = self.repo_root / ".myai"
        self.history_path = self.memory_dir / "chat_history.jsonl"
        self.facts_path = self.memory_dir / "facts.json"
        self.learned_pairs_path = self.repo_root / "data" / "raw" / "memory_conversations.txt"
        self.user_text_path = self.repo_root / "data" / "raw" / "user_messages.txt"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.learned_pairs_path.parent.mkdir(parents=True, exist_ok=True)

    def remember_turn(self, user: str, assistant: str, *, source: str = "chat") -> bool:
        user = " ".join(user.strip().split())
        assistant = " ".join(assistant.strip().split())
        if not user or not assistant:
            return False
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "user": user,
            "assistant": assistant,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        changed = self._extract_facts(user)
        changed = append_plain_text(self.user_text_path, user) or changed
        return changed

    def teach_pair(self, user: str, assistant: str) -> bool:
        before = self.learnable_pair_count()
        append_conversation(self.learned_pairs_path, user, assistant)
        return self.learnable_pair_count() > before

    def auto_teach_from_text(self, text: str) -> tuple[bool, str]:
        pair = extract_teaching_pair(text)
        if pair is None:
            return False, ""
        user, assistant = pair
        added = self.teach_pair(user, assistant)
        if added:
            return True, f"Auto-taught pair saved: {user} => {assistant}"
        return False, "Ye taught pair pehle se saved hai."

    def facts(self) -> dict[str, str]:
        if not self.facts_path.exists():
            return {}
        try:
            payload = json.loads(self.facts_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def answer_from_memory(self, user_text: str) -> str | None:
        normalized = _normalize(user_text)
        facts = self.facts()
        if _is_name_question(normalized):
            name = facts.get("user_name")
            if name:
                return f"Tumhara naam {name} hai."
        if _is_goal_question(normalized):
            goal = facts.get("current_goal")
            if goal:
                return f"Tumhara current goal: {goal}."
        if normalized in {"what do you remember", "memory", "yaad kya hai", "tumhe kya yaad hai"}:
            if not facts:
                return "Abhi koi personal fact yaad nahi hai."
            return "Mujhe yaad hai: " + ", ".join(f"{key}={value}" for key, value in sorted(facts.items()))
        return None

    def summary(self) -> str:
        facts = self.facts()
        history_count = 0
        if self.history_path.exists():
            history_count = sum(1 for _ in self.history_path.open(encoding="utf-8"))
        learned_pairs = self.learnable_pair_count()
        facts_text = ", ".join(f"{key}={value}" for key, value in sorted(facts.items())) or "none"
        return "\n".join(
            [
                f"Chat history turns: {history_count}",
                f"Learnable memory pairs: {learned_pairs}",
                f"User text lines: {self.user_text_count()}",
                f"Facts: {facts_text}",
                f"History file: {self.history_path}",
                f"Training memory file: {self.learned_pairs_path}",
                f"User text file: {self.user_text_path}",
            ]
        )

    def learnable_pair_count(self) -> int:
        if not self.learned_pairs_path.exists():
            return 0
        return self.learned_pairs_path.read_text(encoding="utf-8").count("INPUT:")

    def user_text_count(self) -> int:
        if not self.user_text_path.exists():
            return 0
        return sum(1 for line in self.user_text_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _extract_facts(self, user_text: str) -> bool:
        normalized = _normalize(user_text)
        name = _extract_name(user_text, normalized)
        goal = _extract_goal(user_text, normalized)
        facts = self.facts()
        before = dict(facts)
        if not name:
            if goal:
                facts["current_goal"] = goal
            if facts != before:
                self.facts_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
                return True
            return False
        facts["user_name"] = name
        if facts != before:
            self.facts_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
            return True
        return False


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in lowered)
    return " ".join(cleaned.split())


def extract_teaching_pair(text: str) -> tuple[str, str] | None:
    text = " ".join(text.strip().split())
    if not text:
        return None

    if "=>" in text:
        left, right = [part.strip() for part in text.split("=>", 1)]
        return _valid_pair(left, right)

    qa_match = re.match(r"(?is)^q\s*:\s*(.+?)\s+a\s*:\s*(.+)$", text)
    if qa_match:
        return _valid_pair(qa_match.group(1), qa_match.group(2))

    jawab_match = re.match(r"(?is)^(.+?)\s+(?:ka|k|ke)\s+jawab\s+(.+?)(?:\s+hai)?$", text)
    if jawab_match:
        return _valid_pair(jawab_match.group(1), jawab_match.group(2))

    jab_match = re.match(
        r"(?is)^(?:jab|agar)\s+(?:main|mn|me|mein)\s+(.+?)\s+"
        r"(?:poochun|puchun|pochoon|kaho|kahoon|bolun|likhun)\s+to\s+(.+?)"
        r"(?:\s+(?:jawab\s+dena|bolna|kehna))?$",
        text,
    )
    if jab_match:
        return _valid_pair(jab_match.group(1), jab_match.group(2))

    english_match = re.match(
        r"(?is)^(?:when|if)\s+i\s+(?:ask|say|type)\s+(.+?)\s+"
        r"(?:then\s+)?(?:reply|say|answer)\s+(.+)$",
        text,
    )
    if english_match:
        return _valid_pair(english_match.group(1), english_match.group(2))
    return None


def _valid_pair(user: str, assistant: str) -> tuple[str, str] | None:
    user = user.strip(" .,:;!?؟")
    assistant = assistant.strip(" .,:;!?؟")
    if len(user) < 1 or len(assistant) < 1:
        return None
    if len(user) > 300 or len(assistant) > 1000:
        return None
    return user, assistant


def _is_name_question(normalized: str) -> bool:
    if normalized in {
        "what is my name",
        "whats my name",
        "do you know my name",
        "mera naam kya hai",
        "mera name kya hai",
        "mera naam kia hai",
        "mera name kia hai",
        "mera naam kia hn",
        "mera name kia hn",
        "mera naam kya hn",
        "mera name kya hn",
    }:
        return True
    words = set(normalized.split())
    asks_name = {"name", "naam"} & words
    asks_question = {"what", "kya", "kia", "hai", "hn", "ha", "he"} & words
    owner = {"my", "mera", "meri", "mra"} & words
    return bool(asks_name and asks_question and owner)


def _extract_name(user_text: str, normalized: str) -> str | None:
    prefixes = (
        "my name is ",
        "my name ",
        "mera naam hai ",
        "mera naam ",
        "mera name hai ",
        "mera name ",
        "mra naam ",
        "mra name ",
        "main ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return user_text.strip().split()[-1].strip(".,!?؟")
    return None


def _extract_goal(user_text: str, normalized: str) -> str | None:
    prefixes = (
        "i want to ",
        "i want you to ",
        "i want to you ",
        "i want ",
        "mujhe ",
        "main chahta hoon ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return " ".join(user_text.strip().split())
    return None


def _is_goal_question(normalized: str) -> bool:
    return normalized in {"what is my goal", "mera goal kya hai", "goal kya hai", "current goal kya hai"}
