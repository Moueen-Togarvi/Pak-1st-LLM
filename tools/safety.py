from __future__ import annotations

import re
import shlex
from pathlib import Path


SECRET_EXACT_NAMES = {
    ".env",
    ".env.local",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
}

SECRET_NAME_MARKERS = (
    "secret",
    "private_key",
    "access_token",
    "api_token",
    "auth_token",
)

DESTRUCTIVE_COMMAND_PATTERNS = (
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-fd",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",
    r">\s*/dev/sd[a-z]",
)

RISKY_CODE_FLAGS = {"-c", "--command", "-e", "--eval"}
RISKY_CODE_RUNNERS = {"bash", "dash", "fish", "node", "perl", "python", "python3", "ruby", "sh", "zsh"}
PRIVILEGE_COMMANDS = {"sudo", "su", "doas"}


def is_path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def looks_secret_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    if any(part in SECRET_EXACT_NAMES for part in parts):
        return True
    name = path.name.lower()
    return any(marker in name for marker in SECRET_NAME_MARKERS)


def is_destructive_command(command: str) -> bool:
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in DESTRUCTIVE_COMMAND_PATTERNS):
        return True
    try:
        parts = shlex.split(command)
    except ValueError:
        return True
    if not parts:
        return False

    executable = Path(parts[0]).name.lower()
    flags = [part.lower() for part in parts[1:] if part.startswith("-")]
    joined_flags = "".join(flag.lstrip("-") for flag in flags)
    if executable == "rm" and "r" in joined_flags and "f" in joined_flags:
        return True
    if executable in {"chmod", "chown"} and any("r" in flag for flag in flags):
        return True
    if executable in PRIVILEGE_COMMANDS:
        return True
    if executable in RISKY_CODE_RUNNERS and any(flag in RISKY_CODE_FLAGS for flag in flags):
        return True
    return False


def safe_command_args(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("empty command")
    return parts
