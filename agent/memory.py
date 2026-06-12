from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryFile:
    path: Path
    content: str


class MemoryLoader:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def load(self) -> list[MemoryFile]:
        files: list[MemoryFile] = []
        for relative in ("AGENTS.md", "CLAUDE.md", ".myai/memory.md"):
            path = self.repo_root / relative
            if path.exists() and path.is_file():
                files.append(MemoryFile(path=path, content=path.read_text(encoding="utf-8", errors="replace")))
        return files

    def render(self) -> str:
        chunks: list[str] = []
        for memory_file in self.load():
            rel = memory_file.path.relative_to(self.repo_root)
            chunks.append(f"## {rel}\n{memory_file.content.strip()}")
        return "\n\n".join(chunks) if chunks else "(no memory files found)"

