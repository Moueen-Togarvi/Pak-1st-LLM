from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    content: str


class SkillLoader:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.skills_dir = self.repo_root / ".myai" / "skills"

    def load(self) -> list[Skill]:
        if not self.skills_dir.exists():
            return []
        skills: list[Skill] = []
        for path in sorted(self.skills_dir.glob("*.md")):
            skills.append(Skill(name=path.stem, content=path.read_text(encoding="utf-8", errors="replace")))
        return skills

    def render_index(self) -> str:
        skills = self.load()
        if not skills:
            return "(no local skills)"
        return "\n".join(f"- {skill.name}" for skill in skills)

