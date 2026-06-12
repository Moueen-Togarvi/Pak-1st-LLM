from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.memory import MemoryLoader
from agent.router import DeterministicRouter
from agent.skills import SkillLoader
from tools.registry import ToolRegistry


@dataclass
class AgentReply:
    content: str
    raw_action: str = ""


class LocalAgent:
    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.tools = ToolRegistry(self.repo_root)
        self.memory = MemoryLoader(self.repo_root)
        self.skills = SkillLoader(self.repo_root)
        self.router = DeterministicRouter()
        self.execute_mode = False

    def handle(self, user_text: str) -> AgentReply:
        text = user_text.strip()
        lowered = text.lower()
        if lowered == "/execute":
            self.execute_mode = not self.execute_mode
            state = "enabled" if self.execute_mode else "disabled"
            return AgentReply(f"Execute mode {state}. Bash/edit tools now {'can' if self.execute_mode else 'cannot'} run.")
        if lowered == "/memory":
            return AgentReply(self.memory.render())
        if lowered == "/context":
            return AgentReply(self.context_summary())
        if lowered == "/help":
            return AgentReply(self.router.help_text())

        action = self.router.route(text)
        if action.action == "final":
            return AgentReply(action.content, action.to_json())

        result = self.tools.run(action.tool, action.args or {}, execute=self.execute_mode)
        status = "OK" if result.success else "ERROR"
        if result.requires_permission:
            status = "PERMISSION"
        return AgentReply(f"[{status}] {action.tool}\n{result.content}", action.to_json())

    def context_summary(self) -> str:
        memory_files = self.memory.load()
        skill_index = self.skills.render_index()
        memory_names = ", ".join(str(item.path.relative_to(self.repo_root)) for item in memory_files) or "none"
        return "\n".join(
            [
                f"Repo root: {self.repo_root}",
                f"Execute mode: {'on' if self.execute_mode else 'off'}",
                f"Memory files: {memory_names}",
                "Skills:",
                skill_index,
            ]
        )

