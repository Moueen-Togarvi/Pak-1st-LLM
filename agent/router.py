from __future__ import annotations

import shlex

from .schema import AgentAction, parse_action


class DeterministicRouter:
    """Rule-based first brain for the local agent shell.

    The tiny model can later be trained to emit the same JSON actions. For v1,
    deterministic routing makes tool use dependable while the model is small.
    """

    def route(self, text: str) -> AgentAction:
        stripped = text.strip()
        parsed = parse_action(stripped)
        if parsed is not None:
            return parsed

        lowered = stripped.lower()
        if lowered in {"help", "/help"}:
            return AgentAction("final", content=self.help_text())
        if lowered in {"hi", "hello", "hey", "salam", "assalam o alaikum"}:
            return AgentAction(
                "final",
                content="Hello. Main MyAI hoon, tumhara local coding agent. /help likho tools dekhne ke liye.",
            )
        if lowered in {"what is your name", "your name", "who are you", "tum kaun ho", "tumhara naam kya hai"}:
            return AgentAction(
                "final",
                content="Mera naam MyAI hai. Main local Claude-Code-style coding agent hoon.",
            )
        if lowered.startswith(("my name is ", "mera naam ")):
            name = stripped.split()[-1].strip(".,!?")
            return AgentAction(
                "final",
                content=f"Nice to meet you, {name}. Main abhi session mein tools aur repo context ke saath help kar sakta hoon.",
            )
        if lowered in {"status", "git status", "/status"}:
            return AgentAction("tool", tool="git_status", args={})
        if lowered in {"diff", "git diff", "/diff"}:
            return AgentAction("tool", tool="git_diff", args={})
        if lowered.startswith("list"):
            parts = shlex.split(stripped)
            path = parts[-1] if len(parts) > 1 and parts[-1] not in {"files", "file"} else "."
            return AgentAction("tool", tool="list_files", args={"path": path})
        if lowered.startswith(("read ", "show ", "open ")):
            parts = shlex.split(stripped)
            if len(parts) >= 2:
                return AgentAction("tool", tool="read_file", args={"path": parts[-1]})
        if lowered.startswith(("grep ", "search ")):
            parts = shlex.split(stripped)
            if len(parts) >= 2:
                pattern = parts[1]
                path = parts[2] if len(parts) >= 3 else "."
                return AgentAction("tool", tool="grep", args={"pattern": pattern, "path": path})
        if lowered.startswith(("run ", "bash ")):
            command = stripped.split(" ", 1)[1] if " " in stripped else ""
            return AgentAction("tool", tool="bash", args={"command": command})
        if lowered.startswith("/plan"):
            prompt = stripped[len("/plan") :].strip()
            content = "Plan mode: inspect relevant files, choose the smallest safe change, implement, then verify."
            if prompt:
                content += f"\nRequest: {prompt}"
            return AgentAction("final", content=content)
        if lowered.startswith("/review"):
            return AgentAction("tool", tool="git_diff", args={})

        return AgentAction(
            "final",
            content=(
                "I can help locally with repo tools. Try: list files, read AGENTS.md, "
                "search TransformerLM, status, diff, run python --version, or /help."
            ),
        )

    @staticmethod
    def help_text() -> str:
        return "\n".join(
            [
                "Commands:",
                "  /help       Show this help",
                "  /execute    Toggle bash/edit permission mode",
                "  /teach A => B  Save exact user-taught Q/A",
                "  /learn      Extra manual training boost",
                "  /diff       Show git diff",
                "  /memory     Show loaded memory files",
                "  /context    Show repo context summary",
                "  /plan TEXT  Draft a local action plan",
                "  /review     Review current diff",
                "",
                "Natural tool prompts:",
                "  list files",
                "  read PATH",
                "  search PATTERN [PATH]",
                "  run COMMAND",
                "  status",
                "  diff",
                "",
                "Auto-teach examples:",
                "  hi => hello",
                "  Q: hi A: hello",
                "  hi ka jawab hello hai",
                "  jab main hi kaho to hello bolna",
            ]
        )
