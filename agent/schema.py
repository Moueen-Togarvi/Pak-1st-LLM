from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


ActionType = Literal["final", "tool"]


@dataclass
class AgentAction:
    action: ActionType
    content: str = ""
    tool: str = ""
    args: dict[str, Any] | None = None

    def to_json(self) -> str:
        if self.action == "final":
            return json.dumps({"action": "final", "content": self.content}, ensure_ascii=False)
        return json.dumps(
            {"action": "tool", "tool": self.tool, "args": self.args or {}},
            ensure_ascii=False,
        )


def parse_action(text: str) -> AgentAction | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("action") == "final":
        return AgentAction("final", content=str(payload.get("content", "")))
    if payload.get("action") == "tool":
        args = payload.get("args", {})
        return AgentAction("tool", tool=str(payload.get("tool", "")), args=args if isinstance(args, dict) else {})
    return None

