from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .safety import is_destructive_command, is_path_inside, looks_secret_path, safe_command_args


IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
}


@dataclass
class ToolResult:
    success: bool
    content: str
    requires_permission: bool = False


class ToolRegistry:
    """Safe tool runner for the local coding agent."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or os.getcwd()).resolve()

    def run(self, name: str, args: dict[str, Any] | None = None, *, execute: bool = False) -> ToolResult:
        args = args or {}
        tools = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "grep": self.grep,
            "bash": self.bash,
            "apply_patch": self.apply_patch,
            "write_file": self.write_file,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
        }
        if name not in tools:
            return ToolResult(False, f"Unknown tool: {name}")
        try:
            if name in {"bash", "apply_patch", "write_file"}:
                return tools[name](execute=execute, **args)
            return tools[name](**args)
        except TypeError as exc:
            return ToolResult(False, f"Bad arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - tool errors must stay inside the agent loop.
            return ToolResult(False, f"{name} failed: {exc}")

    def resolve_path(self, path: str | Path = ".") -> Path:
        candidate = (self.repo_root / path).resolve()
        if not is_path_inside(candidate, self.repo_root):
            raise PermissionError(f"path escapes repo root: {path}")
        return candidate

    def list_files(self, path: str = ".", max_files: int = 200) -> ToolResult:
        max_files = max(1, min(int(max_files), 1000))
        root = self.resolve_path(path)
        if looks_secret_path(root):
            return ToolResult(False, "Refusing to list a secret-looking path.")
        if not root.exists():
            return ToolResult(False, f"Path not found: {path}")
        files: list[str] = []
        if root.is_file():
            files.append(str(root.relative_to(self.repo_root)))
        else:
            for current, dirs, filenames in os.walk(root):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not looks_secret_path(Path(d))]
                for filename in sorted(filenames):
                    file_path = Path(current) / filename
                    rel = file_path.relative_to(self.repo_root)
                    if looks_secret_path(rel):
                        continue
                    files.append(str(rel))
                    if len(files) >= max_files:
                        files.append(f"... truncated at {max_files} files")
                        return ToolResult(True, "\n".join(files))
        return ToolResult(True, "\n".join(files) or "(no files)")

    def read_file(self, path: str, max_bytes: int = 12000) -> ToolResult:
        max_bytes = max(1, min(int(max_bytes), 20000))
        file_path = self.resolve_path(path)
        if looks_secret_path(file_path):
            return ToolResult(False, "Refusing to read a secret-looking file.")
        if not file_path.exists() or not file_path.is_file():
            return ToolResult(False, f"File not found: {path}")
        with file_path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", errors="replace")
        numbered = "\n".join(f"{idx + 1:4d}: {line}" for idx, line in enumerate(text.splitlines()))
        if truncated:
            numbered += f"\n... truncated at {max_bytes} bytes"
        return ToolResult(True, numbered)

    def grep(self, pattern: str, path: str = ".", max_matches: int = 50) -> ToolResult:
        max_matches = max(1, min(int(max_matches), 500))
        if len(pattern) > 200:
            return ToolResult(False, "Pattern too long.")
        root = self.resolve_path(path)
        if looks_secret_path(root):
            return ToolResult(False, "Refusing to search a secret-looking path.")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(False, f"Invalid regex: {exc}")
        matches: list[str] = []
        targets = [root] if root.is_file() else self._iter_files(root)
        for file_path in targets:
            if not file_path.is_file() or looks_secret_path(file_path):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    rel = file_path.relative_to(self.repo_root)
                    matches.append(f"{rel}:{line_no}: {line}")
                    if len(matches) >= max_matches:
                        matches.append(f"... truncated at {max_matches} matches")
                        return ToolResult(True, "\n".join(matches))
        return ToolResult(True, "\n".join(matches) or "(no matches)")

    def bash(self, command: str, timeout: int = 10, *, execute: bool = False) -> ToolResult:
        timeout = max(1, min(int(timeout), 30))
        if is_destructive_command(command):
            return ToolResult(False, "Blocked risky or destructive command.")
        if not execute:
            return ToolResult(False, "Permission required. Type /execute to enable bash/edit tools.", True)
        try:
            args = safe_command_args(command)
        except ValueError as exc:
            return ToolResult(False, f"Invalid command: {exc}")
        proc = subprocess.run(
            args,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        if len(output) > 12000:
            output = output[:12000] + "\n... truncated at 12000 chars"
        return ToolResult(proc.returncode == 0, output or f"(exit {proc.returncode})")

    def write_file(self, path: str, content: str, *, execute: bool = False) -> ToolResult:
        if not execute:
            return ToolResult(False, "Permission required. Type /execute to enable write_file.", True)
        file_path = self.resolve_path(path)
        if looks_secret_path(file_path):
            return ToolResult(False, "Refusing to write a secret-looking file.")
        if len(content.encode("utf-8")) > 200000:
            return ToolResult(False, "Refusing to write content larger than 200KB.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {file_path.relative_to(self.repo_root)}")

    def apply_patch(self, patch: str, *, execute: bool = False) -> ToolResult:
        if not execute:
            return ToolResult(False, "Permission required. Type /execute to enable apply_patch.", True)
        binary = shutil.which("apply_patch")
        if binary is None:
            return ToolResult(False, "apply_patch command is not installed in this environment.")
        proc = subprocess.run(
            [binary],
            input=patch,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=20,
        )
        return ToolResult(proc.returncode == 0, proc.stdout + proc.stderr)

    def git_status(self) -> ToolResult:
        return self._git(["status", "--short"])

    def git_diff(self, path: str | None = None) -> ToolResult:
        cmd = ["diff", "--"]
        if path:
            resolved = self.resolve_path(path)
            cmd.append(str(resolved.relative_to(self.repo_root)))
        return self._git(cmd)

    def _git(self, args: list[str]) -> ToolResult:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=10,
        )
        output = proc.stdout + proc.stderr
        return ToolResult(proc.returncode == 0, output or "(empty)")

    def _iter_files(self, root: Path):
        for current, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not looks_secret_path(Path(d))]
            for filename in sorted(filenames):
                yield Path(current) / filename
