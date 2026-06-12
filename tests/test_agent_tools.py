import tempfile
import unittest
from pathlib import Path

from agent.agent import LocalAgent
from tools.registry import ToolRegistry


class AgentToolsTest(unittest.TestCase):
    def test_read_search_and_permission_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("hello\nneedle here\n", encoding="utf-8")
            tools = ToolRegistry(root)

            read = tools.run("read_file", {"path": "sample.txt"})
            self.assertTrue(read.success)
            self.assertIn("needle here", read.content)

            grep = tools.run("grep", {"pattern": "needle", "path": "."})
            self.assertTrue(grep.success)
            self.assertIn("sample.txt:2", grep.content)

            denied = tools.run("write_file", {"path": "out.txt", "content": "x"})
            self.assertTrue(denied.requires_permission)
            self.assertFalse((root / "out.txt").exists())

    def test_destructive_command_blocked_even_with_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry(tmp)
            result = tools.run("bash", {"command": "rm -rf ."}, execute=True)
            self.assertFalse(result.success)
            self.assertIn("Blocked risky", result.content)

            variant = tools.run("bash", {"command": "rm -r -f ."}, execute=True)
            self.assertFalse(variant.success)
            self.assertIn("Blocked risky", variant.content)

            code_exec = tools.run("bash", {"command": "python -c 'print(123)'"}, execute=True)
            self.assertFalse(code_exec.success)
            self.assertIn("Blocked risky", code_exec.content)

    def test_shell_separators_do_not_execute_second_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = ToolRegistry(root)
            result = tools.run("bash", {"command": "echo ok; touch pwned"}, execute=True)
            self.assertTrue(result.success)
            self.assertFalse((root / "pwned").exists())

    def test_grep_skips_venv_and_rejects_bad_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv").mkdir()
            (root / ".venv" / "hidden.txt").write_text("needle", encoding="utf-8")
            (root / "visible.txt").write_text("needle", encoding="utf-8")
            tools = ToolRegistry(root)
            result = tools.run("grep", {"pattern": "needle", "path": "."})
            self.assertTrue(result.success)
            self.assertIn("visible.txt", result.content)
            self.assertNotIn(".venv", result.content)

            bad_regex = tools.run("grep", {"pattern": "(", "path": "."})
            self.assertFalse(bad_regex.success)
            self.assertIn("Invalid regex", bad_regex.content)

    def test_tokenizer_path_is_not_mistaken_for_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenizer").mkdir()
            (root / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
            (root / ".env").write_text("SECRET=x", encoding="utf-8")
            tools = ToolRegistry(root)
            allowed = tools.run("read_file", {"path": "tokenizer/vocab.json"})
            blocked = tools.run("read_file", {"path": ".env"})
            self.assertTrue(allowed.success)
            self.assertFalse(blocked.success)

    def test_agent_execute_toggle_allows_safe_bash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = LocalAgent(tmp)
            before = agent.handle("run python --version")
            self.assertIn("PERMISSION", before.content)
            agent.handle("/execute")
            after = agent.handle("run python --version")
            self.assertIn("bash", after.content)
            self.assertNotIn("PERMISSION", after.content)


if __name__ == "__main__":
    unittest.main()
