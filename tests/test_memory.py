import tempfile
import unittest
from pathlib import Path

from agent.memory import MemoryLoader


class MemoryTest(unittest.TestCase):
    def test_loads_agents_claude_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".myai").mkdir()
            (root / "AGENTS.md").write_text("agents rule", encoding="utf-8")
            (root / "CLAUDE.md").write_text("claude rule", encoding="utf-8")
            (root / ".myai" / "memory.md").write_text("memory rule", encoding="utf-8")
            rendered = MemoryLoader(root).render()
            self.assertIn("agents rule", rendered)
            self.assertIn("claude rule", rendered)
            self.assertIn("memory rule", rendered)


if __name__ == "__main__":
    unittest.main()

