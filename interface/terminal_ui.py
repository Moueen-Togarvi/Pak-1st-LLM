from __future__ import annotations

import os
import shutil
import sys
import textwrap


class TerminalUI:
    def __init__(self) -> None:
        self.use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    def color(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def header(self) -> None:
        width = min(shutil.get_terminal_size((80, 20)).columns, 96)
        line = "=" * width
        print(self.color(line, "96"))
        print(self.color("MYAI Local Coding Agent", "1;96"))
        print("Owned tiny LLM + Claude-Code-style local tools")
        print("Type /help for commands. Type Ctrl-D or /quit to exit.")
        print(self.color(line, "96"))

    def prompt(self) -> str:
        try:
            return input(self.color("myai> ", "92"))
        except EOFError:
            return "/quit"

    def print_reply(self, text: str) -> None:
        width = min(shutil.get_terminal_size((80, 20)).columns, 100)
        print()
        for paragraph in text.splitlines() or [""]:
            if not paragraph:
                print()
                continue
            print(textwrap.fill(paragraph, width=width, replace_whitespace=False))
        print()

