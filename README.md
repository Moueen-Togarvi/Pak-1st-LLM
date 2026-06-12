# Pak-1st-LLM

Local educational LLM plus a Claude-Code-style terminal agent.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py setup
```

## Commands

```bash
.venv/bin/python main.py info
.venv/bin/python main.py train --epochs 5
.venv/bin/python main.py eval
.venv/bin/python main.py chat
.venv/bin/python main.py agent
```

`agent` mode supports `/help`, `/execute`, `/diff`, `/memory`, `/context`, `/plan`, and `/review`.

The first model is intentionally tiny and local. It needs much more curated data and training before chat output becomes useful.
