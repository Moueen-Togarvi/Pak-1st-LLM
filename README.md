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

`chat` mode supports `/help`, `/memory`, `/teach`, `/learn`, and `/quit`, but normal learning does not require commands.

`agent` mode supports `/help`, `/execute`, `/learn`, `/diff`, `/memory`, `/context`, `/plan`, and `/review`.

## Memory And Learning

Normal chat/agent messages are stored locally:

- `.myai/chat_history.jsonl` keeps session history.
- `.myai/facts.json` keeps simple remembered facts like your name.
- `data/raw/memory_conversations.txt` stores learnable conversation pairs.

By default, each new learnable chat/agent turn is saved and then auto-trained for 1 epoch. This can make the next prompt a little slower, but you do not need to run `/learn` manually.

Exact Q/A can be taught conversationally:

```text
hi
hello moueen
```

If the assistant does not know an answer, the next normal line you type becomes the answer for the previous input.

These explicit teaching styles also work if you want them:

```text
hi => hello moueen
hi ka jawab hello moueen hai
```

Useful controls:

```bash
.venv/bin/python main.py chat --no-auto-learn
.venv/bin/python main.py chat --auto-learn-every 3
.venv/bin/python main.py chat --auto-learn-epochs 2
```

Use `/learn` inside `chat` or `agent` only when you want an extra manual training boost, or run:

```bash
.venv/bin/python main.py train --epochs 5
```

Open-ended chat uses memory/fallback replies by default because the tiny byte model is not yet strong enough for reliable free generation. To deliberately test raw model generation later, create:

```bash
touch .myai/use_raw_model
```

Remove that file to return to stable fallback mode.

The first model is intentionally tiny and local. It needs much more curated data and training before chat output becomes useful.
