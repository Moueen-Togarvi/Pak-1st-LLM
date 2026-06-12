from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.agent import LocalAgent
from data.data_prep import build_training_sequences
from interface.terminal_ui import TerminalUI
from myai.runtime import ensure_project_setup, generate_reply, load_model, load_tokenizer, project_info
from training.gradient_check import check_output_projection_gradient
from training.trainer import Trainer, TrainingConfig


def cmd_setup(_: argparse.Namespace) -> int:
    ensure_project_setup(".")
    print("Setup complete.")
    print("Created/verified tokenizer, model weights, sample data, memory, and local skills.")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    ensure_project_setup(".")
    info = project_info(".")
    print(json.dumps(info, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    ensure_project_setup(".")
    tokenizer = load_tokenizer(".")
    model = load_model(".")
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        save_every=args.save_every,
    )
    trainer = Trainer(model, tokenizer, config)
    trainer.train_from_data()
    print("Saved model to model/weights.npz")
    return 0


def cmd_eval(_: argparse.Namespace) -> int:
    ensure_project_setup(".")
    tokenizer = load_tokenizer(".")
    model = load_model(".")
    sequences = build_training_sequences(
        tokenizer,
        conversations_path=Path("data/raw/conversations.txt"),
        raw_folder=Path("data/raw"),
        max_length=model.config.max_seq_len + 1,
    )
    trainer = Trainer(model, tokenizer, TrainingConfig(epochs=1))
    loss = trainer.evaluate_examples(sequences)
    grad = check_output_projection_gradient()
    print(f"eval_loss={loss:.4f}")
    print(
        "gradient_check "
        f"numeric={grad['numeric']:.6f} analytic={grad['analytic']:.6f} "
        f"abs_error={grad['abs_error']:.6f}"
    )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    ensure_project_setup(".")
    tokenizer = load_tokenizer(".")
    model = load_model(".")
    ui = TerminalUI()
    ui.header()
    while True:
        text = ui.prompt().strip()
        if text.lower() in {"/quit", "quit", "exit"}:
            break
        if not text:
            continue
        reply = generate_reply(model, tokenizer, text, max_new_tokens=args.max_new_tokens)
        ui.print_reply(reply)
    return 0


def cmd_agent(_: argparse.Namespace) -> int:
    ensure_project_setup(".")
    ui = TerminalUI()
    agent = LocalAgent(".")
    ui.header()
    while True:
        text = ui.prompt().strip()
        if text.lower() in {"/quit", "quit", "exit"}:
            break
        if not text:
            continue
        reply = agent.handle(text)
        ui.print_reply(reply.content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pak-1st-LLM local model and coding agent")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="create local tokenizer/model/data scaffolding")
    setup.set_defaults(func=cmd_setup)

    info = sub.add_parser("info", help="show model and project information")
    info.set_defaults(func=cmd_info)

    train = sub.add_parser("train", help="train the tiny local model")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--save-every", type=int, default=5)
    train.set_defaults(func=cmd_train)

    chat = sub.add_parser("chat", help="chat with the tiny local model")
    chat.add_argument("--max-new-tokens", type=int, default=80)
    chat.set_defaults(func=cmd_chat)

    agent = sub.add_parser("agent", help="start Claude-Code-style local agent mode")
    agent.set_defaults(func=cmd_agent)

    eval_cmd = sub.add_parser("eval", help="evaluate loss and run a gradient check")
    eval_cmd.set_defaults(func=cmd_eval)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

