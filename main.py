from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.agent import LocalAgent
from data.data_prep import build_training_sequences
from interface.terminal_ui import TerminalUI
from myai.memory_store import MemoryStore
from myai.runtime import ensure_project_setup, generate_reply, is_fallback_reply, load_model, load_tokenizer, project_info
from training.gradient_check import check_output_projection_gradient
from training.trainer import Trainer, TrainingConfig


REPO_ROOT = Path(__file__).resolve().parent


def cmd_setup(_: argparse.Namespace) -> int:
    ensure_project_setup(REPO_ROOT)
    print("Setup complete.")
    print("Created/verified tokenizer, model weights, sample data, memory, and local skills.")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    ensure_project_setup(REPO_ROOT)
    info = project_info(REPO_ROOT)
    print(json.dumps(info, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    ensure_project_setup(REPO_ROOT)
    tokenizer = load_tokenizer(REPO_ROOT)
    model = load_model(REPO_ROOT)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
        save_every=args.save_every,
    )
    trainer = Trainer(model, tokenizer, config)
    trainer.train_from_data(
        data_path=REPO_ROOT / "data" / "raw" / "conversations.txt",
        raw_folder=REPO_ROOT / "data" / "raw",
        weights_path=REPO_ROOT / "model" / "weights.npz",
        config_path=REPO_ROOT / "model" / "weights_config.json",
    )
    print("Saved model to model/weights.npz")
    return 0


def cmd_eval(_: argparse.Namespace) -> int:
    ensure_project_setup(REPO_ROOT)
    tokenizer = load_tokenizer(REPO_ROOT)
    model = load_model(REPO_ROOT)
    sequences = build_training_sequences(
        tokenizer,
        conversations_path=REPO_ROOT / "data" / "raw" / "conversations.txt",
        raw_folder=REPO_ROOT / "data" / "raw",
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
    ensure_project_setup(REPO_ROOT)
    tokenizer = load_tokenizer(REPO_ROOT)
    model = load_model(REPO_ROOT)
    memory = MemoryStore(REPO_ROOT)
    ui = TerminalUI()
    ui.header("MYAI Local Chat", "Tiny local model with starter-data fallback")
    pending_teach_input: str | None = None
    while True:
        text = ui.prompt().strip()
        if text.lower() in {"/quit", "quit", "exit"}:
            break
        if text.lower() == "/help":
            ui.print_reply(
                "\n".join(
                    [
                        "Chat commands:",
                        "  /help    Show this help",
                        "  /memory  Show saved chat memory",
                        "  /teach input => output",
                        "  /learn   Extra manual training boost",
                        "  /quit    Exit chat",
                        "",
                        "Normal baatein user text ki tarah save/train hoti hain.",
                        "Agar jawab nahi aata, next line mein bas sahi jawab likh do.",
                        "For repo tools use: .venv/bin/python main.py agent",
                    ]
                )
            )
            continue
        if text.lower() == "/memory":
            ui.print_reply(memory.summary())
            continue
        if text.lower() == "/learn":
            ui.print_reply(f"Training from saved data and memory for {args.learn_epochs} epoch(s)...")
            train_now(model, tokenizer, args)
            ui.print_reply("Learning complete. New weights saved to model/weights.npz")
            continue
        if text.lower().startswith("/teach "):
            taught = teach_from_command(memory, text)
            ui.print_reply(taught)
            auto_learn_if_needed(ui, model, tokenizer, memory, args, "Saved taught pair" in taught)
            pending_teach_input = None
            continue
        auto_taught, taught_message = memory.auto_teach_from_text(text)
        if taught_message:
            ui.print_reply(taught_message)
            auto_learn_if_needed(ui, model, tokenizer, memory, args, auto_taught)
            pending_teach_input = None
            continue
        if not text:
            continue
        if pending_teach_input and should_capture_pending_answer(text):
            added = memory.teach_pair(pending_teach_input, text)
            if added:
                ui.print_reply(f"Learned: '{pending_teach_input}' ka jawab '{text}'")
            else:
                ui.print_reply("Ye answer pehle se saved hai.")
            auto_learn_if_needed(ui, model, tokenizer, memory, args, added)
            pending_teach_input = None
            continue
        reply = generate_reply(model, tokenizer, text, max_new_tokens=args.max_new_tokens, repo_root=REPO_ROOT)
        ui.print_reply(reply)
        added_pair = memory.remember_turn(text, reply, source="chat")
        auto_learn_if_needed(ui, model, tokenizer, memory, args, added_pair)
        pending_teach_input = text if is_fallback_reply(reply) else None
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    ensure_project_setup(REPO_ROOT)
    ui = TerminalUI()
    agent = LocalAgent(REPO_ROOT)
    memory = MemoryStore(REPO_ROOT)
    ui.header("MYAI Local Coding Agent", "Claude-Code-style repo tools with permission gates")
    while True:
        text = ui.prompt().strip()
        if text.lower() in {"/quit", "quit", "exit"}:
            break
        if text.lower() == "/learn":
            ui.print_reply(f"Training from saved data and memory for {args.learn_epochs} epoch(s)...")
            tokenizer = load_tokenizer(REPO_ROOT)
            model = load_model(REPO_ROOT)
            train_now(model, tokenizer, args)
            ui.print_reply("Learning complete. New weights saved to model/weights.npz")
            continue
        if text.lower().startswith("/teach "):
            memory = MemoryStore(REPO_ROOT)
            taught = teach_from_command(memory, text)
            ui.print_reply(taught)
            auto_learn_if_needed(ui, None, None, memory, args, "Saved taught pair" in taught)
            continue
        if not text:
            continue
        auto_taught, taught_message = memory.auto_teach_from_text(text)
        if taught_message:
            ui.print_reply(taught_message)
            auto_learn_if_needed(ui, None, None, memory, args, auto_taught)
            continue
        before_pairs = memory.learnable_pair_count()
        reply = agent.handle(text)
        ui.print_reply(reply.content)
        after_pairs = memory.learnable_pair_count()
        auto_learn_if_needed(ui, None, None, memory, args, after_pairs > before_pairs)
    return 0


def train_now(model, tokenizer, args: argparse.Namespace) -> None:
    config = TrainingConfig(
        epochs=args.learn_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_clip=args.grad_clip,
    )
    trainer = Trainer(model, tokenizer, config)
    trainer.train_from_data(
        data_path=REPO_ROOT / "data" / "raw" / "conversations.txt",
        raw_folder=REPO_ROOT / "data" / "raw",
        weights_path=REPO_ROOT / "model" / "weights.npz",
        config_path=REPO_ROOT / "model" / "weights_config.json",
    )


def teach_from_command(memory: MemoryStore, text: str) -> str:
    payload = text[len("/teach ") :].strip()
    if "=>" not in payload:
        return "Format: /teach input => output"
    user_text, assistant_text = [part.strip() for part in payload.split("=>", 1)]
    if not user_text or not assistant_text:
        return "Format: /teach input => output"
    added = memory.teach_pair(user_text, assistant_text)
    if added:
        return "Saved taught pair. Yehi Q/A training data mein add hua."
    return "Ye taught pair pehle se saved hai."


def should_capture_pending_answer(text: str) -> bool:
    normalized = text.lower().strip()
    if not normalized or normalized.startswith("/"):
        return False
    if normalized in {"quit", "exit", "skip", "cancel", "choro", "chor do"}:
        return False
    question_starts = ("what ", "why ", "how ", "kya ", "kia ", "kaise ", "kesy ", "mera ", "tum ")
    if text.rstrip().endswith("?") or normalized.startswith(question_starts):
        return False
    return True


def auto_learn_if_needed(
    ui: TerminalUI,
    model,
    tokenizer,
    memory: MemoryStore,
    args: argparse.Namespace,
    added_pair: bool,
) -> None:
    if not getattr(args, "auto_learn", False) or not added_pair:
        return
    every = max(1, int(args.auto_learn_every))
    pair_count = memory.learnable_pair_count()
    if pair_count == 0 or pair_count % every != 0:
        return

    if model is None or tokenizer is None:
        tokenizer = load_tokenizer(REPO_ROOT)
        model = load_model(REPO_ROOT)
    ui.print_reply(
        f"Auto-learning from saved memory: {args.auto_learn_epochs} epoch(s), {pair_count} learnable pair(s)."
    )
    train_args = argparse.Namespace(
        learn_epochs=args.auto_learn_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    train_now(model, tokenizer, train_args)
    ui.print_reply("Auto-learning complete.")


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
    chat.add_argument("--learn-epochs", type=int, default=3)
    chat.add_argument("--auto-learn", action=argparse.BooleanOptionalAction, default=True)
    chat.add_argument("--auto-learn-every", type=int, default=1)
    chat.add_argument("--auto-learn-epochs", type=int, default=1)
    chat.add_argument("--batch-size", type=int, default=4)
    chat.add_argument("--lr", type=float, default=1e-3)
    chat.add_argument("--grad-clip", type=float, default=1.0)
    chat.set_defaults(func=cmd_chat)

    agent = sub.add_parser("agent", help="start Claude-Code-style local agent mode")
    agent.add_argument("--learn-epochs", type=int, default=3)
    agent.add_argument("--auto-learn", action=argparse.BooleanOptionalAction, default=True)
    agent.add_argument("--auto-learn-every", type=int, default=1)
    agent.add_argument("--auto-learn-epochs", type=int, default=1)
    agent.add_argument("--batch-size", type=int, default=4)
    agent.add_argument("--lr", type=float, default=1e-3)
    agent.add_argument("--grad-clip", type=float, default=1.0)
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
