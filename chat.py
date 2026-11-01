"""Interactive terminal chat with Mr. Chatterbox using Keras 3 + MLX.

Talk directly with an eccentric 19th-century British gentleman powered by
KerasHub and Apple Silicon GPU acceleration.
"""

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Set

from chat_formatter import MrChatterboxFormatter, extract_dialogue_act
from model_loader import (
    DEFAULT_HF_REPO,
    generate_stream_tokens,
    load_mr_chatterbox,
)

# ANSI terminal colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ChatSession:
    """Manages the interactive multi-turn chat state and message history."""

    def __init__(
        self,
        system_prompt: str = "You are Mr. Chatterbox, an eccentric 19th-century British gentleman.",
        show_acts: bool = True,
    ):
        self.system_prompt = system_prompt
        self.show_acts = show_acts
        self.history: List[Dict[str, str]] = []
        self.formatter = MrChatterboxFormatter(default_system_prompt=system_prompt)

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self.history.clear()

    def get_prompt(self) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        return self.formatter.format_chat(messages, add_generation_prompt=True)


async def run_chat_loop(
    model: Any,
    tokenizer: Any,
    session: ChatSession,
    max_tokens: int = 512,
    stop_token_ids: Optional[Set[int]] = None,
    max_sequence_length: int = 2048,
):
    """Interactive REPL loop for terminal chat."""
    print(f"\n{BOLD}{CYAN}🎩 Welcome to Mr. Chatterbox! 🎩{RESET}")
    print(f"{DIM}Pretrained on 19th-century text. Accelerated with Apple MLX & Keras 3.{RESET}")
    print(f"{DIM}Commands: /clear (reset history), /acts (toggle acts), /history (view transcript), /exit{RESET}\n")

    if stop_token_ids is None:
        stop_token_ids = session.formatter.get_stop_token_ids(tokenizer)

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{MAGENTA}Good day to you, sir! Farewell.{RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(f"\n{CYAN}Mr. Chatterbox:{RESET} I bid you a very fond adieu, sir! Until our paths cross again.")
            break
        elif cmd == "/clear":
            session.clear()
            print(f"{DIM}[Conversation history cleared.]{RESET}\n")
            continue
        elif cmd == "/acts":
            session.show_acts = not session.show_acts
            state_str = "visible" if session.show_acts else "hidden"
            print(f"{DIM}[Dialogue acts are now {state_str}.]{RESET}\n")
            continue
        elif cmd == "/history":
            print(f"\n{DIM}--- Conversation History ---{RESET}")
            for msg in session.history:
                prefix = f"{GREEN}You:{RESET}" if msg["role"] == "user" else f"{CYAN}Mr. Chatterbox:{RESET}"
                print(f"{prefix} {msg['content']}")
            print(f"{DIM}----------------------------{RESET}\n")
            continue

        session.add_user_message(user_input)
        prompt = session.get_prompt()

        print(f"\n{BOLD}{CYAN}Mr. Chatterbox: {RESET}", end="", flush=True)

        full_response_parts: List[str] = []
        displayed_act = False

        try:
            async for piece in generate_stream_tokens(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                max_sequence_length=max_sequence_length,
            ):
                full_response_parts.append(piece)

                # Check if we need to format the leading dialogue act tag
                current_text = "".join(full_response_parts)
                if not displayed_act:
                    if ">" in current_text:
                        act, remaining = extract_dialogue_act(current_text)
                        if act:
                            if session.show_acts:
                                print(f"{YELLOW}{DIM}[Act: {act}]{RESET} ", end="", flush=True)
                            if remaining:
                                print(remaining, end="", flush=True)
                            displayed_act = True
                        else:
                            # Not an act tag, print accumulated text
                            print(current_text, end="", flush=True)
                            displayed_act = True
                    elif len(current_text) > 30 and not current_text.startswith("<act:"):
                        # No act tag present
                        print(piece, end="", flush=True)
                        displayed_act = True
                else:
                    print(piece, end="", flush=True)

            print("\n")
            full_text = "".join(full_response_parts)
            session.add_assistant_message(full_text)

        except KeyboardInterrupt:
            print(f"\n{DIM}[Interrupted generation]{RESET}\n")
            if full_response_parts:
                session.add_assistant_message("".join(full_response_parts))


def main():
    parser = argparse.ArgumentParser(description="Interactive terminal chat with Mr. Chatterbox (Keras + MLX)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Hugging Face repo or local snapshot path (default: tventurella/Mr.Chatterbox-2.7B-Chat)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute precision (default: bfloat16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per generation turn (default: 512)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable mx.compile Metal kernel fusion",
    )
    parser.add_argument(
        "--no-acts",
        action="store_true",
        help="Hide dialogue act tags (<act:...>)",
    )
    args = parser.parse_args()

    model, tokenizer, preprocessor = load_mr_chatterbox(
        repo_or_path=args.model,
        dtype=args.dtype,
        compile_metal=not args.no_compile,
    )

    session = ChatSession(show_acts=not args.no_acts)
    asyncio.run(
        run_chat_loop(
            model=model,
            tokenizer=tokenizer,
            session=session,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
