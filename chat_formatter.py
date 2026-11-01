"""Chat template formatting for Mr. Chatterbox.

Formats OpenAI-style message lists ([{"role": "...", "content": "..."}]) into
Mr. Chatterbox's 19th-century conversational format:
    <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>

Also provides dialogue act parsing (<act:...>) and stop token resolution.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_text_content(content: Any) -> str:
    """Extracts plain text from string or structured content parts (e.g. AI SDK)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(str(p["text"]))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


# Regex to detect dialogue acts emitted by Mr. Chatterbox
DIALOGUE_ACT_PATTERN = re.compile(r"^\s*<act:([a-zA-Z0-9_]+)>\s*")


def extract_dialogue_act(text: str) -> Tuple[Optional[str], str]:
    """Extracts leading dialogue act tag if present, returning (act_name, remaining_text).

    Example:
        '<act:return_greeting>Good day, sir!' -> ('return_greeting', 'Good day, sir!')
    """
    match = DIALOGUE_ACT_PATTERN.match(text)
    if match:
        act = match.group(1)
        remaining = text[match.end():]
        return act, remaining
    return None, text


def strip_dialogue_act(text: str) -> str:
    """Removes leading dialogue act tag if present."""
    _, remaining = extract_dialogue_act(text)
    return remaining


class MrChatterboxFormatter:
    """Chat formatter for Mr. Chatterbox (2.7B 19th-century conversational model)."""

    name: str = "mr_chatterbox"
    stop_strings: List[str] = ["<|end|>", "<|endoftext|>"]

    def __init__(
        self,
        default_system_prompt: Optional[str] = "You are Mr. Chatterbox, an eccentric 19th-century British gentleman.",
    ):
        self.default_system_prompt = default_system_prompt

    def format_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats OpenAI-style message dicts into Mr. Chatterbox's prompt representation."""
        formatted_parts: List[str] = []

        # Check if first message is a system message
        has_system = bool(messages and messages[0].get("role") == "system")
        if has_system:
            sys_content = extract_text_content(messages[0].get("content", ""))
            if sys_content:
                formatted_parts.append(f"<|system|>{sys_content}<|end|>")
            remaining_messages = messages[1:]
        elif self.default_system_prompt:
            formatted_parts.append(f"<|system|>{self.default_system_prompt}<|end|>")
            remaining_messages = messages
        else:
            remaining_messages = messages

        for msg in remaining_messages:
            role = msg.get("role", "user")
            content = extract_text_content(msg.get("content", ""))

            if role in ("system", "developer"):
                formatted_parts.append(f"<|system|>{content}<|end|>")
            elif role == "user":
                formatted_parts.append(f"<|user|>{content}<|end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>{content}<|end|>")
            else:
                # Fallback to user role for unknown types
                formatted_parts.append(f"<|user|>{content}<|end|>")

        if add_generation_prompt:
            formatted_parts.append("<|assistant|>")

        return "".join(formatted_parts)

    def get_stop_token_ids(self, tokenizer: Any) -> Set[int]:
        """Resolves stop token IDs from the tokenizer."""
        stop_ids: Set[int] = set()

        # Check direct attributes on tokenizer
        for attr in ("end_token_id", "end_token2_id"):
            val = getattr(tokenizer, attr, None)
            if val is not None:
                stop_ids.add(int(val))

        # Check token_to_id lookup
        if hasattr(tokenizer, "token_to_id"):
            for s in self.stop_strings:
                try:
                    tid = tokenizer.token_to_id(s)
                    if tid is not None and tid != -1:
                        stop_ids.add(int(tid))
                except Exception:
                    pass

        # Also fallback to Mr. Chatterbox's known stop token IDs (123: <|end|>, 119: <|endoftext|>)
        stop_ids.add(123)
        stop_ids.add(119)

        return stop_ids
