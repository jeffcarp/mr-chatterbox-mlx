"""Unit tests for chat_formatter.py."""

import pytest
from chat_formatter import (
    MrChatterboxFormatter,
    extract_dialogue_act,
    extract_text_content,
    strip_dialogue_act,
)


def test_extract_text_content_string():
    assert extract_text_content("hello") == "hello"
    assert extract_text_content("") == ""
    assert extract_text_content(None) == ""


def test_extract_text_content_structured_list():
    parts = [
        {"type": "text", "text": "Part 1 "},
        {"type": "other", "text": "Part 2"},
        " and part 3",
    ]
    assert extract_text_content(parts) == "Part 1 Part 2 and part 3"


def test_extract_dialogue_act():
    act, text = extract_dialogue_act("<act:return_greeting>Good day, sir!")
    assert act == "return_greeting"
    assert text == "Good day, sir!"

    act, text = extract_dialogue_act("No act here, sir.")
    assert act is None
    assert text == "No act here, sir."


def test_strip_dialogue_act():
    assert strip_dialogue_act("<act:explain>The matter is quite simple.") == "The matter is quite simple."
    assert strip_dialogue_act("Plain response") == "Plain response"


def test_format_chat_default_system():
    formatter = MrChatterboxFormatter(default_system_prompt="Custom system prompt.")
    messages = [
        {"role": "user", "content": "How do you do?"},
    ]
    prompt = formatter.format_chat(messages, add_generation_prompt=True)
    assert prompt == (
        "<|system|>Custom system prompt.<|end|>"
        "<|user|>How do you do?<|end|>"
        "<|assistant|>"
    )


def test_format_chat_explicit_system():
    formatter = MrChatterboxFormatter(default_system_prompt="Default")
    messages = [
        {"role": "system", "content": "Custom instructions."},
        {"role": "user", "content": "What year is it?"},
        {"role": "assistant", "content": "It is the year of our Lord 1888."},
        {"role": "user", "content": " Splendid."},
    ]
    prompt = formatter.format_chat(messages, add_generation_prompt=True)
    expected = (
        "<|system|>Custom instructions.<|end|>"
        "<|user|>What year is it?<|end|>"
        "<|assistant|>It is the year of our Lord 1888.<|end|>"
        "<|user|> Splendid.<|end|>"
        "<|assistant|>"
    )
    assert prompt == expected


def test_format_chat_no_generation_prompt():
    formatter = MrChatterboxFormatter(default_system_prompt=None)
    messages = [{"role": "user", "content": "Hello"}]
    prompt = formatter.format_chat(messages, add_generation_prompt=False)
    assert prompt == "<|user|>Hello<|end|>"


def test_get_stop_token_ids():
    class DummyTokenizer:
        end_token_id = 123
        end_token2_id = 119

        def token_to_id(self, token):
            if token == "<|end|>":
                return 123
            if token == "<|endoftext|>":
                return 119
            return -1

    formatter = MrChatterboxFormatter()
    stop_ids = formatter.get_stop_token_ids(DummyTokenizer())
    assert 123 in stop_ids
    assert 119 in stop_ids
