"""Unit tests for chat.py."""

import pytest
from chat import ChatSession


def test_chat_session_init():
    session = ChatSession(system_prompt="Custom system.", show_acts=True)
    assert session.system_prompt == "Custom system."
    assert session.show_acts is True
    assert len(session.history) == 0


def test_chat_session_add_messages():
    session = ChatSession()
    session.add_user_message("Good morning.")
    session.add_assistant_message("Good morning to you, sir.")
    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "Good morning."}
    assert session.history[1] == {"role": "assistant", "content": "Good morning to you, sir."}


def test_chat_session_clear():
    session = ChatSession()
    session.add_user_message("Test")
    session.clear()
    assert len(session.history) == 0


def test_chat_session_get_prompt():
    session = ChatSession(system_prompt="Victorian gentlemen only.")
    session.add_user_message("Pray, sir, what is your trade?")
    prompt = session.get_prompt()
    assert "<|system|>Victorian gentlemen only.<|end|>" in prompt
    assert "<|user|>Pray, sir, what is your trade?<|end|>" in prompt
    assert prompt.endswith("<|assistant|>")
