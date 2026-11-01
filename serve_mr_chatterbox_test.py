"""Unit tests for serve_mr_chatterbox.py."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

import serve_mr_chatterbox
from serve_mr_chatterbox import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "mlx"


def test_models_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "mr-chatterbox" in ids
    assert "tventurella/Mr.Chatterbox-2.7B-Chat" in ids


def test_chat_completions_non_streaming():
    with patch("serve_mr_chatterbox.generate_full_text", return_value="Good day to you, sir! I am Mr. Chatterbox."):
        # Set dummy model state
        serve_mr_chatterbox.active_model = MagicMock()
        serve_mr_chatterbox.active_tokenizer = MagicMock()

        payload = {
            "model": "mr-chatterbox",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Good day to you, sir! I am Mr. Chatterbox."
        assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming():
    async def mock_stream_tokens(*args, **kwargs):
        yield "Good "
        yield "day, "
        yield "sir!"

    with patch("serve_mr_chatterbox.generate_stream_tokens", side_effect=mock_stream_tokens):
        serve_mr_chatterbox.active_model = MagicMock()
        serve_mr_chatterbox.active_tokenizer = MagicMock()

        payload = {
            "model": "mr-chatterbox",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = response.text
        assert "data: " in text
        assert "[DONE]" in text
        assert "Good " in text
        assert "sir!" in text
