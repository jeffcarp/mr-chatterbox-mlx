"""Unit tests for model_loader.py."""

import pytest
from model_loader import (
    DEFAULT_HF_REPO,
    KNOWN_MODELS,
    compute_sequence_length,
    download_model_if_needed,
)


def test_known_models():
    assert "2.7b" in KNOWN_MODELS
    assert KNOWN_MODELS["2.7b"] == DEFAULT_HF_REPO
    assert "1.2b" in KNOWN_MODELS
    assert "500m" in KNOWN_MODELS
    assert "345m" in KNOWN_MODELS


def test_compute_sequence_length():
    # Prompt 50 tokens, max_tokens 100 -> needed min(max(50+100+32, 512), 2048) = 512
    assert compute_sequence_length(50, 100, max_sequence_length=2048) == 512

    # Prompt 500 tokens, max_tokens 300 -> 500+300+32 = 832 -> rounded up to 512 boundary = 1024
    assert compute_sequence_length(500, 300, max_sequence_length=2048) == 1024

    # Large prompt capped at max_sequence_length
    assert compute_sequence_length(1800, 500, max_sequence_length=2048) == 2048


def test_download_model_if_needed_existing_path(tmp_path):
    # If path already exists locally, returns it directly
    local_dir = str(tmp_path / "dummy_model")
    import os
    os.makedirs(local_dir, exist_ok=True)
    assert download_model_if_needed(local_dir) == local_dir
