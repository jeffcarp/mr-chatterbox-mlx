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


def test_sample_token_mlx_argmax():
    import mlx.core as mx
    from model_loader import sample_token_mlx

    logits = mx.array([[[1.0, 5.0, 2.0]]])
    # Temperature 0 => pure argmax
    token = sample_token_mlx(logits, context_ids=[], temperature=0.0)
    assert token == 1


def test_sample_token_mlx_repetition_penalty():
    import mlx.core as mx
    from model_loader import sample_token_mlx

    # Token 1 has logit 5.0, Token 2 has logit 4.5
    logits = mx.array([[[1.0, 5.0, 4.5]]])
    # If token 1 is in context_ids with repetition penalty 2.0, its logit drops to 2.5
    # Token 2 (4.5) will now be highest
    token = sample_token_mlx(logits, context_ids=[1], temperature=0.0, repetition_penalty=2.0)
    assert token == 2


def test_sample_token_mlx_choose_from_picks_best_act():
    import mlx.core as mx
    from model_loader import sample_token_mlx

    # Token 0 is the overall favourite, but only tokens 2 and 3 (the "acts") may be chosen.
    logits = mx.array([[[9.0, 1.0, 2.0, 3.0]]])
    token = sample_token_mlx(logits, context_ids=[], temperature=0.3, choose_from=[2, 3])
    assert token == 3


def test_sample_token_mlx_banned_ids():
    import mlx.core as mx
    from model_loader import sample_token_mlx

    # Token 1 would win, but it is banned (an act token mid-reply).
    logits = mx.array([[[1.0, 5.0, 2.0]]])
    token = sample_token_mlx(logits, context_ids=[], temperature=0.0, banned_ids=[1])
    assert token == 2


def test_sample_token_mlx_penalty_spares_exempt_ids():
    import mlx.core as mx
    from model_loader import sample_token_mlx

    # Token 1 stands in for <|end|>: it is in the context (every turn closes with it), but the
    # repetition penalty must not touch it, or the model could never stop.
    logits = mx.array([[[1.0, 5.0, 4.5]]])
    token = sample_token_mlx(
        logits, context_ids=[1], temperature=0.0, repetition_penalty=2.0, exempt_ids={1}
    )
    assert token == 1


def test_special_token_ids():
    from model_loader import special_token_ids

    class DummyTokenizer:
        vocabulary = {
            "hello": 5,
            "<|end|>": 123,
            "<|endoftext|>": 119,
            "<|user|>": 120,
            "<|assistant|>": 121,
            "<act:return_greeting>": 65537,
            "<act:explain>": 65536,
        }

    act_ids, exempt = special_token_ids(DummyTokenizer())
    assert act_ids == [65536, 65537]
    assert {119, 123, 120, 121, 65536, 65537} <= exempt
    assert 5 not in exempt
