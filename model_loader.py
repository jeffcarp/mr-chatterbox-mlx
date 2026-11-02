"""Model loader and MLX generation engine for Mr. Chatterbox.

Loads Mr. Chatterbox models (e.g. tventurella/Mr.Chatterbox-2.7B-Chat) into KerasHub
backed by Apple Silicon Metal GPU acceleration via the keras-mlx pluggable backend.
"""

import atexit
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

# Ensure Keras backend is MLX prior to importing keras/keras_hub
os.environ.setdefault("KERAS_BACKEND", "mlx")

DEFAULT_HF_REPO = "tventurella/Mr.Chatterbox-2.7B-Chat"
KNOWN_MODELS = {
    "2.7b": "tventurella/Mr.Chatterbox-2.7B-Chat",
    "2.7b-base": "tventurella/Mr.Chatterbox-2.7B",
    "1.2b": "tventurella/Mr.Chatterbox-1.2B",
    "500m": "tventurella/Mr.Chatterbox-500M",
    "345m": "tventurella/Mr.Chatterbox-345M",
}


def _register_tf_mlx_converter():
    """Registers mlx.core.array converter for TensorFlow to support KerasHub tokenizers."""
    try:
        import mlx.core as mx
        import numpy as np
        import tensorflow as tf

        tf.register_tensor_conversion_function(
            mx.array,
            lambda val, dtype=None, name=None, as_ref=False: tf.convert_to_tensor(
                np.array(val), dtype=dtype, name=name
            ),
        )
    except Exception:
        pass


def download_model_if_needed(repo_or_path: str) -> str:
    """Downloads model files from Hugging Face Hub if repo_or_path is a HF repo ID.

    Returns the local path containing model snapshot.
    """
    if os.path.exists(repo_or_path):
        return repo_or_path

    # Check shorthand alias
    repo_id = KNOWN_MODELS.get(repo_or_path.lower(), repo_or_path)

    # Strip hf:// prefix if provided
    if repo_id.startswith("hf://"):
        repo_id = repo_id[5:]

    from huggingface_hub import snapshot_download

    print(f"Checking/downloading model weights from Hugging Face: {repo_id}...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "*.safetensors",
        ],
    )
    print(f"Model snapshot located at: {local_dir}")
    return local_dir


def load_tokenizer_and_preprocessor(
    snapshot_path: str,
    sequence_length: int = 2048,
):
    """Loads and configures the BytePairTokenizer with special tokens for Mr. Chatterbox."""
    from keras_hub.src.models.llama3.llama3_causal_lm_preprocessor import (
        Llama3CausalLMPreprocessor,
    )
    from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
    from keras_hub.src.utils.preset_utils import load_json

    tok_path = os.path.join(snapshot_path, "tokenizer.json")
    cfg_path = os.path.join(snapshot_path, "tokenizer_config.json")

    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab = tok_data["model"]["vocab"]
    merges = tok_data["model"]["merges"]
    if merges and isinstance(merges[0], list) and len(merges[0]) == 2:
        merges = [" ".join(m) for m in merges]

    special_tokens = set()
    for item in tok_data.get("added_tokens", []):
        content = item.get("content")
        if content and not content.startswith("<|reserved_special_token_"):
            vocab[content] = item["id"]
            special_tokens.add(content)

    bos_token = None
    eos_token = "<|endoftext|>"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
            bos_token = cfg_data.get("bos_token")
            eos_token = cfg_data.get("eos_token") or "<|endoftext|>"
    # Mr. Chatterbox has no BOS token (bos_token: null). KerasHub's tokenizer puts bos into its list of
    # unsplittable tokens, and a None in that list fails in TensorFlow ("Can't convert Python sequence
    # with mixed types to Tensor"). Stand in with the EOS token: add_start_token=False below, so it is
    # never inserted into a prompt.
    if not bos_token:
        bos_token = eos_token

    misc_tokens = {t for t in special_tokens if t in vocab}
    if bos_token:
        misc_tokens.discard(bos_token)
    if eos_token:
        misc_tokens.discard(eos_token)

    tokenizer = Llama3Tokenizer(
        vocabulary=vocab,
        merges=merges,
        bos_token=bos_token,
        eos_token=eos_token,
        misc_special_tokens=misc_tokens,
    )

    preprocessor = Llama3CausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        add_start_token=False,
        add_end_token=False,
    )
    return tokenizer, preprocessor


def load_mr_chatterbox(
    repo_or_path: str = DEFAULT_HF_REPO,
    dtype: str = "bfloat16",
    compile_metal: bool = True,
    max_sequence_length: int = 2048,
) -> Tuple[Any, Any, Any]:
    """Loads Mr. Chatterbox model into KerasHub with MLX backend.

    Args:
        repo_or_path: Hugging Face repo ID (e.g. 'tventurella/Mr.Chatterbox-2.7B-Chat')
                      or local directory containing model snapshot.
        dtype: Compute precision ('bfloat16', 'float16', 'float32').
        compile_metal: Whether to compile call_with_cache using mx.compile for kernel fusion.
        max_sequence_length: Maximum sequence context buffer.

    Returns:
        Tuple of (model, tokenizer, preprocessor).
    """
    _register_tf_mlx_converter()

    import keras
    import keras_hub
    import mlx.core as mx

    # Register stream cleanup on exit
    if hasattr(mx, "clear_streams"):
        atexit.register(mx.clear_streams)
    if hasattr(mx, "compile_clear_cache"):
        atexit.register(mx.compile_clear_cache)

    snapshot_path = download_model_if_needed(repo_or_path)

    print("=" * 65)
    print(f"Loading Mr. Chatterbox from: {snapshot_path}")
    print(f"Backend: {keras.config.backend()} | Device: {mx.default_device()} | Dtype: {dtype}")
    print("=" * 65)

    t0 = time.time()
    try:
        # Load directly via KerasHub's from_preset loader
        model = keras_hub.models.Llama3CausalLM.from_preset(snapshot_path, dtype=dtype)
        tokenizer = model.preprocessor.tokenizer
        preprocessor = model.preprocessor
    except Exception as e:
        print(f"Direct preset load encountered: {e}. Falling back to explicit architecture assembly...")
        backbone = keras_hub.models.Llama3Backbone.from_preset(snapshot_path, dtype=dtype)
        tokenizer, preprocessor = load_tokenizer_and_preprocessor(
            snapshot_path, sequence_length=max_sequence_length
        )
        model = keras_hub.models.Llama3CausalLM(backbone=backbone, preprocessor=preprocessor)

    elapsed = time.time() - t0
    params = model.count_params()
    print(f"Model successfully loaded in {elapsed:.2f}s! Total parameters: {params:,}")

    if compile_metal:
        print("Fusing Metal decoding kernels via mx.compile(model.call_with_cache)...")
        uncompiled = model.call_with_cache
        model.call_with_cache = mx.compile(uncompiled)
        # Some keras-mlx builds read an array's value inside call_with_cache (slice_update calls .item()
        # on the cache index), which mx.compile forbids ("Attempting to eval an array during function
        # transformations"). The error only appears on the first decode step, so probe one step now and
        # fall back to uncompiled decoding rather than failing mid-conversation.
        try:
            import keras

            probe = mx.zeros((1, 8), dtype=mx.int32)
            _, probe_cache = model._build_cache(probe)
            logits, _, _ = model.call_with_cache(
                keras.ops.slice(probe, [0, 0], [1, 1]), probe_cache, mx.array(0, dtype=mx.int32)
            )
            mx.eval(logits)
            print("Metal kernel fusion complete!")
        except Exception as e:
            model.call_with_cache = uncompiled
            print(f"mx.compile is not supported by this keras-mlx build ({type(e).__name__}); "
                  "decoding without kernel fusion.")

    return model, tokenizer, preprocessor


def compute_sequence_length(
    prompt_tokens: int,
    max_tokens: int,
    max_sequence_length: int = 2048,
) -> int:
    """Computes bucketed sequence length capped at max_sequence_length to avoid unnecessary allocation."""
    needed = min(max(prompt_tokens + max_tokens + 32, 512), max_sequence_length)
    return min(((needed + 511) // 512) * 512, max_sequence_length)


def special_token_ids(tokenizer: Any) -> Tuple[List[int], Set[int]]:
    """(act token ids, ids the repetition penalty must never touch) for a Mr. Chatterbox tokenizer.

    The chat model writes one <act:NAME> token first in every reply, then the reply itself. Following the
    model card (and the Hugging Face Space): the act is the most likely ACT token, chosen from act tokens
    only, and act tokens are banned for the rest of the reply. Sampling the act like ordinary text lets a
    plain word win that slot, and an unbanned act token can appear mid-sentence ("as far as I can<act:recall>").
    """
    vocab = getattr(tokenizer, "vocabulary", None) or {}
    act_ids = sorted(int(i) for t, i in vocab.items() if isinstance(t, str) and t.startswith("<act:"))
    exempt = set(act_ids) | {119, 123}            # <|endoftext|>, <|end|>
    for t in ("<|user|>", "<|assistant|>", "<|system|>"):
        if t in vocab:
            exempt.add(int(vocab[t]))
    return act_ids, exempt


def sample_token_mlx(
    logits: Any,
    context_ids: Optional[List[int]] = None,
    temperature: float = 0.3,
    top_k: int = 50,
    repetition_penalty: float = 1.3,
    exempt_ids: Optional[Set[int]] = None,
    choose_from: Optional[List[int]] = None,
    banned_ids: Optional[List[int]] = None,
) -> int:
    """Samples next token from logits with repetition penalty, temperature, and top-k filtering in MLX.

    choose_from: return the most likely of these ids (used for the act token, first in each reply).
    banned_ids: ids that may not be sampled (act tokens, after the first).
    exempt_ids: ids the repetition penalty leaves alone. The chat template closes every turn with <|end|>,
    so a penalty over the context would also penalise the stop token: the model then cannot end its reply
    and runs on (the Hugging Face built-in penalty has the same flaw; the Space uses a stop-safe one).
    """
    import mlx.core as mx

    l = logits[0, 0].astype(mx.float32)

    if choose_from:
        idx = mx.array(choose_from, dtype=mx.int32)
        return int(choose_from[int(mx.argmax(l[idx]).item())])

    # Apply repetition penalty to recently observed tokens, never to stop/control/act tokens
    if repetition_penalty != 1.0 and context_ids:
        unique_ids = [i for i in set(context_ids) if not exempt_ids or i not in exempt_ids]
        if unique_ids:
            indices = mx.array(unique_ids, dtype=mx.int32)
            vals = l[indices]
            new_vals = mx.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            l[indices] = new_vals

    if banned_ids:
        l[mx.array(banned_ids, dtype=mx.int32)] = -1e9

    if temperature <= 0.01:
        return int(mx.argmax(l).item())

    l = l / temperature
    if 0 < top_k < l.shape[0]:
        top_indices = mx.argpartition(-l, kth=top_k)[:top_k]
        mask = mx.full(l.shape, -1e9, dtype=mx.float32)
        mask[top_indices] = l[top_indices]
        l = mask

    return int(mx.random.categorical(l).item())


async def generate_stream_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    stop_token_ids: Optional[Set[int]] = None,
    max_sequence_length: int = 2048,
    filter_acts: bool = True,
    temperature: float = 0.3,
    top_k: int = 50,
    repetition_penalty: float = 1.3,
) -> AsyncGenerator[str, None]:
    """Asynchronously generates streamed token pieces from prompt using compiled MLX decoding."""
    import asyncio
    import keras
    import mlx.core as mx

    if stop_token_ids is None:
        stop_token_ids = {123, 119}  # Default <|end|> and <|endoftext|>

    act_ids, exempt_ids = special_token_ids(tokenizer)
    raw_tokens = mx.array(tokenizer(prompt), dtype=mx.int32)
    prompt_tokens = len(raw_tokens)

    seq_len = compute_sequence_length(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        max_sequence_length=max_sequence_length,
    )

    # Ensure prompt fits in buffer with budget for generation
    min_gen_budget = min(max_tokens, 256)
    max_prompt_allowed = max(seq_len - min_gen_budget, 1)
    if prompt_tokens > max_prompt_allowed:
        raw_tokens = raw_tokens[-max_prompt_allowed:]
        prompt_tokens = len(raw_tokens)

    pad_len = seq_len - prompt_tokens
    token_ids = mx.concatenate([raw_tokens, mx.zeros((pad_len,), dtype=raw_tokens.dtype)])
    token_ids = mx.expand_dims(token_ids, 0)

    hidden_states, cache = model._build_cache(token_ids)
    current_prompt = token_ids
    current_cache = cache
    index = prompt_tokens
    max_len = min(prompt_tokens + max_tokens, seq_len)

    # Track recent context tokens for repetition penalty
    context_tokens: List[int] = list(raw_tokens[-128:].tolist())

    while index < max_len:
        cache_update_index = index - 1
        prompt_slice = keras.ops.slice(current_prompt, [0, cache_update_index], [1, 1])
        idx_arr = mx.array(cache_update_index, dtype=mx.int32)

        logits, _, current_cache = model.call_with_cache(
            prompt_slice,
            current_cache,
            idx_arr,
        )
        mx.eval(logits, current_cache)

        next_token = sample_token_mlx(
            logits=logits,
            context_ids=context_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            exempt_ids=exempt_ids,
            choose_from=act_ids if (index == prompt_tokens and act_ids) else None,
            banned_ids=act_ids if index > prompt_tokens else None,
        )
        if next_token in stop_token_ids:
            break

        context_tokens.append(next_token)

        piece = tokenizer.detokenize([next_token])
        if (
            piece
            and not piece.startswith("<|")
            and not (filter_acts and piece.startswith("<act:"))
        ):
            yield piece
            # Yield control back to asyncio event loop
            await asyncio.sleep(0)

        token_tensor = mx.reshape(
            mx.array(next_token, dtype=current_prompt.dtype), (1, 1)
        )
        current_prompt = keras.ops.slice_update(
            current_prompt, [0, index], token_tensor
        )
        index += 1


def generate_full_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    stop_token_ids: Optional[Set[int]] = None,
    max_sequence_length: int = 2048,
    filter_acts: bool = True,
    temperature: float = 0.3,
    top_k: int = 50,
    repetition_penalty: float = 1.3,
) -> str:
    """Synchronously generates the full response string for prompt."""
    import keras
    import mlx.core as mx

    if stop_token_ids is None:
        stop_token_ids = {123, 119}

    act_ids, exempt_ids = special_token_ids(tokenizer)
    raw_tokens = mx.array(tokenizer(prompt), dtype=mx.int32)
    prompt_tokens = len(raw_tokens)

    seq_len = compute_sequence_length(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        max_sequence_length=max_sequence_length,
    )

    pad_len = seq_len - prompt_tokens
    token_ids = mx.concatenate([raw_tokens, mx.zeros((pad_len,), dtype=raw_tokens.dtype)])
    token_ids = mx.expand_dims(token_ids, 0)

    hidden_states, cache = model._build_cache(token_ids)
    current_prompt = token_ids
    current_cache = cache
    index = prompt_tokens
    max_len = min(prompt_tokens + max_tokens, seq_len)

    generated_ids: List[int] = []
    context_tokens: List[int] = list(raw_tokens[-128:].tolist())

    while index < max_len:
        cache_update_index = index - 1
        prompt_slice = keras.ops.slice(current_prompt, [0, cache_update_index], [1, 1])
        idx_arr = mx.array(cache_update_index, dtype=mx.int32)

        logits, _, current_cache = model.call_with_cache(
            prompt_slice,
            current_cache,
            idx_arr,
        )
        mx.eval(logits, current_cache)

        next_token = sample_token_mlx(
            logits=logits,
            context_ids=context_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            exempt_ids=exempt_ids,
            choose_from=act_ids if (index == prompt_tokens and act_ids) else None,
            banned_ids=act_ids if index > prompt_tokens else None,
        )
        if next_token in stop_token_ids:
            break

        context_tokens.append(next_token)

        piece = tokenizer.detokenize([next_token])
        if not (filter_acts and piece.startswith("<act:")):
            generated_ids.append(next_token)

        token_tensor = mx.reshape(
            mx.array(next_token, dtype=current_prompt.dtype), (1, 1)
        )
        current_prompt = keras.ops.slice_update(
            current_prompt, [0, index], token_tensor
        )
        index += 1

    return tokenizer.detokenize(generated_ids)
