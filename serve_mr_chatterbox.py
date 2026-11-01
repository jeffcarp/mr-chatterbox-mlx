"""OpenAI-Compatible Local API Server for Mr. Chatterbox (KerasHub + MLX).

Exposes an OpenAI-compatible /v1/chat/completions endpoint (streaming and non-streaming)
and /v1/models accelerated on Apple Silicon GPU via KerasHub and the keras-mlx backend.

Connect OpenCode, Continue, Cursor, Aider, or curl directly to Mr. Chatterbox.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from chat_formatter import MrChatterboxFormatter, strip_dialogue_act
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from model_loader import (
    DEFAULT_HF_REPO,
    generate_full_text,
    generate_stream_tokens,
    load_mr_chatterbox,
)
import uvicorn

app = FastAPI(title="Mr. Chatterbox OpenAI-Compatible MLX Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
active_model: Any = None
active_tokenizer: Any = None
active_preprocessor: Any = None
formatter = MrChatterboxFormatter()
stop_token_ids: Set[int] = set()
model_id_name: str = "mr-chatterbox"
clean_acts_enabled: bool = False


def init_server_model(
    repo_or_path: str = DEFAULT_HF_REPO,
    dtype: str = "bfloat16",
    compile_metal: bool = True,
    clean_acts: bool = False,
):
    """Initializes and caches the Mr. Chatterbox model instance."""
    global \
        active_model, \
        active_tokenizer, \
        active_preprocessor, \
        formatter, \
        stop_token_ids, \
        model_id_name, \
        clean_acts_enabled

    if active_model is not None:
        return active_model

    clean_acts_enabled = clean_acts
    model_id_name = os.path.basename(repo_or_path.rstrip("/")).lower()

    active_model, active_tokenizer, active_preprocessor = load_mr_chatterbox(
        repo_or_path=repo_or_path,
        dtype=dtype,
        compile_metal=compile_metal,
    )

    formatter = MrChatterboxFormatter()
    stop_token_ids = formatter.get_stop_token_ids(active_tokenizer)
    print(f"Registered {len(stop_token_ids)} stop token IDs: {sorted(stop_token_ids)}")
    return active_model


async def warmup_server():
    """Warms up the MLX compiled execution graph with a short dummy generation."""
    if active_model is None:
        return
    print("Warming up compiled Metal graph...")
    dummy_prompt = formatter.format_chat([{"role": "user", "content": "Good day."}])
    async for _ in generate_stream_tokens(
        active_model,
        active_tokenizer,
        dummy_prompt,
        max_tokens=5,
        stop_token_ids=stop_token_ids,
    ):
        pass
    print("Warmup complete! Server ready to serve requests.\n")


@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": model_id_name,
        "backend": "mlx",
        "service": "Mr. Chatterbox 19th Century Conversational Engine",
    }


@app.get("/v1/models")
async def list_models():
    """Returns available model descriptors for OpenAI client compatibility."""
    created_time = int(time.time())
    model_ids = [
        "mr-chatterbox",
        "tventurella/Mr.Chatterbox-2.7B-Chat",
        model_id_name,
        "default",
    ]
    unique_ids = list(dict.fromkeys(model_ids))

    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": created_time,
                "owned_by": "keras-mlx",
            }
            for mid in unique_ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completion endpoint supporting SSE streaming."""
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    raw_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    try:
        max_tokens = int(raw_max_tokens) if raw_max_tokens is not None else 512
    except (ValueError, TypeError):
        max_tokens = 512
    if max_tokens <= 0:
        max_tokens = 512

    model_name = body.get("model", "mr-chatterbox")
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    prompt = formatter.format_chat(messages, add_generation_prompt=True)

    print(f"\n--- [REQUEST] messages: {len(messages)}, max_tokens: {max_tokens}, stream: {stream}")
    if messages:
        last_msg = messages[-1]
        print(f"  Last user message: {repr(last_msg.get('content'))[:100]}")

    if stream:

        async def event_stream():
            try:
                # 1. Emit initial role chunk per OpenAI streaming spec
                role_payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_payload)}\n\n"

                # 2. Stream generation chunks
                full_chunks = []
                async for piece in generate_stream_tokens(
                    model=active_model,
                    tokenizer=active_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                ):
                    full_chunks.append(piece)
                    chunk_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"

                # 3. Emit final stop chunk
                finish_payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(finish_payload)}\n\n"
                yield "data: [DONE]\n\n"

                full_text = "".join(full_chunks)
                print(f"--- [RESPONSE STREAM COMPLETE] length: {len(full_text)} chars")

            except Exception as e:
                print(f"Error during streaming: {e}")
                err_payload = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err_payload)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming response
        response_text = generate_full_text(
            model=active_model,
            tokenizer=active_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )

        if clean_acts_enabled:
            response_text = strip_dialogue_act(response_text)

        payload = {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        }
        return JSONResponse(payload)


def main():
    parser = argparse.ArgumentParser(description="Serve Mr. Chatterbox via OpenAI-compatible API (MLX + KerasHub)")
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
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable mx.compile Metal kernel fusion",
    )
    parser.add_argument(
        "--clean-acts",
        action="store_true",
        help="Strip leading dialogue acts (<act:...>) in responses",
    )
    args = parser.parse_args()

    init_server_model(
        repo_or_path=args.model,
        dtype=args.dtype,
        compile_metal=not args.no_compile,
        clean_acts=args.clean_acts,
    )

    asyncio.run(warmup_server())

    print(f"\n🚀 Mr. Chatterbox MLX Server listening at http://{args.host}:{args.port}")
    print(f"OpenCode / OpenAI API endpoint: http://{args.host}:{args.port}/v1/chat/completions\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
