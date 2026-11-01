# Mr. Chatterbox MLX

Run **[Mr. Chatterbox](https://huggingface.co/tventurella/Mr.Chatterbox-2.7B-Chat)** locally on Apple Silicon (M1/M2/M3/M4) using **[Keras 3](https://github.com/keras-team/keras)**, **[KerasHub](https://github.com/keras-team/keras-hub)**, and the **[Keras MLX](https://github.com/keras-team/keras-mlx)** backend.

Mr. Chatterbox is an eccentric language model created by Trip Venturella, pretrained exclusively on 19th-century British texts (novels, periodicals, parliamentary debates, and historical documents). It speaks and thinks strictly in the manner of a Victorian gentleman!

---

## ⚡ Highlights

- **KerasHub Architecture Representation**: Maps directly to KerasHub's `Llama3CausalLM` / `Llama3Backbone` architecture with 2.7B parameters, 32 transformer layers, 20 attention heads, tied embeddings, and custom BPE vocabulary.
- **Pluggable MLX Metal Backend**: Fully accelerated on Apple Silicon GPU via Metal Performance Shaders using `keras-mlx`.
- **Fused Metal Decoding**: Eliminates dispatch overhead with `mx.compile(model.call_with_cache)` for streaming token decoding.
- **Dialogue Act Detection**: Recognizes and parses conversational speech acts (e.g. `<act:return_greeting>`, `<act:give_opinion>`, `<act:explain>`).
- **Interactive Terminal Chat**: Colorful, streaming REPL interface (`chat.py`) with conversation history and command controls.
- **OpenAI-Compatible Local API**: High-performance FastAPI server (`serve_mr_chatterbox.py`) supporting SSE streaming and drop-in integration with OpenCode, Cursor, Continue, or Aider.

---

## 🏛️ Model Architecture & Specs

| Specification | Value |
| :--- | :--- |
| **Hugging Face Model** | [`tventurella/Mr.Chatterbox-2.7B-Chat`](https://huggingface.co/tventurella/Mr.Chatterbox-2.7B-Chat) |
| **KerasHub Model** | `keras_hub.models.Llama3CausalLM` |
| **KerasHub Backbone** | `keras_hub.models.Llama3Backbone` |
| **Parameters** | 2,705,820,160 (2.7B) |
| **Layers** | 32 Transformer Decoder layers |
| **Hidden Dimension** | 2,560 |
| **Intermediate Dimension** | 6,912 |
| **Attention Heads** | 20 Query Heads, 20 Key/Value Heads |
| **Head Dimension** | 128 |
| **Rotary Embedding ($\theta$)** | 10,000 |
| **Context Window** | 2,048 tokens |
| **Embeddings** | Tied word embeddings (`tie_word_embeddings=True`) |
| **Vocabulary** | 65,664 ByteLevel BPE tokens |
| **Stop Tokens** | `<|end|>` (123), `<|endoftext|>` (119) |

---

## 🚀 Quickstart

### Prerequisites

- macOS 13.5+ (Ventura, Sonoma, or Sequoia)
- Apple Silicon Mac (M1, M2, M3, M4)
- Python 3.11

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jeffcarp/mr-chatterbox-mlx.git
cd mr-chatterbox-mlx

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Terminal Chat

Launch the interactive Victorian chat loop directly in your terminal:

```bash
python chat.py
```

#### Terminal Commands
- `/acts`: Toggle visibility of dialogue act tags (`[Act: give_opinion]`)
- `/clear`: Reset conversation history
- `/history`: Inspect full conversation history
- `/exit`: Conclude conversation

---

## 📡 OpenAI-Compatible API Server (OpenCode Integration)

Start the local API server:

```bash
python serve_mr_chatterbox.py --port 8000
```

### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `tventurella/Mr.Chatterbox-2.7B-Chat` | Hugging Face repo or local snapshot path |
| `--dtype` | `bfloat16` | Precision (`bfloat16`, `float16`, `float32`) |
| `--host` | `127.0.0.1` | Host address to bind |
| `--port` | `8000` | Port to listen on |
| `--clean-acts` | `False` | Strip `<act:...>` tags from completions |
| `--no-compile` | `False` | Disable `mx.compile` Metal kernel fusion |

### Connecting OpenCode

Configure OpenCode (`~/.config/opencode/config.json` or project configuration):

```json
{
  "models": {
    "mr-chatterbox": {
      "provider": "openai",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "model": "mr-chatterbox"
    }
  }
}
```

### Testing with `curl`

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mr-chatterbox",
    "messages": [
      {"role": "user", "content": "Pray tell, what think you of modern computing machines?"}
    ],
    "stream": true
  }'
```

---

## 🧪 Testing

Run the automated test suite:

```bash
pytest -v
```

---

## 📜 Example Conversation

> **User:** What think you of locomotives and steam engines?  
> **Mr. Chatterbox:** `<act:explain>` The locomotive is now a machine which can run upon rails and does run, but it is still the same essentially—an engine with the steam in the cylinder. The steam, which was originally used to lift the piston of the engine, is now used to drive the piston, and the piston and all the other parts of the engine are, in fact, moved and carried by the steam.
