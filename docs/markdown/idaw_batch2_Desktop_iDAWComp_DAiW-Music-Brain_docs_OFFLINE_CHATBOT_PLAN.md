# Offline Chatbot Plan

Goal: run the DAiW Music Brain assistant entirely offline with the same presets/parameters used online.

---

## 1. Architecture Overview

```
┌──────────────────────┐        ┌─────────────────────────┐
│ Offline LLM Runtime  │        │ DAiW Toolkit (Python)   │
│ (llama.cpp / GPT4All)│        │ music_brain APIs        │
│  • Loads local model │        │  • Intent schema        │
│  • Follows prompt    │<──────►│  • Voice/backing tools  │
└──────────────────────┘  JSON  └─────────────────────────┘
          ▲                          ▲
          │                          │
      JUCE/CLI UI  ────────────────  │
```

### Components

| Module | Responsibility |
|--------|----------------|
| `chatbot.agent` | High-level agent orchestrating prompts, tool calls, and persona. |
| `chatbot.llm_runner` | Wraps whichever offline LLM runtime is installed (e.g., llama.cpp binary, GPT4All Python bindings). |
| `chatbot.memory` | Local cache of prior turns, names, and preset selections. |
| `chatbot.tools` | Thin wrappers around `DAiWAPI` methods (voice synth, backing generator, naming helper). |
| UI layer (Streamlit/JUCE/CLI) | Presents conversation, triggers agent actions offline. |

---

## 2. Data & Config

- **Prompt template** stored under `music_brain/data/chatbot_prompt.txt`. Contains system instructions matching current persona.
- **Parameters** (temperature, top_p, etc.) stored in `chatbot/config.yaml`.
- **Local model path** configurable; default e.g., `~/Models/mistral-7b-instruct.Q4_K_M.gguf`.
- **Caches**: conversation history (JSON) saved under `~/.daiw/chat_history/`.

---

## 3. Tooling Flow

1. User message hits agent.
2. Agent decides whether to respond directly or call a tool (using simple regex/keyword or a scratchpad prompt to the LLM).
3. Tool functions run synchronously in Python, returning JSON.
4. Agent composes final reply referencing the tool output.

Example tools:
- `voice_tune`, `voice_modulate`, `voice_speak`
- `backing_generate`
- `name_suggest`
- `intent_new`, `intent_process`

---

## 4. Implementation Steps

1. **Agent Skeleton**
   - `AgentConfig` dataclass (model path, persona, tool bindings).
   - `ChatAgent` class with `chat(user_message)` method.
2. **LLM Runner Stub**
   - Provide `LLMRunner.generate(system_prompt, conversation, params)` interface.
   - In absence of model, return canned response (allows testing offline agent without heavy models).
3. **Toolkit Bridge**
   - Wrap `DAiWAPI` calls in `chatbot/tools.py`, ensuring all inputs/outputs are JSON-serializable.
4. **CLI Integration**
   - `daiw chatbot --model ~/Models/... --persona default`.
   - Optionally, integrate with Streamlit/JUCE UI later.

---

## 5. Offline Distribution

- Package everything in the existing repo; user installs via `pip install -e .[all]`.
- Provide script `scripts/download_offline_model.sh` to fetch recommended gguf/ggml weights (if licensing allows) or instruct user to supply their own.
- Document environment variables for GPU acceleration (Metal, ROCm, CUDA) but default to CPU-friendly settings.

---

## 6. Next Actions

1. Implement `music_brain/chatbot/agent.py` skeleton.
2. Add `chatbot/tools.py` wrapping DAiW APIs.
3. Wire CLI command `daiw chatbot`.
4. Gradually replace stub LLM with user-selected runtime once configured.

