# **Qwen Model WebSocket Server**

A complete WebSocket-based LLM server using **Qwen 2.5**, **vLLM** and **HuggingFace**, with optional **function calling**, **intent decisions**, and a **client for Pipecat audio pipelines**.

This folder contains:

```
qwen_model/
 ├── qwen_server/
 │    ├── qwen.py        # Basic streaming LLM server
 │    ├── client.py      # Simple test client
 │    └── ws_server.py   # Function-calling + disconnect/transfer logic
 └── llm.py              # Pipecat LLM service (audio → streamed text)
```

---

# 🚀 **1. Features**

### ✅ Basic WebSocket LLM Server (`qwen.py`)

* Streams tokens using **vLLM Async engine**
* WebSocket interface (partial + final responses)
* Handles cancel, errors, multi-client management
* Simple system prompt for customer support
* HF token-based authentication support

---

### ✅ Function Calling Server (`ws_server.py`)

Advanced setup for call-center / customer-support automation:

* Injects **strict function calling format**
* Detects `<function_call>` and extracts JSON parameters
* Integrates with a second analysis server (service gateway)
* Can return:

  * `completed`
  * `disconnect`
  * `transfer`
  * `error`
* Automatically **sanitizes hallucinated tags**

---

### ✅ Simple Test Client (`client.py`)

* Connects to the server
* Sends text or structured messages
* Reads:

  * `partial` token streams
  * `completed` responses
  * `disconnect`/`transfer`
* Provides clean examples for testing

---

### ✅ Pipecat Audio → LLM Client (`llm.py`)

* Converts buffered audio transcription → LLM request
* Streams tokens as `TextFrame`, `LLMFullResponseStartFrame`, `LLMFullResponseEndFrame`
* Supports function-calling responses (`disconnect` + `transfer`)
* Integrates into **Pipecat** conversation pipelines
* Fixes cumulative text issue using delta logic

---

# 📦 **2. Installation**

### Install dependencies

```bash
pip install -r requirements.txt
```

*(Ensure `vllm`, `transformers`, `websockets`, `loguru` are included)*

### Optional: Login to HuggingFace

```bash
export HF_TOKEN="your_token_here"
```

---

# ⚙️ **3. Environment Variables**

You can configure the server via:

| Variable     | Description            | Default                    |
| ------------ | ---------------------- | -------------------------- |
| `HF_TOKEN`   | HuggingFace auth token | none                       |
| `QWEN_MODEL` | Model name             | `Qwen/Qwen2.5-7B-Instruct` |
| `QWEN_HOST`  | WebSocket host         | `0.0.0.0`                  |
| `QWEN_PORT`  | WebSocket port         | `8766`                     |

---

# 🚀 **4. Running the Servers**

---

## **A. Run Basic Qwen LLM Server**

(File: `qwen_server/qwen.py`)

```bash
python qwen_model/qwen_server/qwen.py
```

Server starts on:

```
ws://0.0.0.0:8766
```

---

## **B. Run Function-Calling Server**

(File: `qwen_server/ws_server.py`)

```bash
python qwen_model/qwen_server/ws_server.py
```

This server **requires** a second Qwen server at:

```
ws://localhost:8765   # internal function-call resolver
```

---

# 🧪 **5. Testing With Client**

Run the client:

```bash
python qwen_model/qwen_server/client.py
```

You will see:

* Simple text example
* Function call example
* Partial streaming
* Automatic disconnect/transfer handling

---
