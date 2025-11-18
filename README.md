# **Qwen Model WebSocket Server**

A complete WebSocket-based LLM server using **Qwen 2.5**, **vLLM** and **HuggingFace**, with optional **function calling**, **intent decisions**, and a **client for Pipecat audio pipelines**.

This project contains:

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

# 🔊 **6. Using LLM with Pipecat (Audio → Text)**

To integrate into Pipecat, import the service:

```python
from qwen_model.llm import QwenService
```

Example usage inside Pipecat pipeline:

```python
llm = QwenService(
    server_url="ws://localhost:8766",
    language="en",
    temperature=0.7,
    max_tokens=200
)
```

This service:

* Buffers user audio
* Sends final transcript to Qwen
* Streams back text as Pipecat frames
* Supports disconnect/transfer instructions

---

# 🔍 **7. Project Structure Explained**

### **1. qwen.py – Basic streaming server**

Handles:

* Prompt formatting
* vLLM inference
* Token streaming
* Multiple clients
* Cancel logic

### **2. client.py – WebSocket test client**

Lets developers test:

* Text messages
* Chat messages
* Partial responses
* Transfers/disconnects

### **3. ws_server.py – Function Calling Engine**

Adds:

* Strict function calling rules
* SAFE parsing
* Decision routing
* Fallbacks
* Error handling

### **4. llm.py – Pipecat Integration**

* Handles audio → LLM
* Streams output
* Handles function-calls
* Integrates with call workflows

---

# 📝 **8. Example WebSocket Request**

```json
{
  "type": "generate",
  "request_id": "abc123",
  "text": "Hello, tell me about your product.",
  "temperature": 0.7,
  "max_tokens": 100
}
```

---

# 📤 **9. Example Response (Streaming)**

### partial

```json
{
  "type": "partial",
  "request_id": "abc123",
  "text": "Our product is"
}
```

### completed

```json
{
  "type": "completed",
  "request_id": "abc123",
  "text": "Our product is designed to help you manage your communications efficiently."
}
```

---

# 📞 **10. Example Function-Call Response**

```xml
<function_call>
<parameters>
{"transcribe_text":"Tell me about your prices"}
</parameters>
</function_call>
```

Server parses → routes → replaces with final text.

---

# 🛟 **11. Troubleshooting**

### **CUDA error / OOM**

Reduce memory usage:

```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.45
```

### **Tokenizer not found**

Ensure model is downloaded:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### **WebSocket disconnect**

Check ping settings in:

* server
* client
* Pipecat

---

# 📘 **12. License**

Add your license here.

---

# 🎉 **13. Credits**

* Qwen2.5 by Alibaba
* vLLM for serving
* HuggingFace Hub
* Pipecat Framework

---

If you want, I can also create:

✅ A **diagram** showing the full flow
✅ A **Dockerfile**
✅ A **one-click start.sh**
✅ A **Postman collection**
✅ A cleaned-up folder structure

Just tell me!
