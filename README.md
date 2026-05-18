# IT Support Agent

A Gradio chatbot that uses OpenAI or local Ollama to troubleshoot common IT support issues for a small company.

## What It Does

- Guides users through safe, step-by-step troubleshooting.
- Uses the current chat history as short-term session memory.
- Redacts likely secrets and email addresses before sending text to the selected model provider.
- Detects escalation signals like malware, account compromise, data loss, outages, and physical danger.
- Avoids asking for passwords, MFA codes, recovery codes, private keys, or full API keys.
- Supports a no-API-key local mode through Ollama.
- Keeps responses concise by default to reduce token use or local compute.
- Ends each answer with a short checklist.

Session memory is not saved to disk. It is rebuilt from the visible chat history each turn, and users can type `reset memory` to make the assistant ignore earlier context. Gradio feedback flagging and saved chat history are disabled.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project folder. You can copy `.env.example` and replace the placeholder key if you want OpenAI:

```env
AI_PROVIDER=auto
OPENAI_API_KEY=your_api_key_here
```

`AI_PROVIDER=auto` uses OpenAI when `OPENAI_API_KEY` is set. If no OpenAI key is set, it tries local Ollama.

Optional model override:

```env
OPENAI_MODEL=gpt-4o-mini
```

## Local Ollama Mode

Ollama lets the chatbot run against a model on your own computer, with no OpenAI or Claude API token.

Install Ollama, then pull a model:

```powershell
ollama pull llama3.2
```

Start the app in Ollama mode:

```powershell
python run_ollama_agent.py
```

Or set these values in `.env` and run the normal app:

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
OLLAMA_KEEP_ALIVE=5m
OLLAMA_TIMEOUT_SECONDS=120
```

Use any model you have pulled locally by changing `OLLAMA_MODEL`.

Optional token and input limits:

```env
MAX_RESPONSE_TOKENS=600
MAX_INPUT_CHARS=4000
```

Optional local port override:

```env
GRADIO_SERVER_PORT=7861
```

Optional logging level:

```env
LOG_LEVEL=WARNING
```

## Run

```powershell
python it_support_agent.py
```

Gradio will print a local URL. Open it in your browser and describe the IT issue.

To avoid unnecessary API usage, only send real test prompts when you want a model response. Opening the app does not use OpenAI tokens.

## Test

```powershell
python -m unittest
```
