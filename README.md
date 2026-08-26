# IT Support Agent

[![Tests and security](https://github.com/DijarBacaj/IT-Support-Chatbot/actions/workflows/tests.yml/badge.svg)](https://github.com/DijarBacaj/IT-Support-Chatbot/actions/workflows/tests.yml)

A privacy-conscious IT support chatbot with OpenAI and local or explicitly trusted Ollama providers. The application includes a polished Gradio interface, a hardened FastAPI/Uvicorn runtime, bounded session memory, multilingual safety rules, authentication, rate limiting, health checks, and automated security scanning.

## Production baseline

The repository is designed as a secure single-process deployment baseline:

- Production and non-local bindings fail closed unless authentication and a host allowlist are configured.
- Provider API routes are private, the Gradio public API footer is disabled, and uploads/history/analytics/flagging are off.
- A sliding-window limiter, queue bound, concurrency bound, input limit, output limit, and response-size limit constrain abuse and cost.
- Security headers, strict CORS, trusted-host validation, non-stored OpenAI requests, and secret redaction reduce common exposure.
- Remote Ollama requires HTTPS plus an explicit opt-in; credentials inside its URL are rejected.
- Deterministic safety guidance supports English, Albanian, and German for urgent physical, security, account, data-loss, and outage signals.
- CI runs unit/component tests on Python 3.11–3.13, Ruff, Bandit, `pip-audit`, compilation checks, immutable GitHub Action references, and Dependabot.
- `requirements.lock` pins transitive dependencies with hashes for repeatable production installs.

This is not a ticketing platform, CMDB, identity provider, or substitute for a trained IT/security team. See [Production deployment](docs/PRODUCTION.md) and [Security policy](SECURITY.md) before exposing it outside a trusted machine.

## Quick start

```powershell
git clone https://github.com/DijarBacaj/IT-Support-Chatbot.git
cd IT-Support-Chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.lock
Copy-Item .env.example .env
python it_support_agent.py --check
python it_support_agent.py
```

Open `http://127.0.0.1:7860`.

For local Ollama:

```powershell
ollama pull llama3.2
python run_ollama_agent.py --check
python run_ollama_agent.py
```

The `.env.example` API key and authentication fields are intentionally blank. Never commit `.env`; Git ignores it.

## Provider configuration

`AI_PROVIDER=auto` selects OpenAI only when a non-placeholder `OPENAI_API_KEY` exists. Otherwise it selects Ollama.

OpenAI example:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_real_key
OPENAI_MODEL=gpt-4o-mini
```

Local Ollama example:

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

## Production configuration

At minimum, production requires:

```env
APP_ENV=production
GRADIO_SERVER_NAME=0.0.0.0
APP_ALLOWED_HOSTS=support.example.com
APP_AUTH_USERNAME=support
APP_AUTH_PASSWORD=replace-with-a-random-password-of-at-least-16-characters
APP_EXTERNAL_HTTPS=true
FORWARDED_ALLOW_IPS=127.0.0.1
```

Keep TLS at a trusted reverse proxy, keep `FORWARDED_ALLOW_IPS` restricted to that proxy, and store secrets in the deployment platform's secret manager. Startup stops with an error if production authentication or allowed hosts are unsafe.

Run this before every deployment:

```powershell
python it_support_agent.py --check
```

The liveness endpoint is `GET /healthz`; it returns only `{"status":"ok"}` and no provider or secret details.

See [docs/PRODUCTION.md](docs/PRODUCTION.md) for the complete deployment checklist and reverse-proxy guidance.

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | `development`, `test`, or `production` |
| `APP_AUTH_USERNAME` | blank | Required with a production/non-local bind |
| `APP_AUTH_PASSWORD` | blank | Required; minimum 16 characters in production |
| `APP_ALLOWED_HOSTS` | local hosts | Comma-separated trusted HTTP Host values |
| `APP_EXTERNAL_HTTPS` | `false` | Enables HSTS when clients access the app through HTTPS |
| `APP_ROOT_PATH` | blank | Optional reverse-proxy URL prefix |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Exact proxies trusted for forwarded client metadata |
| `AI_PROVIDER` | `auto` | `auto`, `openai`, or `ollama` |
| `OPENAI_API_KEY` | blank | OpenAI credential; never place it in tracked files |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Clamped to 5–300 seconds |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local or explicitly allowed HTTPS Ollama URL |
| `OLLAMA_MODEL` | `llama3.2` | Installed Ollama chat model |
| `OLLAMA_KEEP_ALIVE` | `5m` | Ollama model retention duration |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Clamped to 5–600 seconds |
| `ALLOW_REMOTE_OLLAMA` | `false` | Explicitly permits a remote HTTPS Ollama endpoint |
| `MAX_INPUT_CHARS` | `4000` | Per-message limit, clamped to 500–12000 |
| `MAX_RESPONSE_TOKENS` | `600` | Output limit, clamped to 128–2000 |
| `LOGIN_RATE_LIMIT_REQUESTS` | `5` | Login attempts allowed per source/window |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | `300` | Login-attempt window, 30–3600 seconds |
| `RATE_LIMIT_REQUESTS` | `12` | Messages allowed per sliding window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding-window duration, 10–3600 seconds |
| `MAX_CONCURRENT_REQUESTS` | `4` | Provider calls processed concurrently |
| `MAX_QUEUE_SIZE` | `32` | Maximum queued calls |
| `STATE_SESSION_CAPACITY` | `1000` | Bound on Gradio session state |
| `GRADIO_SERVER_NAME` | local / `0.0.0.0` in production | Listen address |
| `GRADIO_SERVER_PORT` | `7860` | Listen port |
| `LOG_LEVEL` | `WARNING` | Valid Python log level |

## Privacy and safety

- Common email addresses, passwords, MFA/recovery codes, bearer tokens, OpenAI/GitHub/AWS/Slack credentials, URL credentials, and private keys are redacted before model calls.
- Redaction is best-effort. Users must still avoid submitting secrets or unnecessary personal data.
- OpenAI calls use `store=False`; provider-side processing remains governed by the provider account and policy.
- The application does not write chat history to disk. The active browser session and provider still temporarily process the conversation.
- Physical-danger signals return local deterministic instructions without calling a model.
- Model output may be wrong. High-impact resets, deletion, account/security actions, and incident response require human approval.

## Development and verification

Install development tools after the runtime lock:

```powershell
python -m pip install --require-hashes -r requirements.lock
python -m pip install -r requirements-dev.txt
```

Run the same core checks as CI:

```powershell
python -m unittest -v
ruff check .
bandit -c pyproject.toml -r it_support it_support_agent.py run_ollama_agent.py
pip-audit -r requirements.lock
python -m compileall -q it_support it_support_agent.py run_ollama_agent.py test_it_support_agent.py
```

To refresh the lock after intentionally changing a direct dependency:

```powershell
uv --cache-dir .uv-cache pip compile requirements.txt --universal --generate-hashes --output-file requirements.lock
```

Review the diff and rerun the complete audit before merging the lock update.

## Project structure

```text
IT-Support-Chatbot/
├── .github/
│   ├── dependabot.yml
│   └── workflows/tests.yml
├── docs/PRODUCTION.md
├── it_support/
│   ├── config.py       # Validated environment configuration
│   ├── core.py         # Prompt and bounded session context
│   ├── providers.py    # OpenAI/Ollama adapters and readiness checks
│   ├── safety.py       # Multilingual deterministic safety rules
│   ├── security.py     # Redaction, auth, limiter, browser headers
│   ├── service.py      # Chat orchestration
│   └── web.py          # Gradio UI and FastAPI/Uvicorn runtime
├── it_support_agent.py # Stable CLI and compatibility exports
├── run_ollama_agent.py
├── test_it_support_agent.py
├── pyproject.toml
├── requirements.txt
├── requirements.lock
└── SECURITY.md
```

## Known operational limits

- The built-in limiter is intentionally single-process. For multiple replicas/workers, enforce a shared limit at the gateway or use a distributed store.
- Built-in authentication is suitable as a baseline for a small private deployment. Prefer SSO through an identity-aware reverse proxy for an organization.
- There is no persistent memory, ticket database, file/screenshot upload, role management, audit database, or SLA workflow.
- Safety keyword coverage is strongest in English, Albanian, and German and remains a guardrail rather than a full classifier.
- No software license has been selected. The repository owner should choose and add one before granting public reuse rights.
