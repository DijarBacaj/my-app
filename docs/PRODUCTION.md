# Production deployment

This guide describes the supported production baseline: one application process behind a TLS-terminating reverse proxy or private application gateway.

## 1. Prepare a clean environment

- Use Python 3.11, 3.12, or 3.13 in a dedicated virtual environment.
- Install `requirements.lock` with `--require-hashes`.
- Run the application as a dedicated, non-administrator operating-system user.
- Keep `.env`, logs, and any platform secret files readable only by that account.
- Do not use Gradio share links for production.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
```

## 2. Configure fail-closed production settings

```env
APP_ENV=production
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
APP_ALLOWED_HOSTS=support.example.com
APP_AUTH_USERNAME=support
APP_AUTH_PASSWORD=use-a-random-secret-of-at-least-16-characters
APP_EXTERNAL_HTTPS=true
FORWARDED_ALLOW_IPS=127.0.0.1

AI_PROVIDER=openai
OPENAI_API_KEY=use-the-deployment-secret-manager
OPENAI_MODEL=gpt-4o-mini
```

Generate the password with an approved password manager. Do not place a real password or provider key in `.env.example`, source control, CI logs, shell history, container images, or issue reports.

`APP_ALLOWED_HOSTS` must contain every legitimate external host and any explicit internal host used by health probes. Do not add `*`.

`FORWARDED_ALLOW_IPS` controls which reverse proxies Uvicorn trusts for forwarded client metadata. Use the exact proxy IP/CIDR supported by the deployment; never set `*` on an untrusted network.

## 3. Validate before launch

```powershell
.\.venv\Scripts\python.exe it_support_agent.py --check
```

The command validates the provider, limits, remote Ollama policy, production auth, password length, and host allowlist. For Ollama it also checks `/api/tags` and the configured model. For OpenAI it verifies configuration without spending tokens.

Do not start production if this command exits with code `1`.

## 4. Put TLS and an outer limit at the gateway

- Expose only HTTPS to users.
- Redirect HTTP to HTTPS at the proxy.
- Preserve the original `Host` header.
- Forward client metadata only from the trusted proxy network.
- Set a request-body limit appropriate for the 12,000-character hard maximum.
- Add a gateway rate limit keyed by authenticated identity or source IP.
- Apply an idle timeout longer than the configured provider timeout.

The application adds HSTS only when `APP_EXTERNAL_HTTPS=true`. Enable it only after HTTPS is correctly deployed for the domain.

The built-in rate limiter is process-local. A gateway or shared rate-limit service is mandatory when running multiple processes or replicas.

## 5. Authentication

The built-in Gradio authentication is a secure baseline for a small private installation. For an organization, prefer an identity-aware proxy with SSO, MFA, per-user access revocation, and centralized audit logs. Keep the built-in credentials enabled unless the deployment design has been reviewed and tested.

Never expose the application anonymously because unauthenticated traffic can consume provider tokens or local compute.

## 6. Health and operations

- Liveness: `GET /healthz` returns `200` and `{"status":"ok"}`.
- Provider readiness: run `python it_support_agent.py --check` during deployment, not on every liveness probe.
- Logs intentionally contain exception classes instead of raw provider responses or user content.
- Monitor process restarts, HTTP 5xx rates, queue saturation, provider latency, and cost outside the application.
- Restart the process after changing secrets or environment variables.

The default Uvicorn configuration is one process. Use the operating-system service manager or deployment platform for restart policy and graceful replacement.

## 7. Release gate

Run and require all commands to pass:

```powershell
python -m unittest -v
ruff check .
bandit -c pyproject.toml -r it_support it_support_agent.py run_ollama_agent.py
pip-audit -r requirements.lock
python -m compileall -q it_support it_support_agent.py run_ollama_agent.py test_it_support_agent.py
python it_support_agent.py --check
```

Then smoke-test:

- valid and invalid login;
- desktop light/dark themes and a narrow mobile viewport;
- normal support request;
- Albanian, German, and English physical-danger messages;
- rate-limit response;
- reset-memory and clear-chat behavior;
- unavailable-provider error without sensitive details;
- `/healthz` and security headers through the real external hostname.

## 8. Scaling

Before adding multiple workers or replicas:

- move rate limiting to the gateway or a shared datastore;
- use organization SSO instead of shared basic credentials;
- define retention and audit policies;
- add centralized metrics without logging chat content;
- run load and failure tests against the selected provider;
- document incident response and rollback ownership.
