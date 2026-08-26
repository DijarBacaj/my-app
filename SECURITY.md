# Security policy

## Supported versions

Security fixes are applied to the latest version on the `main` branch. Older snapshots and forks are not supported unless their maintainers state otherwise.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, personal data, or private logs. Use GitHub's private vulnerability reporting or Security Advisory feature for this repository. Include:

- the affected commit or version;
- a minimal reproduction with synthetic data;
- expected and observed impact;
- relevant configuration with all secrets removed; and
- a safe way to contact the reporter if follow-up is needed.

Do not test against systems or accounts you do not own or have explicit permission to assess. Do not access other users' data, degrade service, or publish a working exploit before a fix is available.

## Security design

- Production and non-local bindings require authentication and explicit trusted hosts.
- The public Gradio API surface, file uploads/downloads, analytics, saved history, monitoring, MCP, and flagging are disabled.
- Requests are bounded by input, response, queue, concurrency, rate, and provider-response-size limits.
- Common secrets are redacted before model calls; raw provider exception details and chat text are not logged.
- OpenAI requests use `store=False`; remote Ollama requires HTTPS and explicit opt-in.
- Browser responses include no-store, anti-framing, MIME-sniffing, referrer, permissions, and opener-policy headers.
- Dependencies are locked with hashes, audited in CI, and monitored by Dependabot.

These controls reduce risk but do not make model output authoritative. Deployments remain responsible for TLS, secret storage, access reviews, gateway limits, monitoring, incident response, and provider-policy compliance.

