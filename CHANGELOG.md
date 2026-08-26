# Changelog

## 0.2.0 - 2026-08-26

- Split configuration, core logic, providers, safety, security, service, and web runtime into dedicated modules.
- Upgraded Gradio to 6.16.0 and added a hash-pinned transitive dependency lock.
- Added fail-closed production authentication and trusted-host validation.
- Added FastAPI/Uvicorn runtime, liveness endpoint, security headers, and safe proxy configuration.
- Added request, queue, concurrency, input, output, session, and provider-response bounds.
- Added multilingual deterministic safety guidance for English, Albanian, and German.
- Disabled the public Gradio API surface, monitoring, MCP, persistence, analytics, file transfer, and feedback storage.
- Added Ruff, Bandit, dependency auditing, immutable CI action SHAs, Dependabot, and production/security documentation.

