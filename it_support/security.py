"""Secret redaction, authentication, rate limiting, and response headers."""

from __future__ import annotations

import hashlib
import html
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
GITHUB_TOKEN_PATTERN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
AWS_ACCESS_KEY_PATTERN = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
SLACK_TOKEN_PATTERN = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b")
URL_CREDENTIALS_PATTERN = re.compile(r"(?i)\b(?P<scheme>https?://)(?P<credentials>[^/@\s]+)@")
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    flags=re.DOTALL,
)
BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?P<label>"
    r"openai[_ -]?api[_ -]?key|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"bearer[_ -]?token|secret|password|pass|pwd|fjal[eë]kalim|passwort|"
    r"mfa[_ -]?code|2fa[_ -]?code|verification[_ -]?code|recovery[_ -]?code"
    r")\s*(?P<separator>[:=]|\bis\b|\bwas\b|\b[eë]sht[eë]\b|\bist\b)\s*"
    r"(?P<value>['\"]?[^\s,;]{4,}['\"]?)"
)


def redact_sensitive_text(text: str) -> str:
    redacted = PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", text)
    redacted = URL_CREDENTIALS_PATTERN.sub(r"\g<scheme>[REDACTED_CREDENTIALS]@", redacted)
    redacted = OPENAI_KEY_PATTERN.sub("[REDACTED_OPENAI_KEY]", redacted)
    redacted = GITHUB_TOKEN_PATTERN.sub("[REDACTED_GITHUB_TOKEN]", redacted)
    redacted = AWS_ACCESS_KEY_PATTERN.sub("[REDACTED_AWS_ACCESS_KEY]", redacted)
    redacted = SLACK_TOKEN_PATTERN.sub("[REDACTED_SLACK_TOKEN]", redacted)
    redacted = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)

    def replace_assignment(match: re.Match[str]) -> str:
        label = match.group("label")
        separator = match.group("separator")
        spacing = " " if separator.lower() in {"is", "was", "është", "eshte", "ist"} else ""
        return f"{label}{spacing}{separator}{spacing}[REDACTED]"

    return SENSITIVE_ASSIGNMENT_PATTERN.sub(replace_assignment, redacted)


def safe_config_value(value: str, maximum_length: int = 120) -> str:
    redacted = redact_sensitive_text(value)
    single_line = re.sub(r"[\x00-\x20\x7f]+", " ", redacted).strip()
    return single_line[:maximum_length]


def safe_status_value(value: str, maximum_length: int = 120) -> str:
    return html.escape(safe_config_value(value, maximum_length).replace("`", ""), quote=True)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class SlidingWindowRateLimiter:
    """A bounded, thread-safe, in-memory limiter for a single app process."""

    def __init__(self, limit: int, window_seconds: int, max_identities: int = 10000):
        if limit < 1 or window_seconds < 1 or max_identities < 1:
            raise ValueError("Rate limiter values must be positive.")
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_identities = max_identities
        self._events: dict[str, deque[float]] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._salt = secrets.token_bytes(32)

    def _key(self, identity: str) -> str:
        return hashlib.blake2b(
            identity.encode("utf-8", errors="replace"),
            key=self._salt,
            digest_size=16,
        ).hexdigest()

    def check(self, identity: str, now: float | None = None) -> RateLimitDecision:
        timestamp = time.monotonic() if now is None else now
        key = self._key(identity or "anonymous")
        cutoff = timestamp - self.window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (timestamp - events[0]) + 0.999))
                self._last_seen[key] = timestamp
                return RateLimitDecision(False, retry_after)
            events.append(timestamp)
            self._last_seen[key] = timestamp
            self._prune(timestamp)
        return RateLimitDecision(True)

    def _prune(self, now: float) -> None:
        if len(self._events) <= self.max_identities:
            return
        stale_before = now - self.window_seconds
        stale_keys = [key for key, seen in self._last_seen.items() if seen <= stale_before]
        for key in stale_keys:
            self._events.pop(key, None)
            self._last_seen.pop(key, None)
        if len(self._events) > self.max_identities:
            oldest = min(self._last_seen, key=lambda item: self._last_seen[item])
            self._events.pop(oldest, None)
            self._last_seen.pop(oldest, None)


def request_identity(request: Any) -> str:
    username = getattr(request, "username", None)
    if isinstance(username, str) and username:
        return f"user:{username}"
    session_hash = getattr(request, "session_hash", None)
    if isinstance(session_hash, str) and session_hash:
        return f"session:{session_hash}"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if isinstance(host, str) and host:
        return f"client:{host}"
    return "anonymous"


def build_auth_callback(credentials: tuple[str, str] | None) -> Callable[[str, str], bool] | None:
    if credentials is None:
        return None
    expected_username, expected_password = credentials

    def authenticate(username: str, password: str) -> bool:
        return secrets.compare_digest(username, expected_username) and secrets.compare_digest(
            password, expected_password
        )

    return authenticate


class SecurityHeadersMiddleware:
    """Small ASGI middleware that adds browser hardening and no-store headers."""

    def __init__(self, app: Any, *, enable_hsts: bool = False):
        self.app = app
        self.enable_hsts = enable_hsts

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _ in headers}
                additions = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"no-referrer",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
                    b"cross-origin-opener-policy": b"same-origin",
                    b"cache-control": b"no-store",
                    b"x-robots-tag": b"noindex, nofollow",
                }
                if self.enable_hsts:
                    additions[b"strict-transport-security"] = b"max-age=31536000; includeSubDomains"
                for name, value in additions.items():
                    if name not in existing:
                        headers.append((name, value))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class LoginRateLimitMiddleware:
    """Rate-limit authentication attempts without counting normal page assets."""

    def __init__(self, app: Any, *, limit: int, window_seconds: int):
        self.app = app
        self.limiter = SlidingWindowRateLimiter(limit, window_seconds)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        path = str(scope.get("path", "")).rstrip("/")
        is_login_attempt = (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and path.endswith("/login")
        )
        if is_login_attempt:
            client = scope.get("client")
            identity = (
                str(client[0]) if isinstance(client, (list, tuple)) and client else "anonymous"
            )
            decision = self.limiter.check(identity)
            if not decision.allowed:
                body = b'{"detail":"Too many login attempts. Try again later."}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                            (b"retry-after", str(decision.retry_after_seconds).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
