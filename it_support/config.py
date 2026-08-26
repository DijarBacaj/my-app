"""Environment-backed configuration with secure production defaults."""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROVIDER = "auto"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30
DEFAULT_MAX_INPUT_CHARS = 4000
DEFAULT_MAX_RESPONSE_TOKENS = 600
MIN_RESPONSE_TOKENS = 128
MAX_RESPONSE_TOKENS_LIMIT = 2000
SUPPORTED_PROVIDERS = {"auto", "ollama", "openai"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"
OPENAI_KEY_PLACEHOLDERS = {
    "add_your_key_here",
    "change_me",
    "replace_me",
    "sk_your_api_key_here",
    "your_api_key",
    "your_api_key_here",
}

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def clamp_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def get_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def load_project_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "The python-dotenv package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    load_dotenv(dotenv_path=PROJECT_ENV_PATH)


def get_app_environment() -> str:
    return os.getenv("APP_ENV", "development").strip().lower() or "development"


def is_production() -> bool:
    return get_app_environment() == "production"


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def has_openai_api_key() -> bool:
    api_key = get_openai_api_key()
    if not api_key:
        return False
    normalized = "_".join(filter(None, re.split(r"[^a-z0-9]+", api_key.lower())))
    return normalized not in OPENAI_KEY_PLACEHOLDERS


def get_provider_setting() -> str:
    return os.getenv("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER


def is_supported_provider(provider: str) -> bool:
    return provider in SUPPORTED_PROVIDERS


def resolve_provider() -> str:
    provider = get_provider_setting()
    if not is_supported_provider(provider):
        raise ValueError(f"Unsupported provider: {provider}")
    if provider != "auto":
        return provider
    return "openai" if has_openai_api_key() else "ollama"


def get_ollama_base_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
    return (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def get_ollama_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def get_ollama_keep_alive() -> str:
    return (
        os.getenv("OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE).strip()
        or DEFAULT_OLLAMA_KEEP_ALIVE
    )


def get_ollama_timeout_seconds() -> int:
    return clamp_int_env("OLLAMA_TIMEOUT_SECONDS", DEFAULT_OLLAMA_TIMEOUT_SECONDS, 5, 600)


def get_openai_timeout_seconds() -> int:
    return clamp_int_env("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS, 5, 300)


def get_max_input_chars() -> int:
    return clamp_int_env("MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS, 500, 12000)


def get_max_response_tokens() -> int:
    raw_value = os.getenv(
        "MAX_RESPONSE_TOKENS",
        os.getenv("OPENAI_MAX_RESPONSE_TOKENS", str(DEFAULT_MAX_RESPONSE_TOKENS)),
    ).strip()
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_RESPONSE_TOKENS
    return min(max(value, MIN_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS_LIMIT)


def get_server_port() -> int:
    return clamp_int_env("GRADIO_SERVER_PORT", 7860, 1, 65535)


def get_server_name() -> str:
    # Production may listen on the container interface only after validation
    # enforces authentication and an explicit host allowlist.
    default = "0.0.0.0" if is_production() else "127.0.0.1"  # nosec B104
    return os.getenv("GRADIO_SERVER_NAME", default).strip() or default


def get_root_path() -> str | None:
    value = os.getenv("APP_ROOT_PATH", "").strip()
    if not value:
        return None
    return "/" + value.strip("/")


def get_auth_credentials() -> tuple[str, str] | None:
    username = os.getenv("APP_AUTH_USERNAME", "").strip()
    password = os.getenv("APP_AUTH_PASSWORD", "")
    if not username and not password:
        return None
    return username, password


def get_allowed_hosts() -> list[str]:
    raw_value = os.getenv("APP_ALLOWED_HOSTS", "").strip()
    if raw_value:
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return ["localhost", "127.0.0.1", "[::1]"]


def get_forwarded_allow_ips() -> str:
    return os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1").strip() or "127.0.0.1"


def get_rate_limit_requests() -> int:
    return clamp_int_env("RATE_LIMIT_REQUESTS", 12, 1, 120)


def get_rate_limit_window_seconds() -> int:
    return clamp_int_env("RATE_LIMIT_WINDOW_SECONDS", 60, 10, 3600)


def get_login_rate_limit_requests() -> int:
    return clamp_int_env("LOGIN_RATE_LIMIT_REQUESTS", 5, 1, 30)


def get_login_rate_limit_window_seconds() -> int:
    return clamp_int_env("LOGIN_RATE_LIMIT_WINDOW_SECONDS", 300, 30, 3600)


def get_queue_size() -> int:
    return clamp_int_env("MAX_QUEUE_SIZE", 32, 1, 500)


def get_concurrency_limit() -> int:
    return clamp_int_env("MAX_CONCURRENT_REQUESTS", 4, 1, 32)


def get_state_session_capacity() -> int:
    return clamp_int_env("STATE_SESSION_CAPACITY", 1000, 10, 10000)


def get_log_level() -> str:
    level = os.getenv("LOG_LEVEL", "WARNING").strip().upper()
    return level if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "WARNING"


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_ollama_base_url() -> list[str]:
    errors: list[str] = []
    parsed = urlparse(get_ollama_base_url())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ["OLLAMA_BASE_URL must be an absolute http:// or https:// URL."]
    if parsed.username or parsed.password:
        errors.append("OLLAMA_BASE_URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        errors.append("OLLAMA_BASE_URL must not contain a query string or fragment.")
    if parsed.path not in {"", "/"}:
        errors.append("OLLAMA_BASE_URL must not contain a path.")
    is_local = is_loopback_host(parsed.hostname)
    if not is_local and parsed.scheme != "https":
        errors.append("A remote OLLAMA_BASE_URL must use HTTPS.")
    if not is_local and not get_bool_env("ALLOW_REMOTE_OLLAMA", False):
        errors.append("Set ALLOW_REMOTE_OLLAMA=true to explicitly allow a remote Ollama endpoint.")
    return errors


def validate_web_security_configuration() -> list[str]:
    errors: list[str] = []
    if get_app_environment() not in {"development", "production", "test"}:
        errors.append("APP_ENV must be development, production, or test.")

    exposed = not is_loopback_host(get_server_name())
    if is_production() or exposed:
        credentials = get_auth_credentials()
        if credentials is None:
            errors.append(
                "APP_AUTH_USERNAME and APP_AUTH_PASSWORD are required for production or non-local binding."
            )
        else:
            username, password = credentials
            if not username:
                errors.append("APP_AUTH_USERNAME must not be blank.")
            if len(password) < 16:
                errors.append("APP_AUTH_PASSWORD must contain at least 16 characters.")

        allowed_hosts = get_allowed_hosts()
        if not os.getenv("APP_ALLOWED_HOSTS", "").strip():
            errors.append(
                "APP_ALLOWED_HOSTS must be explicitly configured for production or non-local binding."
            )
        if "*" in allowed_hosts:
            errors.append(
                "APP_ALLOWED_HOSTS must not contain '*' for production or non-local binding."
            )

    provider = get_provider_setting()
    if not is_supported_provider(provider):
        errors.append("AI_PROVIDER must be auto, openai, or ollama.")
        return errors
    if resolve_provider() == "ollama":
        errors.extend(validate_ollama_base_url())
    return errors
