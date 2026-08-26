"""Provider adapters and readiness checks."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from . import LOGGER
from .config import (
    get_max_input_chars,
    get_max_response_tokens,
    get_model_name,
    get_ollama_base_url,
    get_ollama_keep_alive,
    get_ollama_model_name,
    get_ollama_timeout_seconds,
    get_openai_api_key,
    get_openai_timeout_seconds,
    get_provider_setting,
    has_openai_api_key,
    is_production,
    is_supported_provider,
    resolve_provider,
    validate_ollama_base_url,
    validate_web_security_configuration,
)
from .core import build_messages
from .security import safe_config_value, safe_status_value

MAX_PROVIDER_RESPONSE_BYTES = 1_000_000


def create_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return OpenAI(api_key=get_openai_api_key(), timeout=get_openai_timeout_seconds())


def ask_it_agent(client: Any, message: str, history: Any) -> str:
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=build_messages(message, history),
        max_completion_tokens=get_max_response_tokens(),
        store=False,
        temperature=0.25,
        timeout=get_openai_timeout_seconds(),
    )
    return response.choices[0].message.content or "I could not generate a response."


def _read_limited_response(response: Any) -> str:
    try:
        body = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    except TypeError:
        # Some compatible test doubles and response wrappers do not accept a size.
        body = response.read()
    if len(body) > MAX_PROVIDER_RESPONSE_BYTES:
        raise RuntimeError("The provider response exceeded the configured safety limit.")
    return body.decode("utf-8")


def ask_ollama_agent(
    message: str,
    history: Any,
    urlopen_func=urllib.request.urlopen,
) -> str:
    errors = validate_ollama_base_url()
    if errors:
        raise RuntimeError("Invalid Ollama endpoint configuration.")
    payload = {
        "model": get_ollama_model_name(),
        "messages": build_messages(message, history),
        "stream": False,
        "keep_alive": get_ollama_keep_alive(),
        "options": {"num_predict": get_max_response_tokens(), "temperature": 0.25},
    }
    request = urllib.request.Request(
        f"{get_ollama_base_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen_func(request, timeout=get_ollama_timeout_seconds()) as response:
            body = _read_limited_response(response)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError("Ollama is not reachable on the configured URL.") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned a response that was not valid JSON.") from exc
    if "error" in data:
        raise RuntimeError("Ollama returned an error. Check that the model is pulled.")
    message_data = data.get("message")
    if isinstance(message_data, dict):
        content = message_data.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return "I could not generate a response."


def format_runtime_error(exc: Exception, provider: str) -> str:
    LOGGER.warning("IT support agent request failed: %s", exc.__class__.__name__)
    if provider == "ollama":
        return (
            "I could not contact the local Ollama service or configured remote endpoint right now.\n\n"
            "Checklist:\n"
            "- Confirm Ollama is installed and running.\n"
            f"- Confirm the model is available with `ollama pull {get_ollama_model_name()}`.\n"
            "- Confirm `OLLAMA_BASE_URL` is correct and passes `python it_support_agent.py --check`.\n"
            "- Try again after fixing the issue."
        )
    return (
        "I could not contact the AI service right now.\n\n"
        "Checklist:\n"
        "- Confirm `OPENAI_API_KEY` is set in your `.env` file.\n"
        "- Confirm dependencies are installed with `pip install -r requirements.txt`.\n"
        "- Confirm your network connection is working.\n"
        "- Try again after fixing the issue."
    )


def get_runtime_status() -> str:
    provider_setting = get_provider_setting()
    if not is_supported_provider(provider_setting):
        return f"configuration error (`AI_PROVIDER={safe_status_value(provider_setting)}`)"
    provider = resolve_provider()
    if provider == "openai":
        return f"OpenAI · model `{safe_status_value(get_model_name())}`"
    base = f"Ollama · model `{safe_status_value(get_ollama_model_name())}`"
    if not is_production():
        base += f" · local URL `{safe_status_value(get_ollama_base_url())}`"
    return base


def check_ollama_connection(urlopen_func=urllib.request.urlopen) -> tuple[bool, str]:
    errors = validate_ollama_base_url()
    if errors:
        return False, " ".join(errors)
    request = urllib.request.Request(
        f"{get_ollama_base_url()}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen_func(request, timeout=5) as response:
            body = _read_limited_response(response)
    except (OSError, urllib.error.URLError) as exc:
        return False, f"Ollama is not reachable ({exc.__class__.__name__})."
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False, "Ollama responded, but `/api/tags` did not return valid JSON."
    models = data.get("models", [])
    installed_names = {
        item.get("name", "")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    configured_model = get_ollama_model_name()
    aliases = (
        {configured_model, f"{configured_model}:latest"}
        if ":" not in configured_model
        else {configured_model}
    )
    if not installed_names.intersection(aliases):
        return False, f"Ollama is running, but model `{configured_model}` is not installed."
    return True, f"Ollama is running and model `{configured_model}` is installed."


def run_configuration_check(
    ollama_urlopen_func=urllib.request.urlopen,
) -> tuple[bool, str]:
    provider_setting = get_provider_setting()
    lines = [f"AI_PROVIDER: {safe_config_value(provider_setting)}"]
    if not is_supported_provider(provider_setting):
        lines.append("ERROR: use one of auto, openai, or ollama.")
        return False, "\n".join(lines)

    security_errors = validate_web_security_configuration()
    lines.append(f"Application environment: {'production' if is_production() else 'development'}")
    if security_errors:
        lines.extend(f"ERROR: {error}" for error in security_errors)

    provider = resolve_provider()
    lines.extend(
        [
            f"Resolved provider: {provider}",
            f"Maximum input characters: {get_max_input_chars()}",
            f"Maximum response tokens: {get_max_response_tokens()}",
        ]
    )
    if provider == "openai":
        lines.append(f"OpenAI model: {safe_config_value(get_model_name())}")
        lines.append(f"OpenAI timeout: {get_openai_timeout_seconds()} seconds")
        if not has_openai_api_key():
            lines.append("ERROR: OPENAI_API_KEY is missing or still a placeholder.")
            return False, "\n".join(lines)
        lines.append("OK: OpenAI configuration is present (no API request was made).")
        return not security_errors, "\n".join(lines)

    lines.append(f"Ollama model: {safe_config_value(get_ollama_model_name())}")
    lines.append(f"Ollama URL: {safe_config_value(get_ollama_base_url())}")
    healthy, detail = check_ollama_connection(ollama_urlopen_func)
    lines.append(("OK: " if healthy else "ERROR: ") + detail)
    return healthy and not security_errors, "\n".join(lines)
