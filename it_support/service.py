"""Chat request orchestration."""

from __future__ import annotations

import urllib.request
from typing import Any

from .config import (
    get_max_input_chars,
    get_provider_setting,
    has_openai_api_key,
    is_supported_provider,
    resolve_provider,
)
from .core import is_memory_reset_message
from .providers import ask_it_agent, ask_ollama_agent, create_openai_client, format_runtime_error
from .safety import (
    apply_escalation_notice,
    build_escalation_notice,
    detect_response_language,
    detect_risk_flags,
)
from .security import redact_sensitive_text

LOCALIZED_UI = {
    "en": {
        "empty": "Please describe the IT issue, including the affected device, app, and what changed recently.",
        "reset": "Memory reset for future replies in this chat. Use the clear-chat button too if you want to remove the visible conversation.",
        "rate": "Too many requests. Please wait {seconds} seconds and try again.",
    },
    "sq": {
        "empty": "Përshkruaje problemin e IT-së, pajisjen ose aplikacionin e prekur dhe çfarë ka ndryshuar së fundmi.",
        "reset": "Memoria u pastrua për përgjigjet e ardhshme. Përdor edhe butonin e pastrimit nëse dëshiron ta heqësh bisedën e dukshme.",
        "rate": "Ka shumë kërkesa. Prit {seconds} sekonda dhe provo përsëri.",
    },
    "de": {
        "empty": "Beschreibe das IT-Problem, das betroffene Gerät oder die App und was sich zuletzt geändert hat.",
        "reset": "Der Sitzungsspeicher wurde für künftige Antworten zurückgesetzt. Lösche zusätzlich den sichtbaren Chat, falls gewünscht.",
        "rate": "Zu viele Anfragen. Bitte warte {seconds} Sekunden und versuche es erneut.",
    },
}


def localized_ui_message(kind: str, message: str = "", **values: int) -> str:
    language = detect_response_language(message)
    return LOCALIZED_UI[language][kind].format(**values)


def chat(
    message: str,
    history: Any,
    client_factory=create_openai_client,
    ollama_urlopen_func=urllib.request.urlopen,
) -> str:
    if not isinstance(message, str) or not message.strip():
        return localized_ui_message("empty", message if isinstance(message, str) else "")
    language = detect_response_language(message)
    if is_memory_reset_message(message):
        return LOCALIZED_UI[language]["reset"]
    if len(message) > get_max_input_chars():
        return (
            f"That message is too long for this support chat. Keep it under {get_max_input_chars()} "
            "characters and include only the relevant error, device/app, and safe steps already tried."
        )

    risk_flags = detect_risk_flags(redact_sensitive_text(message))
    if "possible physical danger" in risk_flags:
        return build_escalation_notice(risk_flags, language)

    provider_setting = get_provider_setting()
    if not is_supported_provider(provider_setting):
        return apply_escalation_notice(
            f"Unsupported `AI_PROVIDER`: `{provider_setting}`.\n\nUse one of: `auto`, `openai`, or `ollama`.",
            risk_flags,
            language,
        )
    provider = resolve_provider()
    if provider == "openai" and not has_openai_api_key():
        return apply_escalation_notice(
            "Missing `OPENAI_API_KEY` or the value is still a placeholder.\n\n"
            "Checklist:\n- Create a `.env` file in this folder.\n"
            "- Add your real key after `OPENAI_API_KEY=`.\n"
            "- Restart the app with `python it_support_agent.py`.",
            risk_flags,
            language,
        )
    try:
        response = (
            ask_ollama_agent(message, history, urlopen_func=ollama_urlopen_func)
            if provider == "ollama"
            else ask_it_agent(client_factory(), message, history)
        )
    except Exception as exc:
        response = format_runtime_error(exc, provider)
    return apply_escalation_notice(response, risk_flags, language)
