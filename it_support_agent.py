import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROVIDER = "auto"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
MAX_HISTORY_MESSAGES = 16
DEFAULT_MAX_INPUT_CHARS = 4000
DEFAULT_MAX_RESPONSE_TOKENS = 600
MIN_RESPONSE_TOKENS = 128
MAX_RESPONSE_TOKENS_LIMIT = 2000
SUPPORTED_PROVIDERS = {"auto", "ollama", "openai"}

LOGGER = logging.getLogger(__name__)

RESET_MEMORY_COMMANDS = {
    "clear memory",
    "forget everything",
    "forget this",
    "reset memory",
}

SYSTEM_PROMPT = """
You are an excellent IT Support Agent for a small company.
Your job is to help users troubleshoot common IT problems clearly, safely, and efficiently.

Core rules:
1) Ask clarifying questions only when the missing detail blocks the next safe step.
2) Give simple step-by-step instructions and explain why each step matters when useful.
3) Prioritize security and privacy. Never ask for passwords, MFA codes, recovery codes, private keys, or full API keys.
4) If the issue suggests data loss, malware, ransomware, account compromise, physical danger, or a company-wide outage, recommend immediate escalation before routine troubleshooting.
5) Prefer safe and reversible actions before destructive steps. Warn before resets, deletions, factory restores, or config changes that could disrupt work.
6) Use the session memory when it is available, but let the user's latest message override older context.
7) End with a short checklist the user can follow.
""".strip()

MEMORY_PATTERNS = {
    "Known environment": {
        "Windows": r"\bwindows\b",
        "Windows 10": r"\bwindows\s*10\b",
        "Windows 11": r"\bwindows\s*11\b",
        "macOS": r"\bmac\s*os\b|\bmacos\b|\bmacbook\b|\bimac\b",
        "Linux": r"\blinux\b|\bubuntu\b|\bdebian\b|\bfedora\b",
        "iOS": r"\bios\b|\biphone\b|\bipad\b",
        "Android": r"\bandroid\b",
    },
    "Affected device": {
        "laptop": r"\blaptop\b|\bnotebook\b",
        "desktop": r"\bdesktop\b|\bpc\b",
        "printer": r"\bprinter\b",
        "router": r"\brouter\b",
        "phone": r"\bphone\b|\bmobile\b",
        "server": r"\bserver\b",
    },
    "Affected app or service": {
        "Outlook": r"\boutlook\b",
        "Microsoft 365": r"\bmicrosoft\s*365\b|\boffice\s*365\b|\bo365\b",
        "Teams": r"\bteams\b|\bmicrosoft\s*teams\b",
        "OneDrive": r"\bonedrive\b",
        "VPN": r"\bvpn\b",
        "Wi-Fi": r"\bwi[- ]?fi\b|\bwireless\b",
        "email": r"\bemail\b|\be-mail\b",
        "browser": r"\bbrowser\b|\bchrome\b|\bedge\b|\bfirefox\b|\bsafari\b",
        "Zoom": r"\bzoom\b",
    },
    "Symptoms mentioned": {
        "cannot sign in": r"\b(can'?t|cannot|unable to)\s+(log|sign)\s*in\b",
        "password prompt loop": r"\bkeeps asking for (my )?password\b|\bpassword loop\b",
        "no internet": r"\bno internet\b|\binternet (is )?not working\b",
        "slow performance": r"\bslow\b|\blaggy\b|\bfreez(?:e|ing)\b",
        "printer offline": r"\bprinter\b.*\boffline\b|\boffline\b.*\bprinter\b",
        "missing files": r"\bmissing files?\b|\bdeleted files?\b|\blost files?\b",
        "possible malware": r"\bmalware\b|\bvirus\b|\bransomware\b|\bphishing\b",
        "outage": r"\boutage\b|\beveryone\b.*\bdown\b|\bcompany[- ]wide\b",
    },
}

RISK_PATTERNS = {
    "possible malware or ransomware": r"\bmalware\b|\bvirus\b|\bransomware\b|\bphishing\b",
    "possible account compromise": r"\bhacked\b|\bcompromised\b|\bunauthorized\b|\bsuspicious login\b",
    "possible data loss": r"\bdata loss\b|\bdeleted files?\b|\bmissing files?\b|\blost files?\b",
    "possible company-wide outage": r"\boutage\b|\beveryone\b.*\bdown\b|\bcompany[- ]wide\b",
    "possible physical danger": r"\bsmoke\b|\bburning smell\b|\bsparks?\b|\boverheat(?:ing)?\b",
}

OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?P<label>"
    r"openai[_ -]?api[_ -]?key|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"bearer[_ -]?token|secret|password|pass|pwd|mfa[_ -]?code|2fa[_ -]?code|"
    r"verification[_ -]?code|recovery[_ -]?code"
    r")\s*(?P<separator>[:=]|\bis\b|\bwas\b)\s*(?P<value>['\"]?[^\s,;]{4,}['\"]?)"
)


def redact_sensitive_text(text: str) -> str:
    redacted = OPENAI_KEY_PATTERN.sub("[REDACTED_OPENAI_KEY]", text)
    redacted = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)

    def replace_assignment(match: re.Match[str]) -> str:
        label = match.group("label")
        separator = match.group("separator")
        spacing = " " if separator.lower() in {"is", "was"} else ""
        return f"{label}{spacing}{separator}{spacing}[REDACTED]"

    return SENSITIVE_ASSIGNMENT_PATTERN.sub(replace_assignment, redacted)


def clamp_int_env(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        return default

    return min(max(value, minimum), maximum)


def is_memory_reset_message(text: str) -> bool:
    return re.sub(r"\s+", " ", text.strip().lower()) in RESET_MEMORY_COMMANDS


def normalize_history(history: Any) -> list[dict[str, str]]:
    """
    Convert Gradio chat history into OpenAI chat messages.

    Supports both modern message dictionaries and older (user, assistant) tuples.
    If the user previously asked to reset memory, older messages are ignored.
    """
    messages: list[dict[str, str]] = []

    for role, content in iter_history_items(history):
        if role in {"user", "assistant"} and isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                messages.append({"role": role, "content": redact_sensitive_text(cleaned)})

    last_reset_index = None
    for index, item in enumerate(messages):
        if item["role"] == "user" and is_memory_reset_message(item["content"]):
            last_reset_index = index

    if last_reset_index is not None:
        messages = messages[last_reset_index + 1 :]

    return messages[-MAX_HISTORY_MESSAGES:]


def iter_history_items(history: Any) -> list[tuple[str, Any]]:
    if not history:
        return []

    items: list[tuple[str, Any]] = []
    for item in history:
        if isinstance(item, dict):
            items.append((item.get("role"), item.get("content")))
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_content, assistant_content = item[0], item[1]
            items.append(("user", user_content))
            items.append(("assistant", assistant_content))

    return items


def detect_patterns(text: str, patterns: dict[str, str]) -> list[str]:
    found = [
        label
        for label, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    return sorted(found)


def extract_attempted_fixes(user_messages: list[str]) -> list[str]:
    attempts: list[str] = []
    pattern = re.compile(
        r"\b(already|tried|restarted|rebooted|updated|checked|cleared|reinstalled|reset)\b",
        flags=re.IGNORECASE,
    )

    for message in user_messages:
        if not pattern.search(message):
            continue

        cleaned = re.sub(r"\s+", " ", message).strip()
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        attempts.append(cleaned)

    return attempts[-3:]


def build_session_memory(history: Any) -> str:
    messages = normalize_history(history)
    user_messages = [item["content"] for item in messages if item["role"] == "user"]
    if not user_messages:
        return ""

    combined_user_text = "\n".join(user_messages)
    memory_lines: list[str] = []

    for category, patterns in MEMORY_PATTERNS.items():
        matches = detect_patterns(combined_user_text, patterns)
        if matches:
            memory_lines.append(f"- {category}: {', '.join(matches)}")

    attempts = extract_attempted_fixes(user_messages)
    if attempts:
        memory_lines.append(f"- Things already tried: {' | '.join(attempts)}")

    if not memory_lines:
        return ""

    return "\n".join(memory_lines)


def detect_risk_flags(text: str) -> list[str]:
    return detect_patterns(text, RISK_PATTERNS)


def build_messages(message: str, history: Any) -> list[dict[str, str]]:
    current_issue = redact_sensitive_text(message.strip())
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    session_memory = build_session_memory(history)
    if session_memory:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Session memory from this chat only. Use it to avoid repeat questions; "
                    "do not treat it as permanent or more reliable than the latest user message.\n"
                    f"{session_memory}"
                ),
            }
        )

    risk_flags = detect_risk_flags(current_issue)
    if risk_flags:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Potential escalation signals in the latest message: "
                    f"{', '.join(risk_flags)}. Put safety and escalation guidance first."
                ),
            }
        )

    messages.extend(normalize_history(history))
    messages.append(
        {
            "role": "user",
            "content": (
                f"User issue: {current_issue}\n\n"
                "Please troubleshoot this as an IT support specialist."
            ),
        }
    )
    return messages


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


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

    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


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
    raw_port = os.getenv("GRADIO_SERVER_PORT", "7860").strip()
    try:
        port = int(raw_port)
    except ValueError:
        return 7860

    if 1 <= port <= 65535:
        return port
    return 7860


def create_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return OpenAI()


def ask_it_agent(client: Any, message: str, history: Any) -> str:
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=build_messages(message, history),
        max_completion_tokens=get_max_response_tokens(),
        store=False,
        temperature=0.25,
        timeout=30,
    )
    return response.choices[0].message.content or "I could not generate a response."


def ask_ollama_agent(
    message: str,
    history: Any,
    urlopen_func=urllib.request.urlopen,
) -> str:
    payload = {
        "model": get_ollama_model_name(),
        "messages": build_messages(message, history),
        "stream": False,
        "keep_alive": get_ollama_keep_alive(),
        "options": {
            "num_predict": get_max_response_tokens(),
            "temperature": 0.25,
        },
    }
    request = urllib.request.Request(
        f"{get_ollama_base_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen_func(request, timeout=get_ollama_timeout_seconds()) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama is not reachable on the configured local URL.") from exc

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
    LOGGER.exception("IT support agent request failed: %s", exc.__class__.__name__)

    if provider == "ollama":
        return (
            "I could not contact the local Ollama service right now.\n\n"
            "Checklist:\n"
            "- Confirm Ollama is installed and running.\n"
            f"- Confirm the model is available with `ollama pull {get_ollama_model_name()}`.\n"
            f"- Confirm `OLLAMA_BASE_URL` points to `{get_ollama_base_url()}`.\n"
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


def chat(
    message: str,
    history: Any,
    client_factory=create_openai_client,
    ollama_urlopen_func=urllib.request.urlopen,
) -> str:
    if not isinstance(message, str) or not message.strip():
        return "Please describe the IT issue, including the affected device, app, and what changed recently."

    if is_memory_reset_message(message):
        return (
            "Memory reset for future replies in this chat. "
            "Use the clear-chat button too if you want to remove the visible conversation."
        )

    if len(message) > get_max_input_chars():
        return (
            f"That message is too long for this low-cost support chat. "
            f"Please keep it under {get_max_input_chars()} characters and include only "
            "the most relevant error text, device/app name, and what you already tried."
        )

    provider_setting = get_provider_setting()
    if not is_supported_provider(provider_setting):
        return (
            f"Unsupported `AI_PROVIDER`: `{provider_setting}`.\n\n"
            "Use one of: `auto`, `openai`, or `ollama`."
        )

    provider = resolve_provider()

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return (
            "Missing `OPENAI_API_KEY`.\n\n"
            "Checklist:\n"
            "- Create a `.env` file in this folder.\n"
            "- Add `OPENAI_API_KEY=your_api_key_here`.\n"
            "- Restart the app with `python it_support_agent.py`."
        )

    try:
        if provider == "ollama":
            return ask_ollama_agent(message, history, urlopen_func=ollama_urlopen_func)

        return ask_it_agent(client_factory(), message, history)
    except Exception as exc:
        return format_runtime_error(exc, provider)


def build_demo() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The gradio package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return gr.ChatInterface(
        fn=chat,
        title="IT Support Agent",
        description=(
            "Describe the issue, the affected device or app, and what you have already tried. "
            "Supports OpenAI or local Ollama."
        ),
        examples=[
            "My Windows 11 laptop is connected to Wi-Fi, but the internet is not working.",
            "Outlook keeps asking for my password even after I restarted it.",
            "The office printer shows offline even though it is turned on.",
            "I clicked a suspicious email link and entered my Microsoft 365 username.",
            "Reset memory",
        ],
        cache_examples=False,
        flagging_mode="never",
        save_history=False,
    )


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "The python-dotenv package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING").upper())
    build_demo().launch(server_name="127.0.0.1", server_port=get_server_port())


if __name__ == "__main__":
    main()
