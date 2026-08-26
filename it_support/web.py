"""Gradio UI and hardened FastAPI/Uvicorn runtime."""

from __future__ import annotations

from typing import Any

from .config import (
    PROJECT_ENV_PATH,
    PROJECT_ROOT,
    get_allowed_hosts,
    get_auth_credentials,
    get_bool_env,
    get_concurrency_limit,
    get_forwarded_allow_ips,
    get_log_level,
    get_login_rate_limit_requests,
    get_login_rate_limit_window_seconds,
    get_queue_size,
    get_rate_limit_requests,
    get_rate_limit_window_seconds,
    get_root_path,
    get_server_name,
    get_server_port,
    get_state_session_capacity,
    validate_web_security_configuration,
)
from .providers import get_runtime_status
from .safety import detect_response_language
from .security import (
    LoginRateLimitMiddleware,
    SecurityHeadersMiddleware,
    SlidingWindowRateLimiter,
    build_auth_callback,
    request_identity,
)
from .service import LOCALIZED_UI, chat

APP_CSS = """
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 24px clamp(16px, 3vw, 38px) 28px !important;
}
h1 {
  letter-spacing: -0.035em !important;
  font-weight: 760 !important;
}
.prose p { line-height: 1.65 !important; }
.message { border-radius: 18px !important; }
.examples { gap: 10px !important; }
footer { display: none !important; }
@media (max-width: 600px) {
  .gradio-container { padding: 16px 12px 20px !important; }
  h1 { font-size: 1.75rem !important; }
}
"""


def build_demo(rate_limiter: SlidingWindowRateLimiter | None = None) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The gradio package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    limiter = rate_limiter or SlidingWindowRateLimiter(
        get_rate_limit_requests(), get_rate_limit_window_seconds()
    )

    def chat_for_ui(message: str, history: Any, request: gr.Request) -> str:
        decision = limiter.check(request_identity(request))
        if not decision.allowed:
            language = detect_response_language(message)
            return LOCALIZED_UI[language]["rate"].format(seconds=decision.retry_after_seconds)
        return chat(message, history)

    # Future annotations make the nested request annotation a string; Gradio needs
    # the concrete class to inject request metadata without exposing it as an input.
    chat_for_ui.__annotations__["request"] = gr.Request

    chatbot = gr.Chatbot(
        label="Support conversation",
        placeholder=(
            "Describe the problem to begin. Do not paste passwords, MFA codes, "
            "recovery codes, private keys, or full API keys."
        ),
        height=540,
        layout="bubble",
        buttons=["copy"],
        sanitize_html=True,
        allow_file_downloads=False,
    )
    textbox = gr.Textbox(
        placeholder="What is affected, what changed, and what have you already tried?",
        lines=2,
        max_lines=6,
        submit_btn="Send",
    )
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    )
    demo = gr.ChatInterface(
        fn=chat_for_ui,
        chatbot=chatbot,
        textbox=textbox,
        title="IT Support Agent",
        description=(
            "Fast, privacy-conscious help for workplace technology issues. "
            "Describe the device or app, the exact symptom, and what you already tried.\n\n"
            f"**Provider:** {get_runtime_status()} · **Session:** not saved\n\n"
            "**Safety:** never paste passwords, MFA/recovery codes, private keys, or full API keys."
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
        analytics_enabled=False,
        fill_width=True,
        save_history=False,
        api_visibility="private",
        concurrency_limit=get_concurrency_limit(),
    )
    demo.theme = theme
    demo.css = APP_CSS
    demo.state_session_capacity = get_state_session_capacity()
    return demo.queue(
        api_open=False,
        max_size=get_queue_size(),
        default_concurrency_limit=get_concurrency_limit(),
    )


def create_app() -> Any:
    errors = validate_web_security_configuration()
    if errors:
        raise RuntimeError("Unsafe application configuration:\n- " + "\n- ".join(errors))
    try:
        import gradio as gr
        from fastapi import FastAPI
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "Install production dependencies with: pip install -r requirements.txt"
        ) from exc

    app = FastAPI(
        title="IT Support Agent",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=get_allowed_hosts())
    app.add_middleware(
        LoginRateLimitMiddleware,
        limit=get_login_rate_limit_requests(),
        window_seconds=get_login_rate_limit_window_seconds(),
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=get_bool_env("APP_EXTERNAL_HTTPS", False),
    )

    @app.get("/healthz", include_in_schema=False)
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    blocked_paths = [str(PROJECT_ENV_PATH), str(PROJECT_ROOT / ".git")]
    return gr.mount_gradio_app(
        app,
        build_demo(),
        path="/",
        server_name=get_server_name(),
        server_port=get_server_port(),
        auth=build_auth_callback(get_auth_credentials()),
        auth_message="Sign in with the support credentials supplied by your administrator.",
        root_path=get_root_path(),
        allowed_paths=[],
        blocked_paths=blocked_paths,
        show_error=False,
        max_file_size="1mb",
        enable_monitoring=False,
        pwa=False,
        mcp_server=False,
        footer_links=[],
        head=('<title>IT Support Agent</title><meta name="robots" content="noindex,nofollow">'),
    )


def launch_app() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "The uvicorn package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    uvicorn.run(
        create_app(),
        host=get_server_name(),
        port=get_server_port(),
        log_level=get_log_level().lower(),
        proxy_headers=True,
        forwarded_allow_ips=get_forwarded_allow_ips(),
        server_header=False,
        date_header=False,
        access_log=True,
    )
