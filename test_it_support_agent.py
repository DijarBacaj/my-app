import asyncio
import gc
import json
import os
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import it_support_agent as agent


class FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Try restarting Outlook."))]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeOllamaResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ITSupportAgentTests(unittest.TestCase):
    def test_redact_sensitive_text_removes_common_secrets(self):
        fake_openai_key = "sk-" + "testsecret123456789"
        text = (
            "Email me at user@example.com. "
            f"OPENAI_API_KEY={fake_openai_key} "
            "password is Hunter2! Bearer abcdefghijklmnop"
        )

        redacted = agent.redact_sensitive_text(text)

        self.assertNotIn("user@example.com", redacted)
        self.assertNotIn(fake_openai_key, redacted)
        self.assertNotIn("Hunter2", redacted)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)

    def test_redact_sensitive_text_removes_additional_token_types(self):
        # Build fake credentials at runtime so repository secret scanners do not
        # mistake these redaction fixtures for live credentials.
        fake_github_token = "gh" + "p_abcdefghijklmnopqrstuvwxyz123456"
        fake_aws_key = "AK" + "IAABCDEFGHIJKLMNOP"
        fake_slack_token = "xox" + "b-123456789012-abcdefghijklmnop"
        fake_private_key = (
            "-----BEGIN PRIVATE" + " KEY-----\nvery-secret-material\n"
            "-----END PRIVATE" + " KEY-----"
        )
        text = (
            f"GitHub {fake_github_token} "
            f"AWS {fake_aws_key} "
            f"Slack {fake_slack_token} "
            "URL https://admin:secret-password@example.com/api "
            f"{fake_private_key}"
        )

        redacted = agent.redact_sensitive_text(text)

        self.assertNotIn("ghp_", redacted)
        self.assertNotIn(fake_aws_key, redacted)
        self.assertNotIn("xoxb-", redacted)
        self.assertNotIn("secret-password", redacted)
        self.assertNotIn("very-secret-material", redacted)
        self.assertIn("[REDACTED_GITHUB_TOKEN]", redacted)
        self.assertIn("[REDACTED_PRIVATE_KEY]", redacted)

    def test_normalize_history_supports_dicts_and_tuples(self):
        history = [
            {"role": "user", "content": "My printer is offline."},
            {"role": "assistant", "content": "Check the printer power."},
            ("I already restarted it.", "Good to know."),
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": ""},
        ]

        self.assertEqual(
            agent.normalize_history(history),
            [
                {"role": "user", "content": "My printer is offline."},
                {"role": "assistant", "content": "Check the printer power."},
                {"role": "user", "content": "I already restarted it."},
                {"role": "assistant", "content": "Good to know."},
            ],
        )

    def test_memory_reset_drops_older_history(self):
        history = [
            {"role": "user", "content": "I use Windows 11 and Outlook."},
            {"role": "assistant", "content": "Thanks."},
            {"role": "user", "content": "reset memory"},
            {"role": "assistant", "content": "Memory reset."},
            {"role": "user", "content": "Now my printer is offline."},
        ]

        memory = agent.build_session_memory(history)

        self.assertNotIn("Windows 11", memory)
        self.assertIn("printer", memory)

    def test_build_messages_adds_session_memory_and_risk_context(self):
        history = [
            {
                "role": "user",
                "content": "My Windows 11 laptop has Outlook password prompts.",
            }
        ]

        messages = agent.build_messages(
            "I clicked a phishing link and now email is acting weird.", history
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("Session memory", combined)
        self.assertIn("Windows 11", combined)
        self.assertIn("Outlook", combined)
        self.assertIn("Potential escalation signals", combined)
        self.assertIn("possible malware or ransomware", combined)

    def test_build_messages_redacts_history_and_latest_issue(self):
        history = [
            {
                "role": "user",
                "content": "My email is person@example.com and password=Secret123.",
            }
        ]

        messages = agent.build_messages("The API key is sk-testsecret123456789.", history)
        combined = "\n".join(message["content"] for message in messages)

        self.assertNotIn("person@example.com", combined)
        self.assertNotIn("Secret123", combined)
        self.assertNotIn("sk-testsecret123456789", combined)
        self.assertIn("[REDACTED_EMAIL]", combined)

    def test_chat_handles_empty_message(self):
        response = agent.chat("", [])

        self.assertIn("Please describe the IT issue", response)

    def test_chat_rejects_oversized_input_before_api_use(self):
        with patch.dict(os.environ, {"MAX_INPUT_CHARS": "500"}, clear=True):
            response = agent.chat("x" * 501, [])

        self.assertIn("too long", response)
        self.assertIn("500 characters", response)

    def test_openai_provider_handles_missing_api_key_without_importing_openai(self):
        with patch.dict(os.environ, {"AI_PROVIDER": "openai"}, clear=True):
            response = agent.chat("My Wi-Fi is down.", [])

        self.assertIn("Missing `OPENAI_API_KEY`", response)

    def test_auto_provider_uses_ollama_when_openai_key_is_missing(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(
                {
                    "url": request.full_url,
                    "timeout": timeout,
                    "payload": json.loads(request.data.decode("utf-8")),
                }
            )
            return FakeOllamaResponse(
                {"message": {"role": "assistant", "content": "Use local steps."}}
            )

        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "auto",
                "OLLAMA_MODEL": "llama3.2",
                "MAX_RESPONSE_TOKENS": "256",
            },
            clear=True,
        ):
            response = agent.chat("My printer is offline.", [], ollama_urlopen_func=fake_urlopen)

        self.assertEqual(response, "Use local steps.")
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(calls[0]["timeout"], 120)
        self.assertEqual(calls[0]["payload"]["model"], "llama3.2")
        self.assertIs(calls[0]["payload"]["stream"], False)
        self.assertEqual(calls[0]["payload"]["options"]["num_predict"], 256)

    def test_auto_provider_ignores_example_api_key_placeholder(self):
        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "auto", "OPENAI_API_KEY": "your_api_key_here"},
            clear=True,
        ):
            self.assertFalse(agent.has_openai_api_key())
            self.assertEqual(agent.resolve_provider(), "ollama")

        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "auto", "OPENAI_API_KEY": "sk-your-api-key-here"},
            clear=True,
        ):
            self.assertFalse(agent.has_openai_api_key())
            self.assertEqual(agent.resolve_provider(), "ollama")

    def test_resolve_provider_prefers_openai_when_key_exists_in_auto_mode(self):
        with patch.dict(os.environ, {"AI_PROVIDER": "auto", "OPENAI_API_KEY": "sk-test"}):
            self.assertEqual(agent.resolve_provider(), "openai")

        with patch.dict(os.environ, {"AI_PROVIDER": "ollama"}, clear=True):
            self.assertEqual(agent.resolve_provider(), "ollama")

    def test_invalid_provider_fails_without_calling_any_model(self):
        with patch.dict(os.environ, {"AI_PROVIDER": "olama"}, clear=True):
            response = agent.chat("My Wi-Fi is down.", [])

        self.assertIn("Unsupported `AI_PROVIDER`", response)
        self.assertIn("auto", response)

    def test_ask_it_agent_sends_messages_to_openai_client(self):
        fake_client = FakeClient()

        with patch.dict(
            os.environ,
            {"OPENAI_MODEL": "test-model", "OPENAI_MAX_RESPONSE_TOKENS": "256"},
            clear=True,
        ):
            response = agent.ask_it_agent(fake_client, "Outlook will not open.", [])

        self.assertEqual(response, "Try restarting Outlook.")
        self.assertEqual(fake_client.chat.completions.last_kwargs["model"], "test-model")
        self.assertEqual(fake_client.chat.completions.last_kwargs["max_completion_tokens"], 256)
        self.assertIs(fake_client.chat.completions.last_kwargs["store"], False)
        self.assertEqual(fake_client.chat.completions.last_kwargs["timeout"], 30)
        self.assertEqual(fake_client.chat.completions.last_kwargs["messages"][-1]["role"], "user")

    def test_physical_danger_returns_immediate_guidance_without_model_call(self):
        def unexpected_client_factory():
            raise AssertionError("model should not be called")

        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            response = agent.chat(
                "The laptop has smoke and sparks.",
                [],
                client_factory=unexpected_client_factory,
            )

        self.assertIn("Safety first", response)
        self.assertIn("Move away", response)
        self.assertIn("emergency services", response)

    def test_albanian_physical_danger_returns_localized_guidance(self):
        def unexpected_client_factory():
            raise AssertionError("model should not be called")

        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            response = agent.chat(
                "Laptopi është duke nxjerrë tym dhe shkëndija.",
                [],
                client_factory=unexpected_client_factory,
            )

        self.assertIn("Siguria së pari", response)
        self.assertIn("Largohu nga pajisja", response)
        self.assertIn("shërbimet emergjente", response)

    def test_german_physical_danger_returns_localized_guidance(self):
        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            response = agent.chat("Mein Gerät hat Rauch und Funken.", [])

        self.assertIn("Sicherheit zuerst", response)
        self.assertIn("Entferne dich", response)

    def test_security_risk_guidance_is_kept_when_model_responds(self):
        fake_client = FakeClient()

        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            response = agent.chat(
                "I clicked a phishing link.",
                [],
                client_factory=lambda: fake_client,
            )

        self.assertTrue(response.startswith("Safety first"))
        self.assertIn("Disconnect the affected device", response)
        self.assertIn("Try restarting Outlook.", response)

    def test_chat_hides_runtime_error_details(self):
        def broken_client_factory():
            raise RuntimeError("provider secret detail")

        with (
            patch.dict(
                os.environ, {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=True
            ),
            patch.object(agent.LOGGER, "warning") as log_warning,
        ):
            response = agent.chat(
                "Outlook will not open.", [], client_factory=broken_client_factory
            )

        self.assertIn("I could not contact the AI service", response)
        self.assertNotIn("provider secret detail", response)
        self.assertNotIn("RuntimeError", response)
        self.assertNotIn("provider secret detail", repr(log_warning.call_args))

    def test_ollama_runtime_error_has_local_setup_guidance(self):
        def broken_urlopen(request, timeout):
            raise OSError("local socket detail")

        with patch.dict(os.environ, {"AI_PROVIDER": "ollama"}, clear=True):
            with patch.object(agent.LOGGER, "warning"):
                response = agent.chat(
                    "Outlook will not open.", [], ollama_urlopen_func=broken_urlopen
                )

        self.assertIn("local Ollama service", response)
        self.assertIn("ollama pull", response)
        self.assertNotIn("local socket detail", response)

    def test_get_server_port_uses_safe_default_for_invalid_values(self):
        with patch.dict(os.environ, {"GRADIO_SERVER_PORT": "not-a-port"}, clear=True):
            self.assertEqual(agent.get_server_port(), 7860)

        with patch.dict(os.environ, {"GRADIO_SERVER_PORT": "7861"}, clear=True):
            self.assertEqual(agent.get_server_port(), 7861)

    def test_get_max_response_tokens_clamps_to_safe_range(self):
        with patch.dict(os.environ, {"MAX_RESPONSE_TOKENS": "50"}, clear=True):
            self.assertEqual(agent.get_max_response_tokens(), 128)

        with patch.dict(os.environ, {"MAX_RESPONSE_TOKENS": "9999"}, clear=True):
            self.assertEqual(agent.get_max_response_tokens(), 2000)

    def test_get_max_response_tokens_accepts_legacy_openai_name(self):
        with patch.dict(os.environ, {"OPENAI_MAX_RESPONSE_TOKENS": "256"}, clear=True):
            self.assertEqual(agent.get_max_response_tokens(), 256)

    def test_get_openai_timeout_clamps_to_safe_range(self):
        with patch.dict(os.environ, {"OPENAI_TIMEOUT_SECONDS": "1"}, clear=True):
            self.assertEqual(agent.get_openai_timeout_seconds(), 5)

        with patch.dict(os.environ, {"OPENAI_TIMEOUT_SECONDS": "999"}, clear=True):
            self.assertEqual(agent.get_openai_timeout_seconds(), 300)

    def test_runtime_status_never_displays_key_and_escapes_model_name(self):
        fake_openai_key = "sk-" + "super-secret-test-key"
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "auto",
                "OPENAI_API_KEY": fake_openai_key,
                "OPENAI_MODEL": "<script>`bad`</script>\nnext",
            },
            clear=True,
        ):
            status = agent.get_runtime_status()

        self.assertNotIn(fake_openai_key, status)
        self.assertNotIn("<script>", status)
        self.assertNotIn("`bad`", status)
        self.assertIn("&lt;script&gt;", status)

    def test_runtime_status_redacts_credentials_inside_ollama_url(self):
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "ollama",
                "OLLAMA_BASE_URL": "https://admin:secret-password@example.com",
            },
            clear=True,
        ):
            status = agent.get_runtime_status()

        self.assertNotIn("secret-password", status)
        self.assertIn("[REDACTED_CREDENTIALS]", status)

    def test_configuration_check_verifies_local_ollama_model(self):
        def fake_urlopen(request, timeout):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/tags")
            self.assertEqual(timeout, 5)
            return FakeOllamaResponse({"models": [{"name": "llama3.2:latest"}]})

        with patch.dict(os.environ, {"AI_PROVIDER": "auto"}, clear=True):
            is_healthy, report = agent.run_configuration_check(fake_urlopen)

        self.assertTrue(is_healthy)
        self.assertIn("Resolved provider: ollama", report)
        self.assertIn("model `llama3.2` is installed", report)

    def test_configuration_check_rejects_placeholder_openai_key(self):
        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "your_api_key_here"},
            clear=True,
        ):
            is_healthy, report = agent.run_configuration_check()

        self.assertFalse(is_healthy)
        self.assertIn("placeholder", report)

    def test_demo_uses_private_chat_defaults(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            demo = agent.build_demo()
            self.addCleanup(demo.close)

            self.assertFalse(demo.analytics_enabled)
            self.assertFalse(demo.save_history)
            self.assertEqual(demo.flagging_mode, "never")
            self.assertTrue(demo.chatbot.sanitize_html)
            self.assertFalse(demo.chatbot.allow_file_downloads)
            self.assertEqual(demo.chatbot.layout, "bubble")
            self.assertEqual(demo.textbox.submit_btn, "Send")
            self.assertEqual(demo.api_visibility, "private")

    def test_rate_limiter_uses_a_sliding_window(self):
        limiter = agent.SlidingWindowRateLimiter(limit=2, window_seconds=10)

        self.assertTrue(limiter.check("same-user", now=100).allowed)
        self.assertTrue(limiter.check("same-user", now=101).allowed)
        blocked = limiter.check("same-user", now=102)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after_seconds, 8)
        self.assertTrue(limiter.check("same-user", now=111).allowed)

    def test_auth_callback_uses_exact_credentials(self):
        callback = agent.build_auth_callback(("support", "correct horse battery"))

        self.assertIsNotNone(callback)
        self.assertTrue(callback("support", "correct horse battery"))
        self.assertFalse(callback("support", "wrong password"))
        self.assertFalse(callback("admin", "correct horse battery"))

    def test_login_attempts_are_rate_limited(self):
        downstream_calls = []

        async def downstream(scope, receive, send):
            downstream_calls.append(scope)
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = agent.LoginRateLimitMiddleware(
            downstream,
            limit=1,
            window_seconds=60,
        )
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/login",
            "client": ("192.0.2.10", 12345),
        }

        async def make_request():
            messages = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            await middleware(scope, receive, send)
            return messages

        first = asyncio.run(make_request())
        second = asyncio.run(make_request())

        self.assertEqual(first[0]["status"], 204)
        self.assertEqual(second[0]["status"], 429)
        self.assertEqual(len(downstream_calls), 1)

    def test_production_configuration_requires_auth_and_allowed_hosts(self):
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
                "GRADIO_SERVER_NAME": "0.0.0.0",
            },
            clear=True,
        ):
            errors = agent.validate_web_security_configuration()

        self.assertTrue(any("APP_AUTH_USERNAME" in error for error in errors))
        self.assertTrue(any("APP_ALLOWED_HOSTS" in error for error in errors))

    def test_production_configuration_accepts_strong_explicit_settings(self):
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "APP_AUTH_USERNAME": "support",
                "APP_AUTH_PASSWORD": "a-long-production-password",
                "APP_ALLOWED_HOSTS": "support.example.com",
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
                "GRADIO_SERVER_NAME": "0.0.0.0",
            },
            clear=True,
        ):
            errors = agent.validate_web_security_configuration()

        self.assertEqual(errors, [])

    def test_remote_ollama_requires_https_and_explicit_opt_in(self):
        with patch.dict(
            os.environ,
            {"OLLAMA_BASE_URL": "http://ollama.example.com"},
            clear=True,
        ):
            errors = agent.validate_ollama_base_url()

        self.assertTrue(any("HTTPS" in error for error in errors))
        self.assertTrue(any("ALLOW_REMOTE_OLLAMA" in error for error in errors))

    def test_healthcheck_has_security_headers(self):
        import httpx

        with patch.dict(
            os.environ,
            {
                "APP_ENV": "test",
                "APP_ALLOWED_HOSTS": "testserver",
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test",
            },
            clear=True,
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                app = agent.create_app()

                async def request_healthcheck(asgi_app):
                    transport = httpx.ASGITransport(app=asgi_app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        return await client.get("/healthz")

                response = asyncio.run(request_healthcheck(app))
                del app
                gc.collect()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-robots-tag"], "noindex, nofollow")

    def test_main_loads_only_the_project_env_file_for_health_check(self):
        with (
            patch("dotenv.load_dotenv") as load_dotenv,
            patch.object(
                agent,
                "run_configuration_check",
                return_value=(True, "configuration ok"),
            ),
            patch("builtins.print"),
        ):
            exit_code = agent.main(["--check"])

        self.assertEqual(exit_code, 0)
        load_dotenv.assert_called_once_with(dotenv_path=agent.PROJECT_ENV_PATH)


if __name__ == "__main__":
    unittest.main()
