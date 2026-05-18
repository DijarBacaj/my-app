import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import it_support_agent as agent


class FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Try restarting Outlook."))
            ]
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
        text = (
            "Email me at user@example.com. "
            "OPENAI_API_KEY=sk-testsecret123456789 "
            "password is Hunter2! Bearer abcdefghijklmnop"
        )

        redacted = agent.redact_sensitive_text(text)

        self.assertNotIn("user@example.com", redacted)
        self.assertNotIn("sk-testsecret123456789", redacted)
        self.assertNotIn("Hunter2", redacted)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)

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

        messages = agent.build_messages(
            "The API key is sk-testsecret123456789.", history
        )
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
            response = agent.chat(
                "My printer is offline.", [], ollama_urlopen_func=fake_urlopen
            )

        self.assertEqual(response, "Use local steps.")
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(calls[0]["timeout"], 120)
        self.assertEqual(calls[0]["payload"]["model"], "llama3.2")
        self.assertIs(calls[0]["payload"]["stream"], False)
        self.assertEqual(calls[0]["payload"]["options"]["num_predict"], 256)

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
        self.assertEqual(
            fake_client.chat.completions.last_kwargs["max_completion_tokens"], 256
        )
        self.assertIs(fake_client.chat.completions.last_kwargs["store"], False)
        self.assertEqual(
            fake_client.chat.completions.last_kwargs["messages"][-1]["role"], "user"
        )

    def test_chat_hides_runtime_error_details(self):
        def broken_client_factory():
            raise RuntimeError("provider secret detail")

        with patch.dict(
            os.environ, {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=True
        ):
            with patch.object(agent.LOGGER, "exception"):
                response = agent.chat(
                    "Outlook will not open.", [], client_factory=broken_client_factory
                )

        self.assertIn("I could not contact the AI service", response)
        self.assertNotIn("provider secret detail", response)
        self.assertNotIn("RuntimeError", response)

    def test_ollama_runtime_error_has_local_setup_guidance(self):
        def broken_urlopen(request, timeout):
            raise OSError("local socket detail")

        with patch.dict(os.environ, {"AI_PROVIDER": "ollama"}, clear=True):
            with patch.object(agent.LOGGER, "exception"):
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


if __name__ == "__main__":
    unittest.main()
