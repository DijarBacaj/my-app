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


class ITSupportAgentTests(unittest.TestCase):
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

    def test_chat_handles_empty_message(self):
        response = agent.chat("", [])

        self.assertIn("Please describe the IT issue", response)

    def test_chat_handles_missing_api_key_without_importing_openai(self):
        with patch.dict(os.environ, {}, clear=True):
            response = agent.chat("My Wi-Fi is down.", [])

        self.assertIn("Missing `OPENAI_API_KEY`", response)

    def test_ask_it_agent_sends_messages_to_openai_client(self):
        fake_client = FakeClient()

        with patch.dict(os.environ, {"OPENAI_MODEL": "test-model"}, clear=True):
            response = agent.ask_it_agent(fake_client, "Outlook will not open.", [])

        self.assertEqual(response, "Try restarting Outlook.")
        self.assertEqual(fake_client.chat.completions.last_kwargs["model"], "test-model")
        self.assertEqual(
            fake_client.chat.completions.last_kwargs["messages"][-1]["role"], "user"
        )

    def test_get_server_port_uses_safe_default_for_invalid_values(self):
        with patch.dict(os.environ, {"GRADIO_SERVER_PORT": "not-a-port"}, clear=True):
            self.assertEqual(agent.get_server_port(), 7860)

        with patch.dict(os.environ, {"GRADIO_SERVER_PORT": "7861"}, clear=True):
            self.assertEqual(agent.get_server_port(), 7861)


if __name__ == "__main__":
    unittest.main()
