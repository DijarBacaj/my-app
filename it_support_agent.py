from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


MODEL_GPT = "gpt-4o-mini"


SYSTEM_PROMPT = """
You are an IT Support Agent for a small company.
Your job is to help users troubleshoot common IT problems clearly and safely.

Rules:
1) Ask clarifying questions when details are missing.
2) Give step-by-step instructions with simple language.
3) Prioritize security and privacy (never ask for passwords).
4) If issue looks critical (data loss, malware, account compromise, outage), suggest immediate escalation.
5) End with a short checklist the user can follow.
""".strip()


def normalize_history(history):
    """
    Convert Gradio history into plain OpenAI chat messages.
    Gradio history items usually look like:
    {"role": "user"|"assistant", "content": "..."}
    """
    messages = []

    for item in history:
        role = item.get("role")
        content = item.get("content")

        # Keep only plain text content for this simple chatbot
        if isinstance(content, str) and role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})

    return messages


def build_messages(message: str, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(normalize_history(history))
    messages.append(
        {
            "role": "user",
            "content": f"User issue: {message}\n\nPlease troubleshoot this as an IT support specialist.",
        }
    )
    return messages


def ask_it_agent(client: OpenAI, message: str, history) -> str:
    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=build_messages(message, history),
        temperature=0.3,
    )
    return response.choices[0].message.content or "I could not generate a response."


def chat(message, history):
    return ask_it_agent(client, message, history)


load_dotenv()
client = OpenAI()

demo = gr.ChatInterface(
    fn=chat,
    title="IT Support Agent",
    description="Describe your IT issue and get troubleshooting help.",
    examples=[
        "My laptop is connected to Wi-Fi but the internet is not working.",
        "Outlook keeps asking for my password.",
        "My printer shows offline even though it is turned on.",
    ],
)

demo.launch()