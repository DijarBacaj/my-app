from dotenv import load_dotenv
from openai import OpenAI


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


def build_messages(user_issue: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User issue: {user_issue}\n\nPlease troubleshoot this as an IT support specialist.",
        },
    ]


def ask_it_agent(client: OpenAI, user_issue: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=build_messages(user_issue),
        temperature=0.3,
    )
    return response.choices[0].message.content or "I could not generate a response."


def main():
    load_dotenv()
    client = OpenAI()

    print("IT Support Agent (type 'exit' to quit)\n")
    while True:
        try:
            issue = input("Describe your IT issue: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not issue:
            print("Please enter a problem description.\n")
            continue
        if issue.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print("\nIT Agent Response:\n")
        answer = ask_it_agent(client, issue)
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
