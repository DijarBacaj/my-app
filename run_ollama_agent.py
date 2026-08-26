import os

from it_support_agent import main

os.environ.setdefault("AI_PROVIDER", "ollama")


if __name__ == "__main__":
    raise SystemExit(main())
