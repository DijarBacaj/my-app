"""Backward-compatible CLI and public API for the IT Support Agent."""

from __future__ import annotations

import argparse
import logging

from it_support import LOGGER as LOGGER
from it_support.config import *
from it_support.core import *
from it_support.providers import *
from it_support.safety import *
from it_support.security import *
from it_support.service import *
from it_support.web import build_demo as build_demo
from it_support.web import create_app as create_app
from it_support.web import launch_app


def main(argv: list[str] | None = None) -> int:
    load_project_environment()
    logging.basicConfig(level=get_log_level())

    parser = argparse.ArgumentParser(description="Run or check the IT Support Agent.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate provider, production security, and local Ollama availability",
    )
    args = parser.parse_args(argv)
    if args.check:
        is_healthy, report = run_configuration_check()
        print(report)
        return 0 if is_healthy else 1

    launch_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
