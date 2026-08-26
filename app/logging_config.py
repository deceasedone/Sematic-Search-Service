"""Minimal structured logging — key=value lines, stdlib only. No need for a
heavier framework (structlog etc.) at this project's scale.
"""
import logging
import sys


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


logger = logging.getLogger("search_service")
