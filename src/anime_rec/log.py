"""Structured logging via structlog.

`console` format is human-friendly for development; `json` emits one JSON object per line,
which is what log aggregators (Loki, CloudWatch, Datadog) expect in production.
Standard-library loggers (uvicorn, httpx, qdrant-client) are routed through the same pipeline.
"""

import logging
import sys

import structlog

from anime_rec.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = logging.getLevelName(settings.log_level.upper())

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
