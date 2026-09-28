"""Command-line entry point. Each pipeline stage runs independently:

python -m anime_rec ingest   # raw CSV      -> data/processed/anime.parquet
python -m anime_rec embed    # parquet      -> on-disk embedding cache
python -m anime_rec index    # cache        -> Qdrant collection (idempotent upsert)
python -m anime_rec serve    # FastAPI backend
python -m anime_rec ui       # Streamlit chat front end
python -m anime_rec eval     # retrieval quality report
"""

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from anime_rec.log import configure_logging, get_logger

log = get_logger(__name__)


def _not_yet(phase: str) -> Callable[[argparse.Namespace], None]:
    def run(_: argparse.Namespace) -> None:
        log.error("stage not implemented yet", phase=phase)
        sys.exit(1)

    return run


def _ingest(_: argparse.Namespace) -> None:
    from anime_rec.config import get_settings
    from anime_rec.ingestion.pipeline import run_ingest

    run_ingest(get_settings())


def _embed(args: argparse.Namespace) -> None:
    from anime_rec.config import get_settings
    from anime_rec.embeddings.gemini import DailyQuotaExceededError
    from anime_rec.embeddings.pipeline import run_embed

    try:
        run_embed(get_settings(), template=args.template, limit=args.limit, max_new=args.max_new)
    except DailyQuotaExceededError as exc:
        # Expected on the free tier: progress is cached. Distinct exit code for scripts.
        log.warning("stopped early", reason=str(exc))
        sys.exit(3)


def _index(args: argparse.Namespace) -> None:
    from anime_rec.config import get_settings
    from anime_rec.vectorstore.indexer import run_index

    run_index(get_settings(), recreate=args.recreate, partial=args.partial)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("anime_rec.api.main:app", host=args.host, port=args.port, log_config=None)


def _ui(args: argparse.Namespace) -> None:
    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", args.host, "--server.port", str(args.port),
        "--server.headless", "true",
    ]  # fmt: skip
    sys.exit(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime_rec",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="clean raw CSVs into parquet").set_defaults(func=_ingest)

    embed = sub.add_parser("embed", help="embed documents with Gemini (cached, resumable)")
    embed.add_argument("--template", help="document template (default: DOCUMENT_TEMPLATE)")
    embed.add_argument("--limit", type=int, help="only embed the top-N ranked anime (trial run)")
    embed.add_argument(
        "--max-new", type=int, help="embed at most N uncached documents (daily quota budget)"
    )
    embed.set_defaults(func=_embed)

    index = sub.add_parser("index", help="sync cached vectors into Qdrant (idempotent)")
    index.add_argument(
        "--recreate", action="store_true", help="drop and rebuild the collection first"
    )
    index.add_argument(
        "--partial", action="store_true", help="index only anime whose vectors are cached"
    )
    index.set_defaults(func=_index)

    # Stages filled in by later phases.
    for name, help_text, phase in [
        ("eval", "run retrieval evaluation", "phase 9"),
    ]:
        sub.add_parser(name, help=help_text).set_defaults(func=_not_yet(phase))

    serve = sub.add_parser("serve", help="run the FastAPI backend")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    ui = sub.add_parser("ui", help="run the Streamlit front end")
    ui.add_argument("--host", default="0.0.0.0")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(func=_ui)
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
    args.func(args)
