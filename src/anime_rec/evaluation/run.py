"""Retrieval evaluation: compare embedding models x document templates on a fixed query set.

Search here is exact brute-force cosine in numpy (the corpus is 10k x <=1024), so no
Qdrant collection is needed per configuration and results are not affected by ANN recall.
Results go to stdout and to data/eval/ as Markdown + JSON.
"""

import gc
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from google.genai import errors as genai_errors

from anime_rec.config import Settings
from anime_rec.embeddings.base import CacheMissError, Embedder, TaskType
from anime_rec.embeddings.cache import EmbeddingCache
from anime_rec.embeddings.documents import build_documents
from anime_rec.embeddings.factory import create_embedder
from anime_rec.embeddings.gemini import DailyQuotaExceededError
from anime_rec.evaluation.dataset import EvalQuery, load_queries, resolve
from anime_rec.evaluation.metrics import MRR_CUTOFF, first_relevant_rank, summarize
from anime_rec.ingestion.pipeline import load_processed
from anime_rec.log import get_logger

log = get_logger(__name__)

TOP_K = MRR_CUTOFF


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "local" | "gemini"
    model: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        provider, sep, model = spec.partition(":")
        if not sep or provider not in ("local", "gemini") or not model:
            raise ValueError(
                f"model spec must look like 'local:<hf-id>' or 'gemini:<id>': {spec!r}"
            )
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class RunResult:
    model: str
    template: str
    status: str  # "ok" or a skip reason
    metrics: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int | None] = field(default_factory=dict)


def build_embedder(settings: Settings, spec: ModelSpec, cache: EmbeddingCache) -> Embedder:
    key = "local_embedding_model" if spec.provider == "local" else "gemini_embedding_model"
    configured = settings.model_copy(update={"embedding_provider": spec.provider, key: spec.model})
    return create_embedder(configured, cache=cache)


def evaluate(
    embedder: Embedder,
    corpus: pd.DataFrame,
    template: str,
    queries: list[EvalQuery],
    relevant: dict[str, set[int]],
    *,
    documents_cache_only: bool,
) -> dict[str, int | None]:
    docs = build_documents(corpus, template)
    doc_matrix = np.asarray(
        embedder.embed(docs, TaskType.DOCUMENT, cache_only=documents_cache_only), dtype=np.float32
    )
    query_matrix = np.asarray(
        embedder.embed([q.query for q in queries], TaskType.QUERY), dtype=np.float32
    )
    ids = corpus["anime_id"].to_numpy()
    scores = query_matrix @ doc_matrix.T  # vectors are unit length -> cosine similarity
    top = np.argsort(-scores, axis=1)[:, :TOP_K]
    return {
        q.id: first_relevant_rank([int(i) for i in ids[row]], relevant[q.id])
        for q, row in zip(queries, top, strict=True)
    }


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_eval(
    settings: Settings,
    *,
    models: list[str] | None = None,
    templates: list[str] | None = None,
    top: int | None = None,
    queries_path: Path = Path("eval/queries.yaml"),
) -> list[RunResult]:
    specs = [ModelSpec.parse(m) for m in models] if models else [_configured_spec(settings)]
    templates = templates or [settings.document_template]
    corpus = load_processed(settings)
    if top:
        corpus = corpus.head(top)  # rank order: top-N best-known anime
    queries = load_queries(queries_path)
    relevant = resolve(queries, corpus)
    log.info("evaluation", corpus=len(corpus), queries=len(queries),
             models=[str(s) for s in specs], templates=templates)  # fmt: skip

    results: list[RunResult] = []
    cache = EmbeddingCache(settings.embedding_cache_path)
    try:
        for spec in specs:
            embedder = build_embedder(settings, spec, cache)
            for template in templates:
                result = RunResult(model=str(spec), template=template, status="ok")
                try:
                    # Never spend API quota on documents; local models just compute them.
                    ranks = evaluate(embedder, corpus, template, queries, relevant,
                                     documents_cache_only=spec.provider == "gemini")  # fmt: skip
                    result.ranks, result.metrics = ranks, summarize(list(ranks.values()))
                except CacheMissError as exc:
                    result.status = f"skipped: documents not embedded ({exc})"
                except (DailyQuotaExceededError, genai_errors.APIError) as exc:
                    # Gemini quota/API trouble shouldn't sink the other models' results.
                    result.status = f"skipped: {type(exc).__name__}"
                log.info("evaluated", model=str(spec), template=template, status=result.status,
                         **{k: round(v, 3) for k, v in result.metrics.items()})  # fmt: skip
                results.append(result)
            del embedder
            _free_gpu_memory()
    finally:
        cache.close()

    report = render_markdown(results, queries, corpus_size=len(corpus))
    print(report)
    _save(settings, results, report)
    return results


def _configured_spec(settings: Settings) -> ModelSpec:
    if settings.embedding_provider == "gemini":
        return ModelSpec("gemini", settings.gemini_embedding_model)
    return ModelSpec("local", settings.local_embedding_model)


def render_markdown(results: list[RunResult], queries: list[EvalQuery], corpus_size: int) -> str:
    lines = [
        f"### Retrieval evaluation ({len(queries)} queries, corpus = top {corpus_size} anime)",
        "",
        "| Model | Template | Hit@1 | Hit@5 | Hit@10 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.status != "ok":
            lines.append(f"| `{r.model}` | {r.template} | {r.status} | | | |")
            continue
        m = r.metrics
        lines.append(
            f"| `{r.model}` | {r.template} | {m['hit@1']:.2f} | {m['hit@5']:.2f} "
            f"| {m['hit@10']:.2f} | {m['mrr@10']:.3f} |"
        )
    ok = [r for r in results if r.status == "ok"]
    if ok:
        lines += ["", "#### Rank of first relevant result per query (- = not in top 10)", ""]
        header = " | ".join(f"{r.model.split('/')[-1]} / {r.template}" for r in ok)
        lines += [f"| Query | {header} |", "|---" * (len(ok) + 1) + "|"]
        for q in queries:
            cells = " | ".join(str(r.ranks[q.id] or "-") for r in ok)
            lines.append(f"| {q.id} | {cells} |")
    return "\n".join(lines)


def _save(settings: Settings, results: list[RunResult], report: str) -> None:
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"{stamp}.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / f"{stamp}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    log.info("saved evaluation report", path=str(out_dir / f"{stamp}.md"))
