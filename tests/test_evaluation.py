from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anime_rec.embeddings.base import Embedder, TaskType
from anime_rec.evaluation.dataset import EvalQuery, load_queries, resolve
from anime_rec.evaluation.metrics import first_relevant_rank, summarize
from anime_rec.evaluation.run import ModelSpec, evaluate

CORPUS = pd.DataFrame(
    {
        "anime_id": [1, 2, 3, 4],
        "title": ["Monster", "Monster Musume", "Steins;Gate", "Steins;Gate 0"],
        "synopsis": ["surgeon thriller", "monster girls", "time travel", "time travel sequel"],
    }
)


def test_first_relevant_rank() -> None:
    assert first_relevant_rank([5, 3, 9], {3, 9}) == 2
    assert first_relevant_rank([5, 6], {3}) is None


def test_summarize() -> None:
    m = summarize([1, 3, None, 10])
    assert m["hit@1"] == 0.25
    assert m["hit@5"] == 0.5
    assert m["hit@10"] == 0.75
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 10) / 4)


def test_exact_titles_vs_franchise_prefixes() -> None:
    exact = EvalQuery(id="a", query="q", titles=["Monster"])
    franchise = EvalQuery(id="b", query="q", franchises=["Steins;Gate"])
    assert exact.relevant_ids(CORPUS) == {1}  # not "Monster Musume"
    assert franchise.relevant_ids(CORPUS) == {3, 4}  # sequels count


def test_query_without_targets_rejected() -> None:
    with pytest.raises(ValueError, match="no relevant"):
        EvalQuery(id="x", query="q")


def test_resolve_rejects_queries_missing_from_corpus() -> None:
    with pytest.raises(ValueError, match="no relevant anime"):
        resolve([EvalQuery(id="x", query="q", titles=["Nope"])], CORPUS)


def test_repository_query_set_is_valid() -> None:
    queries = load_queries(Path("eval/queries.yaml"))
    assert len(queries) >= 25


def test_model_spec_parse() -> None:
    assert ModelSpec.parse("local:BAAI/bge-base-en-v1.5") == ModelSpec(
        "local", "BAAI/bge-base-en-v1.5"
    )
    with pytest.raises(ValueError):
        ModelSpec.parse("bge-base")


class KeywordEmbedder(Embedder):
    """Toy embedder: one dimension per vocabulary word, so rankings are predictable."""

    model = "keyword"
    dim = 4
    batch_size = 10
    VOCAB = ("surgeon", "monster", "time", "girls")

    def __init__(self) -> None:
        super().__init__(cache=None)

    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.array([float(w in t.lower()) for w in self.VOCAB]) + 1e-3
            out.append((v / np.linalg.norm(v)).tolist())
        return out


def test_evaluate_ranks_with_exact_search() -> None:
    queries = [
        EvalQuery(id="thriller", query="surgeon", titles=["Monster"]),
        EvalQuery(id="tt", query="time", franchises=["Steins;Gate"]),
    ]
    ranks = evaluate(
        KeywordEmbedder(),
        CORPUS,
        "synopsis_only",
        queries,
        resolve(queries, CORPUS),
        documents_cache_only=False,
    )
    assert ranks == {"thriller": 1, "tt": 1}
