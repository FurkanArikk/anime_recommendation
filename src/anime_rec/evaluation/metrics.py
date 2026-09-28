"""Ranking metrics for single-intent queries with a set of acceptable answers."""

from collections.abc import Sequence

K_VALUES = (1, 5, 10)
MRR_CUTOFF = 10


def first_relevant_rank(ranked_ids: Sequence[int], relevant: set[int]) -> int | None:
    """1-based rank of the first relevant result, or None if none was retrieved."""
    for rank, anime_id in enumerate(ranked_ids, start=1):
        if anime_id in relevant:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, cutoff: int = MRR_CUTOFF) -> float:
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def summarize(ranks: Sequence[int | None]) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        raise ValueError("no ranks to summarize")
    summary = {f"hit@{k}": sum(hit_at_k(r, k) for r in ranks) / n for k in K_VALUES}
    summary[f"mrr@{MRR_CUTOFF}"] = sum(reciprocal_rank(r) for r in ranks) / n
    return summary
