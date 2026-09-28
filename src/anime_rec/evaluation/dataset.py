"""Evaluation query set: YAML -> queries with resolved relevant anime IDs."""

from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator


class EvalQuery(BaseModel):
    id: str
    query: str
    franchises: list[str] = Field(default_factory=list)  # title prefixes
    titles: list[str] = Field(default_factory=list)  # exact titles

    @model_validator(mode="after")
    def _has_targets(self) -> "EvalQuery":
        if not self.franchises and not self.titles:
            raise ValueError(f"query {self.id!r} lists no relevant franchises or titles")
        return self

    def relevant_ids(self, df: pd.DataFrame) -> set[int]:
        titles = df["title"]
        mask = titles.isin(self.titles)
        for prefix in self.franchises:
            mask |= titles.str.startswith(prefix)
        return set(df.loc[mask, "anime_id"].astype(int))


def load_queries(path: Path) -> list[EvalQuery]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = [EvalQuery(**q) for q in raw["queries"]]
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query ids in evaluation set")
    return queries


def resolve(queries: list[EvalQuery], corpus: pd.DataFrame) -> dict[str, set[int]]:
    """Relevant IDs per query within the corpus; a query with none is a broken test case."""
    resolved = {q.id: q.relevant_ids(corpus) for q in queries}
    empty = [qid for qid, ids in resolved.items() if not ids]
    if empty:
        raise ValueError(f"queries with no relevant anime in the corpus: {empty}")
    return resolved
