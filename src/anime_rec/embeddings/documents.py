"""Document templates: how one anime row becomes the text we embed.

Templates are named so the evaluation can compare them. Volatile fields (score, members,
rank) are deliberately excluded: they change with every re-scrape and would invalidate the
embedding cache without changing what the anime is *about*. They live in the Qdrant payload.
"""

from collections.abc import Callable, Hashable, Mapping
from typing import Any

import pandas as pd

Row = Mapping[Hashable, Any]


def _join(values: Any) -> str:
    return ", ".join(values) if values is not None and len(values) else ""


def _present(value: Any) -> bool:
    return value is not None and not (pd.api.types.is_scalar(value) and pd.isna(value))


def _aired(row: Row) -> str:
    start, end = row.get("start_year"), row.get("end_year")
    if not _present(start):
        return ""
    if row.get("is_ongoing"):
        return f"{start}-ongoing"
    if _present(end) and end != start:
        return f"{start}-{end}"
    return str(start)


def full_document(row: Row) -> str:
    """Title + format + tags + studio + synopsis, one labelled field per line.

    Labelled lines give the model consistent structure across 10k docs, and empty fields
    are omitted rather than written as 'Genres: ' (which would add a shared token pattern
    to every sparse document and pull them together in vector space).
    """
    fmt = row["type"]
    if _present(row.get("episodes")) and row["episodes"] > 1:
        fmt += f", {row['episodes']} episodes"
    aired = _aired(row)
    fields = [
        ("Title", row["title"]),
        ("Format", f"{fmt} ({aired})" if aired else fmt),
        ("Genres", _join(row.get("genres"))),
        ("Themes", _join(row.get("themes"))),
        ("Demographic", _join(row.get("demographics"))),
        ("Studio", _join(row.get("studios"))),
        ("Synopsis", row.get("synopsis") or ""),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def synopsis_only_document(row: Row) -> str:
    """Baseline for the evaluation: plot text only (title as fallback so no doc is empty)."""
    synopsis = row.get("synopsis")
    return synopsis if _present(synopsis) and synopsis else str(row["title"])


TEMPLATES: dict[str, Callable[[Row], str]] = {
    "full": full_document,
    "synopsis_only": synopsis_only_document,
}


def build_documents(df: pd.DataFrame, template: str) -> list[str]:
    try:
        render = TEMPLATES[template]
    except KeyError:
        raise ValueError(
            f"unknown template {template!r}; choose from {sorted(TEMPLATES)}"
        ) from None
    return [render(row) for row in df.to_dict("records")]
