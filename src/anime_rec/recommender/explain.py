"""Grounded LLM re-ranking and explanations with Gemini structured output.

Grounding rules, enforced in code rather than trusted to the prompt:
  * the model sees only the retrieved candidates and refers to them by anime_id;
  * its JSON is validated against a schema; ids that were not retrieved are dropped;
  * the summary must not name titles (it can't then name one we didn't retrieve).
Any failure (quota, network, bad JSON) falls back to the plain retrieval order.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from anime_rec.config import Settings
from anime_rec.log import get_logger
from anime_rec.recommender.llm import GeminiJson
from anime_rec.recommender.schemas import AnimeHit, Recommendation

log = get_logger(__name__)

SYNOPSIS_CHARS = 700  # enough to judge fit; keeps the prompt small and cheap

SYSTEM_INSTRUCTION = """\
You are an anime recommendation assistant.
You receive a user's request and a list of CANDIDATES retrieved from a database.
Rules:
- Choose ONLY from the candidates, identified by their anime_id. Never suggest anything else.
- Rank the candidates that reasonably fit the request, best first, at most {top_n}.
- For each pick write one or two sentences explaining why it fits the request, using only
  facts present in that candidate's data (synopsis, genres, themes, year, episodes, score).
- Only leave out candidates that clearly contradict the request.
- The summary is one sentence about the overall selection and must not name any anime.
"""


class Pick(BaseModel):
    anime_id: int
    reason: str = Field(max_length=600)


class ExplanationOutput(BaseModel):
    picks: list[Pick]
    summary: str = Field(max_length=400)


def _candidate_view(hit: AnimeHit) -> dict[str, Any]:
    synopsis = (hit.synopsis or "")[:SYNOPSIS_CHARS]
    return {
        "anime_id": hit.anime_id,
        "title": hit.title,
        "title_english": hit.title_english,
        "type": hit.type,
        "year": hit.start_year,
        "episodes": hit.episodes,
        "score": hit.score,
        "genres": hit.genres + hit.themes + hit.demographics,
        "synopsis": synopsis,
    }


def build_prompt(request: str, candidates: list[AnimeHit]) -> str:
    lines = "\n".join(json.dumps(_candidate_view(c), ensure_ascii=False) for c in candidates)
    return f"USER REQUEST:\n{request}\n\nCANDIDATES (one JSON object per line):\n{lines}"


def apply_output(
    output: ExplanationOutput, candidates: list[AnimeHit], top_n: int
) -> Recommendation:
    """Map validated LLM picks back onto retrieved candidates, dropping anything invented."""
    by_id = {c.anime_id: c for c in candidates}
    items: list[AnimeHit] = []
    invented = []
    for pick in output.picks:
        hit = by_id.pop(pick.anime_id, None)  # pop: duplicates are ignored too
        if hit is None:
            invented.append(pick.anime_id)
            continue
        items.append(hit.model_copy(update={"reason": pick.reason.strip()}))
        if len(items) == top_n:
            break
    if invented:
        log.warning("dropped picks not in retrieved candidates", anime_ids=invented)
    if not items:
        return Recommendation(items=candidates[:top_n])
    # Fill remaining slots in retrieval order. They carry no reason, so the UI shows which
    # results the LLM vouched for and which are plain vector matches.
    items += list(by_id.values())[: top_n - len(items)]
    return Recommendation(items=items, summary=output.summary.strip(), explained=True)


class Explainer:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        *,
        llm: GeminiJson | None = None,
    ) -> None:
        if llm is None:
            if settings is None:
                raise ValueError("pass settings or an llm")
            llm = GeminiJson(settings, client)
        self._llm = llm

    @property
    def models(self) -> list[str]:
        return self._llm.models

    def explain(self, request: str, candidates: list[AnimeHit], top_n: int = 5) -> Recommendation:
        output = self._llm.generate(
            build_prompt(request, candidates),
            system=SYSTEM_INSTRUCTION.format(top_n=top_n),
            schema=ExplanationOutput,
        )
        if output is None:
            # Every model failed: plain retrieval order is still a useful answer.
            return Recommendation(items=candidates[:top_n])
        return apply_output(output, candidates, top_n)
