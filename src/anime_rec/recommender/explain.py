"""Grounded LLM re-ranking and explanations with Gemini structured output.

Grounding rules, enforced in code rather than trusted to the prompt:
  * the model sees only the retrieved candidates and refers to them by anime_id;
  * its JSON is validated against a schema; ids that were not retrieved are dropped;
  * the summary must not name titles (it can't then name one we didn't retrieve).
Any failure (quota, network, bad JSON) falls back to the plain retrieval order.
"""

import json
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

from anime_rec.config import Settings
from anime_rec.log import get_logger
from anime_rec.recommender.schemas import AnimeHit, Recommendation

log = get_logger(__name__)

SYNOPSIS_CHARS = 700  # enough to judge fit; keeps the prompt small and cheap

SYSTEM_INSTRUCTION = """\
You are an anime recommendation assistant.
You receive a user's request and a list of CANDIDATES retrieved from a database.
Rules:
- Choose ONLY from the candidates, identified by their anime_id. Never suggest anything else.
- Pick the candidates that best fit the request, best first, at most {top_n}.
- For each pick write one or two sentences explaining why it fits the request, using only
  facts present in that candidate's data (synopsis, genres, themes, year, episodes, score).
- Skip candidates that clearly do not fit, even if that leaves fewer than {top_n} picks.
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
    return Recommendation(items=items, summary=output.summary.strip(), explained=True)


FALLBACK_STATUS = {429, 500, 503}


class Explainer:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if client is None:
            if settings.gemini_api_key is None:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._client = client
        self.models = [settings.gemini_chat_model, *settings.gemini_chat_fallback_models]
        self._thinking_budget = settings.gemini_thinking_budget

    def _config(self, model: str, top_n: int) -> types.GenerateContentConfig:
        thinking = None
        if self._thinking_budget is not None and model.startswith("gemini-2"):
            thinking = types.ThinkingConfig(thinking_budget=self._thinking_budget)
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION.format(top_n=top_n),
            response_mime_type="application/json",
            response_schema=ExplanationOutput,
            temperature=0.2,
            thinking_config=thinking,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def explain(self, request: str, candidates: list[AnimeHit], top_n: int = 5) -> Recommendation:
        prompt = build_prompt(request, candidates)
        for model in self.models:
            try:
                response = self._client.models.generate_content(
                    model=model, contents=prompt, config=self._config(model, top_n)
                )
                output = ExplanationOutput.model_validate_json(response.text or "")
            except errors.APIError as exc:
                if exc.code in FALLBACK_STATUS:
                    log.warning("chat model unavailable, trying next", model=model, code=exc.code)
                    continue
                log.warning("explanation failed", model=model, error=str(exc)[:200])
                break
            except (ValidationError, ValueError) as exc:
                log.warning("invalid explanation output", model=model, error=str(exc)[:200])
                break
            return apply_output(output, candidates, top_n)
        # Every model failed: plain retrieval order is still a useful answer.
        return Recommendation(items=candidates[:top_n])
