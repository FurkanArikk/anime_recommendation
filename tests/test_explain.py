"""Explainer tests: the Gemini client is faked; grounding is enforced in code."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from anime_rec.config import Settings
from anime_rec.recommender.explain import (
    Explainer,
    ExplanationOutput,
    Pick,
    apply_output,
    build_prompt,
)
from anime_rec.recommender.schemas import AnimeHit


def hit(anime_id: int, title: str) -> AnimeHit:
    return AnimeHit(
        anime_id=anime_id, title=title, synopsis=f"{title} synopsis " * 100, type="TV",
        score=8.0, rank=anime_id, popularity=anime_id, members=100,
        image_url="https://cdn.myanimelist.net/x.jpg", mal_url=f"https://myanimelist.net/anime/{anime_id}",
        genres=["Drama"],
    )  # fmt: skip


CANDIDATES = [hit(1, "Monster"), hit(2, "Death Note"), hit(3, "Psycho-Pass")]


class FakeModels:
    def __init__(self, result: Any) -> None:
        """`result` is returned for every call; a list gives one result per call."""
        self.results = list(result) if isinstance(result, list) else None
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0) if self.results is not None else self.result
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


def explainer(result: Any) -> tuple[Explainer, FakeModels]:
    models = FakeModels(result)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        gemini_chat_model="gemini-2.5-flash",
        gemini_chat_fallback_models=["gemini-3.5-flash-lite"],
    )
    return Explainer(settings, client=SimpleNamespace(models=models)), models


def output_json(picks: list[tuple[int, str]], summary: str = "Tense thrillers.") -> str:
    return json.dumps(
        {"picks": [{"anime_id": i, "reason": r} for i, r in picks], "summary": summary}
    )


def test_reranks_and_attaches_reasons() -> None:
    exp, models = explainer(output_json([(3, "future police"), (1, "slow-burn chase")]))
    rec = exp.explain("dark thriller", CANDIDATES, top_n=5)
    assert rec.explained
    assert [h.anime_id for h in rec.items] == [3, 1, 2]  # LLM order, then retrieval fill
    assert rec.items[0].reason == "future police"
    assert rec.items[2].reason is None  # filled, not vouched for by the LLM
    assert rec.summary == "Tense thrillers."
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ExplanationOutput


def test_invented_and_duplicate_ids_are_dropped() -> None:
    out = ExplanationOutput(
        picks=[Pick(anime_id=999, reason="hallucinated"), Pick(anime_id=2, reason="a"),
               Pick(anime_id=2, reason="dup")],
        summary="s",
    )  # fmt: skip
    rec = apply_output(out, CANDIDATES, top_n=2)
    assert [h.anime_id for h in rec.items] == [2, 1]  # 999 dropped, duplicate ignored, filled
    assert rec.items[0].reason == "a" and rec.items[1].reason is None


def test_only_invented_ids_falls_back_to_retrieval() -> None:
    rec = apply_output(ExplanationOutput(picks=[Pick(anime_id=999, reason="x")], summary="s"),
                       CANDIDATES, top_n=2)  # fmt: skip
    assert not rec.explained and [h.anime_id for h in rec.items] == [1, 2]


def test_top_n_respected() -> None:
    exp, _ = explainer(output_json([(1, "a"), (2, "b"), (3, "c")]))
    assert len(exp.explain("q", CANDIDATES, top_n=2).items) == 2


@pytest.mark.parametrize(
    "result",
    [
        errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "X"}}),
        "not json at all",
        json.dumps({"picks": "wrong shape"}),
    ],
)
def test_failures_fall_back_to_retrieval_order(result: Any) -> None:
    exp, _ = explainer(result)
    rec = exp.explain("q", CANDIDATES, top_n=2)
    assert not rec.explained and [h.anime_id for h in rec.items] == [1, 2]
    assert all(h.reason is None for h in rec.items)


def test_prompt_contains_only_candidates_and_truncated_synopses() -> None:
    prompt = build_prompt("dark thriller", CANDIDATES)
    assert "dark thriller" in prompt
    lines = [json.loads(line) for line in prompt.split("\n") if line.startswith("{")]
    assert [c["anime_id"] for c in lines] == [1, 2, 3]
    assert all(len(c["synopsis"]) <= 700 for c in lines)


def api_error(code: int) -> errors.APIError:
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": "x", "status": "X"}})


def test_overloaded_primary_falls_back_to_next_model() -> None:
    exp, models = explainer([api_error(503), output_json([(2, "fits")])])
    rec = exp.explain("q", CANDIDATES, top_n=1)
    assert rec.explained and [h.anime_id for h in rec.items] == [2]
    assert [c["model"] for c in models.calls] == ["gemini-2.5-flash", "gemini-3.5-flash-lite"]


def test_thinking_disabled_only_for_gemini_2() -> None:
    exp, models = explainer([api_error(503), output_json([(1, "a")])])
    exp.explain("q", CANDIDATES)
    first, second = (c["config"].thinking_config for c in models.calls)
    assert first is not None and first.thinking_budget == 0
    assert second is None


def test_non_transient_error_does_not_try_other_models() -> None:
    exp, models = explainer([api_error(400), output_json([(1, "a")])])
    assert not exp.explain("q", CANDIDATES).explained
    assert len(models.calls) == 1
