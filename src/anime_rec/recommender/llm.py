"""Structured-output Gemini calls shared by the explainer and the intent parser.

One place for the operational policy: JSON output validated against a pydantic schema,
fallback models on 503/429, thinking disabled for Gemini 2.x (latency), and `None`
instead of an exception on any failure, because both callers have a non-LLM fallback.
"""

from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from anime_rec.config import Settings
from anime_rec.log import get_logger

log = get_logger(__name__)

FALLBACK_STATUS = {429, 500, 503}
T = TypeVar("T", bound=BaseModel)


class GeminiJson:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if client is None:
            if settings.gemini_api_key is None:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._client = client
        self.models = [settings.gemini_chat_model, *settings.gemini_chat_fallback_models]
        self._thinking_budget = settings.gemini_thinking_budget

    def _config(
        self, model: str, system: str, schema: type[BaseModel], temperature: float
    ) -> types.GenerateContentConfig:
        thinking = None
        if self._thinking_budget is not None and model.startswith("gemini-2"):
            thinking = types.ThinkingConfig(thinking_budget=self._thinking_budget)
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            thinking_config=thinking,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(
        self, prompt: str, *, system: str, schema: type[T], temperature: float = 0.2
    ) -> T | None:
        for model in self.models:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=self._config(model, system, schema, temperature),
                )
                return schema.model_validate_json(response.text or "")
            except errors.APIError as exc:
                if exc.code in FALLBACK_STATUS:
                    log.warning("chat model unavailable, trying next", model=model, code=exc.code)
                    continue
                log.warning("llm call failed", model=model, error=str(exc)[:200])
                return None
            except (ValidationError, ValueError) as exc:
                log.warning("invalid llm output", model=model, error=str(exc)[:200])
                return None
        return None
