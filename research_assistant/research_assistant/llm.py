"""LLM clients (OpenAI, Gemini, offline mock) behind one interface."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class LLMError(RuntimeError):
    """Configuration problem, failed API call, or unusable model output."""


# Model names change often; override with env vars or the UI/CLI model field.
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "mock": "mock-1",
}


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and chatter."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMError("Model did not return valid JSON.")


class LLM(ABC):
    name = "base"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS[self.name]

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        json_mode: bool = False,
        max_tokens: int | None = None,
        role: str = "",
    ) -> str:
        """Return the model's text. ``role`` is a label for logging/mocking only."""


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("No OpenAI API key: set OPENAI_API_KEY or enter one in the UI.")
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError("Install the OpenAI SDK: pip install openai") from None
        self._client = OpenAI(api_key=key)

    def complete(self, system, user, *, temperature=0.3, json_mode=False, max_tokens=None, role=""):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model)
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("No Gemini API key: set GEMINI_API_KEY or enter one in the UI.")
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise LLMError("Install the Gemini SDK: pip install google-genai") from None
        self._types = types
        self._client = genai.Client(api_key=key)

    def complete(self, system, user, *, temperature=0.3, json_mode=False, max_tokens=None, role=""):
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=cfg
            )
            text = (resp.text or "").strip()
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        if not text:
            raise LLMError("Gemini returned no text (the response may have been blocked).")
        return text


class MockLLM(LLM):
    """Deterministic offline model. Dispatches on ``role`` so every agent gets valid output."""

    name = "mock"

    def __init__(self, model: str | None = None, **_):
        super().__init__(model)
        self.calls: list[dict] = []

    def complete(self, system, user, *, temperature=0.3, json_mode=False, max_tokens=None, role=""):
        self.calls.append({"role": role, "temperature": temperature, "json_mode": json_mode})
        if role == "planner":
            topic = re.search(r"Topic: (.*)", user)
            t = topic.group(1).strip() if topic else "topic"
            return json.dumps({"queries": [f"{t} overview", f"{t} latest developments",
                                           f"{t} evidence and criticism", f"{t} future outlook"]})
        if role == "summarizer":
            return json.dumps({
                "summary": "The page discusses the topic and offers background and examples.",
                "key_points": ["Key point A with a figure (2025).", "Key point B."],
                "relevance": 4,
                "reliability_notes": "Mock source; treat as illustrative only.",
            })
        if role == "synthesis":
            return json.dumps({
                "themes": ["Adoption is growing [1]"],
                "agreements": ["Sources agree on the basics [1][2]"],
                "disagreements": ["Estimates differ in scale [1][2]"],
                "gaps": ["Long-term data is limited"],
            })
        if role == "reporter":
            ids = sorted({int(i) for i in re.findall(r"^\[(\d+)\]", user, flags=re.M)}) or [1]
            a, b = ids[0], ids[min(1, len(ids) - 1)]
            return json.dumps({
                "title": "Mock Research Report",
                "executive_summary": f"This is a mock summary of the findings [{a}].",
                "key_findings": [
                    {"text": "Finding one.", "source_ids": [a]},
                    {"text": "Finding two.", "source_ids": [a, b]},
                ],
                "sections": [
                    {"heading": "Background", "body": f"Background paragraph [{a}]."},
                    {"heading": "Analysis", "body": f"Analysis paragraph [{b}]."},
                ],
                "limitations": ["Mock data only."],
                "conclusion": f"Mock conclusion [{a}].",
                "follow_up_questions": ["What happens next?"],
            })
        return "Mock response."


def make_llm(provider: str, model: str | None = None, api_key: str | None = None) -> LLM:
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLM(model, api_key)
    if provider == "gemini":
        return GeminiLLM(model, api_key)
    if provider == "mock":
        return MockLLM(model)
    raise LLMError(f"Unknown provider {provider!r}. Choose openai, gemini or mock.")
