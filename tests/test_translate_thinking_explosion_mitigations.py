"""Tests for the gemini-3 thinking-explosion mitigations.

Covers:
- AUTOSUB_GOOGLE_MAX_TOKENS env override of the google output budget (idea 1).
- Per-chunk reasoning-effort fallback when a chunk fails (idea 2).
"""

import pytest

from autosub.core.llm import ReasoningEffort
from autosub.core.llm.pydantic_ai import BaseStructuredLLM
from autosub.pipeline.translate import main as translate_main


# --- idea 1: configurable google max output tokens -------------------------


def test_google_max_tokens_defaults(monkeypatch):
    monkeypatch.delenv("AUTOSUB_GOOGLE_MAX_TOKENS", raising=False)
    assert (
        BaseStructuredLLM._google_max_output_tokens()
        == BaseStructuredLLM._GOOGLE_DEFAULT_MAX_TOKENS
    )


def test_google_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("AUTOSUB_GOOGLE_MAX_TOKENS", "131072")
    assert BaseStructuredLLM._google_max_output_tokens() == 131072


@pytest.mark.parametrize("bad", ["0", "-5", "abc", ""])
def test_google_max_tokens_invalid_falls_back(monkeypatch, bad):
    monkeypatch.setenv("AUTOSUB_GOOGLE_MAX_TOKENS", bad)
    assert (
        BaseStructuredLLM._google_max_output_tokens()
        == BaseStructuredLLM._GOOGLE_DEFAULT_MAX_TOKENS
    )


# --- idea 2: per-chunk reasoning fallback ----------------------------------


class _FakeTranslator:
    """Records translate() calls and the reasoning_effort active for each."""

    def __init__(self, reasoning_effort, fail_first=False):
        self.reasoning_effort = reasoning_effort
        self.fail_first = fail_first
        self.calls = []  # reasoning_effort active at each call

    def translate(self, chunk):
        self.calls.append(self.reasoning_effort)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("Model token limit (65536) exceeded")
        return [f"t:{x}" for x in chunk]


def test_fallback_retries_at_low_and_restores_effort():
    t = _FakeTranslator(ReasoningEffort.MEDIUM, fail_first=True)
    out = translate_main._translate_chunk_with_fallback(t, ["a", "b"], 0, 1)
    assert out == ["t:a", "t:b"]
    # first attempt at MEDIUM, retry at LOW
    assert t.calls == [ReasoningEffort.MEDIUM, ReasoningEffort.LOW]
    # original effort restored afterwards
    assert t.reasoning_effort == ReasoningEffort.MEDIUM


def test_no_fallback_when_already_low_reraises():
    t = _FakeTranslator(ReasoningEffort.LOW, fail_first=True)
    with pytest.raises(RuntimeError):
        translate_main._translate_chunk_with_fallback(t, ["a"], 0, 1)
    # no retry attempted
    assert t.calls == [ReasoningEffort.LOW]


def test_success_path_no_retry():
    t = _FakeTranslator(ReasoningEffort.HIGH, fail_first=False)
    out = translate_main._translate_chunk_with_fallback(t, ["a"], 0, 1)
    assert out == ["t:a"]
    assert t.calls == [ReasoningEffort.HIGH]
    assert t.reasoning_effort == ReasoningEffort.HIGH
