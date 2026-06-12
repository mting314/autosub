import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ContentFilterError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart
from pydantic_ai.usage import RequestUsage

from autosub.core.errors import (
    VertexBlockedResponseError,
    VertexRequestError,
    VertexResponseDiagnostics,
    VertexResponseParseError,
    VertexResponseShapeError,
)
from autosub.core.llm import BaseStructuredLLM
from autosub.pipeline.translate.translator import TranslatedSubtitle, VertexTranslator


class DummyStructuredLLM(BaseStructuredLLM):
    pass


class FakeAgent:
    def __init__(self, *, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def run_sync(self, user_prompt: str):
        if self._error is not None:
            raise self._error
        return self._result


def _make_llm(**kwargs) -> DummyStructuredLLM:
    return DummyStructuredLLM(
        project_id="test-project",
        model="gemini-test",
        location="us-central1",
        **kwargs,
    )


def _make_result(output, *, parts=None):
    usage = SimpleNamespace(input_tokens=5, output_tokens=12)
    response = SimpleNamespace(
        provider_response_id="resp-1",
        model_name="gemini-test-version",
        finish_reason="STOP",
        parts=parts or [],
        timestamp=None,
        run_id="run-1",
    )
    return SimpleNamespace(
        output=output,
        response=response,
        usage=lambda: usage,
    )


def test_run_structured_output_wraps_request_error(monkeypatch):
    llm = _make_llm()
    monkeypatch.setattr(
        llm,
        "_build_agent",
        lambda **kwargs: FakeAgent(error=RuntimeError("simulated transport failure")),
    )

    with pytest.raises(VertexRequestError) as exc_info:
        llm._run_structured_output(
            user_prompt="[]",
            system_prompt="test",
            output_type=list[dict],
            operation_name="Vertex test operation",
            output_name="test_output",
        )

    message = str(exc_info.value)
    assert "Vertex test operation request to LLM failed" in message
    assert "project_id=test-project" in message
    assert "model=gemini-test" in message
    assert "location=us-central1" in message


def test_run_structured_output_raises_blocked_response_error(monkeypatch):
    llm = _make_llm()
    monkeypatch.setattr(
        llm,
        "_build_agent",
        lambda **kwargs: FakeAgent(error=ContentFilterError("blocked by provider")),
    )

    with pytest.raises(VertexBlockedResponseError) as exc_info:
        llm._run_structured_output(
            user_prompt="[]",
            system_prompt="test",
            output_type=list[dict],
            operation_name="Vertex test operation",
            output_name="test_output",
        )

    message = str(exc_info.value)
    assert "Vertex test operation returned blocked output" in message
    assert "text_preview=blocked by provider" in message


def test_run_structured_output_raises_parse_error_with_preview(monkeypatch):
    llm = _make_llm()
    monkeypatch.setattr(
        llm,
        "_build_agent",
        lambda **kwargs: FakeAgent(
            error=UnexpectedModelBehavior("invalid structured output")
        ),
    )

    with pytest.raises(VertexResponseParseError) as exc_info:
        llm._run_structured_output(
            user_prompt="[]",
            system_prompt="test",
            output_type=list[dict],
            operation_name="Vertex test operation",
            output_name="test_output",
        )

    message = str(exc_info.value)
    assert "Vertex test operation returned invalid structured output" in message
    assert "text_preview=invalid structured output" in message


def test_run_structured_output_builds_generic_diagnostics(monkeypatch):
    llm = _make_llm()
    monkeypatch.setattr(
        llm,
        "_build_agent",
        lambda **kwargs: FakeAgent(
            result=_make_result([{"id": 0, "translated": "hi"}])
        ),
    )

    output, diagnostics = llm._run_structured_output(
        user_prompt="[]",
        system_prompt="test",
        output_type=list[dict],
        operation_name="Vertex test operation",
        output_name="test_output",
    )

    assert output == [{"id": 0, "translated": "hi"}]
    assert diagnostics.response_id == "resp-1"
    assert diagnostics.model_version == "gemini-test-version"
    assert diagnostics.candidate_finish_reasons == ("STOP",)
    assert diagnostics.prompt_token_count == 5
    assert diagnostics.candidates_token_count == 12
    assert diagnostics.total_token_count == 17


def test_run_structured_output_writes_trace_file(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    llm = _make_llm(trace_path=trace_path)
    monkeypatch.setattr(
        llm,
        "_build_agent",
        lambda **kwargs: FakeAgent(
            result=_make_result(
                [{"id": 0, "translated": "hi"}],
                parts=[ThinkingPart("reasoning"), TextPart("done")],
            )
        ),
    )

    llm._run_structured_output(
        user_prompt='[{"id": 0, "text": "こんにちは"}]',
        system_prompt="test system",
        output_type=list[dict],
        operation_name="Vertex test operation",
        output_name="test_output",
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["status"] == "success"
    assert entry["thinking_parts"] == [
        {
            "content": "reasoning",
            "id": None,
            "signature": None,
            "provider_name": None,
            "provider_details": None,
        }
    ]
    assert entry["response_parts"][0]["part_kind"] == "thinking"
    assert entry["output"] == [{"id": 0, "translated": "hi"}]


def test_extract_thinking_text_and_thoughts_token_count():
    llm = _make_llm()

    response = SimpleNamespace(
        parts=[ThinkingPart("step one"), TextPart("answer"), ThinkingPart("step two")]
    )
    assert llm._extract_thinking_text(response) == "step one\n\nstep two"
    assert llm._extract_thinking_text(SimpleNamespace(parts=[TextPart("only")])) is None

    assert (
        llm._extract_thoughts_token_count(
            SimpleNamespace(details={"thoughts_tokens": 4096})
        )
        == 4096
    )
    assert (
        llm._extract_thoughts_token_count(SimpleNamespace(thoughts_token_count=7)) == 7
    )
    assert llm._extract_thoughts_token_count(None) is None


def test_success_diagnostics_include_thinking_and_thoughts(monkeypatch):
    llm = _make_llm()
    usage = SimpleNamespace(
        input_tokens=5, output_tokens=12, details={"thoughts_tokens": 8}
    )
    response = SimpleNamespace(
        provider_response_id="resp-1",
        model_name="gemini-test-version",
        finish_reason="STOP",
        parts=[ThinkingPart("because reasons")],
        timestamp=None,
        run_id="run-1",
    )
    result = SimpleNamespace(
        output=[{"id": 0, "translated": "hi"}],
        response=response,
        usage=lambda: usage,
    )
    monkeypatch.setattr(llm, "_build_agent", lambda **kwargs: FakeAgent(result=result))

    _, diagnostics = llm._run_structured_output(
        user_prompt="[]",
        system_prompt="test",
        output_type=list[dict],
        operation_name="Vertex test operation",
        output_name="test_output",
    )

    assert diagnostics.thoughts_token_count == 8
    assert diagnostics.thinking_text == "because reasons"


def test_failure_salvages_partial_response_diagnostics(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    llm = _make_llm(trace_path=trace_path)

    partial = ModelResponse(
        parts=[ThinkingPart(content="runaway thinking")],
        usage=RequestUsage(
            input_tokens=10,
            output_tokens=65536,
            details={"thoughts_tokens": 65000},
        ),
        model_name="gemini-3-flash-preview",
        provider_response_id="resp-x",
        finish_reason="length",
    )

    @contextmanager
    def fake_capture():
        yield [partial]

    monkeypatch.setattr(
        "autosub.core.llm.pydantic_ai.capture_run_messages", fake_capture
    )
    monkeypatch.setattr(
        llm,
        "_build_agent",
        lambda **kwargs: FakeAgent(
            error=UnexpectedModelBehavior("Exceeded maximum retries")
        ),
    )

    with pytest.raises(VertexResponseParseError) as exc_info:
        llm._run_structured_output(
            user_prompt="[]",
            system_prompt="test",
            output_type=list[dict],
            operation_name="Vertex test operation",
            output_name="test_output",
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics.candidate_finish_reasons == ("length",)
    assert diagnostics.thoughts_token_count == 65000
    assert diagnostics.thinking_text == "runaway thinking"

    # The failing call's diagnostics are also persisted to the trace file.
    entry = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["status"] == "error"
    assert entry["finish_reason"] == "length"
    assert entry["diagnostics"]["thoughts_token_count"] == 65000


def test_vertex_translator_wraps_unexpected_json_shape(monkeypatch):
    translator = VertexTranslator(project_id="test-project")
    diagnostics = VertexResponseDiagnostics(response_id="resp-shape")

    monkeypatch.setattr(
        translator,
        "_run_structured_output",
        lambda **kwargs: (
            [TranslatedSubtitle(id=3, translated="hello")],
            diagnostics,
        ),
    )

    with pytest.raises(VertexResponseShapeError) as exc_info:
        translator.translate(["こんにちは"])

    message = str(exc_info.value)
    assert "unexpected structure" in message
    assert "response_id=resp-shape" in message
