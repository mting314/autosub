import json
import subprocess

import pytest
from pydantic import BaseModel

from autosub.core.errors import VertexRequestError, VertexResponseParseError
from autosub.core.llm import LLMResolutionError, resolve_llm_selection
from autosub.core.llm import claude_cli
from autosub.core.llm.pydantic_ai import BaseStructuredLLM


class _Line(BaseModel):
    id: int
    text: str


def _fake_completed(payload: dict, *, returncode: int = 0, banner: str = ""):
    return subprocess.CompletedProcess(
        args=["claude"],
        returncode=returncode,
        stdout=banner + json.dumps(payload) + "\n",
        stderr="",
    )


@pytest.fixture
def fake_claude(monkeypatch):
    monkeypatch.setattr(claude_cli, "resolve_claude_binary", lambda: "/bin/claude")
    calls: list[dict] = []

    def install(payload: dict, **kwargs):
        def fake_run(command, **run_kwargs):
            calls.append({"command": command, **run_kwargs})
            return _fake_completed(payload, **kwargs)

        monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
        return calls

    return install


def test_list_output_type_is_wrapped_and_unwrapped(fake_claude):
    calls = fake_claude(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": {"items": [{"id": 1, "text": "Hello"}]},
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        banner="Claude Code at Meta\nUsing AI Gateway\n",
    )

    result = claude_cli.run_structured(
        model="claude-opus-5-5",
        system_prompt="sys",
        user_prompt="user",
        output_type=list[_Line],
        reasoning_effort="medium",
    )

    assert result.output == [_Line(id=1, text="Hello")]
    command = calls[0]["command"]
    assert "--bare" in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--effort") + 1] == "medium"
    schema = json.loads(command[command.index("--json-schema") + 1])
    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert "$defs" in schema
    assert calls[0]["input"] == "user"


def test_object_output_type_is_not_wrapped():
    schema, wrapped = claude_cli.build_output_schema(_Line)
    assert wrapped is False
    assert schema["type"] == "object"


def test_error_result_raises(fake_claude):
    fake_claude({"type": "result", "subtype": "error_max_turns", "is_error": True})
    with pytest.raises(claude_cli.ClaudeCLIError):
        claude_cli.run_structured(
            model="claude-opus-5-5",
            system_prompt="s",
            user_prompt="u",
            output_type=list[_Line],
        )


def test_invalid_structured_output_raises_output_error(fake_claude):
    fake_claude(
        {
            "type": "result",
            "subtype": "success",
            "structured_output": {"items": [{"id": "x"}]},
        }
    )
    with pytest.raises(claude_cli.ClaudeCLIOutputError):
        claude_cli.run_structured(
            model="claude-opus-5-5",
            system_prompt="s",
            user_prompt="u",
            output_type=list[_Line],
        )


def test_base_structured_llm_routes_to_cli_and_traces(fake_claude, tmp_path):
    fake_claude(
        {
            "type": "result",
            "subtype": "success",
            "structured_output": {"items": [{"id": 2, "text": "Hi"}]},
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "session_id": "abc",
        }
    )
    trace = tmp_path / "trace.jsonl"
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-opus-5-5",
        provider="claude-cli",
        reasoning_effort="low",
        trace_path=trace,
    )

    output, diagnostics = llm._run_structured_output(
        user_prompt="u",
        system_prompt="s",
        output_type=list[_Line],
        operation_name="Test",
        output_name="lines",
    )

    assert output == [_Line(id=2, text="Hi")]
    assert diagnostics.total_token_count == 10
    entry = json.loads(trace.read_text().splitlines()[0])
    assert entry["provider"] == "claude-cli"
    assert entry["status"] == "success"


def test_base_structured_llm_maps_cli_errors(fake_claude):
    llm = BaseStructuredLLM(
        project_id=None, model="claude-opus-5-5", provider="claude-cli"
    )

    fake_claude({"type": "result", "subtype": "success", "structured_output": None})
    with pytest.raises(VertexResponseParseError):
        llm._run_structured_output(
            user_prompt="u",
            system_prompt="s",
            output_type=list[_Line],
            operation_name="Test",
            output_name="lines",
        )

    fake_claude({"type": "result", "subtype": "error", "is_error": True})
    with pytest.raises(VertexRequestError):
        llm._run_structured_output(
            user_prompt="u",
            system_prompt="s",
            output_type=list[_Line],
            operation_name="Test",
            output_name="lines",
        )


def test_resolver_accepts_explicit_claude_cli(monkeypatch):
    monkeypatch.setattr(claude_cli, "resolve_claude_binary", lambda: "/bin/claude")
    selection = resolve_llm_selection(model="claude-opus-5-5", provider="claude-cli")
    assert selection.provider == "claude-cli"
    assert selection.model == "claude-opus-5-5"


def test_resolver_never_auto_selects_claude_cli(monkeypatch):
    for var in (
        "GOOGLE_CLOUD_PROJECT",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(claude_cli, "resolve_claude_binary", lambda: "/bin/claude")
    with pytest.raises(LLMResolutionError):
        resolve_llm_selection(model="claude-opus-5-5", provider=None)


def test_resolver_reports_missing_cli(monkeypatch):
    monkeypatch.setattr(claude_cli, "resolve_claude_binary", lambda: None)
    with pytest.raises(LLMResolutionError, match="claude"):
        resolve_llm_selection(model="claude-opus-5-5", provider="claude-cli")
