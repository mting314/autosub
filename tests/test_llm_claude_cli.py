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
    # System prompt goes through a file, never argv (OS arg-length limits).
    assert "--system-prompt" not in command
    assert "sys" not in command
    assert command[command.index("--system-prompt-file") + 1].endswith(
        "system_prompt.txt"
    )
    assert result.effort == "medium"


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


def test_system_prompt_file_contents(monkeypatch):
    monkeypatch.setattr(claude_cli, "resolve_claude_binary", lambda: "/bin/claude")
    seen = {}

    def fake_run(command, **kwargs):
        path = command[command.index("--system-prompt-file") + 1]
        seen["prompt"] = open(path, encoding="utf-8").read()
        return _fake_completed(
            {"type": "result", "subtype": "success", "structured_output": {"items": []}}
        )

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    big = "日本語のプロンプト" * 20000  # ~540KB UTF-8, far past any argv limit
    claude_cli.run_structured(
        model="opus", system_prompt=big, user_prompt="u", output_type=list[_Line]
    )
    assert seen["prompt"] == big


def test_bare_can_be_disabled(fake_claude, monkeypatch):
    calls = fake_claude(
        {"type": "result", "subtype": "success", "structured_output": {"items": []}}
    )
    monkeypatch.setenv(claude_cli.CLAUDE_CLI_ENV_BARE, "0")
    claude_cli.run_structured(
        model="opus", system_prompt="s", user_prompt="u", output_type=list[_Line]
    )
    assert "--bare" not in calls[0]["command"]


def test_off_effort_maps_to_low():
    assert claude_cli.map_effort("off") == "low"
    assert claude_cli.map_effort("minimal") == "low"
    assert claude_cli.map_effort(None) is None


def test_malformed_timeout_raises_cli_error(fake_claude, monkeypatch):
    fake_claude({"type": "result", "subtype": "success", "structured_output": {}})
    monkeypatch.setenv(claude_cli.CLAUDE_CLI_ENV_TIMEOUT, "30m")
    with pytest.raises(claude_cli.ClaudeCLIError, match="number of seconds"):
        claude_cli.run_structured(
            model="opus", system_prompt="s", user_prompt="u", output_type=list[_Line]
        )


def test_nonzero_exit_with_non_result_json_reports_exit_code(monkeypatch):
    monkeypatch.setattr(claude_cli, "resolve_claude_binary", lambda: "/bin/claude")
    monkeypatch.setattr(
        claude_cli.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 3, stdout='{"banner": true}\n', stderr="boom"
        ),
    )
    with pytest.raises(claude_cli.ClaudeCLIError, match="exited with code 3: boom"):
        claude_cli.run_structured(
            model="opus", system_prompt="s", user_prompt="u", output_type=list[_Line]
        )


def test_non_result_json_is_not_treated_as_result(fake_claude):
    fake_claude({"banner": True})
    with pytest.raises(claude_cli.ClaudeCLIOutputError, match="did not return"):
        claude_cli.run_structured(
            model="opus", system_prompt="s", user_prompt="u", output_type=list[_Line]
        )


def test_auth_error_hints_at_bare_mode(fake_claude):
    fake_claude(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "Invalid API key · Please run /login",
        }
    )
    with pytest.raises(claude_cli.ClaudeCLIError, match="AUTOSUB_CLAUDE_CLI_BARE=0"):
        claude_cli.run_structured(
            model="opus", system_prompt="s", user_prompt="u", output_type=list[_Line]
        )


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("autosub.pipeline.translate.translator", "VertexTranslator"),
        ("autosub.pipeline.translate.reflow_llm", "ReflowSplitter"),
        ("autosub.pipeline.format.normalizer", "LLMKeywordNormalizer"),
        ("autosub.extensions.combined_classifier", "CombinedClassifier"),
    ],
)
def test_default_models_cover_claude_cli(module_path, class_name):
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    assert cls.DEFAULT_MODELS["claude-cli"] in ("opus", "haiku")


def test_resolver_missing_creds_message_suggests_flag_not_env(monkeypatch):
    for var in (
        "GOOGLE_CLOUD_PROJECT",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMResolutionError) as excinfo:
        resolve_llm_selection(model="claude-opus-5-5", provider=None)
    message = str(excinfo.value)
    assert "AUTOSUB_CLAUDE_CLI" not in message
    assert "--llm-provider claude-cli" in message
