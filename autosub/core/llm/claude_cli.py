"""Structured-output backend that shells out to the local ``claude`` CLI.

Lets translation (and the other structured LLM stages) run on whatever Claude
model the locally installed Claude Code is authenticated for, instead of a
separately billed API key or GCP project.

The call is ``claude --bare -p --output-format json --json-schema ...``:

- ``--bare`` drops hooks, plugins, MCP servers, skills and CLAUDE.md discovery,
  cutting per-call overhead from ~65k tokens to under 1k. Bare mode only
  authenticates via ``ANTHROPIC_API_KEY``/``apiKeyHelper`` or a 3P provider
  (Bedrock/Vertex/Foundry) — it never reads an OAuth/keychain login. Set
  ``AUTOSUB_CLAUDE_CLI_BARE=0`` to run without it (slower, but uses OAuth).
- The system prompt is passed via ``--system-prompt-file`` so large profiles
  don't hit OS argument-length limits (Windows ~32K chars, Linux 128KB/arg).
- ``--tools ""`` disables every built-in tool; the model can only answer.
- ``--json-schema`` makes the CLI validate the final answer against the schema
  and return it under ``structured_output``.

The CLI requires an object at the schema root, so non-object output types
(e.g. ``list[TranslatedSubtitle]``) are wrapped as ``{"items": ...}`` and
unwrapped before validation.

Extra launcher flags can be supplied via ``AUTOSUB_CLAUDE_CLI_ARGS`` (shell
syntax), e.g. ``--dangerously-disable-osx-sandbox`` when autosub itself runs
inside a sandbox that forbids nesting another one.

The CLI has no "thinking off" level, so ``off``/``minimal`` map to ``--effort
low``; the effective level is recorded on the result (and in traces).
Temperature, token caps and reasoning budgets are not exposed by the CLI and
are ignored.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

CLAUDE_CLI_ENV_BINARY = "AUTOSUB_CLAUDE_CLI"
CLAUDE_CLI_ENV_ARGS = "AUTOSUB_CLAUDE_CLI_ARGS"
CLAUDE_CLI_ENV_TIMEOUT = "AUTOSUB_CLAUDE_CLI_TIMEOUT"
CLAUDE_CLI_ENV_BARE = "AUTOSUB_CLAUDE_CLI_BARE"
DEFAULT_TIMEOUT_SECONDS = 1800.0

_WRAP_KEY = "items"

# autosub ReasoningEffort value -> `claude --effort` level.
_EFFORT_MAP: dict[str, str] = {
    "off": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class ClaudeCLIError(RuntimeError):
    """The CLI could not be run or returned an error result."""


class ClaudeCLIOutputError(ClaudeCLIError):
    """The CLI ran but its structured output was missing or invalid."""


@dataclass(frozen=True, slots=True)
class ClaudeCLIResult:
    output: Any
    raw: dict[str, Any]
    command: list[str]
    effort: str | None = None

    @property
    def usage(self) -> dict[str, Any]:
        usage = self.raw.get("usage")
        return usage if isinstance(usage, dict) else {}

    @property
    def input_tokens(self) -> int:
        usage = self.usage
        return sum(
            int(usage.get(key) or 0)
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )

    @property
    def output_tokens(self) -> int:
        return int(self.usage.get("output_tokens") or 0)


def resolve_claude_binary() -> str | None:
    override = os.environ.get(CLAUDE_CLI_ENV_BINARY)
    if override:
        return shutil.which(override) or (
            override if os.path.exists(override) else None
        )
    return shutil.which("claude")


def claude_cli_available() -> bool:
    return resolve_claude_binary() is not None


def map_effort(reasoning_effort: str | None) -> str | None:
    return _EFFORT_MAP.get((reasoning_effort or "").lower())


def _bare_enabled() -> bool:
    return os.environ.get(CLAUDE_CLI_ENV_BARE, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    raw = os.environ.get(CLAUDE_CLI_ENV_TIMEOUT)
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError as exc:
        raise ClaudeCLIError(
            f"{CLAUDE_CLI_ENV_TIMEOUT} must be a number of seconds, got {raw!r}"
        ) from exc


def build_output_schema(output_type: Any) -> tuple[dict[str, Any], bool]:
    """Return (schema, wrapped). Wraps non-object roots under ``items``."""
    schema = TypeAdapter(output_type).json_schema()
    if schema.get("type") == "object":
        return schema, False

    defs = schema.pop("$defs", None)
    wrapped: dict[str, Any] = {
        "type": "object",
        "properties": {_WRAP_KEY: schema},
        "required": [_WRAP_KEY],
        "additionalProperties": False,
    }
    if defs:
        wrapped["$defs"] = defs
    return wrapped, True


def build_command(
    *,
    binary: str,
    model: str,
    system_prompt_file: str,
    schema: dict[str, Any],
    reasoning_effort: str | None,
) -> list[str]:
    extra = shlex.split(os.environ.get(CLAUDE_CLI_ENV_ARGS, ""))
    command = [
        binary,
        *extra,
        *(["--bare"] if _bare_enabled() else []),
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--tools",
        "",
        "--system-prompt-file",
        system_prompt_file,
        "--json-schema",
        json.dumps(schema, ensure_ascii=False),
    ]
    effort = map_effort(reasoning_effort)
    if effort:
        command += ["--effort", effort]
    return command


def _parse_result_json(stdout: str) -> dict[str, Any]:
    # The launcher may print banner lines before the JSON object; take the
    # last line that parses as a result object.
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "result":
            return payload
    start = stdout.find('{"')
    if start >= 0:
        try:
            payload = json.loads(stdout[start:])
            if isinstance(payload, dict) and payload.get("type") == "result":
                return payload
        except json.JSONDecodeError:
            pass
    raise ClaudeCLIOutputError(
        f"claude CLI did not return a JSON result. stdout tail: {stdout[-500:]!r}"
    )


def run_structured(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    output_type: Any,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> ClaudeCLIResult:
    binary = resolve_claude_binary()
    if binary is None:
        raise ClaudeCLIError(
            "claude-cli provider requires the `claude` CLI on PATH "
            f"(or set {CLAUDE_CLI_ENV_BINARY})."
        )

    schema, wrapped = build_output_schema(output_type)
    timeout = _resolve_timeout(timeout)

    # Run from an empty directory so no project files leak into context; the
    # system prompt goes in a file there to stay clear of argv length limits.
    with tempfile.TemporaryDirectory(prefix="autosub-claude-") as workdir:
        prompt_file = os.path.join(workdir, "system_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as handle:
            handle.write(system_prompt)
        command = build_command(
            binary=binary,
            model=model,
            system_prompt_file=prompt_file,
            schema=schema,
            reasoning_effort=reasoning_effort,
        )
        try:
            completed = subprocess.run(
                command,
                input=user_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=workdir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"claude CLI timed out after {timeout:.0f}s") from exc
        except OSError as exc:
            raise ClaudeCLIError(f"failed to launch claude CLI: {exc}") from exc

    try:
        payload = _parse_result_json(completed.stdout)
    except ClaudeCLIOutputError:
        if completed.returncode != 0:
            raise ClaudeCLIError(
                f"claude CLI exited with code {completed.returncode}: "
                f"{completed.stderr.strip()[-800:]}"
            ) from None
        raise

    if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
        message = (
            "claude CLI returned an error result: "
            f"subtype={payload.get('subtype')!r} "
            f"api_error_status={payload.get('api_error_status')!r} "
            f"result={str(payload.get('result'))[:500]!r}"
        )
        if _bare_enabled() and _looks_like_auth_error(payload):
            message += (
                f". --bare mode never reads OAuth/keychain logins; set "
                f"{CLAUDE_CLI_ENV_BARE}=0 or provide ANTHROPIC_API_KEY."
            )
        raise ClaudeCLIError(message)

    structured = payload.get("structured_output")
    if structured is None:
        raise ClaudeCLIOutputError(
            "claude CLI result has no structured_output. "
            f"result={str(payload.get('result'))[:500]!r}"
        )
    if wrapped:
        if not isinstance(structured, dict) or _WRAP_KEY not in structured:
            raise ClaudeCLIOutputError(
                f"claude CLI structured_output missing '{_WRAP_KEY}' wrapper key."
            )
        structured = structured[_WRAP_KEY]

    try:
        output = TypeAdapter(output_type).validate_python(structured)
    except Exception as exc:
        raise ClaudeCLIOutputError(
            f"claude CLI structured_output failed validation: {exc}"
        ) from exc

    return ClaudeCLIResult(
        output=output,
        raw=payload,
        command=command,
        effort=map_effort(reasoning_effort),
    )


def _looks_like_auth_error(payload: dict[str, Any]) -> bool:
    status = payload.get("api_error_status")
    if status in (401, 403, "401", "403"):
        return True
    text = str(payload.get("result") or "").lower()
    return any(
        marker in text
        for marker in ("api key", "apikey", "authentication", "not logged in", "login")
    )
