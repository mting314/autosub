from autosub.core.errors import (
    VertexError,
    VertexResponseDiagnostics,
    VertexResponseError,
    _truncate,
)


def test_truncate_none():
    assert _truncate(None) is None


def test_truncate_short_string():
    assert _truncate("hello") == "hello"


def test_truncate_normalizes_whitespace():
    assert _truncate("hello   world") == "hello world"


def test_truncate_long_string():
    result = _truncate("a" * 300)
    assert len(result) == 240
    assert result.endswith("...")


def test_truncate_custom_limit():
    result = _truncate("a" * 20, limit=10)
    assert len(result) == 10
    assert result.endswith("...")


def test_vertex_error_str_no_context():
    err = VertexError("something broke")
    assert str(err) == "something broke"


def test_vertex_error_str_with_context():
    err = VertexError(
        "fail", project_id="proj", model="gemini", location="us-central1"
    )
    assert "project_id=proj" in str(err)
    assert "model=gemini" in str(err)
    assert "location=us-central1" in str(err)


def test_vertex_error_str_partial_context():
    err = VertexError("fail", model="gemini")
    result = str(err)
    assert "model=gemini" in result
    assert "project_id" not in result


def test_diagnostics_summary_parts_empty():
    diag = VertexResponseDiagnostics()
    assert diag.summary_parts() == []


def test_diagnostics_summary_parts_full():
    diag = VertexResponseDiagnostics(
        response_id="resp-1",
        model_version="v1",
        prompt_block_reason="SAFETY",
        prompt_block_reason_message="blocked for safety",
        prompt_safety_ratings=("HIGH",),
        candidate_finish_reasons=("STOP",),
        candidate_finish_messages=("done",),
        candidate_token_counts=(100,),
        candidate_safety_ratings=("LOW",),
        prompt_token_count=50,
        candidates_token_count=100,
        thoughts_token_count=20,
        total_token_count=170,
        text_preview="hello...",
    )
    parts = diag.summary_parts()
    assert any("response_id=resp-1" in p for p in parts)
    assert any("model_version=v1" in p for p in parts)
    assert any("SAFETY" in p for p in parts)
    assert any("tokens=" in p for p in parts)
    assert any("text_preview=" in p for p in parts)


def test_vertex_response_error_includes_diagnostics():
    diag = VertexResponseDiagnostics(response_id="resp-1")
    err = VertexResponseError("bad response", diagnostics=diag, model="gemini")
    result = str(err)
    assert "bad response" in result
    assert "model=gemini" in result
    assert "response_id=resp-1" in result
