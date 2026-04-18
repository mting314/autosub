import pytest
from unittest.mock import patch, MagicMock

from autosub.core.errors import VertexResponseDiagnostics, VertexResponseShapeError
from autosub.core.schemas import SubtitleLine
from autosub.extensions.corners.classifier import (
    CornerDecision,
    VertexCornerClassifier,
    _build_windows,
    classify_corners_with_vertex,
)


SEGMENTS = [
    {"name": "Opening", "description": "Opening skit", "cues": ["始まり"]},
    {"name": "Closing", "description": "Closing segment"},
]


def _make_line(text="テスト", start=0.0, end=1.0):
    return SubtitleLine(text=text, start_time=start, end_time=end)


def _make_lines(n):
    return [_make_line(f"line {i}", i, i + 1) for i in range(n)]


# --- CornerDecision ---


def test_corner_decision_with_corner():
    d = CornerDecision(id=0, corner="Opening")
    assert d.corner == "Opening"


def test_corner_decision_null_corner():
    d = CornerDecision(id=0, corner=None)
    assert d.corner is None


# --- _build_windows ---


def test_build_windows_full_script():
    lines = _make_lines(5)
    windows = _build_windows(lines, {"scope": "full_script"})
    assert len(windows) == 1
    assert len(windows[0]) == 5


def test_build_windows_default_is_full_script():
    lines = _make_lines(5)
    windows = _build_windows(lines, {})
    assert len(windows) == 1


def test_build_windows_sliding():
    lines = _make_lines(10)
    config = {"scope": "sliding", "window_size": 4, "window_overlap": 1}
    windows = _build_windows(lines, config)
    assert len(windows) > 1
    assert all(len(w) <= 4 for w in windows)
    all_ids = set()
    for w in windows:
        for line_id, _ in w:
            all_ids.add(line_id)
    assert all_ids == set(range(10))


def test_build_windows_sliding_covers_end():
    lines = _make_lines(7)
    config = {"scope": "sliding", "window_size": 3, "window_overlap": 1}
    windows = _build_windows(lines, config)
    last_ids = [line_id for line_id, _ in windows[-1]]
    assert 6 in last_ids


def test_build_windows_overlap_larger_than_size():
    lines = _make_lines(5)
    config = {"scope": "sliding", "window_size": 3, "window_overlap": 10}
    windows = _build_windows(lines, config)
    assert len(windows) >= 1


# --- VertexCornerClassifier ---


def test_system_instruction_includes_segments():
    classifier = VertexCornerClassifier(
        project_id="test", segments=SEGMENTS
    )
    instruction = classifier._get_system_instruction(5)
    assert "Opening" in instruction
    assert "Closing" in instruction
    assert "始まり" in instruction
    assert "exactly 5 items" in instruction


def test_classify_window_empty():
    classifier = VertexCornerClassifier(
        project_id="test", segments=SEGMENTS
    )
    assert classifier.classify_window([]) == {}


def test_classify_window_valid_response():
    classifier = VertexCornerClassifier(
        project_id="test", segments=SEGMENTS
    )
    decisions = [
        CornerDecision(id=0, corner="Opening"),
        CornerDecision(id=1, corner=None),
    ]
    diag = VertexResponseDiagnostics()
    lines = [(0, _make_line("line 0")), (1, _make_line("line 1"))]

    with patch.object(
        classifier, "_run_structured_output", return_value=(decisions, diag)
    ):
        result = classifier.classify_window(lines)

    assert result == {0: "Opening", 1: None}


def test_classify_window_filters_invalid_corner_names():
    classifier = VertexCornerClassifier(
        project_id="test", segments=SEGMENTS
    )
    decisions = [
        CornerDecision(id=0, corner="NonexistentSegment"),
    ]
    diag = VertexResponseDiagnostics()
    lines = [(0, _make_line())]

    with patch.object(
        classifier, "_run_structured_output", return_value=(decisions, diag)
    ):
        result = classifier.classify_window(lines)

    assert result[0] is None


def test_classify_window_mismatched_ids_raises():
    classifier = VertexCornerClassifier(
        project_id="test", segments=SEGMENTS
    )
    decisions = [
        CornerDecision(id=99, corner=None),
    ]
    diag = VertexResponseDiagnostics()
    lines = [(0, _make_line())]

    with patch.object(
        classifier, "_run_structured_output", return_value=(decisions, diag)
    ):
        with pytest.raises(VertexResponseShapeError):
            classifier.classify_window(lines)


# --- classify_corners_with_vertex ---


def test_classify_corners_empty_lines():
    assert classify_corners_with_vertex([], SEGMENTS, {}) == []


def test_classify_corners_requires_project_id_for_vertex():
    lines = _make_lines(1)
    with pytest.raises(ValueError, match="project id"):
        classify_corners_with_vertex(lines, SEGMENTS, {"provider": "google-vertex"})


def test_classify_corners_end_to_end():
    lines = _make_lines(3)
    decisions = {0: "Opening", 1: None, 2: "Closing"}
    diag = VertexResponseDiagnostics()

    with patch.object(
        VertexCornerClassifier,
        "classify_window",
        return_value=decisions,
    ):
        result = classify_corners_with_vertex(
            lines, SEGMENTS, {"project_id": "test"}
        )

    assert result == ["Opening", None, "Closing"]


def test_classify_corners_majority_vote():
    lines = _make_lines(2)

    call_count = 0

    def mock_classify(self, window_lines):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {lid: "Opening" for lid, _ in window_lines}
        return {lid: None for lid, _ in window_lines}

    with patch.object(VertexCornerClassifier, "classify_window", mock_classify):
        result = classify_corners_with_vertex(
            lines,
            SEGMENTS,
            {
                "project_id": "test",
                "scope": "sliding",
                "window_size": 2,
                "window_overlap": 1,
            },
        )

    assert result[0] == "Opening"
