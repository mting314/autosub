"""Tests for the combined radio_discourse + corners classifier path."""

from unittest.mock import patch

import pytest

from autosub.core.schemas import SubtitleLine, TranscribedWord
from autosub.pipeline.format.main import _apply_combined_extensions


def _line(text, start=0.0, end=1.0, speaker=None):
    return SubtitleLine(text=text, start_time=start, end_time=end, speaker=speaker)


@pytest.fixture
def sample_lines():
    return [
        _line("皆さんこんにちは！リエラジへようこそ！", 0, 3),
        _line("今日もお便りをいただいています。", 3, 6),
        _line("ラジオネームさくらさんから。", 6, 9),
        _line("先日のライブ最高でした！", 9, 12),
        _line("ありがとうね。嬉しい。", 12, 15),
        _line("続いてのコーナー、投書箱に参りましょう。", 15, 18),
        _line("最初のお題はこちらです。", 18, 21),
    ]


@pytest.fixture
def radio_config():
    return {
        "enabled": True,
        "engine": "hybrid",
        "split_framing_phrases": True,
        "label_roles": True,
    }


@pytest.fixture
def corners_config():
    return {
        "enabled": True,
        "engine": "hybrid",
        "segments": [
            {"name": "Opening", "cues": ["ようこそ"]},
            {"name": "Suggestion Box", "cues": ["投書箱"]},
        ],
    }


class TestApplyCombinedExtensions:
    def test_falls_back_to_rules_and_cues_on_vertex_error(
        self, sample_lines, radio_config, corners_config, tmp_path
    ):
        """When LLM fails with hybrid engines, fallback to rules + cues."""
        from autosub.core.errors import VertexError

        output_path = tmp_path / "test.ass"

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=VertexError("test error"),
        ):
            result = _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Should not raise — hybrid engines allow fallback
        assert len(result) >= len(sample_lines)  # may have more due to framing splits
        # Cue-based corners should be detected
        corners = [line.corner for line in result if line.corner]
        assert any("Opening" in c for c in corners)

    def test_raises_on_vertex_error_with_strict_engine(
        self, sample_lines, corners_config, tmp_path
    ):
        """When radio engine is 'llm' (strict), VertexError should propagate."""
        from autosub.core.errors import VertexError

        radio_config = {
            "enabled": True,
            "engine": "llm",
            "split_framing_phrases": True,
            "label_roles": True,
        }
        output_path = tmp_path / "test.ass"

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=VertexError("test error"),
        ):
            with pytest.raises(VertexError):
                _apply_combined_extensions(
                    sample_lines, radio_config, corners_config, output_path
                )

    def test_successful_combined_classification(
        self, sample_lines, radio_config, corners_config, tmp_path
    ):
        """When LLM succeeds, uses its roles and corners."""
        output_path = tmp_path / "test.ass"

        def mock_classify(lines, fallback_roles, segments, config, speaker_names=None):
            roles = ["host"] * len(lines)
            corners: list[str | None] = [None] * len(lines)
            corners[0] = "Opening"
            return roles, corners, [line.speaker for line in lines]

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            result = _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Should have roles assigned
        roles = [line.role for line in result]
        assert all(r == "host" for r in roles)

        # Should have the LLM corner + cue corners merged
        corners = [line.corner for line in result]
        assert "Opening" in corners

    def test_merges_corners_config_into_combined(self, sample_lines, tmp_path):
        """Corners config settings fill gaps not specified by radio config."""
        radio_config = {
            "enabled": True,
            "engine": "hybrid",
            "split_framing_phrases": True,
            "label_roles": True,
        }
        corners_config = {
            "enabled": True,
            "engine": "hybrid",
            "model": "corners-specific-model",
            "location": "us-central1",
            "segments": [
                {"name": "Opening", "cues": ["ようこそ"]},
            ],
        }
        output_path = tmp_path / "test.ass"

        captured_config = {}

        def mock_classify(lines, fallback_roles, segments, config, speaker_names=None):
            captured_config.update(config)
            return (
                ["host"] * len(lines),
                [None] * len(lines),
                [line.speaker for line in lines],
            )

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Corners-specific settings should be present since radio doesn't set them
        assert captured_config["model"] == "corners-specific-model"
        assert captured_config["location"] == "us-central1"

    def test_radio_config_takes_precedence_over_corners(self, sample_lines, tmp_path):
        """When both configs specify the same key, radio config wins."""
        radio_config = {
            "enabled": True,
            "engine": "hybrid",
            "model": "radio-model",
            "split_framing_phrases": True,
            "label_roles": True,
        }
        corners_config = {
            "enabled": True,
            "engine": "hybrid",
            "model": "corners-model",
            "segments": [
                {"name": "Opening", "cues": ["ようこそ"]},
            ],
        }
        output_path = tmp_path / "test.ass"

        captured_config = {}

        def mock_classify(lines, fallback_roles, segments, config, speaker_names=None):
            captured_config.update(config)
            return (
                ["host"] * len(lines),
                [None] * len(lines),
                [line.speaker for line in lines],
            )

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        assert captured_config["model"] == "radio-model"

    def test_cue_corners_fill_llm_gaps(
        self, sample_lines, radio_config, corners_config, tmp_path
    ):
        """Cue-detected corners fill in where LLM returns None."""
        output_path = tmp_path / "test.ass"

        def mock_classify(lines, fallback_roles, segments, config, speaker_names=None):
            roles = ["host"] * len(lines)
            corners = [None] * len(lines)
            # LLM doesn't detect "Suggestion Box" but cues will
            return roles, corners, [line.speaker for line in lines]

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            result = _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Cue-based detection should find both corners
        corners = [line.corner for line in result if line.corner]
        assert "Opening" in corners
        assert "Suggestion Box" in corners

    def test_combined_extensions_apply_greetings_split_before_classification(
        self, corners_config, tmp_path
    ):
        radio_config = {
            "enabled": True,
            "engine": "hybrid",
            "split_framing_phrases": False,
            "label_roles": True,
            "greetings": ["のんばんは"],
        }
        lines = [
            SubtitleLine(
                text="のんばんは？初メールです。",
                start_time=0.0,
                end_time=2.0,
                words=[
                    TranscribedWord(word="のん", start_time=0.0, end_time=0.3),
                    TranscribedWord(word="ばん", start_time=0.3, end_time=0.6),
                    TranscribedWord(word="は？", start_time=0.6, end_time=1.0),
                    TranscribedWord(word="初", start_time=1.0, end_time=1.2),
                    TranscribedWord(word="メール", start_time=1.2, end_time=1.6),
                    TranscribedWord(word="です。", start_time=1.6, end_time=2.0),
                ],
            )
        ]
        output_path = tmp_path / "test.ass"

        def mock_classify(lines, fallback_roles, segments, config, speaker_names=None):
            roles = ["host"] * len(lines)
            corners = [None] * len(lines)
            return roles, corners, [line.speaker for line in lines]

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            result = _apply_combined_extensions(
                lines, radio_config, corners_config, output_path
            )

        assert len(result) == 2
        assert result[0].text == "のんばんは？"
        assert result[1].text == "初メールです。"


# --- Speaker attribution (TASK 3) ---


_SPEAKER_MAP = {
    "0": {"name": "Date Sayuri", "character": "Shibuya Kanon", "slot": 1},
    "1": {"name": "Liyuu", "character": "Tang Keke", "slot": 2},
}


def _classifier(speaker_names=None):
    from autosub.extensions.combined_classifier import CombinedClassifier

    return CombinedClassifier(
        project_id="p",
        segments=[{"name": "Opening", "description": "d", "cues": ["ようこそ"]}],
        speaker_names=speaker_names,
    )


def test_speaker_task_is_absent_unless_speakers_are_supplied():
    """Shows that do not need the correction must not pay for the extra task."""
    instruction = _classifier()._get_system_instruction(3)

    assert "TASK 3" not in instruction
    assert "TWO tasks" in instruction
    assert "'speaker'" not in instruction


def test_speaker_task_names_the_speakers_and_the_addressee_rule():
    instruction = _classifier(["Date Sayuri", "Liyuu"])._get_system_instruction(3)

    assert "TASK 3" in instruction
    assert "THREE tasks" in instruction
    assert "- Date Sayuri" in instruction and "- Liyuu" in instruction
    # The point of the feedback: a name in the line is the addressee, not the speaker.
    assert "almost never the speaker" in instruction
    assert "鈴木" in instruction


def test_diarized_label_is_sent_as_a_prior_only_when_correcting():
    from autosub.extensions.combined_classifier import CombinedDecision

    lines = [(0, _line("じゃあ鈴木さんはどうですか", speaker="Liyuu"))]
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return [CombinedDecision(id=0, role="host", corner=None)], {}

    plain = _classifier()
    with patch.object(plain, "_run_structured_output", side_effect=fake_run):
        plain.classify_window(lines)
    assert "diarized" not in captured["user_prompt"]

    correcting = _classifier(["Date Sayuri", "Liyuu"])
    with patch.object(correcting, "_run_structured_output", side_effect=fake_run):
        correcting.classify_window(lines)
    assert '"diarized": "Liyuu"' in captured["user_prompt"]


def test_speaker_outside_the_map_is_discarded():
    from autosub.extensions.combined_classifier import CombinedDecision

    lines = [(0, _line("なにか", speaker="Liyuu"))]
    classifier = _classifier(["Date Sayuri", "Liyuu"])

    def fake_run(**kwargs):
        return [
            CombinedDecision(id=0, role="host", corner=None, speaker="Suzuki-san")
        ], {}

    with patch.object(classifier, "_run_structured_output", side_effect=fake_run):
        result = classifier.classify_window(lines)

    assert result[0] == ("host", None, None)


def _run_classify_combined(lines, decisions, speaker_names):
    """Drive classify_combined with a canned per-line speaker answer."""
    from autosub.extensions.combined_classifier import classify_combined

    def fake_window(self, window):
        return {
            line_id: ("host", None, decisions[line_id]) for line_id, _ in window
        }

    with patch(
        "autosub.extensions.combined_classifier.CombinedClassifier.classify_window",
        fake_window,
    ):
        return classify_combined(
            lines,
            [None] * len(lines),
            [],
            {"project_id": "p", "scope": "full_script"},
            speaker_names=speaker_names,
        )


def test_model_override_replaces_the_diarizer_when_it_names_someone_else():
    """The addressee case: the reply belongs to the person who was asked."""
    lines = [
        _line("じゃあ鈴木さんはどうですか", speaker="Liyuu"),
        _line("私はそうですね", speaker="Liyuu"),  # diarizer slipped at the turn
    ]

    _, _, speakers = _run_classify_combined(
        lines, {0: "Liyuu", 1: "Date Sayuri"}, ["Date Sayuri", "Liyuu"]
    )

    assert speakers == ["Liyuu", "Date Sayuri"]


def test_diarizer_stands_when_the_model_abstains():
    lines = [_line("なにか", speaker="Liyuu"), _line("そうだね", speaker="Date Sayuri")]

    _, _, speakers = _run_classify_combined(
        lines, {0: None, 1: None}, ["Date Sayuri", "Liyuu"]
    )

    assert speakers == ["Liyuu", "Date Sayuri"]


def test_speakers_are_untouched_when_attribution_is_off():
    lines = [_line("なにか", speaker="0"), _line("そうだね", speaker="1")]

    _, _, speakers = _run_classify_combined(lines, {0: None, 1: None}, None)

    assert speakers == ["0", "1"]


def test_combined_extensions_resolve_labels_to_names_before_correcting(
    sample_lines, corners_config, tmp_path
):
    """The prior sent to the model and the answer it returns share a vocabulary."""
    radio_config = {
        "enabled": True,
        "engine": "hybrid",
        "split_framing_phrases": False,
        "label_roles": True,
        "correct_speakers": True,
    }
    lines = [_line("こんにちは", 0, 3, speaker="0"), _line("どうも", 3, 6, speaker="1")]
    seen = {}

    def mock_classify(lines_, fallback_roles, segments, config, speaker_names=None):
        seen["names"] = speaker_names
        seen["diarized"] = [line.speaker for line in lines_]
        # Model moves the second line to the other host.
        return (
            ["host"] * len(lines_),
            [None] * len(lines_),
            ["Date Sayuri", "Date Sayuri"],
        )

    with patch(
        "autosub.extensions.combined_classifier.classify_combined",
        side_effect=mock_classify,
    ):
        result = _apply_combined_extensions(
            lines,
            radio_config,
            corners_config,
            tmp_path / "t.ass",
            speaker_map=_SPEAKER_MAP,
        )

    assert seen["names"] == ["Date Sayuri", "Liyuu"]
    # Raw labels "0"/"1" were resolved before the call, not passed through.
    assert seen["diarized"] == ["Date Sayuri", "Liyuu"]
    assert [line.speaker for line in result] == ["Date Sayuri", "Date Sayuri"]


def test_combined_extensions_leave_speakers_alone_without_the_flag(
    corners_config, tmp_path
):
    radio_config = {
        "enabled": True,
        "engine": "hybrid",
        "split_framing_phrases": False,
        "label_roles": True,
    }
    lines = [_line("こんにちは", 0, 3, speaker="0")]

    def mock_classify(lines_, fallback_roles, segments, config, speaker_names=None):
        assert speaker_names is None
        return ["host"], [None], ["Date Sayuri"]

    with patch(
        "autosub.extensions.combined_classifier.classify_combined",
        side_effect=mock_classify,
    ):
        result = _apply_combined_extensions(
            lines,
            radio_config,
            corners_config,
            tmp_path / "t.ass",
            speaker_map=_SPEAKER_MAP,
        )

    assert [line.speaker for line in result] == ["0"]


# --- Per-window resilience ---


def test_a_failed_window_is_retried_before_the_pass_is_abandoned():
    """One transient window failure used to discard every window already done."""
    from autosub.core.errors import VertexRequestError
    from autosub.extensions.combined_classifier import _classify_window_with_retry

    calls = {"n": 0}

    class Flaky:
        def classify_window(self, window):
            calls["n"] += 1
            if calls["n"] == 1:
                raise VertexRequestError("connection cut")
            return {line_id: ("host", None, None) for line_id, _ in window}

    with patch("autosub.extensions.combined_classifier.time.sleep"):
        result = _classify_window_with_retry(Flaky(), [(0, _line("x"))], 0, 1)

    assert calls["n"] == 2
    assert result == {0: ("host", None, None)}


def test_a_window_that_never_succeeds_still_raises():
    from autosub.core.errors import VertexRequestError
    from autosub.extensions.combined_classifier import _classify_window_with_retry

    class Broken:
        def classify_window(self, window):
            raise VertexRequestError("connection cut")

    with patch("autosub.extensions.combined_classifier.time.sleep"):
        with pytest.raises(VertexRequestError):
            _classify_window_with_retry(Broken(), [(0, _line("x"))], 0, 1)


def test_completed_windows_survive_a_later_retry():
    """A retry mid-pass must not lose the windows already classified."""
    from autosub.core.errors import VertexRequestError
    from autosub.extensions.combined_classifier import classify_combined

    lines = [_line(f"line {i}", i, i + 1, speaker="Liyuu") for i in range(6)]
    seen = {"n": 0}

    def flaky_window(self, window):
        seen["n"] += 1
        if seen["n"] == 2:  # second window fails once, then succeeds
            raise VertexRequestError("connection cut")
        return {line_id: ("host", None, None) for line_id, _ in window}

    with patch(
        "autosub.extensions.combined_classifier.CombinedClassifier.classify_window",
        flaky_window,
    ), patch("autosub.extensions.combined_classifier.time.sleep"):
        roles, _, _ = classify_combined(
            lines,
            [None] * len(lines),
            [],
            {"project_id": "p", "scope": "windowed", "window_size": 3,
             "window_overlap": 0},
        )

    assert roles == ["host"] * 6
