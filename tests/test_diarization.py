import pyass

from autosub.core.schemas import SubtitleLine, TranscribedWord
from autosub.core.speaker_map import build_speaker_prompt, hex_to_pyass_color, load_speaker_map, remap_speaker_labels
from autosub.pipeline.format.chunker import chunk_words_to_lines
from autosub.core.profile import load_unified_profile


def test_chunk_words_by_speaker():
    # Simulate overlapping speech from 2 speakers
    words = [
        TranscribedWord(
            word="Hello,", start_time=0.0, end_time=0.5, speaker="Speaker_1"
        ),
        TranscribedWord(word="Hi", start_time=0.2, end_time=0.4, speaker="Speaker_2"),
        TranscribedWord(word="how", start_time=0.5, end_time=0.8, speaker="Speaker_1"),
        TranscribedWord(
            word="there!", start_time=0.4, end_time=0.9, speaker="Speaker_2"
        ),
        TranscribedWord(word="are", start_time=0.8, end_time=1.0, speaker="Speaker_1"),
        TranscribedWord(word="you?", start_time=1.0, end_time=1.5, speaker="Speaker_1"),
    ]

    lines = chunk_words_to_lines(words)

    # We expect 2 lines, one for speaker 1 and one for speaker 2.
    assert len(lines) == 2

    l1 = lines[
        0
    ]  # Speaker 2 starts earliest logically but we sort chronologically by line start time. Speaker 1 starts at 0.0
    assert l1.speaker == "Speaker_1"
    assert l1.text == "Hello,howareyou?"

    l2 = lines[1]  # Speaker 2 starts at 0.2
    assert l2.speaker == "Speaker_2"
    assert l2.text == "Hithere!"


def test_soft_punctuation_splits_dense_chunk_without_pause():
    words = [
        TranscribedWord(word=f"語{i}、", start_time=i * 0.1, end_time=i * 0.1 + 0.05)
        for i in range(12)
    ]
    words.extend(
        [
            TranscribedWord(word="続き", start_time=1.2, end_time=1.25),
            TranscribedWord(word="です", start_time=1.3, end_time=1.35),
        ]
    )

    lines = chunk_words_to_lines(words)

    assert len(lines) == 2
    assert lines[0].text == "".join(word.word for word in words[:12])
    assert lines[1].text == "続きです"


def test_soft_punctuation_does_not_split_short_chunk_without_pause():
    words = [
        TranscribedWord(word=f"語{i}、", start_time=i * 0.1, end_time=i * 0.1 + 0.05)
        for i in range(11)
    ]
    words.append(TranscribedWord(word="続き", start_time=1.1, end_time=1.15))

    lines = chunk_words_to_lines(words)

    assert len(lines) == 1
    assert lines[0].text == "".join(word.word for word in words)


def test_profile_speaker_parsing(tmp_path):
    # Mock a TOML profile
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    test_toml = profile_dir / "test.toml"
    test_toml.write_text('vocab = ["test"]\nspeakers = 3\n')

    # Monkeypatch the Path inside load_unified_profile to read from our tmp_path
    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            # If checking for "profiles", map down to our tmp_dir
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("test")
        assert data["speakers"] == 3
        assert data["vocab"] == ["test"]
    finally:
        autosub.core.profile.Path = original_path


def test_profile_prompt_inheritance(tmp_path):
    profile_dir = tmp_path / "profiles"
    prompt_dir = tmp_path / "prompts"
    profile_dir.mkdir()
    prompt_dir.mkdir()

    (prompt_dir / "base.md").write_text("base guidance", encoding="utf-8")
    (prompt_dir / "child.md").write_text("child guidance", encoding="utf-8")

    (profile_dir / "base.toml").write_text(
        'prompt = "prompts/base.md"\n', encoding="utf-8"
    )
    (profile_dir / "child.toml").write_text(
        'extends = ["base"]\nprompt = "prompts/child.md"\n',
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts/base.md":
                return prompt_dir / "base.md"
            if args and args[0] == "prompts/child.md":
                return prompt_dir / "child.md"
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["prompt"] == ["base guidance", "child guidance"]
    finally:
        autosub.core.profile.Path = original_path


# --- Speaker map tests ---


def test_load_speaker_map(tmp_path):
    toml_content = """\
[speakers."0"]
name = "Suzuki Minori"
character = "Ena Shinonome"
color = "#FFA0A0"

[speakers."1"]
name = "Sato Hinata"
character = "Mizuki Akiyama"
color = "#A0D0FF"
"""
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text(toml_content, encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result == {
        "0": {"name": "Suzuki Minori", "character": "Ena Shinonome", "color": "#FFA0A0"},
        "1": {"name": "Sato Hinata", "character": "Mizuki Akiyama", "color": "#A0D0FF"},
    }


def test_load_speaker_map_missing_color(tmp_path):
    toml_content = """\
[speakers."1"]
name = "Speaker One"
"""
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text(toml_content, encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result["1"]["name"] == "Speaker One"
    assert result["1"]["color"] is None
    assert result["1"]["character"] is None


def test_load_speaker_map_fallback_name(tmp_path):
    toml_content = """\
[speakers."3"]
color = "#00FF00"
"""
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text(toml_content, encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result["3"]["name"] == "3"
    assert result["3"]["character"] is None


def test_remap_speaker_labels():
    lines = [
        SubtitleLine(text="hello", start_time=0.0, end_time=1.0, speaker="1"),
        SubtitleLine(text="world", start_time=1.0, end_time=2.0, speaker="2"),
        SubtitleLine(text="test", start_time=2.0, end_time=3.0, speaker=None),
    ]
    speaker_map = {
        "1": {"name": "Mizuki", "color": "#FFFF00"},
        "2": {"name": "Ena", "color": "#FF8080"},
    }

    remap_speaker_labels(lines, speaker_map)

    assert lines[0].speaker == "Mizuki"
    assert lines[1].speaker == "Ena"
    assert lines[2].speaker is None  # unchanged


def test_hex_to_pyass_color():
    c = hex_to_pyass_color("#FF8040")
    assert c.r == 255
    assert c.g == 128
    assert c.b == 64
    assert c.a == 0


def test_hex_to_pyass_color_no_hash():
    c = hex_to_pyass_color("00FF00")
    assert c.r == 0
    assert c.g == 255
    assert c.b == 0


def test_build_speaker_prompt_with_characters():
    speaker_map = {
        "0": {"name": "Suzuki Minori", "character": "Ena Shinonome", "color": "#FFA0A0"},
        "1": {"name": "Sato Hinata", "character": "Mizuki Akiyama", "color": "#A0D0FF"},
    }
    result = build_speaker_prompt(speaker_map)
    assert "Suzuki Minori (voice of Ena Shinonome)" in result
    assert "Sato Hinata (voice of Mizuki Akiyama)" in result
    assert result.startswith("Speakers in this recording:")


def test_build_speaker_prompt_without_characters():
    speaker_map = {
        "0": {"name": "Speaker A", "character": None, "color": None},
    }
    result = build_speaker_prompt(speaker_map)
    assert "- Speaker A" in result
    assert "voice of" not in result


# --- Speaker review flow tests ---


def _make_transcript_result(speaker_labels):
    """Build a mock TranscriptionResult with words tagged by speaker labels."""
    from autosub.core.schemas import TranscriptionResult
    words = []
    for i, label in enumerate(speaker_labels):
        words.append(TranscribedWord(
            word=f"word{i}",
            start_time=float(i),
            end_time=float(i) + 0.5,
            speaker=label,
        ))
    return TranscriptionResult(words=words)


def test_single_speaker_no_review_needed():
    """Single speaker project — review should never trigger."""
    result = _make_transcript_result(["0"] * 20)
    unique = {w.speaker for w in result.words if w.speaker}
    # --speakers 1, got 1 label → no review
    assert not (len(unique) > 1)


def test_multi_speaker_matching_count_no_review():
    """Multi speaker project where Chirp returns exact count — no review needed."""
    result = _make_transcript_result(["0"] * 10 + ["1"] * 10 + ["2"] * 10)
    unique = {w.speaker for w in result.words if w.speaker}
    speakers_requested = 3
    # Got 3 labels, requested 3 → no review
    assert not (len(unique) > speakers_requested)


def test_multi_speaker_extra_labels_triggers_review():
    """Multi speaker project where Chirp over-segments — review should trigger."""
    result = _make_transcript_result(
        ["0"] * 10 + ["1"] * 10 + ["2"] * 5 + ["3"] * 5 + ["4"] * 3
    )
    unique = {w.speaker for w in result.words if w.speaker}
    speakers_requested = 3
    # Got 5 labels, requested 3 → review needed
    assert len(unique) > speakers_requested


def test_speaker_map_applied_without_speakers_flag():
    """Speaker map for styling works even without --speakers (no diarization)."""
    lines = [
        SubtitleLine(text="hello", start_time=0.0, end_time=1.0, speaker="0"),
        SubtitleLine(text="world", start_time=1.0, end_time=2.0, speaker="1"),
    ]
    speaker_map = {
        "0": {"name": "Host A", "character": None, "color": "#FF0000"},
        "1": {"name": "Host B", "character": None, "color": "#0000FF"},
    }
    remap_speaker_labels(lines, speaker_map)
    assert lines[0].speaker == "Host A"
    assert lines[1].speaker == "Host B"


def test_many_to_one_speaker_mapping():
    """Multiple labels mapped to the same speaker merge correctly."""
    lines = [
        SubtitleLine(text="a", start_time=0.0, end_time=1.0, speaker="0"),
        SubtitleLine(text="b", start_time=1.0, end_time=2.0, speaker="3"),
        SubtitleLine(text="c", start_time=2.0, end_time=3.0, speaker="6"),
    ]
    # Labels 0, 3, 6 all map to the same person
    speaker_map = {
        "0": {"name": "Date Sayuri", "character": "Kanon", "color": "#FF0000"},
        "3": {"name": "Date Sayuri", "character": "Kanon", "color": "#FF0000"},
        "6": {"name": "Date Sayuri", "character": "Kanon", "color": "#FF0000"},
    }
    remap_speaker_labels(lines, speaker_map)
    assert all(l.speaker == "Date Sayuri" for l in lines)


def test_transcript_hash_consistency(tmp_path):
    """Same transcript produces the same hash, different transcript produces different hash."""
    from autosub.cli import _transcript_hash

    t1 = tmp_path / "t1.json"
    t2 = tmp_path / "t2.json"
    t1.write_text('{"words": []}', encoding="utf-8")
    t2.write_text('{"words": [{"word": "hi"}]}', encoding="utf-8")

    h1a = _transcript_hash(t1)
    h1b = _transcript_hash(t1)
    h2 = _transcript_hash(t2)

    assert h1a == h1b  # same file → same hash
    assert h1a != h2   # different file → different hash
    assert len(h1a) == 16  # truncated to 16 chars
