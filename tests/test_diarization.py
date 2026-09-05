from pathlib import Path

import pyass

from autosub.core.schemas import SubtitleLine, TranscribedWord
from autosub.core.speaker_map import (
    build_speaker_prompt,
    hex_to_pyass_color,
    load_speaker_map,
    remap_speaker_labels,
)
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
        for i in range(16)
    ]
    words.extend(
        [
            TranscribedWord(word="続き", start_time=1.6, end_time=1.65),
            TranscribedWord(word="です", start_time=1.7, end_time=1.75),
        ]
    )

    lines = chunk_words_to_lines(words)

    assert len(lines) == 2
    assert lines[0].text == "".join(word.word for word in words[:16])
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


def test_profile_prompt_inheritance(tmp_path):
    profile_dir = tmp_path / "profiles"
    prompt_dir = tmp_path / "prompts"
    prompt_examples_dir = prompt_dir / "examples"
    profile_dir.mkdir()
    prompt_examples_dir.mkdir(parents=True)

    (prompt_examples_dir / "base.md").write_text("base guidance", encoding="utf-8")
    (prompt_examples_dir / "child.md").write_text("child guidance", encoding="utf-8")

    (profile_dir / "base.toml").write_text(
        '[translate]\nprompt = "prompts/base.md"\n', encoding="utf-8"
    )
    (profile_dir / "child.toml").write_text(
        'extends = ["base"]\n[translate]\nprompt = "prompts/child.md"\n',
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts":
                return prompt_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["translate"]["prompt"] == ["base guidance", "child guidance"]
        assert data["prompt"] == ["base guidance", "child guidance"]
    finally:
        autosub.core.profile.Path = original_path


def test_legacy_flat_profile_keys_are_mapped_to_stages(tmp_path):
    profile_dir = tmp_path / "profiles"
    prompt_dir = tmp_path / "prompts"
    prompt_examples_dir = prompt_dir / "examples"
    profile_dir.mkdir()
    prompt_examples_dir.mkdir(parents=True)

    (prompt_examples_dir / "legacy.md").write_text("legacy guidance", encoding="utf-8")
    (profile_dir / "legacy.toml").write_text(
        """
prompt = "prompts/legacy.md"
vocab = ["鈴原希実"]

[timing]
min_duration_ms = 900

[extensions.radio_discourse]
enabled = true

[glossary]
"鈴原希実" = "Suzuhara Nozomi"

[replacements]
"鈴原のぞみ" = "鈴原希実"
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts":
                return prompt_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("legacy")
        assert data["transcribe"]["vocab"] == ["鈴原希実"]
        assert data["translate"]["prompt"] == ["legacy guidance"]
        assert data["translate"]["glossary"] == {"鈴原希実": "Suzuhara Nozomi"}
        assert data["format"]["min_duration_ms"] == 900
        assert data["format"]["replacements"] == {"鈴原のぞみ": "鈴原希実"}
        assert data["normalizer"] == {}
        assert data["format"]["extensions"]["radio_discourse"]["enabled"] is True
        assert data["postprocess"]["extensions"]["radio_discourse"]["enabled"] is True
    finally:
        autosub.core.profile.Path = original_path


def test_local_prompt_overrides_example_prompt(tmp_path):
    profiles_root = tmp_path / "profiles"
    profile_examples_dir = profiles_root / "examples"
    prompts_root = tmp_path / "prompts"
    prompt_local_dir = prompts_root / "local"
    prompt_examples_dir = prompts_root / "examples"
    profile_examples_dir.mkdir(parents=True)
    prompt_local_dir.mkdir(parents=True)
    prompt_examples_dir.mkdir(parents=True)

    (profile_examples_dir / "child.toml").write_text(
        '[translate]\nprompt = "prompts/child.md"\n', encoding="utf-8"
    )
    (prompt_examples_dir / "child.md").write_text("example guidance", encoding="utf-8")
    (prompt_local_dir / "child.md").write_text("local guidance", encoding="utf-8")

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profiles_root
            if args and args[0] == "prompts":
                return prompts_root
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["translate"]["prompt"] == ["local guidance"]
        assert data["prompt"] == ["local guidance"]
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
    assert result["0"]["name"] == "Suzuki Minori"
    assert result["0"]["character"] == "Ena Shinonome"
    assert result["0"]["color"] == "#FFA0A0"
    assert result["1"]["name"] == "Sato Hinata"
    assert result["1"]["character"] == "Mizuki Akiyama"
    assert result["1"]["color"] == "#A0D0FF"


def test_speaker_map_avatar_resolves_beside_the_map(tmp_path, monkeypatch):
    """An avatar next to the speaker map is found from any working directory.

    Speaker maps get read from wherever the pipeline happens to be running — a git
    worktree, or a remote box the project folder was copied onto — so an avatar path
    that only resolves from the repo root is a silently blank card waiting to happen.
    """
    project = tmp_path / "project"
    (project / "assets").mkdir(parents=True)
    avatar = project / "assets" / "host.png"
    avatar.write_bytes(b"not really a png")

    map_file = project / "speaker_map.toml"
    map_file.write_text(
        '[speakers."0"]\nname = "A Host"\navatar = "assets/host.png"\n',
        encoding="utf-8",
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = load_speaker_map(map_file)
    assert Path(result["0"]["avatar"]) == avatar


def test_speaker_map_avatar_prefers_the_working_directory(tmp_path, monkeypatch):
    """The existing repo-root-relative maps keep winning over the beside-map fallback."""
    project = tmp_path / "project"
    project.mkdir()
    map_file = project / "speaker_map.toml"
    map_file.write_text(
        '[speakers."0"]\nname = "A Host"\navatar = "assets/host.png"\n',
        encoding="utf-8",
    )
    # Same relative path exists both beside the map and under the working directory.
    for root in (project, tmp_path / "cwd"):
        (root / "assets").mkdir(parents=True, exist_ok=True)
        (root / "assets" / "host.png").write_bytes(b"not really a png")

    monkeypatch.chdir(tmp_path / "cwd")
    result = load_speaker_map(map_file)
    assert Path(result["0"]["avatar"]) == Path("assets/host.png")


def test_speaker_map_avatar_kept_verbatim_when_missing(tmp_path):
    """An unresolvable avatar is reported as written, so the warning names the real path."""
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text(
        '[speakers."0"]\nname = "A Host"\navatar = "assets/nope.png"\n',
        encoding="utf-8",
    )
    result = load_speaker_map(map_file)
    assert result["0"]["avatar"] == "assets/nope.png"


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
        "0": {
            "name": "Suzuki Minori",
            "character": "Ena Shinonome",
            "color": "#FFA0A0",
        },
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
        words.append(
            TranscribedWord(
                word=f"word{i}",
                start_time=float(i),
                end_time=float(i) + 0.5,
                speaker=label,
            )
        )
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
    assert all(line.speaker == "Date Sayuri" for line in lines)


def test_generator_slot_ass_positions(tmp_path):
    """Slot position must live in the style, so retagging in Aegisub moves the line.

    A \\pos in the event text is derived from the cue's speaker, not from the
    Style the event actually references, so the two drift apart the moment a
    human corrects a speaker.
    """
    from autosub.pipeline.format.generator import generate_ass_file

    lines = [
        SubtitleLine(text="Kanon talking", start_time=0.0, end_time=2.0, speaker="0"),
        SubtitleLine(text="Keke talking", start_time=1.0, end_time=3.0, speaker="1"),
    ]
    speaker_map = {
        "0": {
            "name": "Date Sayuri",
            "character": "Kanon",
            "color": "#FF9E00",
            "slot": 1,
        },
        "1": {"name": "Liyuu", "character": "Keke", "color": "#00A3E0", "slot": 2},
    }
    ass_path = tmp_path / "test.ass"
    generate_ass_file(lines, ass_path, speaker_map=speaker_map)

    with open(ass_path, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    events = script.events
    assert len(events) == 2
    assert events[0].style == "Date Sayuri"
    assert events[1].style == "Liyuu"
    assert all(r"\pos(" not in event.text for event in events)

    styles = {style.name: style for style in script.styles}
    assert styles["Date Sayuri"].alignment == pyass.Alignment.TOP_LEFT
    # Slot 1 sits above slot 2, and both clear the avatar card column.
    assert styles["Date Sayuri"].marginV < styles["Liyuu"].marginV
    assert styles["Date Sayuri"].marginL == styles["Liyuu"].marginL > 0


def test_per_speaker_timing_rules_overlapping():
    """Verify apply_timing_rules retains overlapping timestamps when speakers differ."""
    from autosub.pipeline.format.timing import apply_timing_rules

    lines = [
        SubtitleLine(text="Speaker A line", start_time=1.0, end_time=4.0, speaker="A"),
        SubtitleLine(text="Speaker B line", start_time=2.0, end_time=5.0, speaker="B"),
    ]
    processed = apply_timing_rules(lines)

    assert len(processed) == 2
    # Ensure timing was not collapsed between A and B
    a_line = next(ln for ln in processed if ln.speaker == "A")
    b_line = next(ln for ln in processed if ln.speaker == "B")
    assert a_line.start_time == 1.0
    assert b_line.start_time == 2.0
    assert b_line.start_time < a_line.end_time  # Overlapping dialogue preserved!


def test_chirp2_diarization_rejected():
    """chirp_2 + --speakers must fail fast with a clear error (API rejects it)."""
    import pytest
    from pathlib import Path
    from autosub.pipeline.transcribe.main import transcribe

    with pytest.raises(ValueError, match="not supported by the chirp_2"):
        transcribe(
            Path("nonexistent.mkv"),
            Path("out.json"),
            num_speakers=2,
            transcription_backend="chirp_2",
        )


def test_labels_sharing_a_slot_are_not_allowed_to_overlap():
    """Two diarization labels for one person share a box and must not stack."""
    from autosub.pipeline.format.timing import apply_timing_rules

    # Labels "0" and "4" are the same person, so both render in slot 1.
    speaker_map = {
        "0": {"name": "Sakakura Sakura", "slot": 1},
        "4": {"name": "Sakakura Sakura", "slot": 1},
        "2": {"name": "Ookuma Wakana", "slot": 2},
    }
    lines = [
        SubtitleLine(text="first", start_time=1.0, end_time=9.0, speaker="0"),
        SubtitleLine(text="second", start_time=4.0, end_time=6.0, speaker="4"),
    ]
    processed = apply_timing_rules(lines, speaker_map=speaker_map)

    first = next(ln for ln in processed if ln.text == "first")
    second = next(ln for ln in processed if ln.text == "second")
    assert first.end_time <= second.start_time


def test_different_slots_still_allow_concurrent_dialogue():
    """De-overlapping is per slot, so separate boxes keep talking over each other."""
    from autosub.pipeline.format.timing import apply_timing_rules

    speaker_map = {
        "0": {"name": "Sakakura Sakura", "slot": 1},
        "2": {"name": "Ookuma Wakana", "slot": 2},
    }
    lines = [
        SubtitleLine(text="slot one", start_time=1.0, end_time=9.0, speaker="0"),
        SubtitleLine(text="slot two", start_time=4.0, end_time=6.0, speaker="2"),
    ]
    processed = apply_timing_rules(lines, speaker_map=speaker_map)

    one = next(ln for ln in processed if ln.text == "slot one")
    two = next(ln for ln in processed if ln.text == "slot two")
    assert two.start_time < one.end_time


def test_simultaneous_lines_in_one_slot_merge_instead_of_stacking():
    """With no room to truncate into, both texts survive in a single event."""
    from autosub.pipeline.format.timing import apply_timing_rules

    speaker_map = {
        "0": {"name": "Sakakura Sakura", "slot": 1},
        "4": {"name": "Sakakura Sakura", "slot": 1},
    }
    lines = [
        SubtitleLine(text="alpha", start_time=5.00, end_time=5.80, speaker="0"),
        SubtitleLine(text="bravo", start_time=5.02, end_time=5.90, speaker="4"),
    ]
    processed = apply_timing_rules(lines, speaker_map=speaker_map)

    assert len(processed) == 1
    assert "alpha" in processed[0].text
    assert "bravo" in processed[0].text
