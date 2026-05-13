from autosub.core.project_config import (
    build_title_prompt,
    load_project_config,
    merge_glossary,
    merge_vocab,
    resolve_cast,
)


def test_load_project_config_returns_none_when_file_missing(tmp_path):
    audio = tmp_path / "audio.mkv"
    audio.touch()
    assert load_project_config(audio) is None


def test_load_project_config_returns_none_when_reference_is_none():
    assert load_project_config(None) is None


def test_load_project_config_parses_toml_in_same_directory(tmp_path):
    audio = tmp_path / "audio.mkv"
    audio.touch()
    (tmp_path / "project.toml").write_text(
        """
title = "My Event"

vocab = ["foo", "bar"]

[glossary]
"アクエラ" = "aqu3ra"
""".strip(),
        encoding="utf-8",
    )

    data = load_project_config(audio)

    assert data["title"] == "My Event"
    assert data["vocab"] == ["foo", "bar"]
    assert data["glossary"] == {"アクエラ": "aqu3ra"}


def test_load_project_config_returns_none_on_invalid_toml(tmp_path):
    audio = tmp_path / "audio.mkv"
    audio.touch()
    (tmp_path / "project.toml").write_text("not = valid = toml", encoding="utf-8")
    assert load_project_config(audio) is None


def test_merge_glossary_event_overrides_profile():
    profile = {"特訓前": "untrained", "アクエラ": "akuera"}
    event = {"glossary": {"アクエラ": "aqu3ra", "新曲": "new song"}}
    merged = merge_glossary(profile, event)
    assert merged == {
        "特訓前": "untrained",
        "アクエラ": "aqu3ra",
        "新曲": "new song",
    }


def test_merge_glossary_handles_none_event():
    profile = {"特訓前": "untrained"}
    assert merge_glossary(profile, None) == {"特訓前": "untrained"}


def test_merge_glossary_handles_missing_glossary_section():
    assert merge_glossary({"a": "b"}, {"title": "T"}) == {"a": "b"}


def test_merge_glossary_does_not_mutate_input():
    profile = {"a": "b"}
    merge_glossary(profile, {"glossary": {"c": "d"}})
    assert profile == {"a": "b"}


def test_merge_vocab_appends_and_dedupes():
    profile_vocab = ["foo", "bar"]
    event = {"vocab": ["bar", "baz"]}
    assert merge_vocab(profile_vocab, event) == ["foo", "bar", "baz"]


def test_merge_vocab_handles_none_event():
    assert merge_vocab(["x"], None) == ["x"]


def test_merge_vocab_handles_empty_profile():
    assert merge_vocab([], {"vocab": ["y"]}) == ["y"]


def test_merge_vocab_skips_non_string_terms():
    assert merge_vocab(["x"], {"vocab": [1, "y", None]}) == ["x", "y"]


def test_build_title_prompt_includes_canonical_title():
    prompt = build_title_prompt({"title": "Unsteady, still steady step"})
    assert prompt is not None
    assert '"Unsteady, still steady step"' in prompt
    assert "transcription" in prompt.lower()


def test_build_title_prompt_returns_none_when_event_meta_missing():
    assert build_title_prompt(None) is None


def test_build_title_prompt_returns_none_when_title_missing():
    assert build_title_prompt({"glossary": {"a": "b"}}) is None


def test_build_title_prompt_returns_none_when_title_blank():
    assert build_title_prompt({"title": "   "}) is None


def test_build_title_prompt_returns_none_when_title_not_string():
    assert build_title_prompt({"title": 123}) is None


def test_resolve_cast_uses_profile_when_project_missing():
    profile_cast = [
        {"name": "Yuki", "character": "Shiho", "color": "#22DDBB"},
    ]
    cast = resolve_cast(profile_cast, None)
    assert cast == [
        {"name": "Yuki", "character": "Shiho", "color": "#22DDBB"},
    ]


def test_resolve_cast_uses_profile_when_project_has_no_cast():
    profile_cast = [{"name": "A"}]
    cast = resolve_cast(profile_cast, {"title": "T"})
    assert cast == [{"name": "A", "character": None, "color": None}]


def test_resolve_cast_project_overrides_profile():
    profile_cast = [{"name": "Default Host", "character": "X"}]
    project = {
        "speakers": {
            "cast": [
                {"name": "Guest", "character": "Y", "color": "#FF0000"},
            ]
        }
    }
    cast = resolve_cast(profile_cast, project)
    assert cast == [
        {"name": "Guest", "character": "Y", "color": "#FF0000"},
    ]


def test_resolve_cast_returns_empty_when_neither_present():
    assert resolve_cast(None, None) == []
    assert resolve_cast([], {"title": "T"}) == []


def test_resolve_cast_normalizes_partial_entries():
    cast = resolve_cast([{"name": "A"}], None)
    assert cast == [{"name": "A", "character": None, "color": None}]


def test_load_project_config_loads_speakers_cast(tmp_path):
    audio = tmp_path / "audio.mkv"
    audio.touch()
    (tmp_path / "project.toml").write_text(
        """
title = "T"

[[speakers.cast]]
name = "Guest"
character = "G"
color = "#000000"
""".strip(),
        encoding="utf-8",
    )
    data = load_project_config(audio)
    assert data["speakers"]["cast"] == [
        {"name": "Guest", "character": "G", "color": "#000000"},
    ]
