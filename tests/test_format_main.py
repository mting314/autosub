import json

import pyass
import autosub.extensions.radio_discourse.main as radio_discourse_main

from autosub.pipeline.format.main import format_subtitles


def test_format_subtitles_does_not_insert_ass_line_breaks(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [
        {
            "word": "語",
            "start_time": index * 0.1,
            "end_time": index * 0.1 + 0.05,
        }
        for index in range(20)
    ]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    format_subtitles(transcript_path, output_path)

    ass_text = output_path.read_text(encoding="utf-8")
    assert r"\N" not in ass_text


def test_format_subtitles_applies_radio_discourse_extension_and_preserves_role(
    tmp_path,
):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [
        {"word": "おすすめ", "start_time": 0.0, "end_time": 0.6},
        {"word": "です", "start_time": 0.6, "end_time": 1.0},
        {"word": "といただきました。", "start_time": 1.0, "end_time": 2.0},
    ]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    format_subtitles(
        transcript_path,
        output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
    )

    with open(output_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    dialogue_events = [
        event for event in script.events if isinstance(event, pyass.Event)
    ]
    assert len(dialogue_events) == 2
    assert dialogue_events[0].text == "おすすめです。"
    assert dialogue_events[0].name == "listener_mail"
    assert dialogue_events[1].text == "といただきました。"
    assert dialogue_events[1].name == "host_meta"


def test_format_subtitles_sets_radio_discourse_trace_path(tmp_path, monkeypatch):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [{"word": "こんにちは", "start_time": 0.0, "end_time": 1.0}]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_apply(lines, config):
        captured["llm_trace_path"] = config.get("llm_trace_path")
        return lines

    monkeypatch.setattr(radio_discourse_main, "apply_radio_discourse", fake_apply)

    format_subtitles(
        transcript_path,
        output_path,
        extensions_config={"radio_discourse": {"enabled": True, "engine": "hybrid"}},
    )

    assert captured["llm_trace_path"] == output_path.with_suffix(".llm_trace.jsonl")
