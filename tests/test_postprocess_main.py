import re

import pytest

from autosub.core.schemas import SubtitleCue, SubtitleDocument, TranscribedWord
from autosub.pipeline.postprocess.main import _ensure_quoted, postprocess_subtitles


def _write_translated_document(path, cues: list[SubtitleCue]) -> None:
    document = SubtitleDocument(stage="translated", cues=cues)
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def test_postprocess_quotes_listener_mail_replace_mode(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールです。",
                translated_text="This is a listener message.",
                role="listener_mail",
            ),
            SubtitleCue(
                id="cue-00002",
                start_time=1,
                end_time=2,
                source_text="ありがとう。",
                translated_text="Thanks for writing in.",
                role="host",
            ),
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is a listener message."'
    assert document.cues[1].final_text == "Thanks for writing in."


def test_postprocess_quotes_only_translated_line_in_bilingual_mode(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールを送るのは初めてです。",
                translated_text="This is my first message.",
                role="listener_mail",
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=True,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is my first message."'


def test_postprocess_preserves_source_words(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールです。",
                translated_text="This is a listener message.",
                role="listener_mail",
                words=[
                    TranscribedWord(word="メール", start_time=0.0, end_time=0.5),
                    TranscribedWord(word="です。", start_time=0.5, end_time=1.0),
                ],
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.stage == "postprocessed"
    assert document.cues[0].final_text == '"This is a listener message."'
    assert [word.word for word in document.cues[0].words] == ["メール", "です。"]


def test_postprocess_requires_translated_document(tmp_path):
    input_path = tmp_path / "formatted.json"
    output_path = tmp_path / "postprocessed.json"
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="source",
            )
        ],
    )
    input_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(
        ValueError, match="postprocess expects stage='translated', got 'formatted'"
    ):
        postprocess_subtitles(input_path, output_json_path=output_path)


def test_postprocess_preserves_and_renders_chunk_boundaries(tmp_path):
    input_path = tmp_path / "translated.json"
    output_json_path = tmp_path / "postprocessed.json"
    output_ass_path = tmp_path / "postprocessed.ass"
    document = SubtitleDocument(
        stage="translated",
        chunk_boundaries=[1],
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="first",
                translated_text="First.",
            ),
            SubtitleCue(
                id="cue-00002",
                start_time=1,
                end_time=2,
                source_text="second",
                translated_text="Second.",
            ),
        ],
    )
    input_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    postprocess_subtitles(
        input_path,
        output_json_path=output_json_path,
        output_ass_path=output_ass_path,
        bilingual=False,
    )

    processed = SubtitleDocument.model_validate_json(
        output_json_path.read_text(encoding="utf-8")
    )
    assert processed.chunk_boundaries == [1]
    assert (
        "[autosub] Chunk boundary - review translation around this line"
        in output_ass_path.read_text(encoding="utf-8")
    )


def test_postprocess_collapses_double_outer_quotes_in_replace_mode(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールです。",
                translated_text='""This is a listener message.""',
                role="listener_mail",
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is a listener message."'


def test_postprocess_collapses_double_outer_quotes_on_bilingual_translation(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールを送るのは初めてです。",
                translated_text='""This is my first message.""',
                role="listener_mail",
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=True,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is my first message."'


def _slot_speaker_map() -> dict[str, dict]:
    return {
        "0": {
            "name": "Date Sayuri",
            "character": "Shibuya Kanon",
            "color": "#FF9E00",
            "slot": 1,
        },
        "1": {
            "name": "Liyuu",
            "character": "Tang Keke",
            "color": "#00A3E0",
            "slot": 2,
        },
    }


def _two_speaker_document(path) -> None:
    _write_translated_document(
        path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=2,
                source_text="こんばんは。",
                translated_text="Good evening.",
                speaker="Date Sayuri",
            ),
            SubtitleCue(
                id="cue-00002",
                start_time=1,
                end_time=3,
                source_text="よろしくね。",
                translated_text="Nice to meet you.",
                speaker="Liyuu",
            ),
        ],
    )


def test_postprocess_renders_slot_styles_when_speaker_map_supplied(tmp_path):
    """The final .ass must carry the overlay layout, not just original.ass.

    speaker_map used to stop at the format stage, so the file that actually gets
    hardsubbed fell back to bottom-centred Arial and every slot piled into one box.
    """
    import pyass

    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    ass_path = tmp_path / "final.ass"
    _two_speaker_document(input_path)

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        output_ass_path=ass_path,
        bilingual=False,
        speaker_map=_slot_speaker_map(),
    )

    with open(ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    styles = {style.name: style for style in script.styles}
    assert set(styles) == {"Date Sayuri", "Liyuu"}

    # Slot geometry: text clears the avatar card column, and the character colour
    # sits in the outline with a white fill.
    for style in styles.values():
        assert style.alignment == pyass.Alignment.CENTER_LEFT
        assert style.marginL == 330
        assert style.primaryColor == pyass.Color(r=255, g=255, b=255, a=0)
    assert styles["Date Sayuri"].outlineColor == pyass.Color(r=255, g=158, b=0, a=0)
    assert styles["Liyuu"].outlineColor == pyass.Color(r=0, g=163, b=224, a=0)

    # Each slot is positioned, and the two slots land at different heights.
    positions = [
        re.search(r"\\pos\((\d+),(\d+)\)", event.text) for event in script.events
    ]
    assert all(match is not None for match in positions)
    assert positions[0].group(2) != positions[1].group(2)


def test_postprocess_without_speaker_map_keeps_flat_layout(tmp_path):
    """No speaker map means no overlay - the plain bottom-centred styling."""
    import pyass

    input_path = tmp_path / "translated.json"
    ass_path = tmp_path / "final.ass"
    _two_speaker_document(input_path)

    postprocess_subtitles(
        input_path,
        output_json_path=tmp_path / "postprocessed.json",
        output_ass_path=ass_path,
        bilingual=False,
    )

    with open(ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    assert all(
        style.alignment == pyass.Alignment.BOTTOM for style in script.styles
    )
    assert all(r"\pos(" not in event.text for event in script.events)


def test_ensure_quoted_collapses_duplicate_quotes_on_first_and_last_visual_lines():
    text = r'""aaaaaa"\N"aaaaa"\N"aaaa""'

    assert _ensure_quoted(text) == r'"aaaaaa"\N"aaaaa"\N"aaaa"'


def test_ensure_quoted_collapses_duplicate_quotes_on_first_and_last_newline_lines():
    text = '""aaaaaa"\n"aaaaa"\n"aaaa""'

    assert _ensure_quoted(text) == '"aaaaaa"\n"aaaaa"\n"aaaa"'


def test_ensure_quoted_preserves_duplicate_quotes_at_interior_visual_line_edges():
    text = r'"aaaa"\N""bbbb""\N"cccc"'

    assert _ensure_quoted(text) == r'"aaaa"\N""bbbb""\N"cccc"'
