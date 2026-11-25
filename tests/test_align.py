from pathlib import Path

from autosub.core.schemas import (
    SubtitleCue,
    SubtitleDocument,
    SubtitleMetadata,
    TranscriptionSegment,
)
from autosub.pipeline.transcribe.align import (
    align_segments_to_vtt,
    align_subtitle_document,
    ensure_companion_vtt,
    find_companion_vtt,
    parse_vtt_words,
    resolve_youtube_url,
    to_phonetic_hiragana,
)

_SAMPLE_VTT = """WEBVTT
Kind: captions
Language: ja

00:00:05.000 --> 00:00:09.500 align:start position:0%
こんばんは<00:00:05.800><c>ビビット</c><00:00:06.400><c>バッド</c><00:00:07.100><c>白石杏役の</c><00:00:08.400><c>須藤友美ジェナです</c>

00:00:14.200 --> 00:00:19.000 align:start position:0%
はい<00:00:14.500><c>スマートフォン</c><00:00:15.300><c>ゲーム</c><00:00:15.800><c>プロジェクトセカイ</c>
"""


def test_to_phonetic_hiragana_normalizes_katakana_and_punct():
    assert to_phonetic_hiragana("「コンビバ！」") == "こんびば"
    assert to_phonetic_hiragana("プロジェクト・セカイ") == "ぷろじぇくとせかい"


def test_parse_vtt_words_extracts_inline_timestamps(tmp_path: Path):
    vtt = tmp_path / "ep.auto-captions.ja.vtt"
    vtt.write_text(_SAMPLE_VTT, encoding="utf-8")
    words = parse_vtt_words(vtt)
    assert len(words) == 9
    assert words[0].word == "こんばんは"
    assert words[0].start_time == 5.0
    assert words[5].word == "はい"
    assert words[5].start_time == 14.2


def test_align_segments_to_vtt_corrects_drifted_timestamps_and_populates_words(
    tmp_path: Path,
):
    vtt = tmp_path / "test.auto-captions.ja.vtt"
    vtt.write_text(_SAMPLE_VTT, encoding="utf-8")

    # Simulate drifted Gemini timestamps (+1.2s and +2.5s late) with words=[]
    segs = [
        TranscriptionSegment(
            text="こんばんは！Vivid BAD 白石杏役の鷲見友美ジェナです！",
            start_time=6.20,
            end_time=10.80,
            speaker="Tomomi Jiena Sumi",
            words=[],
        ),
        TranscriptionSegment(
            text="はい、スマートフォンゲーム『プロジェクトセカイ』",
            start_time=16.70,
            end_time=21.00,
            speaker="Tomomi Jiena Sumi",
            words=[],
        ),
    ]
    aligned = align_segments_to_vtt(segs, vtt, chunk_boundaries=[(0.0, 60.0)])
    assert aligned[0].start_time == 5.0
    assert len(aligned[0].words) >= 3
    assert aligned[1].start_time == 14.2
    assert len(aligned[1].words) == 4
    assert aligned[1].words[0].speaker == "Tomomi Jiena Sumi"


def test_align_subtitle_document_preserves_ass_edits_and_aligns(tmp_path: Path):
    vtt = tmp_path / "ep.auto-captions.ja.vtt"
    vtt.write_text(_SAMPLE_VTT, encoding="utf-8")

    doc = SubtitleDocument(
        stage="postprocessed",
        metadata=SubtitleMetadata(),
        cues=[
            SubtitleCue(
                id="cue-00000001",
                start_time=6.50,
                end_time=11.00,
                source_text="こんばんは！白石杏役の鷲見友美ジェナです！",
                translated_text="Old translation 1",
                final_text="Old translation 1",
                speaker="Tomomi Jiena Sumi",
                role="host",
            ),
            SubtitleCue(
                id="cue-00000002",
                start_time=16.80,
                end_time=21.20,
                source_text="はい、スマートフォンゲーム『プロジェクトセカイ』",
                translated_text="Old translation 2",
                final_text="Old translation 2",
                speaker="Tomomi Jiena Sumi",
                role="host",
            ),
        ],
    )
    json_path = tmp_path / "ep_postprocessed.json"
    json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    ass_path = tmp_path / "ep_final.ass"
    ass_path.write_text(
        "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:06.50,0:00:11.00,Default,host,0,0,0,,QC edited line 1\n"
        "Dialogue: 0,0:00:16.80,0:00:21.20,Default,host,0,0,0,,QC edited line 2\n",
        encoding="utf-8",
    )

    assert find_companion_vtt(json_path) == vtt
    updated = align_subtitle_document(json_path, ass_path=ass_path)
    assert updated.cues[0].final_text == "QC edited line 1"
    assert updated.cues[1].final_text == "QC edited line 2"
    assert updated.cues[0].start_time == 5.0
    assert updated.cues[1].start_time == 14.2
    assert len(updated.cues[1].words) == 4


def test_resolve_youtube_url_from_id_or_notes_and_auto_fetch(
    tmp_path: Path, monkeypatch
):
    # 1. Bare 11-char ID
    assert (
        resolve_youtube_url(explicit_url="MTAKEOzS5Ac")
        == "https://www.youtube.com/watch?v=MTAKEOzS5Ac"
    )
    # 2. Auto-discovered from notes.md in the project folder
    (tmp_path / "notes.md").write_text(
        "# Sekaraji Ep 78\n- **Source**: <https://www.youtube.com/watch?v=MTAKEOzS5Ac>\n",
        encoding="utf-8",
    )
    mkv = tmp_path / "Sekaraji Ep 78.mkv"
    mkv.write_bytes(b"video")
    assert (
        resolve_youtube_url(mkv) == "https://www.youtube.com/watch?v=MTAKEOzS5Ac"
    )

    # 3. ensure_companion_vtt invokes yt-dlp when .vtt is not on disk yet
    calls = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / "Sekaraji Ep 78.auto-captions.ja.vtt").write_text(
            _SAMPLE_VTT, encoding="utf-8"
        )
        return _Proc()

    monkeypatch.setattr("autosub.pipeline.transcribe.align.shutil.which", lambda _: "/usr/local/bin/yt-dlp")
    monkeypatch.setattr("autosub.pipeline.transcribe.align.subprocess.run", fake_run)

    fetched = ensure_companion_vtt(mkv)
    assert fetched == tmp_path / "Sekaraji Ep 78.auto-captions.ja.vtt"
    assert len(calls) == 1
    assert "--write-auto-sub" in calls[0]
    assert "https://www.youtube.com/watch?v=MTAKEOzS5Ac" in calls[0]



def test_partial_match_in_later_window_does_not_truncate_end():
    """Regression: the coverage gate must compare cue positions, not global VTT indices."""
    from autosub.core.schemas import TranscribedWord as W
    from autosub.pipeline.transcribe.align import _align_items_in_windows

    vtt = [W(word="あ", start_time=i * 0.25, end_time=i * 0.25 + 0.1) for i in range(200)]
    base = 62.0
    for k, ch in enumerate("かきくけこ"):
        vtt.append(W(word=ch, start_time=base + k * 0.1, end_time=base + k * 0.1 + 0.1))
    for k, ch in enumerate("らりるれろ" * 6):
        t = base + 3 + k * 0.1
        vtt.append(W(word=ch, start_time=t, end_time=t + 0.1))
    text = "かきくけこ" + "さしすせそ" * 6
    _, ends, _ = _align_items_in_windows(
        [text], [62.0], [64.5], [None], vtt, [(0, 60), (60, 120)]
    )
    assert ends[0] >= 64.0


def test_find_companion_vtt_ignores_other_episodes(tmp_path):
    from autosub.pipeline.transcribe.align import find_companion_vtt

    (tmp_path / "Show Ep 75.auto-captions.ja.vtt").write_text("WEBVTT\n")
    assert find_companion_vtt(tmp_path / "Show Ep 76.mkv") is None
    own = tmp_path / "Show Ep 76.auto-captions.ja.vtt"
    own.write_text("WEBVTT\n")
    assert find_companion_vtt(tmp_path / "Show Ep 76.mkv") == own
