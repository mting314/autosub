from pathlib import Path

from autosub.tools.font_collector import (
    build_fonts_section,
    embed_fonts,
    fonts_required,
    uuencode,
)

HEADER = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: A,Lato ExtraBold,70,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: B,Noto Serif JP,70,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,A,,0,0,0,,plain line
Dialogue: 0,0:00:01.00,0:00:02.00,A,,0,0,0,,{\\fnSome Other Face}inline override
"""


def test_fonts_required_reads_styles_and_inline_overrides():
    assert fonts_required(HEADER) == {
        "Lato ExtraBold",
        "Noto Serif JP",
        "Some Other Face",
    }


def test_vertical_writing_prefix_is_stripped():
    assert fonts_required("Style: A,@MS Gothic,40,") == {"MS Gothic"}


def test_uuencode_matches_the_ass_attachment_scheme():
    # 3 bytes -> 4 chars, each 6-bit group offset by 33.
    assert uuencode(b"\x00\x00\x00") == ["!!!!"]
    # A trailing partial group emits one char per byte, plus one.
    assert len(uuencode(b"\x00")[0]) == 2
    assert len(uuencode(b"\x00\x00")[0]) == 3
    # Lines wrap at 80 characters.
    assert all(len(line) <= 80 for line in uuencode(b"\x01" * 500))


def test_build_fonts_section_names_attachments_like_aegisub(tmp_path):
    font = tmp_path / "MyFont.ttf"
    font.write_bytes(b"not a real font")
    section = build_fonts_section([font])
    assert section.startswith("[Fonts]\n")
    assert "fontname: MyFont_0.ttf" in section


def test_embed_reports_fonts_it_cannot_find(tmp_path):
    ass = tmp_path / "s.ass"
    ass.write_text(
        "[V4+ Styles]\nStyle: A,No Such Font Family Here,40,\n\n[Events]\n",
        encoding="utf-8",
    )
    embedded, missing = embed_fonts(ass, tmp_path / "out.ass")
    assert embedded == []
    assert missing == ["No Such Font Family Here"]


def test_embedding_is_idempotent(tmp_path):
    font_src = Path(
        "/Users/michaelting/github/autosub/projects/projects/Project Sekai/"
        "This story continues with hope Aftertalk/NotoSerifJP-VF.ttf"
    )
    if not font_src.exists():  # asset-dependent; skip where unavailable
        return
    ass = tmp_path / "s.ass"
    ass.write_text(
        "[V4+ Styles]\nStyle: A,Noto Serif JP,40,\n\n[Events]\n", encoding="utf-8"
    )
    (tmp_path / font_src.name).write_bytes(font_src.read_bytes())

    embed_fonts(ass)
    once = ass.read_text(encoding="utf-8")
    embed_fonts(ass)
    twice = ass.read_text(encoding="utf-8")
    assert once == twice
    assert once.count("[Fonts]") == 1


def test_strip_survives_font_data_that_looks_like_a_section_header():
    """The attachment alphabet includes '[', so data lines can start with one."""
    from autosub.tools.font_collector import strip_fonts_section

    text = (
        "[Script Info]\nTitle: t\n\n"
        "[Fonts]\nfontname: F_0.ttf\n"
        "[_):'O-$@RUH\"3)C`5^4KW-?7;F35\n"   # real-looking encoded line
        "[!!,!#]!/Q\"+!!!8)C9V.$9T-B96\n"
        "\n[Events]\nFormat: Layer\n"
    )
    out = strip_fonts_section(text)
    assert "[Fonts]" not in out
    assert "fontname:" not in out
    assert "[_):" not in out          # the data went with it
    assert "[Script Info]" in out and "[Events]" in out
