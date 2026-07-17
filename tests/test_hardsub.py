import pyass
import pytest

from autosub.core.utils import parse_timestamp
from autosub.pipeline.hardsub import main as hs

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,6,6,2,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _write_ass(path, events):
    lines = _ASS_HEADER + "".join(
        f"Dialogue: 0,{s},{e},Default,,0,0,0,,{t}\n" for s, e, t in events
    )
    path.write_text(lines, encoding="utf-8")


class _Result:
    def __init__(self, returncode=0, stderr=b""):
        self.returncode = returncode
        self.stderr = stderr


# --- command construction ---


def test_burn_whole_cmd_has_no_seek():
    cmd = hs._burn_whole_cmd("ffmpeg", "/v.mkv", "/tmp/subs.ass", "/out.mp4", 18, "medium")
    assert "ass=/tmp/subs.ass" in cmd
    assert "-ss" not in cmd and "-t" not in cmd
    assert "libx264" in cmd and "18" in cmd


def test_seg_burn_cmd_uses_input_seek():
    cmd = hs._seg_burn_cmd("ffmpeg", "/v.mkv", "/tmp/seg.ass", "/o.mp4", "00:44:00", 25.0, 20, "fast")
    # -ss must come BEFORE -i (input seek), and -t bounds the duration
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "00:44:00"
    assert cmd[cmd.index("-t") + 1] == "25.000"
    assert cmd[cmd.index("-crf") + 1] == "20"


def test_resolve_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        hs._resolve_ffmpeg()


# --- subtitle shifting ---


def test_shift_ass_shifts_drops_and_clamps(tmp_path):
    src = tmp_path / "s.ass"
    _write_ass(src, [
        ("0:43:00.00", "0:43:05.00", "before"),   # entirely before the segment -> dropped
        ("0:43:58.00", "0:44:03.00", "straddle"),  # straddles start -> clamped to 0
        ("0:44:07.00", "0:44:11.00", "inside"),    # inside -> shifted
    ])
    dst = tmp_path / "shifted.ass"
    hs._shift_ass(src, parse_timestamp("0:44:00"), dst)

    with open(dst, encoding="utf-8") as f:
        events = pyass.load(f).events
    spans = {e.text: (e.start.total_seconds(), e.end.total_seconds()) for e in events}
    assert "before" not in spans          # dropped
    assert spans["straddle"] == (0.0, 3.0)  # start clamped to 0
    assert spans["inside"] == (7.0, 11.0)   # shifted by 44:00


# --- input validation ---


def test_missing_video_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    ass = tmp_path / "s.ass"
    ass.write_text("x")
    with pytest.raises(FileNotFoundError, match="video"):
        hs.hardsub_video(tmp_path / "nope.mkv", ass, tmp_path / "o.mp4")


def test_missing_ass_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="subtitle"):
        hs.hardsub_video(video, tmp_path / "nope.ass", tmp_path / "o.mp4")


# --- single / whole-video burn ---


def test_single_segment_input_seek_and_shifted_subs(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    ass = tmp_path / "my subs.ass"  # spaces in name
    _write_ass(ass, [("0:44:07.00", "0:44:11.00", "hi")])
    calls = []
    monkeypatch.setattr(hs.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _Result())

    hs.hardsub_video(video, ass, tmp_path / "o.mp4", segments=[("0:44:00", "0:44:25")], detect_black=False)

    burn = next(c for c in calls if "-vf" in c)
    assert burn.index("-ss") < burn.index("-i")  # input seek
    vf = burn[burn.index("-vf") + 1]
    assert vf.startswith("ass=") and " " not in vf and vf.endswith(".ass")  # shifted subs, clean path


def test_whole_video_no_seek(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    ass = tmp_path / "s.ass"
    _write_ass(ass, [("0:00:01.00", "0:00:02.00", "hi")])
    calls = []
    monkeypatch.setattr(hs.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _Result())

    hs.hardsub_video(video, ass, tmp_path / "o.mp4", segments=[], detect_black=False)

    burn = next(c for c in calls if "-vf" in c)
    assert "-ss" not in burn and "-t" not in burn


def test_burn_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    ass = tmp_path / "s.ass"
    _write_ass(ass, [("0:00:01.00", "0:00:02.00", "hi")])
    monkeypatch.setattr(hs.subprocess, "run", lambda cmd, **k: _Result(returncode=1, stderr=b"boom"))
    with pytest.raises(RuntimeError, match="hardsub burn failed"):
        hs.hardsub_video(video, ass, tmp_path / "o.mp4", detect_black=False)


# --- ass= filter escaping (Windows-safe) ---


def test_ass_filter_arg_escapes_windows_path():
    assert hs._ass_filter_arg("/tmp/x/subs.ass") == "ass=/tmp/x/subs.ass"  # posix: no-op
    assert hs._ass_filter_arg("C:\\Users\\x\\subs.ass") == "ass=C\\:/Users/x/subs.ass"


# --- join offsets ---


def test_join_offsets_are_cumulative_part_durations():
    offs = hs._join_offsets([("0:00:00", "0:00:10"), ("0:01:00", "0:01:05"), ("0:02:00", "0:02:20")])
    assert offs == [10.0, 15.0]  # parts are 10s, 5s, 20s -> joins at 10, 15


# --- multi-segment bounded-parallel + concat ---


def _run_recorder():
    import threading

    lock = threading.Lock()
    cmds: list[list[str]] = []

    def fake_run(cmd, **k):
        with lock:
            cmds.append(cmd)
        return _Result()

    return cmds, fake_run


def test_multi_segment_parallel_then_concat(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    ass = tmp_path / "s.ass"
    _write_ass(ass, [("0:10:00.00", "0:10:05.00", "a"), ("0:20:00.00", "0:20:05.00", "b")])
    cmds, fake_run = _run_recorder()
    monkeypatch.setattr(hs.subprocess, "run", fake_run)

    hs.hardsub_video(
        video, ass, tmp_path / "o.mp4",
        segments=[("0:10:00", "0:11:00"), ("0:20:00", "0:21:00"), ("0:30:00", "0:31:00")],
        detect_black=False,
    )

    burns = [c for c in cmds if "-vf" in c]
    assert len(burns) == 3
    assert all(c.index("-ss") < c.index("-i") for c in burns)  # each uses input seek
    concat = next(c for c in cmds if "concat" in c)
    assert concat[concat.index("-c") + 1] == "copy"


def test_multi_segment_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    ass = tmp_path / "s.ass"
    _write_ass(ass, [("0:10:00.00", "0:10:05.00", "a")])
    monkeypatch.setattr(hs.subprocess, "run", lambda cmd, **k: _Result(returncode=1, stderr=b"seg boom"))
    with pytest.raises(RuntimeError, match="hardsub segment"):
        hs.hardsub_video(video, ass, tmp_path / "o.mp4", segments=[("0:10:00", "0:11:00"), ("0:20:00", "0:21:00")], detect_black=False)


def test_black_scan_only_around_joins(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")
    ass = tmp_path / "s.ass"
    _write_ass(ass, [("0:10:00.00", "0:10:05.00", "a")])
    cmds, fake_run = _run_recorder()
    monkeypatch.setattr(hs.subprocess, "run", fake_run)

    hs.hardsub_video(
        video, ass, tmp_path / "o.mp4",
        segments=[("0:10:00", "0:11:00"), ("0:20:00", "0:21:00"), ("0:30:00", "0:31:00")],
        detect_black=True,
    )

    black = [c for c in cmds if any("blackdetect" in tok for tok in c)]
    assert len(black) == 2  # 3 segments -> 2 joins, one windowed scan each
    assert all("-ss" in c and "-t" in c for c in black)  # windowed, not a full-output decode
