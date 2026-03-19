from autosub.core.schemas import SubtitleLine
from autosub.extensions.radio_discourse.main import apply_radio_discourse


def test_radio_discourse_splits_framing_phrase_and_labels_roles():
    lines = [
        SubtitleLine(
            text="メールを送るのは初めてです。",
            start_time=0.0,
            end_time=2.0,
        ),
        SubtitleLine(
            text="おお、嬉しい。",
            start_time=2.0,
            end_time=3.0,
        ),
        SubtitleLine(
            text="いろんなコーデに合わせられるのでおすすめですといただきました。",
            start_time=3.0,
            end_time=6.0,
        ),
    ]

    result = apply_radio_discourse(lines, {"enabled": True})

    assert [line.role for line in result] == [
        "listener_mail",
        "host",
        "listener_mail",
        "host_meta",
    ]
    assert result[2].text == "いろんなコーデに合わせられるのでおすすめです。"
    assert result[3].text == "といただきました。"
    assert result[2].end_time <= result[3].start_time
