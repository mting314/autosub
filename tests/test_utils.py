import pytest
from autosub.core.utils import parse_timestamp


def test_parse_empty_string():
    assert parse_timestamp("") == 0.0


def test_parse_plain_seconds():
    assert parse_timestamp("42.5") == 42.5


def test_parse_mm_ss():
    assert parse_timestamp("1:30.0") == 90.0


def test_parse_hh_mm_ss():
    assert parse_timestamp("1:02:03.5") == 3723.5


def test_parse_single_colon_part():
    assert parse_timestamp("5") == 5.0


def test_parse_invalid_format():
    with pytest.raises(ValueError, match="Invalid timestamp format"):
        parse_timestamp("1:2:3:4")
