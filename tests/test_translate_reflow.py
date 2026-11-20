from autosub.pipeline.translate.reflow import (
    _canonical,
    _ends_sentence,
    _forbidden_end,
    _group_indices,
    _resplit_group,
    _split_score,
    reflow_line_breaks,
)


# --- grouping ---


def test_group_splits_on_terminal_punctuation():
    texts = ["This is done.", "And here is more"]
    assert _group_indices(texts, set()) == [[0], [1]]


def test_group_keeps_continuation_lines_together():
    texts = ["it's very refreshing, but", "in the A and B sections."]
    assert _group_indices(texts, set()) == [[0, 1]]


def test_group_breaks_on_hard_boundary():
    texts = ["one two", "three four", "five six"]
    # a hard boundary (speaker change / corner) before index 1
    assert _group_indices(texts, {1}) == [[0], [1, 2]]


def test_ends_sentence_ignores_trailing_quote():
    assert _ends_sentence('he said "hello."')
    assert _ends_sentence("done!")
    assert not _ends_sentence("refreshing, but")


# --- dangling connective moves to next line ---


def test_moves_trailing_conjunction_to_next_line():
    out = reflow_line_breaks(
        ["what kind of path they will take, but", "I want to keep watching."],
        [3.0, 3.0],
    )
    assert out == [
        "what kind of path they will take,",
        "but I want to keep watching.",
    ]


def test_moves_trailing_so():
    out = reflow_line_breaks(
        ["The chart is available right away, so", "please give it a try."],
        [3.5, 1.0],
    )
    assert out == [
        "The chart is available right away,",
        "so please give it a try.",
    ]


def test_three_line_group_moves_each_connective():
    out = reflow_line_breaks(
        ["I stayed home because", "it was raining and", "I had no umbrella."],
        [2.0, 2.0, 2.0],
    )
    assert out == [
        "I stayed home",
        "because it was raining",
        "and I had no umbrella.",
    ]


# --- casing ---


def test_lowercases_capitalized_conjunction_at_seam():
    out = reflow_line_breaks(["I went home,", "But then I left."], [2.0, 2.0])
    assert out == ["I went home,", "but then I left."]


def test_preserves_single_letter_labels():
    # "A" is a section label, not an article — must not be lowercased.
    out = reflow_line_breaks(
        ["it's very refreshing, but", "in the A and B sections."],
        [4.0, 4.0],
    )
    assert out == [
        "it's very refreshing,",
        "but in the A and B sections.",
    ]


# --- no-op / preservation ---


def test_separate_sentences_unchanged():
    texts = ["This is a complete sentence.", "And here is another one."]
    assert reflow_line_breaks(texts, [3.0, 3.0]) == texts


def test_single_line_group_unchanged():
    texts = ["just one line here"]
    assert reflow_line_breaks(texts, [3.0]) == texts


def test_empty_input():
    assert reflow_line_breaks([], []) == []


def test_no_words_added_or_dropped():
    texts = ["the reason I came here today, and", "the thing I wanted to say"]
    out = reflow_line_breaks(texts, [3.0, 3.0])
    assert _canonical(out) == _canonical(texts)


def test_duration_weighting_favors_longer_slot():
    # Second slot is tiny, so most text should stay in the first slot.
    out = reflow_line_breaks(
        ["I really wanted to say something important but", "no."],
        [4.0, 0.5],
    )
    assert len(out[0]) > len(out[1])
    assert not _forbidden_end(out[0].split()[-1])


# --- quality gate ---


def test_does_not_churn_a_clean_comma_break():
    # Original ends on a comma with a plain next word — reflow must not move the
    # break to a weaker position (ending on a plain word) just to rebalance.
    texts = ["the new song and Miku's song,", "symbolizing the future choices."]
    assert reflow_line_breaks(texts, [3.0, 3.0]) == texts


def test_does_not_churn_two_clean_comma_breaks():
    texts = [
        "for those who haven't finished the story yet,",
        "even after the next event starts,",
    ]
    assert reflow_line_breaks(texts, [3.0, 3.0]) == texts


def test_split_score_penalizes_dangling_and_rewards_punctuation():
    dangling = ["what kind of path they take, but", "I want to watch."]
    clean = ["what kind of path they take,", "but I want to watch."]
    assert _split_score(clean) > _split_score(dangling)


# --- guardrail ---


def test_forbidden_end_detects_bare_function_words():
    assert _forbidden_end("but")
    assert _forbidden_end("the")
    assert _forbidden_end("of")
    # punctuation-terminated words are always allowed
    assert not _forbidden_end("refreshing,")
    assert not _forbidden_end("home.")
    assert not _forbidden_end("cat")


def test_resplit_never_ends_a_piece_on_a_connective():
    out = _resplit_group(
        ["the cat sat on", "the mat and then slept"],
        [2.0, 2.0],
    )
    assert out is not None
    for piece in out[:-1]:
        assert not _forbidden_end(piece.split()[-1])


def test_length_and_duration_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        reflow_line_breaks(["a", "b"], [1.0])
