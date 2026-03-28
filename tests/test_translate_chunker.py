from autosub.pipeline.translate.chunker import (
    make_chunks,
    _find_corner_boundaries,
    _chunk_by_corners,
    MIN_CHUNK_SIZE,
)


def _texts(n):
    """Generate n dummy subtitle lines."""
    return [f"line {i}" for i in range(n)]


# --- _find_corner_boundaries ---


def test_find_boundaries_basic():
    texts = ["hello", "welcome to the corner", "more text", "another cue here"]
    boundaries = _find_corner_boundaries(texts, ["corner", "cue"])
    assert boundaries == [1, 3]


def test_find_boundaries_no_matches():
    texts = ["hello", "world", "foo"]
    assert _find_corner_boundaries(texts, ["zzz"]) == []


def test_find_boundaries_multiple_cues_same_line():
    texts = ["this has cue1 and cue2"]
    assert _find_corner_boundaries(texts, ["cue1", "cue2"]) == [0]


# --- _chunk_by_corners ---


def test_chunk_by_corners_basic():
    texts = _texts(30)
    # boundary at 15 splits into two chunks
    chunks = _chunk_by_corners(texts, [15], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 15
    assert len(chunks[1]) == 15


def test_chunk_by_corners_merges_tiny_segments():
    texts = _texts(30)
    # boundaries at 10 and 12 — gap of 2 is below MIN_CHUNK_SIZE, so 12 is dropped
    chunks = _chunk_by_corners(texts, [10, 12], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 10
    assert len(chunks[1]) == 20  # 12 merged forward


def test_chunk_by_corners_adjacent_boundaries():
    texts = _texts(50)
    # boundaries at 20 and 21 — only 1 line apart, second is dropped
    chunks = _chunk_by_corners(texts, [20, 21], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 20
    assert len(chunks[1]) == 30


def test_chunk_by_corners_subsplits_oversized():
    texts = _texts(100)
    # single boundary at 0 means one giant segment, max_chunk_size=30
    chunks = _chunk_by_corners(texts, [], max_chunk_size=30)
    # no boundaries means filtered = [0], all_breaks = [0, 100]
    # segment of 100 gets sub-split at 30: 30+30+30+10
    assert len(chunks) == 4
    assert [len(c) for c in chunks] == [30, 30, 30, 10]


def test_chunk_by_corners_boundary_at_zero():
    texts = _texts(25)
    # boundary at 0 just reinforces the start, boundary at 15 splits
    chunks = _chunk_by_corners(texts, [0, 15], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 15
    assert len(chunks[1]) == 10


def test_chunk_by_corners_preserves_content():
    texts = ["a", "b", "c", "d", "e"] * 6  # 30 items
    chunks = _chunk_by_corners(texts, [15], max_chunk_size=80)
    flat = [item for chunk in chunks for item in chunk]
    assert flat == texts


# --- make_chunks (integration) ---


def test_make_chunks_no_cues_fixed_chunking():
    texts = _texts(25)
    chunks = make_chunks(texts, chunk_size=10)
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [10, 10, 5]


def test_make_chunks_cues_no_matches_falls_back():
    texts = _texts(25)
    chunks = make_chunks(texts, chunk_size=10, corner_cues=["zzz_no_match"])
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [10, 10, 5]


def test_make_chunks_none_cues_fixed_chunking():
    texts = _texts(25)
    chunks = make_chunks(texts, chunk_size=10, corner_cues=None)
    assert len(chunks) == 3


def test_make_chunks_corner_aware():
    # Plant a cue at line 12
    texts = _texts(30)
    texts[12] = "さあ、カードイラストのコーナーです"
    chunks = make_chunks(texts, chunk_size=80, corner_cues=["カードイラスト"])
    assert len(chunks) == 2
    assert len(chunks[0]) == 12
    assert len(chunks[1]) == 18


def test_make_chunks_preserves_all_lines():
    texts = _texts(50)
    texts[20] = "cue phrase here"
    chunks = make_chunks(texts, chunk_size=80, corner_cues=["cue phrase"])
    flat = [item for chunk in chunks for item in chunk]
    assert flat == texts


# --- edge case: max_chunk_size < MIN_CHUNK_SIZE ---


def test_chunk_by_corners_small_max_chunk_size():
    texts = _texts(20)
    # boundary at 5, max_chunk_size=3 (smaller than MIN_CHUNK_SIZE)
    # min_size should clamp to 3, so boundary at 5 is kept
    chunks = _chunk_by_corners(texts, [5], max_chunk_size=3)
    # chunk 0: lines 0-4 (5 lines) — oversized, sub-split at 3: [3, 2]
    # chunk 1: lines 5-19 (15 lines) — oversized, sub-split at 3: [3, 3, 3, 3, 3]
    assert sum(len(c) for c in chunks) == 20
    assert all(len(c) <= 3 for c in chunks)


def test_make_chunks_small_chunk_size_with_cues():
    texts = _texts(20)
    texts[8] = "transition cue"
    chunks = make_chunks(texts, chunk_size=5, corner_cues=["transition cue"])
    # boundary at 8 is kept (8 - 0 = 8 >= min(10, 5) = 5)
    flat = [item for chunk in chunks for item in chunk]
    assert flat == texts
    # no chunk should exceed chunk_size=5
    assert all(len(c) <= 5 for c in chunks)
