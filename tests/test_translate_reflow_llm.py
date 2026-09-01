"""Tests for the pluggable reflow engine and the LLM-backed re-split.

No network: the LLM call is exercised via a stubbed `resplit_batch` /
`_run_structured_output`, and the pluggable engine via plain callables.
"""

from typing import Any

from autosub.pipeline.translate import reflow_llm
from autosub.pipeline.translate.reflow import reflow_line_breaks
from autosub.pipeline.translate.reflow_llm import (
    ReflowSplit,
    ReflowSplitter,
    build_llm_resplitter,
)


# --- pluggable engine + shared gate ---


def test_custom_engine_proposal_is_applied_when_better():
    def engine(batch):
        return [["I saw it,", "but she left."]]

    out = reflow_line_breaks(
        ["I saw it, but", "she left."], [2.0, 2.0], resplitter=engine
    )
    assert out == ["I saw it,", "but she left."]


def test_custom_engine_word_tampering_is_rejected():
    def engine(batch):
        return [["totally", "different words"]]

    texts = ["I saw it, but", "she left."]
    assert reflow_line_breaks(texts, [2.0, 2.0], resplitter=engine) == texts


def test_custom_engine_worse_split_is_rejected():
    # Original already breaks cleanly on the comma; a proposal that ends a piece
    # on the bare "and" scores worse and must be rejected.
    def engine(batch):
        return [["the new song and", "Miku's song, symbolizing it."]]

    texts = ["the new song and Miku's song,", "symbolizing it."]
    assert reflow_line_breaks(texts, [3.0, 3.0], resplitter=engine) == texts


def test_engine_returning_none_per_group_keeps_original():
    def engine(batch):
        return [None]

    texts = ["I saw it, but", "she left."]
    assert reflow_line_breaks(texts, [2.0, 2.0], resplitter=engine) == texts


def test_engine_proposal_count_mismatch_skips_reflow():
    def engine(batch):
        return []  # wrong length vs number of groups

    texts = ["I saw it, but", "she left."]
    assert reflow_line_breaks(texts, [2.0, 2.0], resplitter=engine) == texts


# --- ReflowSplitter.resplit_batch shaping ---


class _StubSplitter(ReflowSplitter):
    def __init__(self, results):
        self._results = results  # skip real LLM init

    def _run_structured_output(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        output_type: Any,
        operation_name: str,
        output_name: str,
    ) -> tuple[Any, Any]:
        return self._results, None


def test_resplit_batch_maps_by_id_and_enforces_slot_count():
    batch = [
        (["a b", "c d"], [1.0, 1.0]),  # id 0, wants 2 slots
        (["e f g"], [1.0]),  # id 1, single-line (not sent, but keep aligned)
    ]
    # Only group 0 is realistic here; return good pieces for it, wrong count for 1.
    results = [
        ReflowSplit(id=0, pieces=["a", "b c d"]),
        ReflowSplit(id=1, pieces=["e", "f g"]),  # wrong slot count (wants 1)
    ]
    proposals = _StubSplitter(results).resplit_batch(batch)
    assert proposals[0] == ["a", "b c d"]
    assert proposals[1] is None  # slot-count mismatch rejected


def test_resplit_batch_missing_id_yields_none():
    batch = [(["a b", "c d"], [1.0, 1.0])]
    proposals = _StubSplitter([]).resplit_batch(batch)
    assert proposals == [None]


# --- build_llm_resplitter fallback ---


def test_llm_resplitter_falls_back_to_deterministic_on_error(monkeypatch):
    def boom(self, batch):
        raise RuntimeError("api down")

    monkeypatch.setattr(ReflowSplitter, "resplit_batch", boom)
    monkeypatch.setattr(reflow_llm.ReflowSplitter, "__init__", lambda self, **kw: None)

    resplitter = build_llm_resplitter(project_id="p")
    out = reflow_line_breaks(
        ["what path they take, but", "I want to watch."],
        [3.0, 3.0],
        resplitter=resplitter,
    )
    # deterministic engine still moves the dangling "but"
    assert out == ["what path they take,", "but I want to watch."]


def test_llm_resplitter_uses_deterministic_where_llm_declines(monkeypatch):
    def decline_all(self, batch):
        return [None for _ in batch]

    monkeypatch.setattr(ReflowSplitter, "resplit_batch", decline_all)
    monkeypatch.setattr(reflow_llm.ReflowSplitter, "__init__", lambda self, **kw: None)

    resplitter = build_llm_resplitter(project_id="p")
    out = reflow_line_breaks(
        ["what path they take, but", "I want to watch."],
        [3.0, 3.0],
        resplitter=resplitter,
    )
    assert out == ["what path they take,", "but I want to watch."]
