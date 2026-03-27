# Deferred AutoSub Planned Work

With the core MVP (Minimum Viable Product) now complete—featuring a fully connected `autosub run` pipeline that handles Transcription, Formatting, and Translation via the composable TOML `--profile` system—the following architectural plans have been mapped out for Phase 2 of the `autosub` toolchain.


## 1. Corner Comment Markers in .ass Output
Insert Aegisub Comment lines into the formatted `.ass` file to mark where recurring show corners/segments start (e.g. "Card Illustrations", "3DMV", "Fan Letter Corner"). Comment lines appear as green rows in Aegisub's subtitle grid, making segment boundaries visible to editors.

**Approach:**
- Match corner cue phrases (Japanese, from profile `[[corners]]`) against subtitle text during the **format** step in `generator.py`, not the translate step — corners are a radio show concept scoped to the `solo_seiyuu_radio` extension.
- Insert `pyass.Event(format=EventFormat.COMMENT)` before matching Dialogue events with `effect="corner"` and `text="=== Corner: {name} ==="`.
- Track active corner to avoid duplicate markers for the same segment.
- Add a defensive filter in `translate/main.py` to skip Comment events so they aren't sent to the LLM.

**Open decisions:**
- Whether to pass corners to `format_subtitles()` as a new parameter or nested inside `extensions_config`.
- Corners data currently lives in profiles as a top-level field, not under `[extensions]`.

**Files to modify:** `autosub/pipeline/format/generator.py`, `autosub/pipeline/format/main.py`, `autosub/pipeline/translate/main.py`, `autosub/cli.py`

## 2. Checkpoint Metadata Validation
Add a hash of the input texts and chunk size to the checkpoint JSON so mismatches (e.g. re-running with a different source file or chunk size) are detected and the checkpoint is discarded instead of silently producing wrong results.

**File:** `autosub/pipeline/translate/main.py`

## 3. Advanced Timing Rules
The single-speaker timing pipeline now supports minimum duration padding, gap snapping, optional keyframe-aware scene snapping, and automatic wrapping to keep subtitles within two visible lines.

Remaining work in this area:
*   Improve line breaking so semantic chunking and visual wrapping cooperate better on very dense speech.
*   Tune scene-aware snapping heuristics against real subtitle editing workflows in Aegisub.
*   Add profile presets for different content densities instead of relying on one global timing profile.

## 3. Audio Extraction & Segmentation Pipeline
*   **Singing Filtering**: Intelligently detect and ignore singing sections in concert videos (e.g. leveraging `spleeter` or similar vocal detection tech), so the primary transcription module exclusively subtitiles the MC / spoken sections.

## 4. On-Screen Text OCR
*   **Visual Pipeline**: Implement optical character recognition (OCR) on the raw video footage.
*   **Integration**: Seamlessly interleave OCR-generated `.ass` lines (e.g., lower thirds, on-screen signs) with the speech-generated `.ass` lines, ensuring visual styles do not clash and timestamps overlap cleanly.

## 5. Review UX
*   Add a lightweight review flow around `original.ass` so users can edit timings and line breaks in Aegisub before running translation.
