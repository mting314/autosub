# Deferred AutoSub Planned Work

With the core MVP (Minimum Viable Product) now complete—featuring a fully connected `autosub run` pipeline that handles Transcription, Formatting, and Translation via the composable TOML `--profile` system—the following architectural plans have been mapped out for Phase 2 of the `autosub` toolchain.


## 2. Advanced Timing Rules
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
*   `autosub report` generates a self-contained HTML review page with side-by-side JP/EN, embedded video player, click-to-seek, auto-highlight, and issue detection filters.
*   **Next milestone**: Add editable text boxes that manipulate the original/translated `.ass` files directly from the report page.

## 6. Radio Overlay

*   Work is already being done in the subtitling-projects branch feat/radio-overlay, but a couple thoughts:
*   **Line-by-line Speaker Designation**: Diarization is very very finnicky and makes lots of mistakes. We need a more
*   This is especially true because for the radio overlay, there's nothing that ties the position tag to the speaker style you use. So if you just do the standard "go through aegisub and update the style", the position tag will be wrong and out of sync.
*   **More intelligent diarization correction through semantic analysis**: Basically, if someone says something like "how about you, Suzuki-san", the speaker shouldn't be identified as Suzuki-san since they're obviously talking about a third person. Don't know if this needs to be like a second separate LLM call though; if so that's probably too expensive.
*   **Radio Overlay doesn't seem to be respecting flashing subtitle and overlap rules**
*   This might be a problem with the prompt but I don't think the translation realizes the semantic of this radio being the seiyuu talking about their own lives, not their characters.
