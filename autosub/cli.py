import typer
import logging
from pathlib import Path

from autosub.pipeline.transcribe import main as transcribe_main
from autosub.pipeline.format import main as format_module
from autosub.pipeline.postprocess import main as postprocess_module
from autosub.pipeline.translate import main as translate_module
from autosub.core.profile import load_unified_profile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="AutoSub CLI for Japanese subtitle generation and translation")

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"


def _add_file_logger(log_path: Path) -> None:
    """Add a FileHandler to the root logger so all output is also saved to disk."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # allow DEBUG through to file handler
    # Keep console at INFO
    for h in root.handlers:
        h.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    logger.info(f"Saving log to {log_path}")


@app.callback()
def main():
    pass


def _transcript_hash(transcript_path: Path) -> str:
    """Compute a short hash of a transcript file for staleness detection."""
    import hashlib
    with open(transcript_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _interactive_speaker_review(
    result, speaker_map: dict[str, dict], output_path: Path,
    sample_count: int = 3,
    transcript_path: Path | None = None,
) -> dict[str, dict] | None:
    """Run interactive speaker label review and write assignments.

    Returns the loaded assignments dict, or None if no assignments were made.
    """
    from autosub.core.speaker_map import load_speaker_map

    # Group words by speaker label
    speaker_words: dict[str, list] = {}
    for w in result.words:
        label = w.speaker or "unknown"
        if label not in speaker_words:
            speaker_words[label] = []
        speaker_words[label].append(w)

    # Deduplicate real speakers from the map
    real_speakers: list[dict] = []
    seen_names: set[str] = set()
    for entry in speaker_map.values():
        if entry["name"] not in seen_names:
            real_speakers.append(entry)
            seen_names.add(entry["name"])

    if not real_speakers:
        logger.error("Speaker map has no speaker entries.")
        return None

    # Display samples per label
    sorted_labels = sorted(speaker_words.keys(), key=lambda x: (not x.isdigit(), x))
    typer.echo("\n=== Speaker Labels in Transcript ===\n")

    for label in sorted_labels:
        words = speaker_words[label]
        if len(words) <= sample_count * 10:
            indices = [0]
        else:
            step = len(words) // sample_count
            indices = [i * step for i in range(sample_count)]

        typer.echo(f"Speaker {label} ({len(words)} words):")
        for idx in indices:
            snippet_words = words[idx:idx + 10]
            t = snippet_words[0].start_time
            mins = int(t // 60)
            secs = t % 60
            text = "".join(w.word for w in snippet_words)
            if len(text) > 50:
                text = text[:47] + "..."
            typer.echo(f"  [{mins:02d}:{secs:04.1f}] {text}")
        typer.echo()

    # Interactive mapping
    typer.echo("=== Known Speakers ===\n")
    for i, sp in enumerate(real_speakers, 1):
        char = f" ({sp['character']})" if sp.get("character") else ""
        typer.echo(f"  {i}. {sp['name']}{char}")
    typer.echo()

    assignments: dict[str, dict] = {}
    for label in sorted_labels:
        while True:
            choice = typer.prompt(
                f"Assign label \"{label}\" → [1-{len(real_speakers)}/skip]",
                default="skip",
            )
            if choice.lower() == "skip":
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(real_speakers):
                    assignments[label] = real_speakers[idx - 1]
                    typer.echo(f"  → {real_speakers[idx - 1]['name']}")
                    break
                else:
                    typer.echo(f"  Enter 1-{len(real_speakers)} or 'skip'")
            except ValueError:
                typer.echo(f"  Enter 1-{len(real_speakers)} or 'skip'")

    if not assignments:
        logger.info("No assignments made.")
        return None

    # Log summary
    skipped = [l for l in sorted_labels if l not in assignments]
    logger.info(
        f"Assigned {len(assignments)}/{len(sorted_labels)} labels, "
        f"skipped {len(skipped)}: {skipped if skipped else 'none'}"
    )
    for label, entry in sorted(assignments.items()):
        word_count = len(speaker_words[label])
        logger.info(f"  Label {label} ({word_count} words) → {entry['name']}")

    # Write assignments file
    lines = [
        f"# Speaker assignments generated by review-speakers",
    ]
    if transcript_path:
        lines.append(f'transcript_hash = "{_transcript_hash(transcript_path)}"')
    lines.append("")
    for label, entry in sorted(assignments.items(), key=lambda x: x[0]):
        lines.append(f'[speakers."{label}"]')
        lines.append(f'name = "{entry["name"]}"')
        if entry.get("character"):
            lines.append(f'character = "{entry["character"]}"')
        if entry.get("color"):
            lines.append(f'color = "{entry["color"]}"')
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote speaker assignments to {output_path}")
    return load_speaker_map(output_path)


@app.command()
def transcribe(
    video_path: Path = typer.Argument(
        ..., help="Path to the input video or audio file"
    ),
    output: Path = typer.Option(
        Path("transcript.json"),
        "--out",
        "-o",
        help="Path to save the output JSON transcript",
    ),
    language: str = typer.Option(
        "ja-JP",
        "--language",
        "-l",
        help="Language code for transcription (e.g. ja-JP, en-US)",
    ),
    vocab: list[str] = typer.Option(
        None,
        "--vocab",
        "-v",
        help="Custom context terminology to increase probability of recognition (can be passed multiple times).",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load transcription vocabulary hints.",
    ),
    speakers: int = typer.Option(
        None,
        "--speakers",
        "-s",
        help="Number of speakers for diarization. Enables per-word speaker labels in the transcript.",
    ),
    start: str = typer.Option(
        None,
        "--start",
        help="Start time for transcription (e.g. 00:01:00 or 60).",
    ),
    end: str = typer.Option(
        None,
        "--end",
        help="End time for transcription (e.g. 00:04:00 or 240).",
    ),
):
    """
    Extracts audio and transcribes Japanese speech with Google Cloud Speech-to-Text.
    """
    logger.info(f"Starting transcription pipeline for: {video_path}")

    final_vocab = []
    if profile:
        profile_data = load_unified_profile(profile)
        final_vocab.extend(profile_data["vocab"])
    if vocab:
        final_vocab.extend(vocab)

    try:
        result = transcribe_main.transcribe(
            video_path,
            output,
            language,
            final_vocab,
            num_speakers=speakers,
            start_time=start,
            end_time=end,
        )
        logger.info(f"Success! Saved {len(result.words)} words to {output}")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise typer.Exit(code=1)


@app.command()
def format(
    input_transcript: Path = typer.Argument(
        ...,
        help="Path to the input transcript.json file generated by the transcribe command.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path to save the generated output .ass file (defaults to original.ass in the same directory).",
    ),
    keyframes: Path = typer.Option(
        None, "--keyframes", help="Path to Aegisub keyframes log for snapping."
    ),
    fps: float = typer.Option(
        0.0, "--fps", help="Video FPS, required if --keyframes is used."
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load timing and formatting extension configuration.",
    ),
    speaker_map: Path = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker_map.toml mapping API speaker labels to character names and colors.",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Step 2: Converts a transcript JSON into timed single-lane .ass subtitles.
    """
    if not out:
        out = input_transcript.with_name("original.ass")

    if keyframes and fps <= 0:
        logger.error("--fps is required when --keyframes is provided.")
        raise typer.Exit(code=1)

    timing_config = {}
    extensions_config = {}
    replacements = {}
    if profile:
        profile_data = load_unified_profile(profile)
        timing_config = profile_data.get("timing", {})
        extensions_config = profile_data.get("extensions", {})
        replacements = profile_data.get("replacements", {})

    loaded_speaker_map = None
    if speaker_map:
        from autosub.core.speaker_map import load_speaker_map
        loaded_speaker_map = load_speaker_map(speaker_map)

    kf_ms = None
    if keyframes and fps > 0:
        from autosub.pipeline.video.keyframes import parse_aegisub_keyframes

        kf_ms = parse_aegisub_keyframes(keyframes, fps)

    try:
        format_module.format_subtitles(
            input_transcript,
            out,
            keyframes=kf_ms,
            timing_config=timing_config,
            extensions_config=extensions_config,
            replacements=replacements,
            speaker_map=loaded_speaker_map,
        )
    except Exception as e:
        logger.error(f"Error during formatting: {e}")
        raise typer.Exit(code=1)


@app.command()
def translate(
    input_ass: Path = typer.Argument(
        ...,
        help="Path to the original .ass file generated by the format command.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path to save the translated .ass file (defaults to translated.ass in the same directory).",
    ),
    engine: str = typer.Option(
        "vertex",
        "--engine",
        "-e",
        help="Translation engine to use ('vertex' or 'cloud-v3').",
    ),
    prompt: str = typer.Option(
        None, "--prompt", "-p", help="System prompt to guide the LLM translation."
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load translation prompt settings.",
    ),
    target_lang: str = typer.Option("en", "--target", help="Target language code."),
    source_lang: str = typer.Option("ja", "--source", help="Source language code."),
    vertex_model: str = typer.Option(
        "gemini-3-flash-preview",
        "--vertex-model",
        help="Vertex model name for LLM translation.",
    ),
    vertex_location: str = typer.Option(
        "global",
        "--vertex-location",
        help="Vertex region for LLM translation.",
    ),
    bilingual: bool = typer.Option(
        False,
        "--bilingual/--replace",
        help="Include original text on top, or replace completely.",
    ),
    chunk: bool = typer.Option(
        False,
        "--chunk/--no-chunk",
        help="Split translation into smaller chunks with retry logic. Useful for long files.",
    ),
    chunk_size: int = typer.Option(
        80,
        "--chunk-size",
        help="Number of subtitle lines per chunk when --chunk is enabled.",
    ),
    mark_chunks: bool = typer.Option(
        False,
        "--mark-chunks/--no-mark-chunks",
        help="Insert comment events at artificial chunk boundaries for review.",
    ),
    save_log: bool = typer.Option(
        False,
        "--save-log/--no-save-log",
        help="Save full log output to a .log file next to the output file.",
    ),
    speaker_map: Path = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker_map.toml to inject speaker identity context into the translation prompt.",
        exists=True,
        dir_okay=False,
    ),
    retry_chunk: list[int] = typer.Option(
        None,
        "--retry-chunk",
        help="Re-translate specific chunk(s) by number (1-based). Can be passed multiple times.",
    ),
):
    """
    Step 3: Translates a .ass subtitle file using the configured Translation Engine.
    """
    if not out:
        out = input_ass.with_name("translated.ass")

    translate_log_dir = None
    if save_log:
        translate_log_dir = out.parent / f"{out.stem}_logs"
        translate_log_dir.mkdir(parents=True, exist_ok=True)
        _add_file_logger(translate_log_dir / "run.log")

    loaded_speaker_map = None
    if speaker_map:
        from autosub.core.speaker_map import load_speaker_map
        loaded_speaker_map = load_speaker_map(speaker_map)

    final_prompt_parts = []
    final_corner_names = []
    final_corner_cues = []
    if profile:
        profile_data = load_unified_profile(profile)
        final_prompt_parts.extend(profile_data["prompt"])

        if profile_data.get("glossary"):
            glossary_text = "Glossary (Always translate these exact phrases):\n"
            for ja, en in profile_data["glossary"].items():
                glossary_text += f'- "{ja}" -> "{en}"\n'
            final_prompt_parts.append(glossary_text)

        if profile_data.get("corners"):
            final_corner_names = [c["name"] for c in profile_data["corners"]]
            for c in profile_data["corners"]:
                final_corner_cues.extend(c.get("cues", []))
            corners_text = "Recurring corners/segments in this program:\n"
            for corner in profile_data["corners"]:
                corners_text += f'- {corner["name"]}: {corner["description"]}\n'
                if corner.get("cues"):
                    corners_text += f'  Common cue phrases: {", ".join(corner["cues"])}\n'
            corners_text += (
                "\nWhen you detect a transition to a new corner/segment, "
                "prepend [CORNER: <corner name>] to the FIRST translated line of that segment. "
                "Only mark the transition once per segment, not every line. "
                "Example: [CORNER: Card Illustrations] Let's take a look at the card illustrations.\n\n"
                "IMPORTANT: Do NOT mark a corner just because a cue phrase appears in passing. "
                "The host may mention a segment topic briefly (e.g. previewing what's coming up later) "
                "without actually transitioning to that segment. Only mark a corner when the overall "
                "topic genuinely changes and the conversation shifts to that segment's content. "
                "Look at the surrounding context, not just a single line."
            )
            final_prompt_parts.append(corners_text)

    if prompt:
        final_prompt_parts.append(prompt)

    if loaded_speaker_map:
        from autosub.core.speaker_map import build_speaker_prompt
        final_prompt_parts.append(build_speaker_prompt(loaded_speaker_map))

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None

    chunk_size = chunk_size if chunk else 0

    try:
        translate_module.translate_subtitles(
            input_ass,
            out,
            engine=engine,
            system_prompt=final_prompt,
            target_lang=target_lang,
            source_lang=source_lang,
            bilingual=bilingual,
            model=vertex_model,
            location=vertex_location,
            chunk_size=chunk_size,
            corner_names=final_corner_names or None,
            corner_cues=final_corner_cues or None,
            debug=mark_chunks,
            retry_chunks=retry_chunk or None,
            log_dir=translate_log_dir,
        )
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        raise typer.Exit(code=1)


@app.command()
def postprocess(
    input_ass: Path = typer.Argument(
        ...,
        help="Path to the translated .ass file generated by the translate command.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path to save the postprocessed .ass file (defaults to overwriting the input file).",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load postprocessing extension configuration.",
    ),
    bilingual: bool = typer.Option(
        False,
        "--bilingual/--replace",
        help="Treat the input as bilingual stacked subtitles, or translated-only subtitles.",
    ),
):
    """
    Step 4: Applies optional post-translation subtitle cleanup and editorial transforms.
    """
    final_extensions = {}
    if profile:
        profile_data = load_unified_profile(profile)
        final_extensions = profile_data.get("extensions", {})

    try:
        postprocess_module.postprocess_subtitles(
            input_ass,
            output_ass_path=out,
            extensions_config=final_extensions,
            bilingual=bilingual,
        )
    except Exception as e:
        logger.error(f"Error during postprocessing: {e}")
        raise typer.Exit(code=1)


@app.command(name="review-speakers")
def review_speakers(
    transcript: Path = typer.Argument(
        ...,
        help="Path to the transcript JSON file from the transcribe step.",
        exists=True,
        dir_okay=False,
    ),
    speaker_map_path: Path = typer.Option(
        ...,
        "--speaker-map",
        help="Path to speaker_map.toml with the real speaker identities (not modified).",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        None,
        "--out",
        help="Output path for speaker assignments (default: speaker_assignments.toml next to transcript).",
    ),
    sample_lines: int = typer.Option(
        3,
        "--sample-lines",
        help="Number of sample lines to show per speaker label.",
    ),
):
    """
    Review diarized speaker labels and interactively map them to real speakers.

    Shows sample transcript lines per speaker label, then prompts you to assign
    each label to a speaker from the speaker map. Writes assignments to a
    separate file (speaker_assignments.toml) — the original speaker map is not modified.
    """
    import json

    from autosub.core.schemas import TranscriptionResult
    from autosub.core.speaker_map import load_speaker_map

    logger.info(f"Loading transcript from {transcript}...")
    with open(transcript, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = TranscriptionResult(**data)

    if not result.words:
        logger.error("Transcript has no words.")
        raise typer.Exit(code=1)

    logger.info(f"Loaded {len(result.words)} words from transcript.")

    existing_map = load_speaker_map(speaker_map_path)

    if not output:
        output = transcript.parent / "speaker_assignments.toml"

    assigned = _interactive_speaker_review(
        result, existing_map, output, sample_count=sample_lines,
        transcript_path=transcript,
    )
    if not assigned:
        raise typer.Exit()

    logger.info(f"Use with: autosub format ... --speaker-map {output}")


@app.command()
def run(
    video_path: Path = typer.Argument(
        ..., help="Path to the input video or audio file"
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        help="Directory to save generated files. Defaults to the input video's directory.",
    ),
    language: str = typer.Option(
        "ja-JP", "--language", "-l", help="Language code for transcription (e.g. ja-JP)"
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load vocabulary, timing, prompt, and extension settings.",
    ),
    vocab: list[str] = typer.Option(
        None, "--vocab", "-v", help="Custom transcription hints."
    ),
    engine: str = typer.Option(
        "vertex", "--engine", "-e", help="Translation engine ('vertex' or 'cloud-v3')."
    ),
    prompt: str = typer.Option(
        None, "--prompt", "-p", help="System prompt to guide the LLM translation."
    ),
    target_lang: str = typer.Option("en", "--target", help="Target language code."),
    source_lang: str = typer.Option("ja", "--source", help="Source language code."),
    vertex_model: str = typer.Option(
        "gemini-3-flash-preview",
        "--vertex-model",
        help="Vertex model name for LLM translation.",
    ),
    vertex_location: str = typer.Option(
        "global",
        "--vertex-location",
        help="Vertex region for LLM translation.",
    ),
    bilingual: bool = typer.Option(
        False, "--bilingual/--replace", help="Include original text on top."
    ),
    speakers: int = typer.Option(
        None,
        "--speakers",
        "-s",
        help="Number of speakers for diarization. Enables per-word speaker labels and per-speaker styles.",
    ),
    keyframes: Path = typer.Option(
        None,
        "--keyframes",
        help="Path to existing keyframes log (overrides extraction).",
    ),
    extract_keyframes: bool = typer.Option(
        True,
        "--extract-keyframes/--no-extract-keyframes",
        help="Automatically extract keyframes when optional external tooling is installed.",
    ),
    start: str = typer.Option(
        None,
        "--start",
        help="Start time for transcription (e.g. 00:01:00 or 60).",
    ),
    end: str = typer.Option(
        None,
        "--end",
        help="End time for transcription (e.g. 00:04:00 or 240).",
    ),
    chunk: bool = typer.Option(
        False,
        "--chunk/--no-chunk",
        help="Split translation into smaller chunks with retry logic. Useful for long files.",
    ),
    chunk_size: int = typer.Option(
        80,
        "--chunk-size",
        help="Number of subtitle lines per chunk when --chunk is enabled.",
    ),
    mark_chunks: bool = typer.Option(
        False,
        "--mark-chunks/--no-mark-chunks",
        help="Insert comment events at artificial chunk boundaries for review.",
    ),
    save_log: bool = typer.Option(
        False,
        "--save-log/--no-save-log",
        help="Save full log output to a .log file in the output directory.",
    ),
    speaker_map: Path = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker_map.toml mapping API speaker labels to character names and colors.",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Runs the end-to-end Japanese pipeline (Transcribe -> Format -> Translate -> Postprocess).
    """
    logger.info(f"Starting full autosub pipeline for: {video_path}")

    if not out_dir:
        out_dir = video_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    translate_log_dir = None
    if save_log:
        translate_log_dir = out_dir / f"{stem}_logs"
        translate_log_dir.mkdir(parents=True, exist_ok=True)
        _add_file_logger(translate_log_dir / "run.log")

    transcript_out = out_dir / f"{stem}_transcript.json"
    original_ass_out = out_dir / f"{stem}_original.ass"
    translated_ass_out = out_dir / f"{stem}_translated.ass"

    # Resolve Profile
    final_vocab = []
    final_prompt_parts = []
    final_timing = {}
    final_extensions = {}
    replacements = {}
    final_corner_names = []
    final_corner_cues = []
    if profile:
        profile_data = load_unified_profile(profile)
        final_vocab.extend(profile_data["vocab"])
        final_prompt_parts.extend(profile_data["prompt"])
        final_timing = profile_data.get("timing", {})
        final_extensions = profile_data.get("extensions", {})
        replacements = profile_data.get("replacements", {})
        if not speakers and profile_data.get("speakers"):
            speakers = profile_data["speakers"]

        if profile_data.get("glossary"):
            glossary_text = "Glossary (Always translate these exact phrases):\n"
            for ja, en in profile_data["glossary"].items():
                glossary_text += f'- "{ja}" -> "{en}"\n'
            final_prompt_parts.append(glossary_text)

        if profile_data.get("corners"):
            final_corner_names = [c["name"] for c in profile_data["corners"]]
            for c in profile_data["corners"]:
                final_corner_cues.extend(c.get("cues", []))
            corners_text = "Recurring corners/segments in this program:\n"
            for corner in profile_data["corners"]:
                corners_text += f'- {corner["name"]}: {corner["description"]}\n'
                if corner.get("cues"):
                    corners_text += f'  Common cue phrases: {", ".join(corner["cues"])}\n'
            corners_text += (
                "\nWhen you detect a transition to a new corner/segment, "
                "prepend [CORNER: <corner name>] to the FIRST translated line of that segment. "
                "Only mark the transition once per segment, not every line. "
                "Example: [CORNER: Card Illustrations] Let's take a look at the card illustrations.\n\n"
                "IMPORTANT: Do NOT mark a corner just because a cue phrase appears in passing. "
                "The host may mention a segment topic briefly (e.g. previewing what's coming up later) "
                "without actually transitioning to that segment. Only mark a corner when the overall "
                "topic genuinely changes and the conversation shifts to that segment's content. "
                "Look at the surrounding context, not just a single line."
            )
            final_prompt_parts.append(corners_text)

    if vocab:
        final_vocab.extend(vocab)
    if prompt:
        final_prompt_parts.append(prompt)

    # Load speaker map early so it can be used in both prompt and format steps
    loaded_speaker_map = None
    if speaker_map:
        from autosub.core.speaker_map import load_speaker_map, build_speaker_prompt
        loaded_speaker_map = load_speaker_map(speaker_map)
        final_prompt_parts.append(build_speaker_prompt(loaded_speaker_map))

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None

    # Step 1: Transcribe
    try:
        logger.info("[Step 1/4] Transcribing...")
        result = transcribe_main.transcribe(
            video_path,
            transcript_out,
            language,
            final_vocab,
            num_speakers=speakers,
            start_time=start,
            end_time=end,
            replacements=replacements or None,
        )
    except Exception as e:
        logger.error(f"Failed during transcription: {e}")
        raise typer.Exit(code=1)

    # Interactive speaker review when extra labels are detected
    if speakers and result and loaded_speaker_map:
        unique_speakers = {w.speaker for w in result.words if w.speaker}
        if len(unique_speakers) > speakers:
            logger.warning(
                f"Requested {speakers} speakers but transcription returned "
                f"{len(unique_speakers)} labels: {sorted(unique_speakers)}."
            )
            assignments_path = transcript_out.parent / "speaker_assignments.toml"

            # Check for existing assignments from a previous review
            run_review = True
            if assignments_path.exists():
                import tomllib as _tomllib
                with open(assignments_path, "rb") as _f:
                    _adata = _tomllib.load(_f)
                stored_hash = _adata.get("transcript_hash")
                current_hash = _transcript_hash(transcript_out)
                if stored_hash == current_hash:
                    from autosub.core.speaker_map import load_speaker_map as _load_map
                    loaded_speaker_map = _load_map(assignments_path)
                    logger.info(f"Using existing speaker assignments from {assignments_path}")
                    run_review = False
                else:
                    logger.warning(
                        f"Speaker assignments in {assignments_path} are stale "
                        f"(transcript changed). Re-running speaker review."
                    )

            if run_review:
                logger.info("Starting interactive speaker review...")
                assigned = _interactive_speaker_review(
                    result, loaded_speaker_map, assignments_path,
                    transcript_path=transcript_out,
                )
                if assigned:
                    loaded_speaker_map = assigned

    # Step 1.5: Keyframes
    kf_ms = None
    vid_duration_ms = None
    try:
        logger.info("[Step 1.5/4] Processing Video Keyframes & Metadata...")
        from autosub.pipeline.video.keyframes import (
            get_fps,
            extract_keyframes as extract_keyframes_func,
            parse_aegisub_keyframes,
        )

        fps = get_fps(video_path)
        if fps > 0:
            import ffmpeg

            probe = ffmpeg.probe(str(video_path))
            try:
                vid_duration_ms = int(float(probe["format"]["duration"]) * 1000)
            except Exception:
                pass

            if keyframes and keyframes.exists():
                logger.info(f"Using provided keyframes: {keyframes}")
                kf_ms = parse_aegisub_keyframes(keyframes, fps)
            elif extract_keyframes:
                kf_out = out_dir / f"{video_path.stem}_keyframes.log"
                logger.info(f"Extracting keyframes to {kf_out}...")
                extract_keyframes_func(video_path, kf_out)
                kf_ms = parse_aegisub_keyframes(kf_out, fps)
        else:
            logger.warning("No video stream found. Keyframe extraction disabled.")
    except Exception as e:
        logger.warning(f"Failed to process keyframes: {e}")

    # Step 2: Format
    try:
        logger.info("[Step 2/4] Formatting...")
        format_module.format_subtitles(
            transcript_out,
            original_ass_out,
            keyframes=kf_ms,
            video_duration_ms=vid_duration_ms,
            timing_config=final_timing,
            extensions_config=final_extensions,
            replacements=replacements,
            speaker_map=loaded_speaker_map,
        )
    except Exception as e:
        logger.error(f"Failed during formatting: {e}")
        raise typer.Exit(code=1)

    # Step 3: Translate
    chunk_size = chunk_size if chunk else 0

    try:
        logger.info("[Step 3/4] Translating...")
        translate_module.translate_subtitles(
            original_ass_out,
            translated_ass_out,
            engine=engine,
            system_prompt=final_prompt,
            target_lang=target_lang,
            source_lang=source_lang,
            bilingual=bilingual,
            model=vertex_model,
            location=vertex_location,
            chunk_size=chunk_size,
            corner_names=final_corner_names or None,
            corner_cues=final_corner_cues or None,
            debug=mark_chunks,
            log_dir=translate_log_dir,
        )
    except Exception as e:
        logger.error(f"Failed during translation: {e}")
        raise typer.Exit(code=1)

    # Step 4: Postprocess
    try:
        logger.info("[Step 4/4] Postprocessing...")
        postprocess_module.postprocess_subtitles(
            translated_ass_out,
            extensions_config=final_extensions,
            bilingual=bilingual,
        )
    except Exception as e:
        logger.error(f"Failed during postprocessing: {e}")
        raise typer.Exit(code=1)

    # Link video in ASS files for Aegisub
    from autosub.pipeline.format.generator import inject_aegisub_metadata
    inject_aegisub_metadata(original_ass_out, video_path)
    inject_aegisub_metadata(translated_ass_out, video_path)

    logger.info(
        f"Pipeline completed successfully! Final output saved to {translated_ass_out}"
    )


if __name__ == "__main__":
    app()
