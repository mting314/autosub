# autosub

Automatic Japanese subtitle generation and translation pipeline for speech-heavy video and audio.

## Current Pipeline

`autosub` currently runs a four-stage CLI pipeline:

1. **Transcribe**: Extract audio, send it to Google Cloud Speech-to-Text (`chirp_2`), and write a word-timed `transcript.json`.
2. **Format**: Chunk words into subtitle lines, optionally apply discourse-aware radio segmentation, apply timing and optional keyframe snapping, and write `original.ass`.
3. **Translate**: Translate subtitle events with either Vertex AI (`gemini-3-flash-preview`) or Cloud Translation v3, then write `translated.ass`.
4. **Postprocess**: Apply profile-driven editorial cleanup to the translated `.ass` file. The built-in `run` command includes this step automatically.

```mermaid
graph TD
    A[Video or Audio Input] --> B[Transcribe<br/>Google Speech-to-Text chirp_2]
    B --> C[Format<br/>Chunking + Timing + Optional Keyframe Snapping]
    C --> D[Translate<br/>Vertex Gemini 3 Flash or Cloud Translation v3]
    D --> E[Postprocess<br/>Profile Extensions]
    E --> F[Final translated.ass]
```

## Current Capabilities

- Word-level transcript timing from Google Speech-to-Text.
- Automatic short-audio local transcription and long-audio GCS batch transcription.
- Subtitle timing cleanup with minimum-duration padding, gap snapping, and optional keyframe alignment.
- Radio-show discourse extensions that can split listener mail framing and label subtitle roles.
- Optional bilingual output with original Japanese stacked above the translation.
- Profile inheritance for prompts, vocabulary, timing, and extensions.

## Current Limits

- The CLI is still documented and exposed as a **single-speaker** pipeline.
- The transcript and formatter can preserve `speaker` labels if they are already present in `transcript.json`, and `.ass` generation will create per-speaker styles, but diarization is not wired through the transcription commands yet.
- The formatter does **not** currently insert ASS line breaks (`\N`). Layout helpers exist in the codebase, but profile options such as `max_line_width` and `max_lines_per_subtitle` are not currently consumed by the CLI pipeline.

## Prerequisites

1. Python 3.12+
2. `uv`
3. FFmpeg available on `PATH`
4. Google Cloud credentials with:
   - `GOOGLE_APPLICATION_CREDENTIALS`
   - `GOOGLE_CLOUD_PROJECT`
   - `AUTOSUB_GCS_BUCKET` for audio longer than about 60 seconds
5. Optional: `SCXvid` if you want automatic keyframe extraction for scene-aware timing

## Installation

```powershell
git clone https://github.com/yourusername/autosub.git
Set-Location autosub
uv sync
```

## Configuration

Create a `.env` file in the repo root:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
AUTOSUB_GCS_BUCKET=your-staging-bucket
```

## Quick Start

Run the full pipeline:

```powershell
uv run autosub run .\video.mp4 --profile suzuhara_nozomi
```

For bilingual output:

```powershell
uv run autosub run .\video.mp4 --profile suzuhara_nozomi --bilingual
```

By default, `run` writes these files next to the input media:

- `transcript.json`
- `original.ass`
- `original.llm_trace.jsonl` when radio-discourse classification uses an LLM-backed engine
- `translated.ass`
- `translated.llm_trace.jsonl` when translation uses the Vertex LLM engine

If keyframe extraction is enabled and succeeds, it also writes `<input-stem>_keyframes.log`.

## Running Stages Individually

Transcribe:

```powershell
uv run autosub transcribe .\video.mp4 `
  --out .\transcript.json `
  --profile suzuhara_nozomi
```

Format with an existing keyframe log:

```powershell
uv run autosub format .\transcript.json `
  --out .\original.ass `
  --keyframes .\video_keyframes.log `
  --fps 23.976 `
  --profile suzuhara_nozomi
```

Translate with Vertex AI:

```powershell
uv run autosub translate .\original.ass `
  --out .\translated.ass `
  --profile suzuhara_nozomi `
  --engine vertex `
  --vertex-reasoning-effort low `
  --vertex-reasoning-budget 1024 `
  --bilingual
```

Postprocess a translated file explicitly:

```powershell
uv run autosub postprocess .\translated.ass `
  --profile suzuhara_nozomi `
  --bilingual
```

## Command Reference

### `autosub transcribe`

- `--out`, `-o`: Output transcript path. Default: `transcript.json`
- `--language`, `-l`: Speech recognition language code. Default: `ja-JP`
- `--vocab`, `-v`: Additional speech adaptation hints. Can be passed multiple times.
- `--profile`: Loads profile vocabulary.

Behavior notes:

- Audio shorter than about 60 seconds is sent directly to the API.
- Longer audio is uploaded to GCS first and transcribed as a batch job.

### `autosub format`

- `--out`: Output `.ass` path. Default: `original.ass` in the transcript directory
- `--keyframes`: Path to an Aegisub keyframe log
- `--fps`: Required when `--keyframes` is used
- `--profile`: Loads `[timing]` and `[extensions]`

Behavior notes:

- Chunking is punctuation- and pause-aware.
- If speaker labels are already present in the transcript JSON, chunking is done per speaker and the generated `.ass` file gets one style per speaker.
- The `radio_discourse` extension runs here when enabled.

### `autosub translate`

- `--out`: Output `.ass` path. Default: `translated.ass`
- `--engine`, `-e`: `vertex` or `cloud-v3`
- `--prompt`, `-p`: Extra translation guidance appended after profile prompts
- `--profile`: Loads prompt text from the selected profile
- `--target`: Target language code. Default: `en`
- `--source`: Source language code. Default: `ja`
- `--vertex-model`: Override the default Vertex model
- `--vertex-location`: Override the default Vertex region
- `--vertex-reasoning-effort`: Provider-agnostic reasoning effort for Vertex-backed LLM calls. Current Google support varies by model family and can include `off`, `minimal`, `low`, `medium`, `high`
- `--vertex-reasoning-budget`: Optional token-budget override. For Gemini 2.5 this is passed as thinking budget; for level-only Gemini families it is converted heuristically
- `--vertex-reasoning-dynamic` / `--no-vertex-reasoning-dynamic`: Request dynamic reasoning budget on supported model families
- `--bilingual` / `--replace`: Stack Japanese above the translation, or replace text entirely
- `--chunk-size`: Number of subtitle lines per translation chunk. Use `0` to disable chunking. Default: `0`

Behavior notes:

- `vertex` uses Vertex AI with `gemini-3-flash-preview`.
- `cloud-v3` uses Google Cloud Translation v3 and ignores custom prompt text.

### `autosub postprocess`

- `--out`: Output `.ass` path. Default: overwrite the input file
- `--profile`: Loads `[extensions]`
- `--bilingual` / `--replace`: Tells postprocess whether it is operating on stacked bilingual text or translated-only text

Behavior notes:

- Postprocessing only changes files when an enabled extension actually makes edits.
- The built-in `run` command postprocesses `translated.ass` in place.

### `autosub run`

`run` combines the full pipeline above and keeps the common end-to-end options:

- `--out-dir`
- `--language`
- `--profile`
- `--vocab`
- `--prompt`
- `--target`
- `--source`
- `--vertex-reasoning-effort`
- `--bilingual` / `--replace`
- `--keyframes`
- `--extract-keyframes` / `--no-extract-keyframes`
- `--start`
- `--end`
- `--chunk-size`

Behavior notes:

- `run` always uses the default Vertex translation path.
- If you need `cloud-v3` or advanced Vertex overrides such as model, location, or dynamic reasoning settings, run the stages separately and use `autosub translate`.

## Unified Profile Format

Profiles live in [`profiles`](./profiles) and are loaded by name with `--profile <name>`.

Example:

```toml
extends = ["solo_seiyuu_radio"]

prompt = "prompts/suzuhara_nozomi.md"
vocab = [
    "鈴原希実",
    "のんちゃん",
]

[timing]
min_duration_ms = 700
snap_threshold_ms = 250
conditional_snap_threshold_ms = 500

[extensions.radio_discourse]
enabled = true
engine = "hybrid"
model = "gemini-3-flash-preview"
reasoning_effort = "low"
reasoning_budget_tokens = 1024
scope = "full_script"
window_size = 10
window_overlap = 3
split_framing_phrases = true
label_roles = true
```

### Profile Keys

- `extends`: List of base profile names. Base profiles are loaded first.
- `prompt`: Either inline text or a path ending in `.md` or `.txt`. File contents are loaded into the translation prompt.
- `vocab`: List of speech adaptation hints. Inherited lists are appended.
- `[timing]`: Timing options for the formatter.
- `[extensions]`: Nested extension configuration shared by formatting and postprocessing.

### Prompt and Vocab Merge Rules

- Prompt fragments are concatenated in inheritance order: base profile first, child profile after that, then CLI `--prompt` last.
- Vocabulary entries are appended in the same order: base profile, child profile, then CLI `--vocab`.

### Timing Options Currently Wired Up

These keys are currently consumed by the formatter:

- `min_duration_ms`
- `snap_threshold_ms`
- `conditional_snap_threshold_ms`

No layout-related profile keys are currently wired into the CLI formatter.

## `radio_discourse` Extension

The built-in `radio_discourse` extension is designed for solo seiyuu radio or listener-mail style content.

During **formatting**, it can:

- split trailing framing phrases such as `といただきました。` into their own subtitle lines
- classify lines as `host`, `listener_mail`, or `host_meta`
- store the resolved role in the ASS event `Name` field

During **postprocessing**, it can:

- wrap translated `listener_mail` lines in quotation marks
- in bilingual mode, quote only the translated bottom line and leave the original Japanese untouched

Supported options:

- `enabled`: Turn the extension on
- `engine`: `rules`, `vertex`, or `hybrid`
- `model`: Vertex model name for `vertex` or `hybrid`
- `reasoning_effort`: Provider-agnostic reasoning effort for LLM-backed classification. Current Google support varies by model family and can include `off`, `minimal`, `low`, `medium`, `high`
- `reasoning_budget_tokens`: Optional token-budget override. For Gemini 2.5 this maps directly to thinking budget; for level-only Gemini families it is converted heuristically
- `reasoning_dynamic`: Request dynamic reasoning budget when the selected model family supports it
- `location`: Vertex region. Default: `us-central1`
- `scope`: `full_script` or windowed classification
- `window_size`: Window size for non-`full_script` classification
- `window_overlap`: Window overlap for non-`full_script` classification
- `split_framing_phrases`: Split host framing suffixes into separate lines before classification
- `label_roles`: Persist the resolved role onto subtitle events

`hybrid` uses rule-based labels first and falls back to them if Vertex classification fails.
