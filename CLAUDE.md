# CLAUDE.md

Automatic Japanese subtitle generation and translation pipeline. Fork of `ahuei123456/autosub`, hosted at `mting314/autosub`.

## Quick Reference

- **Language**: Python 3.12+, managed with `uv`
- **CLI framework**: Typer, entry point `autosub.cli:app`
- **Run**: `uv run autosub <command>`
- **Tests**: `uv run pytest` (or `uv run pytest tests/<file>` for a subset)
- **Lint**: `uv run ruff check`

## Project Layout

```
autosub/
├── cli.py                     # Typer CLI commands
├── core/
│   ├── llm/                   # LLM provider abstraction (Vertex, Anthropic, OpenAI, OpenRouter)
│   ├── profile.py             # TOML profile loader with inheritance (extends)
│   ├── speaker_map.py         # Speaker map TOML parsing and assignment
│   └── ...
├── extensions/
│   ├── corners/               # Corner (program segment) detection
│   └── radio_discourse/       # Radio show discourse classification
├── pipeline/
│   ├── transcribe/            # Google STT (Chirp 2/3) and WhisperX backends
│   ├── format/                # Transcript → ASS subtitle formatting
│   ├── translate/             # LLM-based subtitle translation
│   ├── postprocess/           # Profile-driven editorial cleanup
│   └── report/                # HTML review report generator
profiles/                      # TOML profiles (examples/ tracked, local/ gitignored)
prompts/                       # Prompt files (examples/ tracked, local/ gitignored)
tests/                         # pytest test suite
```

## Pipeline

1. **Transcribe** - Extract audio, send to STT backend, produce `transcript.json`
2. **Format** - Chunk words into subtitle lines, apply timing/extensions, produce `original.ass`
3. **Translate** - LLM translation of subtitles, produce `translated.ass`
4. **Postprocess** - Profile-driven cleanup of translated output

`autosub run` executes all four stages. Each stage can also run independently.

## Key Concepts

- **Profiles**: TOML files with `extends` inheritance. Searched: `profiles/local/` > `profiles/examples/` > `profiles/`
- **Prompts**: Markdown/text files loaded by profiles. Searched: `prompts/local/` > `prompts/examples/` > `prompts/`
- **Extensions**: `radio_discourse` (listener mail classification) and `corners` (program segment detection) run at format/postprocess time
- **Speaker maps**: Per-project TOML files mapping diarization labels to character names/colors
- **Config**: `config.toml` (gitignored) provides default CLI flags per stage

## Chirp 3 Backend (this branch)

- `--backend chirp_3` selects Chirp 3 transcription
- Requires Opus encoding (WAV/AAC return empty results — observed behavior, not documented)
- Audio > 18 min is split into chunks and transcribed in parallel via `ThreadPoolExecutor`
- Chirp 3 returns bogus word timestamps at internal 18-min boundaries; `_clamp_word_timestamps` fixes these before offset is applied
- SpeechAdaptation PhraseSet is incompatible with `enable_word_time_offsets` on Chirp 3, so vocabulary hints are skipped (logged as warning)
- Chirp 2 remains the default and uses WAV (pcm_s16le) encoding

## Docker Remote Execution

`scripts/remote.sh` (gitignored) runs the full pipeline on a temporary GCP VM. It handles VM creation, Docker build, video staging via GCS, pipeline execution, result download, and VM cleanup.

### Usage

```bash
./scripts/remote.sh <video_path> <autosub_command> [autosub_args...]
```

> **Prerequisite:** the deployed branch (`AUTOSUB_GIT_BRANCH`, default = current branch) must contain the `Dockerfile` — i.e. be at or merged/rebased past PR #19 (Docker support). Feature branches cut earlier will fail the remote build; merge master first or run with `AUTOSUB_GIT_BRANCH=master`. The script pre-flights this and fails fast.

### What it does

1. Uploads video to `gs://subtitling-projects/remote-staging/`
2. Creates a GCE VM (`e2-medium`, Ubuntu 24.04, 30GB disk) in `us-west1-b` — sized to the measured footprint (translate peaks ~160MB RAM / ~0% CPU; all inference is remote)
3. Ensures IAP SSH firewall rule exists, installs Docker
4. Clones the repo (current branch), copies `profiles/local/` and `prompts/local/` via scp
5. Builds the Docker image (profiles baked in via `COPY . .`)
6. Pulls video from GCS, runs the pipeline
7. Downloads outputs back to the local project directory
8. Deletes VM and cleans up GCS staging

### Flags

- `--keep-vm`: Skip VM deletion (reuse on next run, pay ~$2/month disk when stopped)
- Environment overrides: `AUTOSUB_VM_NAME`, `AUTOSUB_VM_ZONE`, `AUTOSUB_VM_MACHINE_TYPE`, `AUTOSUB_VM_DISK_SIZE`, `AUTOSUB_GIT_BRANCH`, `AUTOSUB_GIT_REPO`

### GCP config

- Project: `future-name-201021` (from `.env`)
- GCS bucket: `subtitling-projects`
- SSH: IAP tunneling (`--tunnel-through-iap`) — required on networks that block direct SSH to GCE
- VM gets `cloud-platform` scope for GCS and Vertex API access

### Filenames with special characters

The script escapes single quotes in filenames for nested shell commands (e.g. "Won't"). Tested with ProSeka AfterTalk paths.

### Example (mizu6)

```bash
./scripts/remote.sh \
  "projects/projects/Project Sekai/Reaching Out to a Tomorrow That Won't Come Unraveled Aftertalk/Reaching Out to a Tomorrow That Won't Come Unraveled.mkv" \
  run --profile proseka/n25 --backend chirp_3 \
  --start 9:30 --end 15:00 \
  --start 24:45 --end 35:45 \
  --start 37:55 --end 55:00 \
  --chunk-size 80 --llm-reasoning-effort medium --mark-chunks --save-log
```

## Conventions

- Commit messages: imperative mood, lowercase, concise
- Branch per feature - don't tangle unrelated features
- Local/user-specific files go in gitignored dirs (`profiles/local/`, `prompts/local/`, `config.toml`)
- `scripts/remote.sh` is gitignored (contains hardcoded GCP config)
