"""Combined radio_discourse + corners classifier.

When both extensions use LLM/hybrid engine, this module runs a single LLM call
that classifies discourse roles AND detects corner transitions simultaneously.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from autosub.core.errors import VertexResponseShapeError
from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.core.schemas import SubtitleLine
from autosub.extensions.radio_discourse.classifier import ROLE_VALUES

logger = logging.getLogger(__name__)


class CombinedDecision(BaseModel):
    id: int
    role: Literal["host", "listener_mail", "host_meta"]
    corner: str | None = None
    speaker: str | None = None


class CombinedClassifier(BaseStructuredLLM):
    DEFAULT_MODELS = {
        "google-vertex": "gemini-2.5-flash-lite",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
    }

    def __init__(
        self,
        *,
        project_id: str | None,
        segments: list[dict],
        model: str | None = None,
        location: str = "global",
        temperature: float = 0.1,
        provider: str = "google-vertex",
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
        reasoning_budget_tokens: int | None = None,
        reasoning_dynamic: bool | None = None,
        provider_options: dict[str, object] | None = None,
        trace_path: Path | str | None = None,
        speaker_names: list[str] | None = None,
    ):
        resolved_model = model or self.DEFAULT_MODELS.get(
            provider, "gemini-2.5-flash-lite"
        )
        super().__init__(
            project_id=project_id,
            model=resolved_model,
            location=location,
            temperature=temperature,
            provider=provider,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_dynamic=reasoning_dynamic,
            provider_options=provider_options,
            trace_path=trace_path,
        )
        self._segments = segments
        self._valid_corner_names = {s["name"] for s in segments}
        self._speaker_names = list(speaker_names or [])
        self._valid_speaker_names = set(self._speaker_names)

    def _get_system_instruction(self, num_lines: int) -> str:
        segments_text = ""
        for seg in self._segments:
            segments_text += f"- {seg['name']}: {seg.get('description', '')}\n"
            if seg.get("cues"):
                segments_text += f"  Common cue phrases: {', '.join(seg['cues'])}\n"

        speaker_task = ""
        speaker_field = ""
        if self._speaker_names:
            speaker_task = (
                "TASK 3 — Speaker Attribution:\n"
                "Each line arrives with a 'diarized' field: the speaker the audio\n"
                "diarizer guessed. Diarization is acoustic and frequently wrong at\n"
                "turn boundaries. Correct it from the conversation itself.\n\n"
                "Speakers in this recording:\n"
                + "".join(f"- {name}\n" for name in self._speaker_names)
                + "\n"
                "Speaker attribution rules:\n"
                "1. A personal name spoken inside a line is almost never the speaker\n"
                "   of that line. It is who they are addressing or talking about.\n"
                "   In 「じゃあ鈴木さんはどうですか」 the speaker is NOT Suzuki — she is\n"
                "   asking Suzuki a question, so Suzuki most likely speaks the NEXT line.\n"
                "2. A direct question is usually followed by a turn change: the person\n"
                "   addressed answers next.\n"
                "3. First-person self-reference identifies the speaker. Someone saying\n"
                "   their own name in a self-introduction is the speaker of that line.\n"
                "4. Keep the diarized value unless the conversation clearly contradicts\n"
                "   it. Prefer the diarizer when the evidence is weak; it hears the\n"
                "   voices and you do not.\n"
                "5. 'speaker' must be exactly one of the names listed above, or null\n"
                "   when you cannot tell.\n\n"
            )
            speaker_field = ", and 'speaker'"

        task_count = "THREE" if self._speaker_names else "TWO"
        field_count = "four" if self._speaker_names else "three"

        return (
            "You are analyzing a Japanese voice-actress radio show transcript.\n"
            f"Your output must consist of exactly {num_lines} items.\n\n"
            f"You have {task_count} tasks for each subtitle line:\n\n"
            "TASK 1 — Discourse Role Classification:\n"
            "Classify each line as one of three discourse roles.\n"
            "Role definitions:\n"
            "1. host: the host's own live speech, reactions, commentary, ad-libbing, or monologue.\n"
            "2. listener_mail: text from a listener message, question, or submission that the host is reading aloud.\n"
            "3. host_meta: the host's framing around listener mail, such as introducing a message, naming the sender, or quotative wrap-up like といただきました.\n\n"
            "Role classification rules:\n"
            "1. Use neighboring lines for context. This is sequential dialogue from one radio episode.\n"
            "2. Do not rewrite or normalize the text. Only classify each input line.\n"
            "3. A listener-mail block can span multiple consecutive lines.\n"
            "4. Short host reactions like おお。ありがとう。なるほど。 are usually host.\n"
            "5. Lines that introduce or close out a listener message are usually host_meta, not listener_mail.\n"
            "6. If a line contains only a quotative wrap-up such as といただきました, label it host_meta.\n"
            "7. Greeting edge case: if a listener greeting is immediately echoed or answered by the host in the next line, the echo/answer is host, not listener_mail.\n\n"
            "TASK 2 — Segment (Corner) Detection:\n"
            "Identify when the program transitions from one recurring segment to another.\n"
            "If a line marks the START of a new segment, set 'corner' to that segment's name.\n"
            "Otherwise, set 'corner' to null.\n\n"
            f"Segments in this program:\n{segments_text}\n"
            "Corner detection rules:\n"
            "1. Only mark the FIRST line of a new segment, not every line within it.\n"
            "2. A transition can be signaled by an explicit cue phrase, a change in topic, or the host introducing a new segment.\n"
            "3. Do not mark a segment that is already in progress.\n"
            "4. Each segment transition should appear AT MOST ONCE. If you have already marked a line as starting a segment, do not mark any subsequent lines as starting the same segment — they are continuations, not new transitions.\n"
            "5. If unsure whether a transition occurred, prefer null.\n"
            "6. 'corner' must be one of the segment names listed above, or null.\n\n"
            f"{speaker_task}"
            "Output format:\n"
            "1. Return valid JSON only.\n"
            "2. Return the exact same number of items as the input.\n"
            f"3. Each item must contain exactly {field_count} fields: 'id', 'role', "
            f"'corner'{speaker_field}.\n"
        )

    def classify_window(
        self, lines: list[tuple[int, SubtitleLine]]
    ) -> dict[int, tuple[str, str | None, str | None]]:
        """Classify a window of lines, returning (role, corner, speaker) per line."""
        if not lines:
            return {}

        system_instruction = self._get_system_instruction(len(lines))
        payload = []
        for line_id, line in lines:
            item: dict[str, object] = {"id": line_id, "text": line.text}
            if self._speaker_names:
                # The acoustic guess goes in as a prior, not as ground truth.
                item["diarized"] = line.speaker
            payload.append(item)
        contents = json.dumps(payload, ensure_ascii=False, indent=2)

        decisions, diagnostics = self._run_structured_output(
            user_prompt=contents,
            system_prompt=system_instruction,
            output_type=list[CombinedDecision],
            operation_name="LLM combined classifier",
            output_name="combined_decisions",
        )

        try:
            ordered = sorted(decisions, key=lambda item: item.id)
            returned_ids = [item.id for item in ordered]

            if returned_ids != [line_id for line_id, _ in lines]:
                raise ValueError(f"returned ids were {returned_ids!r}")

            result: dict[int, tuple[str, str | None, str | None]] = {}
            for item in ordered:
                corner = item.corner
                if corner and corner not in self._valid_corner_names:
                    corner = None
                speaker = item.speaker
                if speaker and speaker not in self._valid_speaker_names:
                    speaker = None
                result[item.id] = (item.role, corner, speaker)
            return result
        except Exception as exc:
            raise VertexResponseShapeError(
                "LLM combined classifier returned JSON with an unexpected structure: "
                f"{exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc


def classify_combined(
    lines: list[SubtitleLine],
    fallback_roles: list[str | None],
    segments: list[dict],
    config: dict,
    speaker_names: list[str] | None = None,
) -> tuple[list[str | None], list[str | None], list[str | None]]:
    """Run combined classification, returning (roles, corners, speakers) lists.

    Speakers are only attributed when speaker_names is supplied; otherwise the
    returned speaker list is all None and the prompt never mentions the task, so
    shows that do not need the correction pay nothing for it.
    """
    if not lines:
        return [], [], []

    provider = config.get("provider", "google-vertex")
    project_id = config.get("project_id")
    if provider == "google-vertex" and not project_id:
        raise ValueError(
            "Combined classifier Vertex mode requires a Google Cloud project id."
        )

    classifier = CombinedClassifier(
        project_id=project_id,
        segments=segments,
        model=config.get("model"),
        location=config.get("location", "global"),
        provider=provider,
        reasoning_effort=config.get("reasoning_effort", ReasoningEffort.MEDIUM),
        reasoning_budget_tokens=config.get("reasoning_budget_tokens"),
        reasoning_dynamic=config.get("reasoning_dynamic"),
        provider_options=config.get("provider_options"),
        trace_path=config.get("llm_trace_path"),
        speaker_names=speaker_names,
    )

    windows = _build_windows(lines, config)

    role_votes: dict[int, list[str]] = {i: [] for i in range(len(lines))}
    corner_votes: dict[int, list[str | None]] = {i: [] for i in range(len(lines))}
    speaker_votes: dict[int, list[str]] = {i: [] for i in range(len(lines))}

    for window in windows:
        decisions = classifier.classify_window(window)
        for line_id, (role, corner, speaker) in decisions.items():
            if 0 <= line_id < len(lines):
                if role in ROLE_VALUES:
                    role_votes[line_id].append(role)
                corner_votes[line_id].append(corner)
                if speaker:
                    speaker_votes[line_id].append(speaker)

    # Resolve roles (same logic as radio_discourse)
    resolved_roles: list[str | None] = []
    for i in range(len(lines)):
        resolved_roles.append(_resolve_role(role_votes[i], fallback_roles[i]))

    # Resolve corners (any non-null vote wins)
    resolved_corners: list[str | None] = []
    for i in range(len(lines)):
        non_null = [v for v in corner_votes[i] if v is not None]
        if non_null:
            counts = Counter(non_null)
            resolved_corners.append(counts.most_common(1)[0][0])
        else:
            resolved_corners.append(None)

    # Resolve speakers. The diarizer's label stands unless the model named a
    # different valid speaker; it heard the voices and the model only read the
    # words, so it wins ties and abstentions.
    resolved_speakers: list[str | None] = []
    overrides = 0
    for i, line in enumerate(lines):
        votes = speaker_votes[i]
        if not votes:
            resolved_speakers.append(line.speaker)
            continue
        winner = Counter(votes).most_common(1)[0][0]
        resolved_speakers.append(winner)
        if line.speaker and winner != line.speaker:
            overrides += 1

    if speaker_names:
        logger.info(
            "Speaker attribution reassigned %d of %d line(s) from the diarizer.",
            overrides,
            len(lines),
        )

    return resolved_roles, resolved_corners, resolved_speakers


def _build_windows(
    lines: list[SubtitleLine], config: dict
) -> list[list[tuple[int, SubtitleLine]]]:
    scope = str(config.get("scope", "full_script")).lower()
    if scope == "full_script":
        return [[(i, line) for i, line in enumerate(lines)]]

    window_size = max(int(config.get("window_size", 10)), 1)
    window_overlap = max(int(config.get("window_overlap", 3)), 0)

    step = max(window_size - window_overlap, 1)
    windows: list[list[tuple[int, SubtitleLine]]] = []
    start = 0
    while start < len(lines):
        end = min(start + window_size, len(lines))
        windows.append([(i, lines[i]) for i in range(start, end)])
        if end >= len(lines):
            break
        start += step
    return windows


def _resolve_role(votes: list[str], fallback_role: str | None) -> str | None:
    if not votes:
        return fallback_role

    counts = Counter(votes)
    top_count = max(counts.values())
    top_roles = {role for role, count in counts.items() if count == top_count}

    if fallback_role in top_roles:
        return fallback_role

    for role in ("host_meta", "listener_mail", "host"):
        if role in top_roles:
            return role

    return fallback_role
