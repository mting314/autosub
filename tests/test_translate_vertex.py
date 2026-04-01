import pytest

from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.pipeline.translate.translator import VertexTranslator


def test_vertex_prompt_includes_line_ending_style_guidance():
    translator = VertexTranslator(
        project_id="test-project",
        source_lang="ja",
        target_lang="en",
        system_prompt="Keep the host warm and conversational.",
    )

    instruction = translator._get_system_instruction(0)

    assert (
        "Prefer ending subtitle lines on natural punctuation whenever possible"
        in instruction
    )
    assert "Move trailing connectives such as 'but', 'and', 'so'" in instruction
    assert "Speaker and style context:" in instruction
    assert "Keep the host warm and conversational." in instruction


def test_vertex_translator_accepts_model_and_location_overrides():
    translator = VertexTranslator(
        project_id="test-project",
        model="gemini-custom",
        location="us-central1",
        reasoning_effort=ReasoningEffort.HIGH,
    )

    assert translator.model == "gemini-custom"
    assert translator.location == "us-central1"
    assert translator.reasoning_effort == ReasoningEffort.HIGH


def test_vertex_translator_defaults_to_anthropic_haiku():
    translator = VertexTranslator(
        project_id=None,
        provider="anthropic",
    )

    assert translator.model == "claude-haiku-4-5"


def test_vertex_translator_defaults_to_medium_reasoning_effort():
    translator = VertexTranslator(project_id="test-project")

    assert translator.reasoning_effort == ReasoningEffort.MEDIUM


def test_google_reasoning_effort_maps_to_thinking_config():
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-3-flash-preview",
        reasoning_effort=ReasoningEffort.LOW,
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"] == {"thinking_level": "LOW"}


def test_google_reasoning_effort_maps_medium_for_gemini_3():
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-3-flash-preview",
        reasoning_effort=ReasoningEffort.MEDIUM,
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"] == {"thinking_level": "MEDIUM"}


def test_google_reasoning_effort_maps_minimal_for_supported_gemini_3_model():
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-3-flash-preview",
        reasoning_effort=ReasoningEffort.MINIMAL,
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"] == {"thinking_level": "MINIMAL"}


def test_google_reasoning_effort_maps_to_budget_for_gemini_2_5():
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-2.5-flash",
        reasoning_effort=ReasoningEffort.MEDIUM,
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"] == {"thinking_budget": 8192}


def test_google_reasoning_budget_maps_to_level_for_gemini_3():
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-3-flash-preview",
        reasoning_budget_tokens=1024,
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"] == {"thinking_level": "LOW"}


def test_google_reasoning_dynamic_maps_to_budget_for_gemini_2_5():
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-2.5-flash",
        reasoning_dynamic=True,
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"] == {"thinking_budget": -1}


def test_google_settings_enable_include_thoughts_when_trace_path_is_set(tmp_path):
    llm = BaseStructuredLLM(
        project_id="test-project",
        model="gemini-3-flash-preview",
        trace_path=tmp_path / "trace.jsonl",
    )

    settings = llm._build_google_model_settings(llm._get_model_config())

    assert settings["google_thinking_config"]["include_thoughts"] is True


def test_anthropic_reasoning_effort_maps_to_unified_thinking():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.LOW,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] == "low"
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 16384


def test_anthropic_reasoning_off_disables_thinking():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.OFF,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] is False


def test_anthropic_reasoning_budget_maps_to_anthropic_thinking():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_budget_tokens=2048,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["anthropic_thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
    }
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 4096


def test_anthropic_medium_reasoning_uses_higher_budget_and_double_max_tokens():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.MEDIUM,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] == "medium"
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 32768


def test_anthropic_minimal_reasoning_uses_requested_budget_and_max_tokens():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.MINIMAL,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] == "minimal"
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 16384


def test_anthropic_high_reasoning_uses_requested_budget_and_max_tokens():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.HIGH,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] == "high"
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 65536


def test_anthropic_adaptive_models_use_same_max_tokens_policy():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-sonnet-4-6",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.MEDIUM,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] == "medium"
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 32768


def test_anthropic_temperature_is_preserved_when_thinking_is_off():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_effort=ReasoningEffort.OFF,
        temperature=0.2,
    )

    settings = llm._build_anthropic_model_settings(llm._get_model_config())

    assert settings["thinking"] is False
    assert settings["temperature"] == 0.2


def test_anthropic_reasoning_dynamic_is_rejected():
    llm = BaseStructuredLLM(
        project_id=None,
        model="claude-haiku-4-5",
        provider="anthropic",
        reasoning_dynamic=True,
    )

    with pytest.raises(ValueError, match="does not support reasoning_dynamic"):
        llm._build_anthropic_model_settings(llm._get_model_config())
