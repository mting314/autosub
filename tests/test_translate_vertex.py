from autosub.pipeline.translate.vertex import VertexTranslator


def test_vertex_prompt_includes_line_ending_style_guidance():
    translator = VertexTranslator(
        project_id="test-project",
        source_lang="ja",
        target_lang="en",
        system_prompt="Keep the host warm and conversational.",
    )

    prompt = translator._build_prompt(
        [
            "Also, my hair today... I wonder if you can tell. It might be hard to tell from a photo, but",
            "I made it a bit brown.",
        ]
    )

    assert (
        "Prefer ending subtitle lines on natural punctuation whenever possible"
        in prompt
    )
    assert "Move trailing connectives such as 'but', 'and', 'so'" in prompt
    assert "Speaker and style context:" in prompt
