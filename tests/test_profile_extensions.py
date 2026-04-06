from autosub.core.profile import load_unified_profile


def test_profile_extensions_are_inherited_and_overridden(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "base.toml").write_text(
        """
[format.extensions.radio_discourse]
enabled = true
split_framing_phrases = true
label_roles = false

[postprocess.extensions.radio_discourse]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    (profile_dir / "child.toml").write_text(
        """
extends = ["base"]

[format.extensions.radio_discourse]
label_roles = true
window_size = 8

[postprocess.extensions.radio_discourse]
preserve_quotes = true
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["format"]["extensions"]["radio_discourse"] == {
            "enabled": True,
            "split_framing_phrases": True,
            "label_roles": True,
            "window_size": 8,
        }
        assert data["postprocess"]["extensions"]["radio_discourse"] == {
            "enabled": True,
            "preserve_quotes": True,
        }
    finally:
        autosub.core.profile.Path = original_path
