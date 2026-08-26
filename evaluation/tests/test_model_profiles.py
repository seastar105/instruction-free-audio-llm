from audio_lfm_eval.model_profiles import (
    load_model_profiles,
    validate_reference_scores,
)
from audio_lfm_eval.server import ServerSettings, build_server_command


def test_reference_models_are_immutable_and_explicitly_preflighted() -> None:
    profiles = load_model_profiles()
    assert profiles["lfm25-audio-en-reference"].revision == (
        "c362a0625dfe45aa588dce5f0ada28a7e5707628"
    )
    assert profiles["lfm25-audio-jp-reference"].revision == (
        "6c34b4d590f80563f8cb2939c2ebd7686d952394"
    )
    assert profiles["qwen25-omni-3b-smoke"].revision == (
        "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    )
    assert profiles["qwen25-omni-3b-smoke"].vllm_supported
    assert not profiles["lfm25-audio-en-reference"].vllm_supported
    assert not profiles["lfm25-audio-jp-reference"].vllm_supported
    for profile in profiles.values():
        command = build_server_command(
            vllm_executable="vllm",
            model_export=profile.model_id,
            model_name=profile.name,
            settings=ServerSettings(),
            revision=profile.revision,
            load_format=None,
        )
        assert command[:3] == ["vllm", "serve", profile.model_id]
        assert command[command.index("--revision") + 1] == profile.revision


def test_reference_range_check_separates_pass_fail_and_missing() -> None:
    profile = load_model_profiles()["lfm25-audio-jp-reference"]
    result = validate_reference_scores(
        profile,
        {
            "voicebench-ja.elyza": 2.2,
            "voicebench-ja.spoken-elyza": 4.9,
        },
    )
    assert result["voicebench-ja.elyza"]["status"] == "pass"
    assert result["voicebench-ja.spoken-elyza"]["status"] == "fail"
    assert result["voicebench-ja.m-ifeval"]["status"] == "missing"
