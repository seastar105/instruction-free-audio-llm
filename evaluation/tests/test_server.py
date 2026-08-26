from audio_lfm_eval.server import ServerSettings, build_server_command


def test_server_command_is_persistent_openai_server() -> None:
    command = build_server_command(
        vllm_executable="vllm",
        model_export="export",
        model_name="audio-lfm-eval",
        settings=ServerSettings(
            api_server_count=4,
            max_num_seqs=64,
            max_num_batched_tokens=32768,
        ),
        allowed_local_media_path="evaluation-data",
    )
    assert command[:3] == ["vllm", "serve", "export"]
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert command[command.index("--api-server-count") + 1] == "4"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--limit-mm-per-prompt") + 1] == '{"audio":3}'
    assert command[command.index("--allowed-local-media-path") + 1].endswith(
        "/evaluation-data"
    )
    assert "--generation-config" in command


def test_server_audio_decode_limit_is_explicit() -> None:
    settings = ServerSettings(
        max_audio_decode_duration_seconds=720,
        audio_encoder_microbatch_size=16,
    )
    assert settings.max_audio_decode_duration_seconds == 720
    assert settings.audio_encoder_microbatch_size == 16
