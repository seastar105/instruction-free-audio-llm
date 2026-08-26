from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
processing = pytest.importorskip("audio_lfm.vllm_plugin.processing")
nn = torch.nn
AudioLfm2Processor = processing.AudioLfm2Processor
build_processing_classes = processing.build_processing_classes


class _Tokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [len(prompt)]


class _Info:
    def __init__(self) -> None:
        from transformers import WhisperFeatureExtractor

        config = SimpleNamespace(
            audio_sample_rate=16_000,
            max_audio_seconds=0.5,
            projector_config={"stack_factor": 4},
        )
        self.processor = AudioLfm2Processor(
            tokenizer=_Tokenizer(),
            feature_extractor=WhisperFeatureExtractor(),
            config=config,
        )

    def get_hf_processor(self, **kwargs: object) -> AudioLfm2Processor:
        return self.processor


def test_vllm_processor_keeps_long_chunks_and_multiple_items_distinct() -> None:
    _, _, processor_class = build_processing_classes()
    processor = object.__new__(processor_class)
    processor.info = _Info()
    result = processor._call_hf_processor(
        "prompt",
        {
            "audios": [
                np.zeros(18_001, dtype=np.float32),
                np.zeros(4_000, dtype=np.float32),
            ]
        },
        {},
        {},
    )
    assert result.input_ids == [[6]]
    assert len(result.audio_features) == 2
    assert [value.shape for value in result.audio_features] == [
        (80, 113),
        (80, 25),
    ]
    assert [value.tolist() for value in result.audio_chunk_length] == [
        [50, 50, 13],
        [25],
    ]
    assert result.audio_feature_length.tolist() == [113, 25]
    assert result.audio_token_length.tolist() == [17, 6]


class _FakeTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(80, 2, 1, bias=False)

    def forward(
        self, values: torch.Tensor, lengths: torch.Tensor
    ) -> list[torch.Tensor]:
        return [
            torch.full(((int(length) + 1) // 2, 2), float(index + 1))
            for index, length in enumerate(lengths.tolist())
        ]


class _FakeProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_start = nn.Parameter(torch.tensor([[-1.0, -1.0]]).squeeze(0))
        self.audio_end = nn.Parameter(torch.tensor([[-2.0, -2.0]]).squeeze(0))
        self.seen_lengths: list[int] = []

    def project_frames(self, hidden: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(hidden.shape[0])
        indices = torch.arange(0, hidden.shape[0], 4)
        return hidden.index_select(0, indices)


def test_vllm_model_concatenates_chunks_with_one_boundary_per_item() -> None:
    from audio_lfm.vllm_plugin.model import AudioLfm2ForConditionalGeneration

    model = object.__new__(AudioLfm2ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2, audio_encoder_microbatch_size=4)
    model.audio_tower = _FakeTower()
    model.multi_modal_projector = _FakeProjector()
    outputs = model.embed_multimodal(
        audio_features=[torch.zeros(80, 113), torch.zeros(80, 25)],
        audio_feature_length=torch.tensor([113, 25]),
        audio_chunk_length=[torch.tensor([50, 50, 13]), torch.tensor([25])],
        audio_token_length=torch.tensor([17, 6]),
    )
    assert [output.shape for output in outputs] == [(17, 2), (6, 2)]
    assert model.multi_modal_projector.seen_lengths == [57, 13]
    for output in outputs:
        assert output[0].tolist() == [-1.0, -1.0]
        assert output[-1].tolist() == [-2.0, -2.0]


def test_vllm_model_honors_audio_encoder_microbatch_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_lfm.vllm_plugin.model import AudioLfm2ForConditionalGeneration

    model = object.__new__(AudioLfm2ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2, audio_encoder_microbatch_size=1)
    model.audio_tower = _FakeTower()
    model.multi_modal_projector = _FakeProjector()
    monkeypatch.setenv("AUDIO_LFM_AUDIO_ENCODER_MICROBATCH_SIZE", "2")

    outputs = model.embed_multimodal(
        audio_features=[torch.zeros(80, 100)],
        audio_feature_length=torch.tensor([100]),
        audio_chunk_length=[torch.tensor([50, 50])],
        audio_token_length=torch.tensor([15]),
    )

    assert [output.shape for output in outputs] == [(15, 2)]
    assert model.multi_modal_projector.seen_lengths == [50]
