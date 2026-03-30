import json
import math
import tempfile
import types
import unittest
import wave
from pathlib import Path

import torch
from torch import nn

from audiox.data.ifcaps import (
    IFCapsFineTuneDataset,
    build_text_prompt,
    normalize_training_metadata,
    select_prompt_variant,
    serialize_ifcaps_to_xml,
)
from audiox.models.conditioners import MultiConditioner
from audiox.models.diffusion import ConditionedDiffusionModelWrapper
from audiox.training.diffusion import DiffusionCondTrainingWrapper
from audiox.training.finetune import apply_finetune_defaults


def _write_wav(path: Path, sample_rate: int, duration_seconds: float = 0.5) -> None:
    t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    waveform = 0.25 * torch.sin(2 * math.pi * 220 * t)
    stereo = torch.stack([waveform, waveform], dim=0)
    pcm = (stereo.t().reshape(-1).clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


class RecorderConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, device):
        self.calls.append(list(inputs))
        batch = len(inputs)
        return torch.zeros(batch, 1, 4), torch.ones(batch, 1)


class RecordingMAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = None

    def forward(self, video, text, audio):
        self.args = (video, text, audio)
        return {"video": video, "text": text, "audio": audio}


class DummyConditionedDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_objective = "v"
        self.pretransform = None
        self.conditioner = DummyConditioner()
        self.model = nn.Conv1d(2, 2, kernel_size=1)

    def forward(self, x, t, cond=None, cfg_dropout_prob=0.0, **kwargs):
        return self.model(x)


class DummyConditioner(nn.Module):
    def forward(self, batch_metadata, device):
        batch = len(batch_metadata)
        return {
            "video_prompt": (torch.zeros(batch, 2, 8, device=device), torch.ones(batch, 2, device=device)),
            "text_prompt": (torch.zeros(batch, 4, 8, device=device), torch.ones(batch, 4, device=device)),
            "audio_prompt": (torch.zeros(batch, 3, 8, device=device), torch.ones(batch, 3, device=device)),
        }


class IFCapsFineTuneTests(unittest.TestCase):
    def test_serializer_covers_category_count_order_and_timestamp(self):
        record = {
            "caption": "Crowd cheering before two dog barks.",
            "SED": [
                {"label": "crowd cheering", "start": 2, "end": 6},
                {"label": "dog bark", "start": 6.5, "end": 7.0},
                {"label": "dog bark", "start": 6.5, "end": 7.0},
            ],
            "time_relation": [{"first": "crowd cheering", "relation": "before", "second": "dog bark"}],
        }
        xml_prompt = serialize_ifcaps_to_xml(record)
        self.assertIn('<caption>Crowd cheering before two dog barks.</caption>', xml_prompt)
        self.assertIn('name="crowd cheering" start="2.0" end="6.0"', xml_prompt)
        self.assertIn('name="dog bark" start="6.5" end="7.0" count="2"', xml_prompt)
        self.assertIn("<order>crowd cheering before dog bark</order>", xml_prompt)

    def test_select_prompt_variant_uses_fixed_mixed_curriculum(self):
        variants = [select_prompt_variant(index) for index in range(8)]
        self.assertEqual(
            variants,
            ["natural", "natural", "xml", "xml_natural", "natural", "natural", "xml", "xml_natural"],
        )

    def test_build_text_prompt_supports_mixed_xml_plus_caption(self):
        record = {"caption": "A distant bell rings.", "category": ["bell"]}
        prompt = build_text_prompt(record, prompt_format="mixed", mixed_variant="xml_natural")
        self.assertIn("<audio>", prompt)
        self.assertIn("A distant bell rings.", prompt)

    def test_multi_conditioner_falls_back_to_legacy_prompt_key(self):
        recorder = RecorderConditioner()
        conditioner = MultiConditioner({"text_prompt": recorder}, default_keys={"text_prompt": "prompt"})
        conditioner([{"prompt": "legacy prompt"}], "cpu")
        self.assertEqual(recorder.calls[0], ["legacy prompt"])

    def test_apply_finetune_defaults_sets_text_fallback_and_t5_length(self):
        model_config = {
            "sample_rate": 44100,
            "sample_size": 485100,
            "model": {
                "conditioning": {
                    "configs": [
                        {"id": "text_prompt", "type": "t5", "config": {"t5_model_name": "t5-base", "max_length": 128}}
                    ]
                }
            },
        }
        patched = apply_finetune_defaults(model_config, text_max_length=256)
        self.assertEqual(
            patched["model"]["conditioning"]["default_keys"]["text_prompt"],
            "prompt",
        )
        self.assertEqual(
            patched["model"]["conditioning"]["configs"][0]["config"]["max_length"],
            256,
        )

    def test_maf_branch_order_matches_video_text_audio(self):
        wrapper = ConditionedDiffusionModelWrapper(
            model=nn.Identity(),
            conditioner=None,
            io_channels=2,
            sample_rate=16000,
            min_input_length=1,
            gate=True,
            gate_type="MAF",
            cross_attn_cond_ids=["video_prompt", "text_prompt", "audio_prompt"],
        )
        maf = RecordingMAF()
        wrapper.maf_block = maf
        conditioning_tensors = {
            "video_prompt": (torch.full((1, 2, 3), 1.0), torch.ones(1, 2)),
            "text_prompt": (torch.full((1, 4, 3), 2.0), torch.ones(1, 4)),
            "audio_prompt": (torch.full((1, 3, 3), 3.0), torch.ones(1, 3)),
        }
        wrapper.get_conditioning_inputs(conditioning_tensors)
        video, text, audio = maf.args
        self.assertTrue(torch.all(video == 1.0))
        self.assertTrue(torch.all(text == 2.0))
        self.assertTrue(torch.all(audio == 3.0))

    def test_dataset_emits_text_video_audio_and_padding_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            _write_wav(audio_path, sample_rate=16000)
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(json.dumps({"audio_path": str(audio_path), "caption": "Bell ring", "category": ["bell"]}) + "\n")

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="mixed",
                include_video_conditioning=False,
                include_audio_conditioning=False,
                video_fps=2,
            )
            audio, metadata = dataset[0]
            normalized = normalize_training_metadata({"prompt": "legacy"})

            self.assertEqual(audio.shape, (2, 8000))
            self.assertIn("text_prompt", metadata)
            self.assertEqual(metadata["video_prompt"]["video_tensors"].shape, (1, 1, 3, 224, 224))
            self.assertEqual(metadata["audio_prompt"].shape, (1, 2, 8000))
            self.assertEqual(metadata["padding_mask"].shape, (8000,))
            self.assertEqual(normalized["text_prompt"], "legacy")

    def test_training_wrapper_smoke_step_handles_1d_padding_masks(self):
        model = DummyConditionedDiffusion()
        wrapper = DiffusionCondTrainingWrapper(
            model,
            lr=1e-4,
            mask_padding=True,
            use_ema=False,
            optimizer_configs=None,
        )
        wrapper.log_loss_info = False
        wrapper.log_dict = lambda *args, **kwargs: None
        wrapper._trainer = types.SimpleNamespace(
            optimizers=[types.SimpleNamespace(param_groups=[{"lr": 1e-4}])]
        )

        batch = (
            torch.randn(2, 2, 16),
            [
                {"padding_mask": torch.ones(16), "text_prompt": "a"},
                {"padding_mask": torch.ones(16), "text_prompt": "b"},
            ],
        )
        loss = wrapper.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
