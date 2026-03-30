import json
import os
import tempfile
import types
import unittest
import wave

import torch
import torchaudio

from audiox.data.ifcaps import (
    IFCapsFineTuneDataset,
    build_text_prompt,
    normalize_training_metadata,
    serialize_ifcaps_to_xml,
)
from audiox.models.conditioners import Conditioner, MultiConditioner
from audiox.models.diffusion import ConditionedDiffusionModel, ConditionedDiffusionModelWrapper
from audiox.training.diffusion import DiffusionCondTrainingWrapper
from audiox.training.finetune import apply_finetune_defaults


class DummyScalarConditioner(Conditioner):
    def __init__(self, output_dim: int = 4):
        super().__init__(dim=output_dim, output_dim=output_dim)

    def forward(self, inputs, device=None):
        embeddings = torch.ones((len(inputs), 1, self.output_dim), device=device)
        mask = torch.ones((len(inputs), 1), device=device, dtype=torch.bool)
        return embeddings, mask


class DummyTextConditioner(Conditioner):
    def __init__(self, output_dim: int = 4):
        super().__init__(dim=output_dim, output_dim=output_dim)

    def forward(self, texts, device=None):
        embeddings = []
        masks = []
        for text in texts:
            seq_len = max(1, min(3, len(str(text).split())))
            embeddings.append(torch.ones((seq_len, self.output_dim), device=device))
            masks.append(torch.ones((seq_len,), device=device, dtype=torch.bool))
        max_len = max(item.shape[0] for item in embeddings)
        padded = []
        padded_masks = []
        for embedding, mask in zip(embeddings, masks):
            if embedding.shape[0] < max_len:
                pad = torch.zeros((max_len - embedding.shape[0], self.output_dim), device=device)
                embedding = torch.cat([embedding, pad], dim=0)
                mask = torch.cat([mask, torch.zeros((max_len - mask.shape[0],), device=device, dtype=torch.bool)], dim=0)
            padded.append(embedding)
            padded_masks.append(mask)
        return torch.stack(padded, dim=0), torch.stack(padded_masks, dim=0)


class DummyDiffusionModel(ConditionedDiffusionModel):
    def __init__(self):
        super().__init__(supports_cross_attention=True)
        self.proj = torch.nn.Conv1d(2, 2, kernel_size=1)

    def forward(self, x, t, **kwargs):
        return self.proj(x)


class RecordingMAF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, video_tokens, text_tokens, audio_tokens):
        self.seen = (video_tokens, text_tokens, audio_tokens)
        return {
            "video": video_tokens,
            "text": text_tokens,
            "audio": audio_tokens,
        }


class IFCapsFineTuneTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
        pcm = waveform.clamp(-1, 1).mul(32767).to(torch.int16).t().contiguous().numpy()
        with wave.open(path, "wb") as handle:
            handle.setnchannels(waveform.shape[0])
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    def test_xml_serializer_handles_counts_timestamps_and_order(self):
        record = {
            "caption": "Brief cheering followed by two distant barks.",
            "category": {"crowd cheering": None, "dog bark": 2},
            "SED": [
                {"name": "crowd cheering", "start": 2.0, "end": 6.0},
                {"name": "dog bark", "start": 6.5, "end": 7.0}
            ],
            "time_relation": "crowd cheering before dog bark"
        }
        xml = serialize_ifcaps_to_xml(record, compact=True, include_caption=True, max_events=8)
        self.assertIn("<audio>", xml)
        self.assertIn('<caption>Brief cheering followed by two distant barks.</caption>', xml)
        self.assertIn('name="crowd cheering" start="2.0" end="6.0"', xml)
        self.assertIn('name="dog bark" start="6.5" end="7.0" count="2"', xml)
        self.assertIn("<order>crowd cheering before dog bark</order>", xml)

    def test_prompt_builder_mixed_distribution_is_deterministic(self):
        record = {"caption": "A single dog bark."}
        natural, natural_variant = build_text_prompt(record, prompt_format="mixed", selector=0)
        xml_only, xml_variant = build_text_prompt(record, prompt_format="mixed", selector=2)
        hybrid, hybrid_variant = build_text_prompt(record, prompt_format="mixed", selector=3)
        self.assertEqual(natural_variant, "natural")
        self.assertEqual(xml_variant, "xml")
        self.assertEqual(hybrid_variant, "mixed")
        self.assertEqual(natural, "A single dog bark.")
        self.assertTrue(xml_only.startswith("<audio>"))
        self.assertTrue(hybrid.startswith("<audio>"))
        self.assertIn("A single dog bark.", hybrid)

    def test_multi_conditioner_accepts_prompt_as_text_prompt_fallback(self):
        conditioner = MultiConditioner(
            {"text_prompt": DummyTextConditioner()},
            default_keys={"text_prompt": "prompt"},
        )
        outputs = conditioner([{"prompt": "legacy prompt"}], device="cpu")
        self.assertEqual(outputs["text_prompt"][0].shape[0], 1)
        self.assertEqual(outputs["text_prompt"][0].shape[2], 4)

    def test_apply_finetune_defaults_patches_text_conditioner(self):
        raw_config = {
            "model": {
                "conditioning": {
                    "configs": [
                        {
                            "id": "text_prompt",
                            "type": "t5",
                            "config": {"t5_model_name": "t5-base", "max_length": 128},
                        }
                    ]
                }
            }
        }
        patched = apply_finetune_defaults(raw_config, text_max_length=256)
        conditioning = patched["model"]["conditioning"]
        self.assertEqual(conditioning["default_keys"]["text_prompt"], "prompt")
        self.assertEqual(conditioning["configs"][0]["config"]["max_length"], 256)

    def test_maf_order_uses_video_text_audio(self):
        wrapper = ConditionedDiffusionModelWrapper(
            DummyDiffusionModel(),
            conditioner=None,
            io_channels=2,
            sample_rate=16000,
            min_input_length=1,
            diffusion_objective="v",
            gate=True,
            gate_type="MAF",
            cross_attn_cond_ids=["video_prompt", "text_prompt", "audio_prompt"],
        )
        recorder = RecordingMAF()
        wrapper.maf_block = recorder

        video = torch.randn(1, 2, 8)
        text = torch.randn(1, 3, 8)
        audio = torch.randn(1, 4, 8)
        conditioning_inputs = wrapper.get_conditioning_inputs(
            {
                "video_prompt": (video, torch.ones(1, 2, dtype=torch.bool)),
                "text_prompt": (text, torch.ones(1, 3, dtype=torch.bool)),
                "audio_prompt": (audio, torch.ones(1, 4, dtype=torch.bool)),
            }
        )
        self.assertIs(recorder.seen[0], video)
        self.assertIs(recorder.seen[1], text)
        self.assertIs(recorder.seen[2], audio)
        self.assertEqual(conditioning_inputs["cross_attn_cond"].shape[1], 9)

    def test_ifcaps_dataset_emits_training_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "sample.wav")
            waveform = torch.linspace(-0.1, 0.1, steps=16000, dtype=torch.float32).repeat(2, 1)
            self._write_wav(audio_path, waveform, sample_rate=16000)

            manifest_path = os.path.join(tmpdir, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "id": "sample-1",
                            "audio_path": audio_path,
                            "caption": "A short synthetic tone.",
                            "category": {"synthetic tone": 1},
                            "SED": [{"name": "synthetic tone", "start": 0.0, "end": 1.0}],
                            "time_relation": "synthetic tone"
                        }
                    )
                    + "\n"
                )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=16000,
                prompt_format="xml",
                prompt_seed=0,
                random_crop=False,
            )
            audio_chunk, metadata = dataset[0]
            self.assertEqual(audio_chunk.shape, (2, 16000))
            self.assertTrue(metadata["text_prompt"].startswith("<audio>"))
            self.assertEqual(metadata["video_prompt"]["video_tensors"].shape, (1, 5, 3, 224, 224))
            self.assertEqual(metadata["audio_prompt"].shape, (1, 2, 16000))
            self.assertEqual(metadata["padding_mask"].shape[0], 16000)
            self.assertEqual(metadata["prompt_variant"], "xml")

    def test_training_wrapper_smoke_step(self):
        conditioner = MultiConditioner(
            {
                "video_prompt": DummyScalarConditioner(),
                "text_prompt": DummyTextConditioner(),
                "audio_prompt": DummyScalarConditioner(),
            },
            default_keys={"text_prompt": "prompt"},
        )
        model = ConditionedDiffusionModelWrapper(
            DummyDiffusionModel(),
            conditioner=conditioner,
            io_channels=2,
            sample_rate=16000,
            min_input_length=1,
            diffusion_objective="v",
            cross_attn_cond_ids=["video_prompt", "text_prompt", "audio_prompt"],
        )
        wrapper = DiffusionCondTrainingWrapper(
            model,
            lr=1e-3,
            use_ema=False,
            log_loss_info=False,
            optimizer_configs=None,
        )
        optimizer = wrapper.configure_optimizers()[0]
        wrapper._trainer = types.SimpleNamespace(optimizers=[optimizer])
        wrapper.log_dict = lambda *args, **kwargs: None

        reals = torch.randn(2, 2, 64)
        metadata = [
            normalize_training_metadata(
                {
                    "prompt": "xml training smoke one",
                    "video_prompt": 1,
                    "audio_prompt": 1,
                    "padding_mask": torch.ones(64),
                }
            ),
            normalize_training_metadata(
                {
                    "text_prompt": "xml training smoke two",
                    "video_prompt": 1,
                    "audio_prompt": 1,
                    "padding_mask": torch.ones(64),
                }
            ),
        ]
        loss = wrapper.training_step((reals, metadata), batch_idx=0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
