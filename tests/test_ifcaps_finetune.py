import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from audiox.data.asmr import (
    ASMR_SECONDS_TOTAL,
    build_asmr_manifest_rows,
    split_manifest_rows_by_sequence,
)
from audiox.data.mixed_preference import (
    DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES,
    build_mixed_preference_manifest_rows,
    build_sound_effect_manifest_rows,
    collect_text_prompt_candidates,
    prepare_mixed_preference_manifests,
)
from audiox.data.ifcaps import (
    IFCapsFineTuneDataset,
    _load_audio_waveform,
    _load_wav_with_wave,
    build_text_prompt,
    normalize_training_metadata,
    resolve_text_prompt_record,
    select_prompt_variant,
    serialize_ifcaps_to_xml,
)
from audiox.inference.asmr import build_continuation_conditioning, crossfade_stitch, prepare_audio_prompt
from audiox.inference.asmr import (
    build_standalone_conditioning,
    load_prompt_manifest,
    run_soundfx_eval_batch,
)
from audiox.models.lora import (
    LoRALinear,
    count_parameters,
    extract_lora_state_dict,
    extract_parameter_state_dict,
    inject_lora,
    load_lora_checkpoint,
)
from audiox.models.conditioners import MultiConditioner
from audiox.models.diffusion import ConditionedDiffusionModelWrapper
from audiox.training.diffusion import DiffusionCondTrainingWrapper
from audiox.training.finetune import (
    EpochLoRACheckpointCallback,
    apply_finetune_defaults,
    apply_trainable_scope,
    build_sample_weights,
    create_trainer,
    maybe_upload_huggingface_artifacts,
    maybe_apply_lora,
)


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


class TinyLoRAModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)
        self.proj = nn.Linear(8, 8, bias=False)


class TinyConditionerModule(nn.Module):
    def __init__(self, with_empty_audio_feat=False):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.proj_out = nn.Linear(4, 4, bias=False)
        self.proj_features_128 = nn.Linear(4, 4, bias=False)
        if with_empty_audio_feat:
            self.empty_audio_feat = nn.Parameter(torch.zeros(1, 1, 4))


class TinyConditionerContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleDict(
            {
                "text_prompt": TinyConditionerModule(),
                "audio_prompt": TinyConditionerModule(with_empty_audio_feat=True),
                "video_prompt": TinyConditionerModule(),
            }
        )


class TinyScopeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        self.backbone = nn.Linear(4, 4, bias=False)
        self.maf_block = nn.Sequential(nn.Linear(4, 4, bias=False))
        self.conditioner = TinyConditionerContainer()


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

    def test_load_wav_with_wave_truncates_incomplete_trailing_sample(self):
        wav_path = Path("/tmp/malformed.wav")
        mocked_wave = mock.MagicMock()
        mocked_wave.getframerate.return_value = 16000
        mocked_wave.getnchannels.return_value = 2
        mocked_wave.getsampwidth.return_value = 2
        mocked_wave.getnframes.return_value = 2
        mocked_wave.readframes.return_value = torch.arange(5, dtype=torch.int16).numpy().tobytes()
        mocked_wave.__enter__.return_value = mocked_wave
        mocked_wave.__exit__.return_value = False

        with mock.patch("audiox.data.ifcaps.wave.open", return_value=mocked_wave):
            with self.assertWarnsRegex(RuntimeWarning, "Truncating 1 trailing PCM sample"):
                waveform, sample_rate = _load_wav_with_wave(wav_path)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(tuple(waveform.shape), (2, 2))
        self.assertTrue(torch.allclose(waveform[0], torch.tensor([0.0, 2.0 / 32768.0])))
        self.assertTrue(torch.allclose(waveform[1], torch.tensor([1.0 / 32768.0, 3.0 / 32768.0])))

    def test_load_audio_waveform_uses_ffmpeg_for_mp3_when_torchaudio_is_unavailable(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            wav_path = tmpdir_path / "source.wav"
            mp3_path = tmpdir_path / "source.mp3"
            _write_wav(wav_path, sample_rate=16000)
            subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(wav_path), str(mp3_path)],
                check=True,
            )

            with mock.patch.dict("sys.modules", {"torchaudio": None}):
                waveform = _load_audio_waveform(mp3_path, 16000)

        self.assertEqual(waveform.shape[0], 2)
        self.assertGreater(waveform.shape[-1], 0)

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

    def test_apply_finetune_defaults_caps_t5_length_for_mmdit_maf_budget(self):
        model_config = {
            "sample_rate": 44100,
            "sample_size": 485100,
            "model": {
                "conditioning": {
                    "configs": [
                        {"id": "video_prompt", "type": "clip-with-sync-w-empty-feat", "config": {}},
                        {"id": "text_prompt", "type": "t5", "config": {"t5_model_name": "t5-base", "max_length": 128}},
                        {"id": "audio_prompt", "type": "audio_autoencoder_v2", "config": {}},
                    ]
                },
                "diffusion": {
                    "type": "mmdit",
                    "cross_attention_cond_ids": ["video_prompt", "text_prompt", "audio_prompt"],
                },
            },
        }
        patched = apply_finetune_defaults(model_config, text_max_length=256)
        self.assertEqual(
            patched["model"]["conditioning"]["configs"][1]["config"]["max_length"],
            128,
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

    def test_inject_lora_wraps_target_modules_and_freezes_base_parameters(self):
        module = TinyLoRAModule()
        replaced = inject_lora(module, rank=4, alpha=8.0, target_patterns=("to_q",))
        trainable, total = count_parameters(module)

        self.assertEqual(replaced, ["to_q"])
        self.assertIsInstance(module.to_q, LoRALinear)
        self.assertIsInstance(module.proj, nn.Linear)
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)
        self.assertTrue(any(parameter.requires_grad for parameter in module.to_q.parameters()))
        self.assertFalse(module.proj.weight.requires_grad)

    def test_maybe_apply_lora_returns_replaced_module_names(self):
        model = TinyLoRAModule()
        lora_info = maybe_apply_lora(
            model,
            {
                "lora": {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4.0,
                    "target_patterns": ["to_q"],
                }
            },
        )
        self.assertEqual(lora_info["replaced_modules"], ["to_q"])

    def test_extract_lora_state_dict_excludes_frozen_base_weights(self):
        model = TinyLoRAModule()
        inject_lora(model, rank=2, alpha=4.0, target_patterns=("to_q",))
        lora_state = extract_lora_state_dict(model)
        self.assertTrue(lora_state)
        self.assertTrue(all(".lora_a." in key or ".lora_b." in key for key in lora_state))
        self.assertFalse(any(".base." in key for key in lora_state))

    def test_load_lora_checkpoint_round_trips_plain_lora_without_explicit_target_metadata(self):
        model = TinyLoRAModule()
        inject_lora(model, rank=2, alpha=4.0, target_patterns=("to_q",))
        with torch.no_grad():
            model.to_q.lora_a.weight.fill_(0.25)
            model.to_q.lora_b.weight.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "plain-lora.pt"
            torch.save({"lora_state_dict": extract_lora_state_dict(model)}, checkpoint_path)

            reloaded = TinyLoRAModule()
            checkpoint = load_lora_checkpoint(reloaded, checkpoint_path)

        self.assertIsInstance(reloaded.to_q, LoRALinear)
        self.assertTrue(torch.allclose(reloaded.to_q.lora_a.weight, model.to_q.lora_a.weight))
        self.assertTrue(torch.allclose(reloaded.to_q.lora_b.weight, model.to_q.lora_b.weight))
        self.assertEqual(checkpoint["lora_config"]["target_patterns"], ["to_q"])

    def test_load_lora_checkpoint_rejects_missing_target_metadata_when_keys_are_not_inferable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "broken-lora.pt"
            torch.save({"lora_state_dict": {"invalid_key": torch.ones(1)}}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "target_patterns"):
                load_lora_checkpoint(TinyLoRAModule(), checkpoint_path)

    def test_load_lora_checkpoint_round_trips_trainable_scope_weights(self):
        run_config = {
            "lora": {
                "enabled": True,
                "rank": 2,
                "alpha": 4.0,
                "target_patterns": ["to_q"],
            },
            "training": {"trainable_scope": "asmr_continuation_lora"},
            "data": {"include_audio_conditioning": True, "include_video_conditioning": False},
        }
        model_config = {
            "model": {
                "conditioning": {
                    "configs": [
                        {"id": "text_prompt", "type": "t5", "config": {}},
                        {"id": "audio_prompt", "type": "audio_autoencoder_v2", "config": {}},
                        {"id": "video_prompt", "type": "clip-with-sync-w-empty-feat", "config": {}},
                    ]
                }
            }
        }
        model = TinyScopeModel()
        maybe_apply_lora(model, run_config)
        scope_info = apply_trainable_scope(model, model_config, run_config)
        with torch.no_grad():
            model.to_q.lora_a.weight.fill_(0.125)
            model.to_q.lora_b.weight.fill_(0.25)
            model.maf_block[0].weight.fill_(0.5)
            model.conditioner.conditioners["text_prompt"].proj_out.weight.fill_(0.75)
            model.conditioner.conditioners["audio_prompt"].proj_features_128.weight.fill_(1.0)
            model.conditioner.conditioners["audio_prompt"].empty_audio_feat.fill_(1.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "scope-lora.pt"
            torch.save(
                {
                    "lora_config": run_config["lora"],
                    "lora_state_dict": extract_lora_state_dict(model),
                    "trainable_scope_info": scope_info,
                    "trainable_scope_state_dict": extract_parameter_state_dict(
                        model, scope_info["non_lora_parameter_names"]
                    ),
                },
                checkpoint_path,
            )

            reloaded = TinyScopeModel()
            load_lora_checkpoint(reloaded, checkpoint_path)

        self.assertTrue(torch.allclose(reloaded.maf_block[0].weight, model.maf_block[0].weight))
        self.assertTrue(
            torch.allclose(
                reloaded.conditioner.conditioners["text_prompt"].proj_out.weight,
                model.conditioner.conditioners["text_prompt"].proj_out.weight,
            )
        )
        self.assertTrue(
            torch.allclose(
                reloaded.conditioner.conditioners["audio_prompt"].proj_features_128.weight,
                model.conditioner.conditioners["audio_prompt"].proj_features_128.weight,
            )
        )
        self.assertTrue(
            torch.allclose(
                reloaded.conditioner.conditioners["audio_prompt"].empty_audio_feat,
                model.conditioner.conditioners["audio_prompt"].empty_audio_feat,
            )
        )

    def test_load_lora_checkpoint_rejects_scope_checkpoint_without_scope_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "missing-scope-state.pt"
            model = TinyLoRAModule()
            inject_lora(model, rank=2, alpha=4.0, target_patterns=("to_q",))
            torch.save(
                {
                    "lora_state_dict": extract_lora_state_dict(model),
                    "trainable_scope_info": {"scope": "asmr_continuation_lora", "non_lora_parameter_names": ["maf_block.0.weight"]},
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "trainable_scope_state_dict"):
                load_lora_checkpoint(TinyLoRAModule(), checkpoint_path)

    def test_build_asmr_manifest_rows_creates_standalone_and_continuation_samples(self):
        records = [
            {
                "sequence_id": "seq-a",
                "chunk_index": 0,
                "audio_path": "seq-a_000.wav",
                "caption": "Soft brushing near the ear.",
            },
            {
                "sequence_id": "seq-a",
                "chunk_index": 1,
                "audio_path": "seq-a_001.wav",
                "caption": "Gentle tapping on a glass bottle.",
                "video_path": "seq-a_001.mp4",
            },
        ]
        rows = build_asmr_manifest_rows(records, base_dir="/tmp")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["sample_type"], "standalone")
        self.assertIsNone(rows[0]["audio_prompt_path"])
        self.assertEqual(rows[1]["sample_type"], "standalone")
        self.assertEqual(rows[2]["sample_type"], "continuation")
        self.assertTrue(rows[2]["audio_prompt_path"].endswith("seq-a_000.wav"))
        self.assertEqual(rows[2]["seconds_total"], ASMR_SECONDS_TOTAL)
        self.assertTrue(rows[1]["video_path"].endswith("seq-a_001.mp4"))

    def test_build_asmr_manifest_rows_emits_text_prompt_candidates(self):
        records = [
            {
                "sequence_id": "seq-a",
                "chunk_index": 0,
                "audio_path": "seq-a_000.wav",
                "text_prompt": "Primary brushing prompt.",
                "alternate_captions": ["Alternate brushing prompt.", "Primary brushing prompt."],
                "augmented_captions": [{"style": "timeline", "text": "Timeline brushing prompt."}],
            },
        ]

        rows = build_asmr_manifest_rows(records, base_dir="/tmp")

        self.assertEqual(rows[0]["text_prompt"], "Primary brushing prompt.")
        self.assertEqual(
            rows[0]["text_prompt_candidates"],
            ["Primary brushing prompt.", "Alternate brushing prompt.", "Timeline brushing prompt."],
        )

    def test_build_asmr_manifest_rows_keeps_base_prompt_as_default_and_includes_tagged_alternates(self):
        records = [
            {
                "sequence_id": "seq-a",
                "chunk_index": 0,
                "audio_path": "seq-a_000.wav",
                "caption": "Base folk prompt.",
                "tagged_caption": "Base folk prompt.",
                "tagged_alternate_captions": [
                    "Alternate folk prompt one.",
                    "Alternate folk prompt two.",
                ],
            },
        ]

        rows = build_asmr_manifest_rows(records, base_dir="/tmp")

        self.assertEqual(rows[0]["text_prompt"], "Base folk prompt.")
        self.assertEqual(
            rows[0]["text_prompt_candidates"],
            [
                "Base folk prompt.",
                "Alternate folk prompt one.",
                "Alternate folk prompt two.",
            ],
        )

    def test_split_manifest_rows_by_sequence_avoids_leakage(self):
        rows = [
            {"sequence_id": "seq-a", "sample_type": "standalone"},
            {"sequence_id": "seq-a", "sample_type": "continuation"},
            {"sequence_id": "seq-b", "sample_type": "standalone"},
            {"sequence_id": "seq-b", "sample_type": "continuation"},
        ]
        train_rows, val_rows = split_manifest_rows_by_sequence(rows, val_ratio=0.5, seed=0)
        self.assertTrue(train_rows)
        self.assertTrue(val_rows)
        self.assertTrue({row["sequence_id"] for row in train_rows}.isdisjoint({row["sequence_id"] for row in val_rows}))

    def test_build_sample_weights_prefers_continuation_when_requested(self):
        records = [
            {"sample_type": "standalone"},
            {"sample_type": "standalone"},
            {"sample_type": "continuation"},
            {"sample_type": "continuation"},
            {"sample_type": "continuation"},
        ]
        weights = build_sample_weights(
            records,
            sample_strategy="weighted",
            standalone_ratio=0.3,
            continuation_ratio=0.7,
        )
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(weights[0], 0.15)
        self.assertAlmostEqual(weights[2], 0.7 / 3.0)

    def test_build_sample_weights_returns_none_without_complete_sample_types(self):
        weights = build_sample_weights(
            [{"sample_type": "standalone"}, {"sample_type": "other"}],
            sample_strategy="weighted",
        )
        self.assertIsNone(weights)

    def test_apply_trainable_scope_keeps_lora_and_asmr_modules_trainable(self):
        model = TinyScopeModel()
        maybe_apply_lora(
            model,
            {
                "lora": {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4.0,
                    "target_patterns": ["to_q"],
                }
            },
        )
        scope_info = apply_trainable_scope(
            model,
            {
                "model": {
                    "conditioning": {
                        "configs": [
                            {"id": "text_prompt", "type": "t5", "config": {}},
                            {"id": "audio_prompt", "type": "audio_autoencoder_v2", "config": {}},
                            {"id": "video_prompt", "type": "clip-with-sync-w-empty-feat", "config": {}},
                        ]
                    }
                }
            },
            {
                "training": {"trainable_scope": "asmr_continuation_lora"},
                "data": {"include_audio_conditioning": True, "include_video_conditioning": False},
            },
        )
        self.assertEqual(scope_info["scope"], "asmr_continuation_lora")
        self.assertTrue(model.to_q.lora_a.weight.requires_grad)
        self.assertTrue(model.maf_block[0].weight.requires_grad)
        self.assertTrue(model.conditioner.conditioners["text_prompt"].proj_out.weight.requires_grad)
        self.assertTrue(model.conditioner.conditioners["audio_prompt"].proj_features_128.weight.requires_grad)
        self.assertTrue(model.conditioner.conditioners["audio_prompt"].empty_audio_feat.requires_grad)
        self.assertFalse(model.conditioner.conditioners["audio_prompt"].encoder.weight.requires_grad)
        self.assertFalse(model.conditioner.conditioners["video_prompt"].proj_out.weight.requires_grad)

    def test_apply_trainable_scope_supports_multimodal_alias(self):
        model = TinyScopeModel()
        maybe_apply_lora(
            model,
            {
                "lora": {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4.0,
                    "target_patterns": ["to_q"],
                }
            },
        )
        scope_info = apply_trainable_scope(
            model,
            {
                "model": {
                    "conditioning": {
                        "configs": [
                            {"id": "text_prompt", "type": "t5", "config": {}},
                            {"id": "audio_prompt", "type": "audio_autoencoder_v2", "config": {}},
                            {"id": "video_prompt", "type": "clip-with-sync-w-empty-feat", "config": {}},
                        ]
                    }
                }
            },
            {
                "training": {"trainable_scope": "multimodal_continuation_lora"},
                "data": {"include_audio_conditioning": True, "include_video_conditioning": True},
            },
        )
        self.assertEqual(scope_info["scope"], "multimodal_continuation_lora")
        self.assertTrue(model.maf_block[0].weight.requires_grad)
        self.assertTrue(model.conditioner.conditioners["video_prompt"].proj_out.weight.requires_grad)

    def test_apply_trainable_scope_supports_maf_only_continuation_scope(self):
        model = TinyScopeModel()
        maybe_apply_lora(
            model,
            {
                "lora": {
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4.0,
                    "target_patterns": ["to_q"],
                }
            },
        )
        scope_info = apply_trainable_scope(
            model,
            {
                "model": {
                    "conditioning": {
                        "configs": [
                            {"id": "text_prompt", "type": "t5", "config": {}},
                            {"id": "audio_prompt", "type": "audio_autoencoder_v2", "config": {}},
                            {"id": "video_prompt", "type": "clip-with-sync-w-empty-feat", "config": {}},
                        ]
                    }
                }
            },
            {
                "training": {"trainable_scope": "maf_continuation_lora"},
                "data": {"include_audio_conditioning": True, "include_video_conditioning": True},
            },
        )
        self.assertEqual(scope_info["scope"], "maf_continuation_lora")
        self.assertTrue(model.to_q.lora_a.weight.requires_grad)
        self.assertTrue(model.maf_block[0].weight.requires_grad)
        self.assertFalse(model.conditioner.conditioners["text_prompt"].proj_out.weight.requires_grad)
        self.assertFalse(model.conditioner.conditioners["audio_prompt"].proj_features_128.weight.requires_grad)
        self.assertFalse(model.conditioner.conditioners["audio_prompt"].empty_audio_feat.requires_grad)
        self.assertFalse(model.conditioner.conditioners["video_prompt"].proj_out.weight.requires_grad)

    def test_create_trainer_skips_checkpoint_callback_when_disabled(self):
        trainer = create_trainer(
            trainer_config={"accelerator": "cpu", "devices": 1, "max_steps": 1},
            checkpoint_config={"enabled": False},
            wandb_config={"enabled": False},
            output_dir=tempfile.mkdtemp(),
        )
        checkpoint_callbacks = [callback for callback in trainer.callbacks if callback.__class__.__name__ == "ModelCheckpoint"]
        self.assertEqual(checkpoint_callbacks, [])

    def test_create_trainer_adds_resume_checkpoint_callback_for_true_resume(self):
        trainer = create_trainer(
            trainer_config={"accelerator": "cpu", "devices": 1, "max_steps": 1},
            checkpoint_config={
                "enabled": True,
                "save_lora_only": True,
                "save_resume_checkpoints": True,
                "dirpath": tempfile.mkdtemp(),
                "filename": "epoch={epoch}-step={step}",
                "resume_checkpoint_filename": "resume-epoch={epoch}-step={step}",
                "every_n_epochs": 1,
            },
            wandb_config={"enabled": False},
            output_dir=tempfile.mkdtemp(),
        )
        checkpoint_callbacks = [callback for callback in trainer.callbacks if callback.__class__.__name__ == "ModelCheckpoint"]
        self.assertEqual(len(checkpoint_callbacks), 1)
        self.assertEqual(checkpoint_callbacks[0].filename, "resume-epoch={epoch}-step={step}")

    def test_maybe_upload_huggingface_artifacts_uploads_final_checkpoint_and_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_config_path = tmpdir_path / "config_mixed_preference.json"
            source_config_path.write_text(json.dumps({"output_dir": "unused"}))
            resolved_model_config_path = tmpdir_path / "resolved_model_config.json"
            resolved_model_config_path.write_text(json.dumps({"sample_rate": 44100}))
            final_checkpoint_path = tmpdir_path / "final-lora-state.pt"
            final_checkpoint_path.write_bytes(b"checkpoint")
            manifest_summary_path = tmpdir_path / "manifest_summary.json"
            manifest_summary_path.write_text(json.dumps({"train_rows": 10}))

            api = mock.Mock()
            with mock.patch("audiox.training.finetune.HfApi", return_value=api), mock.patch.dict(
                "os.environ", {"HF_TOKEN": "test-token"}, clear=False
            ):
                upload_info = maybe_upload_huggingface_artifacts(
                    {
                        "huggingface": {
                            "enabled": True,
                            "repo_id": "owenisas/audiox-mixed-preference-lora-test",
                            "repo_type": "model",
                            "private": True,
                            "path_prefix": "runs/test-run",
                        }
                    },
                    source_config_path=source_config_path,
                    resolved_model_config_path=resolved_model_config_path,
                    final_checkpoint_path=final_checkpoint_path,
                )

        api.create_repo.assert_called_once_with(
            repo_id="owenisas/audiox-mixed-preference-lora-test",
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        uploaded_paths = [call.kwargs["path_in_repo"] for call in api.upload_file.call_args_list]
        self.assertEqual(
            uploaded_paths,
            [
                "runs/test-run/final-lora-state.pt",
                "runs/test-run/resolved_model_config.json",
                "runs/test-run/run_config.json",
                "runs/test-run/manifest_summary.json",
            ],
        )
        self.assertEqual(upload_info["repo_url"], "https://huggingface.co/owenisas/audiox-mixed-preference-lora-test")

    def test_maybe_upload_huggingface_artifacts_requires_repo_id_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_config_path = tmpdir_path / "config.json"
            source_config_path.write_text("{}")
            resolved_model_config_path = tmpdir_path / "resolved_model_config.json"
            resolved_model_config_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "repo_id"):
                maybe_upload_huggingface_artifacts(
                    {"huggingface": {"enabled": True}},
                    source_config_path=source_config_path,
                    resolved_model_config_path=resolved_model_config_path,
                    final_checkpoint_path=None,
                )

    def test_maybe_upload_huggingface_artifacts_uploads_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_config_path = tmpdir_path / "config.json"
            source_config_path.write_text("{}")
            resolved_model_config_path = tmpdir_path / "resolved_model_config.json"
            resolved_model_config_path.write_text("{}")
            checkpoint_dir = tmpdir_path / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "epoch=1-step=10.pt").write_bytes(b"epoch1")
            (checkpoint_dir / "last-lora-state.pt").write_bytes(b"last")

            api = mock.Mock()
            with mock.patch("audiox.training.finetune.HfApi", return_value=api), mock.patch.dict(
                "os.environ", {"HF_TOKEN": "test-token"}, clear=False
            ):
                maybe_upload_huggingface_artifacts(
                    {
                        "checkpointing": {"dirpath": str(checkpoint_dir)},
                        "huggingface": {
                            "enabled": True,
                            "repo_id": "owenisas/audiox-mixed-preference-lora-test",
                            "repo_type": "model",
                            "private": True,
                            "path_prefix": "runs/test-run",
                            "upload_final_checkpoint": False,
                            "upload_resolved_config": False,
                            "upload_run_config": False,
                            "upload_manifest_summary": False,
                            "upload_checkpoint_dir": True,
                            "checkpoint_dir_path_in_repo": "checkpoints",
                        },
                    },
                    source_config_path=source_config_path,
                    resolved_model_config_path=resolved_model_config_path,
                    final_checkpoint_path=None,
                )

        uploaded_paths = [call.kwargs["path_in_repo"] for call in api.upload_file.call_args_list]
        self.assertEqual(
            uploaded_paths,
            [
                "runs/test-run/checkpoints/epoch=1-step=10.pt",
                "runs/test-run/checkpoints/last-lora-state.pt",
            ],
        )

    def test_epoch_lora_checkpoint_callback_saves_and_uploads_epoch_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            callback = EpochLoRACheckpointCallback(
                checkpoint_config={
                    "dirpath": str(tmpdir_path / "checkpoints"),
                    "resume_checkpoint_dirpath": str(tmpdir_path / "checkpoints"),
                    "filename": "epoch={epoch}-step={step}",
                    "save_last": True,
                    "every_n_epochs": 1,
                    "save_on_train_epoch_end": True,
                },
                output_dir=tmpdir_path,
                run_config={
                    "lora": {"enabled": True},
                    "huggingface": {
                        "enabled": True,
                        "repo_id": "owenisas/audiox-mixed-preference-lora-test",
                        "repo_type": "model",
                        "private": True,
                        "path_prefix": "runs/test-run",
                        "upload_checkpoint_dir": True,
                        "checkpoint_dir_path_in_repo": "checkpoints",
                    },
                },
                lora_info={"enabled": True},
                trainable_scope_info={"non_lora_parameter_names": []},
            )
            trainer = types.SimpleNamespace(current_epoch=0, global_step=12)
            module = types.SimpleNamespace(diffusion=TinyLoRAModule())
            (tmpdir_path / "checkpoints").mkdir(parents=True, exist_ok=True)
            (tmpdir_path / "checkpoints" / "resume-epoch=1-step=12.ckpt").write_bytes(b"resume")
            api = mock.Mock()
            with mock.patch("audiox.training.finetune.HfApi", return_value=api), mock.patch.dict(
                "os.environ", {"HF_TOKEN": "test-token"}, clear=False
            ):
                callback.on_train_epoch_end(trainer, module)

            checkpoint_dir = tmpdir_path / "checkpoints"
            self.assertTrue((checkpoint_dir / "epoch=1-step=12.pt").exists())
            self.assertTrue((checkpoint_dir / "last-lora-state.pt").exists())
            uploaded_paths = [call.kwargs["path_in_repo"] for call in api.upload_file.call_args_list]
            self.assertEqual(
                uploaded_paths,
                [
                    "runs/test-run/checkpoints/epoch=1-step=12.pt",
                    "runs/test-run/checkpoints/last-lora-state.pt",
                    "runs/test-run/checkpoints/resume-epoch=1-step=12.ckpt",
                ],
            )

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
            self.assertEqual(metadata["video_prompt"]["video_tensors"].shape, (1, 20, 3, 224, 224))
            self.assertEqual(metadata["audio_prompt"].shape, (1, 2, 8000))
            self.assertEqual(metadata["padding_mask"].shape, (8000,))
            self.assertEqual(normalized["text_prompt"], "legacy")

    def test_dataset_explicit_null_audio_prompt_uses_zero_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            _write_wav(audio_path, sample_rate=16000)
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "caption": "Soft rain.",
                        "audio_prompt_path": None,
                    }
                )
                + "\n"
            )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=False,
                include_audio_conditioning=True,
                audio_prompt_num_samples=8000,
            )
            _, metadata = dataset[0]
            self.assertTrue(torch.all(metadata["audio_prompt"] == 0))

    def test_dataset_absent_audio_prompt_uses_zero_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            _write_wav(audio_path, sample_rate=16000)
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(json.dumps({"audio_path": str(audio_path), "caption": "Soft rain."}) + "\n")

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=False,
                include_audio_conditioning=True,
                audio_prompt_num_samples=8000,
            )
            _, metadata = dataset[0]
            self.assertTrue(torch.all(metadata["audio_prompt"] == 0))

    def test_dataset_invalid_audio_prompt_falls_back_to_zero_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            prompt_path = tmpdir_path / "broken.wav"
            _write_wav(audio_path, sample_rate=16000)
            prompt_path.write_bytes(b"not-a-real-wav")
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "audio_prompt_path": str(prompt_path),
                        "caption": "Soft rain.",
                    }
                )
                + "\n"
            )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=False,
                include_audio_conditioning=True,
                audio_prompt_num_samples=8000,
            )
            with mock.patch(
                "audiox.data.ifcaps.load_and_process_audio",
                side_effect=RuntimeError("decode failure"),
            ):
                _, metadata = dataset[0]
            self.assertTrue(torch.all(metadata["audio_prompt"] == 0))

    def test_resolve_text_prompt_record_samples_from_candidate_pool(self):
        record = {
            "text_prompt": "primary",
            "text_prompt_candidates": ["primary", "alternate", "timeline"],
        }
        with mock.patch("audiox.data.ifcaps.torch.randint", return_value=torch.tensor([1])):
            resolved = resolve_text_prompt_record(record, sample_text_prompt_candidates=True)
        self.assertEqual(resolved["text_prompt"], "alternate")
        self.assertEqual(
            resolved["text_prompt_candidates"],
            ["primary", "alternate", "timeline"],
        )

    def test_dataset_keeps_eval_caption_deterministic_when_sampling_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            _write_wav(audio_path, sample_rate=16000)
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "text_prompt": "primary prompt",
                        "text_prompt_candidates": ["primary prompt", "alternate prompt"],
                    }
                )
                + "\n"
            )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=False,
                include_audio_conditioning=False,
                sample_text_prompt_candidates=False,
            )
            _, metadata = dataset[0]
            self.assertEqual(metadata["text_prompt"], "primary prompt")

    def test_build_mixed_preference_manifest_rows_rewrites_paths_and_ignores_gap_links(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            media_root = Path(tmpdir) / "remote"
            audio_dir = dataset_root / "audio" / "audio_targets"
            video_dir = dataset_root / "ifcaps" / "clips"
            screenshot_dir = dataset_root / "ifcaps" / "screenshots"
            audio_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            screenshot_dir.mkdir(parents=True)

            _write_wav(audio_dir / "clip_0000.wav", sample_rate=16000)
            _write_wav(audio_dir / "clip_0001.wav", sample_rate=16000)
            _write_wav(audio_dir / "clip_0004.wav", sample_rate=16000)
            (video_dir / "clip_0000.mp4").touch()
            (video_dir / "clip_0001.mp4").touch()
            (screenshot_dir / "clip_0000.jpg").touch()
            (screenshot_dir / "clip_0001.jpg").touch()

            records = [
                {
                    "clip_id": "clip_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0000.wav"),
                    "video_path": str(video_dir / "clip_0000.mp4"),
                    "training_caption": "plain zero",
                    "tagged_training_caption": "[asmr]: zero",
                    "preference_tags": ["asmr"],
                    "source_family": "asmr",
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "split": "train",
                },
                {
                    "clip_id": "clip_0001",
                    "clip_index": 1,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0001.wav"),
                    "video_path": str(video_dir / "clip_0001.mp4"),
                    "training_caption": "plain one",
                    "tagged_training_caption": "[asmr]: one",
                    "tagged_alternate_captions": ["[asmr]: one alt", "[asmr]: one"],
                    "alternate_captions": ["plain one alt"],
                    "augmented_captions": [
                        {"style": "timeline", "text": "timeline one"},
                        {"style": "short", "text": "plain one alt"},
                    ],
                    "preference_tags": ["asmr"],
                    "source_family": "asmr",
                    "start_s": 10.0,
                    "end_s": 20.0,
                    "prev_clip_id": "clip_0000",
                    "split": "train",
                },
                {
                    "clip_id": "clip_0004",
                    "clip_index": 4,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0004.wav"),
                    "training_caption": "plain four",
                    "preference_tags": ["asmr"],
                    "source_family": "asmr",
                    "start_s": 40.0,
                    "end_s": 50.0,
                    "prev_clip_id": "clip_0001",
                    "split": "val",
                },
            ]

            rows = build_mixed_preference_manifest_rows(
                records,
                dataset_root=dataset_root,
                media_root=media_root,
                caption_field="tagged_training_caption",
                source_families=DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES,
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(1 for row in rows if row["sample_type"] == "continuation"), 1)
            self.assertEqual(rows[0]["text_prompt"], "[asmr]: zero")
            self.assertTrue(rows[0]["audio_path"].startswith(str(media_root)))
            self.assertTrue(rows[0]["video_path"].startswith(str(media_root)))
            self.assertTrue(rows[0]["screenshot_path"].startswith(str(media_root)))
            self.assertEqual(rows[-1]["text_prompt"], "plain four")
            self.assertNotIn("video_path", rows[-1])
            self.assertNotIn("screenshot_path", rows[-1])
            self.assertEqual(
                rows[1]["text_prompt_candidates"],
                ["[asmr]: one", "plain one", "[asmr]: one alt", "plain one alt", "timeline one"],
            )

    def test_dataset_composes_screenshot_into_first_visual_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            video_path = tmpdir_path / "sample.mp4"
            screenshot_path = tmpdir_path / "sample.jpg"
            _write_wav(audio_path, sample_rate=16000)
            video_path.touch()
            screenshot_path.touch()
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "video_path": str(video_path),
                        "screenshot_path": str(screenshot_path),
                        "caption": "Bell ring",
                    }
                )
                + "\n"
            )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=True,
                include_audio_conditioning=False,
                video_fps=2,
            )

            clip_frames = torch.full((1, 3, 224, 224), 2.0)
            screenshot_frame = torch.full((1, 3, 224, 224), 7.0)

            def fake_read_video(path, seek_time=0.0, duration=-1, target_fps=2):
                if path.endswith(".jpg"):
                    return screenshot_frame.clone()
                return clip_frames.repeat(max(1, int(round(duration * target_fps))), 1, 1, 1)

            with mock.patch("audiox.data.ifcaps.read_video", side_effect=fake_read_video):
                _, metadata = dataset[0]

            video_tensors = metadata["video_prompt"]["video_tensors"]
            self.assertEqual(video_tensors.shape, (1, 10, 3, 224, 224))
            self.assertTrue(torch.all(video_tensors[0, 0] == 7.0))
            self.assertTrue(torch.all(video_tensors[0, 1:] == 2.0))

    def test_dataset_uses_screenshot_only_when_clip_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            screenshot_path = tmpdir_path / "sample.jpg"
            _write_wav(audio_path, sample_rate=16000)
            screenshot_path.touch()
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "screenshot_path": str(screenshot_path),
                        "caption": "Bell ring",
                    }
                )
                + "\n"
            )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=True,
                include_audio_conditioning=False,
                video_fps=2,
            )

            screenshot_frame = torch.full((1, 3, 224, 224), 5.0)
            with mock.patch("audiox.data.ifcaps.read_video", return_value=screenshot_frame.clone()):
                _, metadata = dataset[0]

            video_tensors = metadata["video_prompt"]["video_tensors"]
            self.assertEqual(video_tensors.shape, (1, 10, 3, 224, 224))
            self.assertTrue(torch.all(video_tensors == 5.0))

    def test_dataset_uses_clip_only_when_screenshot_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            video_path = tmpdir_path / "sample.mp4"
            _write_wav(audio_path, sample_rate=16000)
            video_path.touch()
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "audio_path": str(audio_path),
                        "video_path": str(video_path),
                        "caption": "Bell ring",
                    }
                )
                + "\n"
            )

            dataset = IFCapsFineTuneDataset(
                manifest_path=manifest_path,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=True,
                include_audio_conditioning=False,
                video_fps=2,
            )

            clip_frames = torch.full((10, 3, 224, 224), 3.0)
            with mock.patch("audiox.data.ifcaps.read_video", return_value=clip_frames.clone()):
                _, metadata = dataset[0]

            video_tensors = metadata["video_prompt"]["video_tensors"]
            self.assertEqual(video_tensors.shape, (1, 10, 3, 224, 224))
            self.assertTrue(torch.all(video_tensors == 3.0))

    def test_collect_text_prompt_candidates_prefers_requested_caption_and_deduplicates_alternates(self):
        record = {
            "training_caption": "plain base",
            "tagged_training_caption": "[asmr]: base",
            "tagged_alternate_captions": ["[asmr]: alt one", "[asmr]: base"],
            "alternate_captions": ["plain alt one", "plain base"],
            "augmented_captions": [
                {"style": "timeline", "text": "timeline caption"},
                {"style": "short", "text": "plain alt one"},
            ],
        }
        self.assertEqual(
            collect_text_prompt_candidates(record, "tagged_training_caption"),
            ["[asmr]: base", "plain base", "[asmr]: alt one", "plain alt one", "timeline caption"],
        )

    def test_prepare_mixed_preference_manifests_preserves_dataset_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            audio_dir = dataset_root / "audio" / "audio_targets"
            manifest_dir = dataset_root / "audio"
            video_dir = dataset_root / "ifcaps" / "clips"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)
            video_dir.mkdir(parents=True)

            for name in ("train_0000", "train_0001", "val_0000", "test_0000"):
                _write_wav(audio_dir / f"{name}.wav", sample_rate=16000)

            records = [
                {
                    "clip_id": "train_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-train",
                    "audio_path": str(audio_dir / "train_0000.wav"),
                    "tagged_training_caption": "[asmr]: train zero",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "split": "train",
                },
                {
                    "clip_id": "train_0001",
                    "clip_index": 1,
                    "sample_group_id": "group-train",
                    "audio_path": str(audio_dir / "train_0001.wav"),
                    "tagged_training_caption": "[asmr]: train one",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 10.0,
                    "end_s": 20.0,
                    "split": "train",
                },
                {
                    "clip_id": "val_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-val",
                    "audio_path": str(audio_dir / "val_0000.wav"),
                    "tagged_training_caption": "[ambience]: val zero",
                    "source_family": "ambience",
                    "preference_tags": ["ambience"],
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "split": "val",
                },
                {
                    "clip_id": "test_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-test",
                    "audio_path": str(audio_dir / "test_0000.wav"),
                    "tagged_training_caption": "[music]: test zero",
                    "source_family": "music",
                    "preference_tags": ["music"],
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "split": "test",
                },
            ]
            manifest_path = manifest_dir / "audio_manifest_split.jsonl"
            with manifest_path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "prepared"
            summary = prepare_mixed_preference_manifests(dataset_root, output_dir)

            self.assertEqual(summary["split_counts"]["train"], 3)
            self.assertEqual(summary["split_counts"]["val"], 1)
            self.assertEqual(summary["split_counts"]["test"], 1)
            self.assertEqual(summary["sample_type_counts"]["continuation"], 1)
            self.assertTrue(Path(summary["train_manifest_path"]).exists())
            self.assertTrue(Path(summary["val_manifest_path"]).exists())
            self.assertTrue(Path(summary["test_manifest_path"]).exists())

    def test_build_sound_effect_manifest_rows_scans_directory_and_builds_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "sound_effects"
            barber_dir = root / "Barber"
            barber_dir.mkdir(parents=True)
            sound_path = barber_dir / "buzzing-electric-razor-joshua-chivers-1-00-10.mp3"
            sound_path.write_bytes(b"fake mp3 bytes")

            rows = build_sound_effect_manifest_rows(root, media_root=Path(tmpdir) / "remote" / "sound_effects")

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["sample_type"], "standalone")
            self.assertEqual(row["source_family"], "sound_effects")
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["sound_effects_category"], "Barber")
            self.assertTrue(row["audio_path"].endswith("sound_effects/Barber/buzzing-electric-razor-joshua-chivers-1-00-10.mp3"))
            self.assertEqual(
                row["text_prompt_candidates"][:2],
                [
                    "barber sound effects, buzzing electric razor joshua chivers",
                    "buzzing electric razor joshua chivers",
                ],
            )

    def test_prepare_mixed_preference_manifests_can_include_sound_effects_and_drop_continuations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            audio_dir = dataset_root / "audio" / "audio_targets"
            manifest_dir = dataset_root / "audio"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)

            _write_wav(audio_dir / "asmr_0000.wav", sample_rate=16000)
            _write_wav(audio_dir / "asmr_0001.wav", sample_rate=16000)
            records = [
                {
                    "clip_id": "asmr_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-asmr",
                    "audio_path": str(audio_dir / "asmr_0000.wav"),
                    "tagged_training_caption": "[asmr]: zero",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "split": "train",
                },
                {
                    "clip_id": "asmr_0001",
                    "clip_index": 1,
                    "sample_group_id": "group-asmr",
                    "audio_path": str(audio_dir / "asmr_0001.wav"),
                    "tagged_training_caption": "[asmr]: one",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 10.0,
                    "end_s": 20.0,
                    "split": "train",
                },
            ]
            manifest_path = manifest_dir / "audio_manifest_split.jsonl"
            with manifest_path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            sound_effects_root = Path(tmpdir) / "Audio"
            forest_dir = sound_effects_root / "Forest"
            forest_dir.mkdir(parents=True)
            (forest_dir / "coyotes-howling-in-the-jungle-felix-blume-1-00-28.mp3").write_bytes(b"fake mp3 bytes")

            output_dir = Path(tmpdir) / "prepared"
            summary = prepare_mixed_preference_manifests(
                dataset_root,
                output_dir,
                media_root=Path(tmpdir) / "remote_dataset",
                sound_effects_root_or_manifest=sound_effects_root,
                standalone_only=True,
            )

            self.assertEqual(summary["sample_type_counts"], {"standalone": 3})
            self.assertEqual(summary["source_family_counts"]["sound_effects"], 1)
            self.assertEqual(summary["split_counts"]["train"], 3)
            with Path(summary["train_manifest_path"]).open() as handle:
                train_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(train_rows), 3)
            sound_effect_row = next(row for row in train_rows if row["source_family"] == "sound_effects")
            self.assertTrue(sound_effect_row["audio_path"].endswith("remote_dataset/sound_effects/Forest/coyotes-howling-in-the-jungle-felix-blume-1-00-28.mp3"))
            self.assertEqual(sound_effect_row["sample_type"], "standalone")
            self.assertIn("coyotes howling in the jungle felix blume", sound_effect_row["text_prompt"])

    def test_prepare_mixed_preference_manifests_remaps_missing_absolute_source_paths_to_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            audio_dir = dataset_root / "audio" / "audio_targets"
            manifest_dir = dataset_root / "audio"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)

            _write_wav(audio_dir / "asmr_0000.wav", sample_rate=16000)
            fake_local_root = Path("/Users/user/PycharmProjects/Scraper/test_runs/hf_collection_20260330")
            records = [
                {
                    "clip_id": "asmr_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-asmr",
                    "audio_path": str(fake_local_root / "audio" / "audio_targets" / "asmr_0000.wav"),
                    "tagged_training_caption": "[asmr]: zero",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "split": "train",
                },
            ]
            manifest_path = manifest_dir / "audio_manifest_split.jsonl"
            with manifest_path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "prepared"
            summary = prepare_mixed_preference_manifests(dataset_root, output_dir)

            with Path(summary["train_manifest_path"]).open() as handle:
                train_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(train_rows[0]["audio_path"], str((audio_dir / "asmr_0000.wav").resolve()))

    def test_prepare_mixed_preference_cli_preserves_template_lora_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            audio_dir = dataset_root / "audio" / "audio_targets"
            manifest_dir = dataset_root / "audio"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)

            _write_wav(audio_dir / "clip_0000.wav", sample_rate=16000)
            records = [
                {
                    "clip_id": "clip_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0000.wav"),
                    "tagged_training_caption": "[asmr]: clip zero",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "split": "train",
                },
            ]
            manifest_path = manifest_dir / "audio_manifest_split.jsonl"
            with manifest_path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            template_path = Path(tmpdir) / "template.json"
            template_path.write_text(
                json.dumps(
                    {
                        "pretrained_name": "HKUSTAudio/AudioX-MAF-MMDiT",
                        "output_dir": "./outputs/template",
                        "data": {
                            "train_manifest": "./data/train.jsonl",
                            "val_manifest": "./data/val.jsonl",
                            "include_video_conditioning": True,
                            "include_audio_conditioning": True,
                        },
                        "evaluation": {
                            "test_manifest": "./data/test.jsonl",
                        },
                        "trainer": {
                            "max_epochs": 10,
                        },
                        "checkpointing": {
                            "enabled": True,
                            "save_resume_checkpoints": True,
                            "monitor": "valid/loss",
                            "mode": "min",
                            "save_top_k": 3,
                        },
                        "lora": {
                            "enabled": True,
                            "rank": 16,
                            "alpha": 32.0,
                        },
                        "wandb": {
                            "enabled": True,
                            "project": "audiox-finetune",
                            "name": "template-run",
                            "offline": False,
                        },
                    }
                )
            )

            output_dir = Path(tmpdir) / "prepared"
            script_path = Path(__file__).resolve().parents[1] / "example" / "prepare_mixed_preference_run.py"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--dataset-root",
                    str(dataset_root),
                    "--output-dir",
                    str(output_dir),
                    "--config-template",
                    str(template_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            config = json.loads((output_dir / "config_mixed_preference.json").read_text())
            self.assertEqual(config["lora"]["rank"], 16)
            self.assertEqual(config["lora"]["alpha"], 32.0)
            self.assertEqual(config["trainer"]["max_epochs"], 10)
            self.assertEqual(config["data"]["train_manifest"], str(output_dir / "manifests" / "train.jsonl"))
            self.assertEqual(config["evaluation"]["test_manifest"], str(output_dir / "manifests" / "test.jsonl"))
            self.assertEqual(config["checkpointing"]["monitor"], "valid/loss")
            self.assertEqual(config["checkpointing"]["mode"], "min")
            self.assertEqual(config["checkpointing"]["save_top_k"], 3)

    def test_prepare_mixed_preference_cli_rejects_conflicting_template_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            audio_dir = dataset_root / "audio" / "audio_targets"
            manifest_dir = dataset_root / "audio"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)

            _write_wav(audio_dir / "clip_0000.wav", sample_rate=16000)
            records = [
                {
                    "clip_id": "clip_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0000.wav"),
                    "tagged_training_caption": "[asmr]: clip zero",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "split": "train",
                },
            ]
            manifest_path = manifest_dir / "audio_manifest_split.jsonl"
            with manifest_path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            template_path = Path(tmpdir) / "template.json"
            template_path.write_text(
                json.dumps(
                    {
                        "pretrained_name": "HKUSTAudio/AudioX-MAF-MMDiT",
                        "output_dir": "./outputs/template",
                        "data": {
                            "train_manifest": "./data/train.jsonl",
                            "val_manifest": "./data/val.jsonl",
                            "include_video_conditioning": True,
                            "include_audio_conditioning": True,
                            "sample_strategy": "weighted",
                            "standalone_ratio": 0.3,
                            "continuation_ratio": 0.7,
                        },
                        "evaluation": {"test_manifest": "./data/test.jsonl"},
                        "training": {"trainable_scope": "multimodal_continuation_lora"},
                        "lora": {"enabled": True, "rank": 16, "alpha": 32.0},
                        "wandb": {"enabled": True, "project": "audiox-finetune", "name": "template-run", "offline": False},
                    }
                )
            )

            output_dir = Path(tmpdir) / "prepared"
            script_path = Path(__file__).resolve().parents[1] / "example" / "prepare_mixed_preference_run.py"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--dataset-root",
                    str(dataset_root),
                    "--output-dir",
                    str(output_dir),
                    "--config-template",
                    str(template_path),
                    "--disable-audio-conditioning",
                ],
                check=False,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("include_audio_conditioning=true", result.stderr)

    def test_mixed_preference_manifest_smoke_dataset_shapes_with_video_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            audio_dir = dataset_root / "audio" / "audio_targets"
            manifest_dir = dataset_root / "audio"
            audio_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)

            _write_wav(audio_dir / "clip_0000.wav", sample_rate=16000)
            _write_wav(audio_dir / "clip_0001.wav", sample_rate=16000)

            records = [
                {
                    "clip_id": "clip_0000",
                    "clip_index": 0,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0000.wav"),
                    "tagged_training_caption": "[asmr]: clip zero",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "split": "train",
                },
                {
                    "clip_id": "clip_0001",
                    "clip_index": 1,
                    "sample_group_id": "group-a",
                    "audio_path": str(audio_dir / "clip_0001.wav"),
                    "tagged_training_caption": "[asmr]: clip one",
                    "source_family": "asmr",
                    "preference_tags": ["asmr"],
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "split": "train",
                },
            ]
            manifest_path = manifest_dir / "audio_manifest_split.jsonl"
            with manifest_path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "prepared"
            summary = prepare_mixed_preference_manifests(dataset_root, output_dir, include_video=True)
            train_manifest = Path(summary["train_manifest_path"])
            with train_manifest.open() as handle:
                train_rows = [json.loads(line) for line in handle]
            continuation_row = next(row for row in train_rows if row["sample_type"] == "continuation")

            prepared_manifest = Path(tmpdir) / "prepared_manifest.jsonl"
            prepared_manifest.write_text(json.dumps(continuation_row) + "\n")
            dataset = IFCapsFineTuneDataset(
                manifest_path=prepared_manifest,
                sample_rate=16000,
                sample_size=8000,
                prompt_format="natural",
                include_video_conditioning=True,
                include_audio_conditioning=True,
                audio_prompt_num_samples=8000,
                video_fps=2,
                video_duration_seconds=1.0,
            )
            with mock.patch(
                "audiox.data.ifcaps.load_and_process_audio",
                return_value=torch.ones(2, 8000),
            ):
                audio, metadata = dataset[0]
            self.assertEqual(audio.shape, (2, 8000))
            self.assertEqual(metadata["text_prompt"], "[asmr]: clip one")
            self.assertEqual(metadata["video_prompt"]["video_tensors"].shape, (1, 2, 3, 224, 224))
            self.assertGreater(float(torch.abs(metadata["audio_prompt"]).sum()), 0.0)

    def test_build_continuation_conditioning_uses_zero_audio_for_first_chunk(self):
        conditioning = build_continuation_conditioning(
            "Soft whispering close to the microphone.",
            sample_rate=44100,
            sample_size=485100,
            video_fps=5,
            audio_prompt_num_samples=440320,
            previous_audio=None,
            device="cpu",
        )
        sample = conditioning[0]
        self.assertEqual(sample["audio_prompt"].shape, (1, 2, 440320))
        self.assertTrue(torch.all(sample["audio_prompt"] == 0))
        self.assertEqual(sample["video_prompt"]["video_tensors"].shape, (1, 55, 3, 224, 224))

    def test_build_continuation_conditioning_uses_previous_audio_for_later_chunks(self):
        previous_audio = torch.ones(1, 2, 64)
        conditioning = build_continuation_conditioning(
            "Light tapping on a ceramic bowl.",
            sample_rate=44100,
            sample_size=485100,
            video_fps=5,
            audio_prompt_num_samples=128,
            previous_audio=previous_audio,
            device="cpu",
        )
        sample = conditioning[0]
        self.assertTrue(torch.all(sample["audio_prompt"][:, :, :64] == 1))
        self.assertTrue(torch.all(sample["audio_prompt"][:, :, 64:] == 0))

    def test_build_standalone_conditioning_uses_zero_audio_and_video(self):
        conditioning = build_standalone_conditioning(
            "Gentle page turning and soft tapping.",
            sample_rate=48000,
            sample_size=480000,
            video_fps=5,
            audio_prompt_num_samples=128,
            device="cpu",
        )
        sample = conditioning[0]
        self.assertEqual(sample["audio_prompt"].shape, (1, 2, 128))
        self.assertTrue(torch.all(sample["audio_prompt"] == 0))
        self.assertEqual(sample["video_prompt"]["video_tensors"].shape, (1, 50, 3, 224, 224))

    def test_load_prompt_manifest_supports_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "prompts.jsonl"
            manifest_path.write_text(
                json.dumps({"clip_id": "clip_a", "text_prompt": "soft tapping"}) + "\n" +
                json.dumps({"clip_id": "clip_b", "prompt": "gentle brushing"}) + "\n"
            )
            records = load_prompt_manifest(manifest_path)
        self.assertEqual(
            records,
            [
                {"clip_id": "clip_a", "text_prompt": "soft tapping"},
                {"clip_id": "clip_b", "text_prompt": "gentle brushing"},
            ],
        )

    def test_run_soundfx_eval_batch_writes_prompt_aligned_outputs(self):
        class DummyEvalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pretransform = None
                self.sample_rate = 48000
                self.io_channels = 2
                self.conditioner = lambda conditioning, device: conditioning
                self.get_conditioning_inputs = lambda tensors, negative=False: tensors

            def to(self, device):
                return self

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            manifest_path = tmpdir_path / "prompts.jsonl"
            manifest_path.write_text(
                json.dumps({"clip_id": "clip/a", "text_prompt": "soft tapping"}) + "\n" +
                json.dumps({"clip_id": "clip_b", "text_prompt": "gentle brushing"}) + "\n"
            )

            save_calls = []

            def fake_save(path, tensor, sample_rate):
                save_calls.append((Path(path).name, tuple(tensor.shape), sample_rate))

            with mock.patch(
                "audiox.inference.asmr.get_pretrained_model",
                return_value=(DummyEvalModel(), {"sample_rate": 48000, "sample_size": 480000, "video_fps": 5}),
            ), mock.patch(
                "audiox.inference.asmr.generate_diffusion_cond",
                side_effect=lambda *args, **kwargs: torch.ones(1, 2, 32) * kwargs["seed"],
            ), mock.patch(
                "torchaudio.save",
                side_effect=fake_save,
            ):
                summary = run_soundfx_eval_batch(
                    manifest_path,
                    output_dir=tmpdir_path / "eval",
                    pretrained_name="HKUSTAudio/AudioX-MAF-MMDiT",
                    lora_path="/tmp/final-lora-state.pt",
                    device="cpu",
                    steps=5,
                    cfg_scale=4.0,
                    sigma_min=0.1,
                    sigma_max=1.0,
                    sampler_type="dpmpp-3m-sde",
                    seed_base=100,
                )

            outputs = [json.loads(line) for line in Path(summary["outputs_path"]).read_text().splitlines()]
            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(outputs[0]["clip_id"], "clip_a")
            self.assertEqual(outputs[0]["seed"], 100)
            self.assertEqual(outputs[1]["clip_id"], "clip_b")
            self.assertEqual(outputs[1]["seed"], 101)
            self.assertEqual(save_calls[0][0], "clip_a.wav")
            self.assertEqual(save_calls[1][0], "clip_b.wav")

    def test_crossfade_stitch_outputs_expected_length(self):
        first = torch.ones(1, 2, 10)
        second = torch.zeros(1, 2, 10)
        stitched = crossfade_stitch([first, second], overlap_samples=4)
        self.assertEqual(stitched.shape, (2, 16))

    def test_prepare_audio_prompt_truncates_to_expected_length(self):
        prompt = prepare_audio_prompt(torch.ones(1, 2, 16), audio_prompt_num_samples=8)
        self.assertEqual(prompt.shape, (1, 2, 8))

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
