import json
import math
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
    collect_text_prompt_candidates,
    prepare_mixed_preference_manifests,
)
from audiox.data.ifcaps import (
    IFCapsFineTuneDataset,
    build_text_prompt,
    normalize_training_metadata,
    resolve_text_prompt_record,
    select_prompt_variant,
    serialize_ifcaps_to_xml,
)
from audiox.inference.asmr import build_continuation_conditioning, crossfade_stitch, prepare_audio_prompt
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
    apply_finetune_defaults,
    apply_trainable_scope,
    build_sample_weights,
    create_trainer,
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

    def test_create_trainer_skips_checkpoint_callback_when_disabled(self):
        trainer = create_trainer(
            trainer_config={"accelerator": "cpu", "devices": 1, "max_steps": 1},
            checkpoint_config={"enabled": False},
            wandb_config={"enabled": False},
            output_dir=tempfile.mkdtemp(),
        )
        checkpoint_callbacks = [callback for callback in trainer.callbacks if callback.__class__.__name__ == "ModelCheckpoint"]
        self.assertEqual(checkpoint_callbacks, [])

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

    def test_dataset_absent_audio_prompt_falls_back_to_audio_path(self):
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
            with mock.patch(
                "audiox.data.ifcaps.load_and_process_audio",
                return_value=torch.ones(2, 8000),
            ):
                _, metadata = dataset[0]
            self.assertGreater(float(torch.abs(metadata["audio_prompt"]).sum()), 0.0)

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
            audio_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)

            _write_wav(audio_dir / "clip_0000.wav", sample_rate=16000)
            _write_wav(audio_dir / "clip_0001.wav", sample_rate=16000)
            _write_wav(audio_dir / "clip_0004.wav", sample_rate=16000)
            (video_dir / "clip_0000.mp4").touch()
            (video_dir / "clip_0001.mp4").touch()

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
            self.assertEqual(rows[-1]["text_prompt"], "plain four")
            self.assertNotIn("video_path", rows[-1])
            self.assertEqual(
                rows[1]["text_prompt_candidates"],
                ["[asmr]: one", "plain one", "[asmr]: one alt", "plain one alt", "timeline one"],
            )

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
