import copy
import json
import os
import typing as tp
from pathlib import Path

import pytorch_lightning as pl
from huggingface_hub import HfApi, hf_hub_download
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.ifcaps import IFCapsFineTuneDataset, collate_audiox_batch
from ..models.factory import create_model_from_config
from ..models.lora import (
    LoRALinear,
    count_parameters,
    extract_lora_state_dict,
    extract_parameter_state_dict,
    inject_lora,
    load_lora_checkpoint,
    is_lora_parameter_name,
)
from ..models.utils import load_ckpt_state_dict
from .factory import create_training_wrapper_from_config
from .utils import copy_state_dict


def _load_json(path: tp.Union[str, Path]) -> tp.Dict[str, tp.Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def _download_optional_file(
    repo_id: str, filename: str, cache_dir: tp.Optional[tp.Union[str, Path]] = None
) -> tp.Optional[str]:
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", cache_dir=cache_dir)
    except Exception:
        return None


def resolve_pretrained_artifacts(
    pretrained_name: str, cache_dir: tp.Optional[tp.Union[str, Path]] = None
) -> tp.Dict[str, tp.Optional[str]]:
    model_ckpt_path = _download_optional_file(pretrained_name, "model.safetensors", cache_dir)
    if model_ckpt_path is None:
        model_ckpt_path = hf_hub_download(
            repo_id=pretrained_name,
            filename="model.ckpt",
            repo_type="model",
            cache_dir=cache_dir,
        )

    return {
        "config_path": hf_hub_download(
            repo_id=pretrained_name,
            filename="config.json",
            repo_type="model",
            cache_dir=cache_dir,
        ),
        "model_ckpt_path": model_ckpt_path,
        "vae_ckpt_path": _download_optional_file(pretrained_name, "VAE.ckpt", cache_dir),
        "synchformer_ckpt_path": _download_optional_file(
            pretrained_name, "synchformer_state_dict.pth", cache_dir
        ),
    }


def _patch_pretransform_ckpt_paths(node: tp.Any, vae_ckpt_path: tp.Optional[str]) -> None:
    if vae_ckpt_path is None:
        return

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pretransform_ckpt_path" and isinstance(value, str) and value.endswith("VAE.ckpt"):
                node[key] = vae_ckpt_path
            else:
                _patch_pretransform_ckpt_paths(value, vae_ckpt_path)
    elif isinstance(node, list):
        for item in node:
            _patch_pretransform_ckpt_paths(item, vae_ckpt_path)


def _conditioner_seq_len_for_mmdit(conditioner: tp.Dict[str, tp.Any]) -> tp.Optional[int]:
    conditioner_type = conditioner.get("type")
    conditioner_config = conditioner.get("config", {})

    if conditioner_type == "clip-with-sync-w-empty-feat":
        return conditioner_config.get("out_features", 128)

    if conditioner_type == "audio_autoencoder_v2":
        return 128

    return None


def _cap_text_length_for_mmdit_budget(model_config: tp.Dict[str, tp.Any], requested_text_max_length: int) -> int:
    model_section = model_config.get("model", {})
    diffusion = model_section.get("diffusion", {})
    conditioning = model_section.get("conditioning", {})
    cross_attention_ids = diffusion.get("cross_attention_cond_ids", [])

    if diffusion.get("type") != "mmdit" or "text_prompt" not in cross_attention_ids:
        return requested_text_max_length

    # The current MMDiT implementation projects multimodal conditioning from a fixed
    # sequence length of 384 tokens before matching the latent sequence length.
    fixed_mm_token_budget = 384
    remaining_budget = fixed_mm_token_budget

    for conditioner in conditioning.get("configs", []):
        conditioner_id = conditioner.get("id")
        if conditioner_id not in cross_attention_ids or conditioner_id == "text_prompt":
            continue

        conditioner_seq_len = _conditioner_seq_len_for_mmdit(conditioner)
        if conditioner_seq_len is not None:
            remaining_budget -= conditioner_seq_len

    if remaining_budget <= 0:
        return requested_text_max_length

    return min(requested_text_max_length, remaining_budget)


def apply_finetune_defaults(
    model_config: tp.Dict[str, tp.Any],
    text_max_length: int = 256,
    training_overrides: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> tp.Dict[str, tp.Any]:
    training_overrides = training_overrides or {}
    patched_config = copy.deepcopy(model_config)
    model_section = patched_config.setdefault("model", {})
    conditioning = model_section.setdefault("conditioning", {})
    default_keys = conditioning.setdefault("default_keys", {})
    default_keys.setdefault("text_prompt", "prompt")
    text_max_length = _cap_text_length_for_mmdit_budget(patched_config, text_max_length)

    for conditioner in conditioning.get("configs", []):
        if conditioner.get("id") == "text_prompt" and conditioner.get("type") == "t5":
            conditioner.setdefault("config", {})
            current_max_length = conditioner["config"].get("max_length", 0)
            conditioner["config"]["max_length"] = max(current_max_length, text_max_length)

    training = patched_config.setdefault("training", {})
    training_defaults = {
        "learning_rate": 1e-5,
        "lora_learning_rate": 1e-4,
        "use_ema": True,
        "mask_padding": True,
        "cfg_dropout_prob": 0.1,
        "timestep_sampler": "uniform",
        "log_loss_info": False,
    }
    training_defaults.update(training_overrides)
    training.update(training_defaults)

    return patched_config


def _resolve_local_artifacts(run_config: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Optional[str]]:
    auxiliary_files = run_config.get("auxiliary_files", {})
    return {
        "config_path": run_config.get("model_config_path"),
        "model_ckpt_path": run_config.get("model_ckpt_path"),
        "vae_ckpt_path": auxiliary_files.get("vae_ckpt_path"),
        "synchformer_ckpt_path": auxiliary_files.get("synchformer_ckpt_path"),
    }


def extract_audio_prompt_num_samples(
    model_config: tp.Dict[str, tp.Any], fallback_sample_size: int
) -> int:
    conditioning_configs = model_config.get("model", {}).get("conditioning", {}).get("configs", [])
    for conditioner in conditioning_configs:
        if conditioner.get("id") != "audio_prompt":
            continue

        conditioner_config = conditioner.get("config", {})
        latent_seq_len = conditioner_config.get("latent_seq_len")
        downsampling_ratio = (
            conditioner_config.get("pretransform_config", {})
            .get("config", {})
            .get("downsampling_ratio")
        )
        if isinstance(latent_seq_len, int) and latent_seq_len > 0 and isinstance(downsampling_ratio, int) and downsampling_ratio > 0:
            return latent_seq_len * downsampling_ratio

    return fallback_sample_size


def _unfreeze_module(module: tp.Any) -> tp.List[str]:
    unfrozen = []
    if module is None:
        return unfrozen
    # Identify base weights inside LoRA wrappers so they stay frozen.
    lora_base_param_ids: tp.Set[int] = set()
    for child in module.modules():
        if isinstance(child, LoRALinear):
            for param in child.base.parameters():
                lora_base_param_ids.add(id(param))
    for name, parameter in module.named_parameters():
        if id(parameter) in lora_base_param_ids:
            continue
        parameter.requires_grad = True
        unfrozen.append(name)
    return unfrozen


def _active_conditioner_ids(
    model_config: tp.Dict[str, tp.Any], data_config: tp.Optional[tp.Dict[str, tp.Any]] = None
) -> tp.Set[str]:
    active = {"text_prompt"}
    data_config = data_config or {}
    if data_config.get("include_audio_conditioning", False):
        active.add("audio_prompt")
    if data_config.get("include_video_conditioning", False):
        active.add("video_prompt")

    conditioning = model_config.get("model", {}).get("conditioning", {})
    defined_ids = {conditioner.get("id") for conditioner in conditioning.get("configs", [])}
    return {conditioner_id for conditioner_id in active if conditioner_id in defined_ids}


def apply_trainable_scope(
    model: tp.Any,
    model_config: tp.Dict[str, tp.Any],
    run_config: tp.Dict[str, tp.Any],
) -> tp.Optional[tp.Dict[str, tp.Any]]:
    training_config = run_config.get("training") or {}
    scope = training_config.get("trainable_scope")
    if not scope:
        return None

    if scope not in {"asmr_continuation_lora", "multimodal_continuation_lora", "maf_continuation_lora"}:
        raise ValueError(f"Unsupported training.trainable_scope: {scope}")

    for _, parameter in model.named_parameters():
        parameter.requires_grad = False

    lora_parameter_names = []
    for name, parameter in model.named_parameters():
        if is_lora_parameter_name(name):
            parameter.requires_grad = True
            lora_parameter_names.append(name)

    unfrozen_modules: tp.List[str] = []
    if hasattr(model, "maf_block"):
        _unfreeze_module(model.maf_block)
        unfrozen_modules.append("maf_block")

    if scope in {"asmr_continuation_lora", "multimodal_continuation_lora"}:
        conditioner_registry = getattr(getattr(model, "conditioner", None), "conditioners", {})
        for conditioner_id in _active_conditioner_ids(model_config, run_config.get("data")):
            conditioner = conditioner_registry[conditioner_id] if conditioner_id in conditioner_registry else None
            if conditioner is None:
                continue

            for child_name, child_module in conditioner.named_children():
                if child_name.startswith("proj"):
                    _unfreeze_module(child_module)
                    unfrozen_modules.append(f"conditioner.{conditioner_id}.{child_name}")

            if hasattr(conditioner, "empty_audio_feat"):
                conditioner.empty_audio_feat.requires_grad = True
                unfrozen_modules.append(f"conditioner.{conditioner_id}.empty_audio_feat")

    trainable_params, total_params = count_parameters(model)
    all_trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    non_lora_parameter_names = [
        name for name in all_trainable_parameter_names if not is_lora_parameter_name(name)
    ]
    return {
        "scope": scope,
        "lora_parameter_names": lora_parameter_names,
        "non_lora_parameter_names": non_lora_parameter_names,
        "all_trainable_parameter_names": all_trainable_parameter_names,
        "unfrozen_modules": unfrozen_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def build_sample_weights(
    records: tp.Sequence[tp.Dict[str, tp.Any]],
    *,
    sample_strategy: str = "uniform",
    standalone_ratio: float = 0.3,
    continuation_ratio: float = 0.7,
) -> tp.Optional[tp.List[float]]:
    if sample_strategy != "weighted":
        return None

    type_to_records: tp.Dict[str, tp.List[int]] = {"standalone": [], "continuation": []}
    for index, record in enumerate(records):
        sample_type = record.get("sample_type")
        if sample_type in type_to_records:
            type_to_records[sample_type].append(index)

    if not type_to_records["standalone"] or not type_to_records["continuation"]:
        return None

    desired_ratios = {
        "standalone": float(standalone_ratio),
        "continuation": float(continuation_ratio),
    }
    weights = [1.0] * len(records)
    for sample_type, indices in type_to_records.items():
        if not indices:
            continue
        per_sample_weight = desired_ratios[sample_type] / len(indices)
        for index in indices:
            weights[index] = per_sample_weight
    return weights


def load_model_and_config(
    run_config: tp.Dict[str, tp.Any]
) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any], tp.Dict[str, tp.Optional[str]]]:
    pretrained_name = run_config.get("pretrained_name")
    cache_dir = run_config.get("cache_dir")
    if pretrained_name:
        artifact_paths = resolve_pretrained_artifacts(pretrained_name, cache_dir=cache_dir)
    else:
        artifact_paths = _resolve_local_artifacts(run_config)

    if artifact_paths["config_path"] is None or artifact_paths["model_ckpt_path"] is None:
        raise ValueError("Either pretrained_name or both model_config_path/model_ckpt_path must be provided.")

    model_config = _load_json(artifact_paths["config_path"])
    _patch_pretransform_ckpt_paths(model_config, artifact_paths.get("vae_ckpt_path"))
    model_config = apply_finetune_defaults(
        model_config,
        text_max_length=run_config.get("training", {}).get("text_max_length", 256),
        training_overrides=run_config.get("training", {}),
    )

    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(artifact_paths["model_ckpt_path"]))
    adapter_checkpoint_path = run_config.get("adapter_ckpt_path") or run_config.get("lora_path")
    if adapter_checkpoint_path:
        load_lora_checkpoint(model, adapter_checkpoint_path)
    return model, model_config, artifact_paths


def maybe_apply_lora(model: tp.Any, run_config: tp.Dict[str, tp.Any]) -> tp.Optional[tp.Dict[str, tp.Any]]:
    lora_config = run_config.get("lora") or {}
    if not lora_config.get("enabled", False):
        return None

    replaced_modules = inject_lora(
        model,
        rank=lora_config.get("rank", 8),
        alpha=lora_config.get("alpha", 16.0),
        dropout=lora_config.get("dropout", 0.0),
        target_patterns=lora_config.get("target_patterns"),
        exclude_patterns=lora_config.get("exclude_patterns"),
        freeze_non_lora=lora_config.get("freeze_non_lora", True),
    )
    if not replaced_modules:
        reused_modules = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
        if reused_modules:
            replaced_modules = reused_modules
        else:
            raise ValueError("LoRA was enabled, but no matching Linear modules were found for injection.")

    trainable_params, total_params = count_parameters(model)
    return {
        "replaced_modules": replaced_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def create_finetune_dataloaders(
    data_config: tp.Dict[str, tp.Any],
    model_config: tp.Dict[str, tp.Any],
    artifact_paths: tp.Dict[str, tp.Optional[str]],
    pretrained_name: tp.Optional[str] = None,
) -> tp.Tuple[DataLoader, tp.Optional[DataLoader]]:
    if "train_manifest" not in data_config:
        raise ValueError("data.train_manifest must be provided.")

    model_name = pretrained_name or data_config.get("model_name", "AudioX")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    sample_seconds = sample_size / sample_rate
    video_fps = model_config.get("video_fps", data_config.get("video_fps", 5))
    video_duration_seconds = data_config.get(
        "video_duration_seconds",
        model_config.get("video_duration_seconds", 10.0),
    )
    num_workers = data_config.get("num_workers", 0)
    audio_prompt_num_samples = extract_audio_prompt_num_samples(model_config, sample_size)

    shared_dataset_kwargs = {
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "prompt_format": data_config.get("prompt_format", "mixed"),
        "xml_compact": data_config.get("xml_compact", True),
        "xml_include_caption": data_config.get("xml_include_caption", True),
        "xml_max_events": data_config.get("xml_max_events", 8),
        "model_name": model_name,
        "include_video_conditioning": data_config.get("include_video_conditioning", False),
        "include_audio_conditioning": data_config.get("include_audio_conditioning", False),
        "video_fps": video_fps,
        "video_duration_seconds": video_duration_seconds,
        "audio_prompt_num_samples": data_config.get("audio_prompt_num_samples", audio_prompt_num_samples),
        "synchformer_ckpt_path": artifact_paths.get("synchformer_ckpt_path"),
        "compute_video_sync_on_the_fly": data_config.get("compute_video_sync_on_the_fly", False),
    }

    train_dataset = IFCapsFineTuneDataset(
        manifest_path=data_config["train_manifest"],
        random_crop=data_config.get("random_crop", True),
        seed=data_config.get("seed", 0),
        sample_text_prompt_candidates=data_config.get("sample_text_prompt_candidates", True),
        **shared_dataset_kwargs,
    )
    sample_weights = build_sample_weights(
        train_dataset.records,
        sample_strategy=data_config.get("sample_strategy", "uniform"),
        standalone_ratio=data_config.get("standalone_ratio", 0.3),
        continuation_ratio=data_config.get("continuation_ratio", 0.7),
    )
    train_sampler = None
    if sample_weights is not None:
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get("batch_size", 1),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_audiox_batch,
        drop_last=data_config.get("drop_last", False),
        pin_memory=data_config.get("pin_memory", False),
        persistent_workers=num_workers > 0 and data_config.get("persistent_workers", False),
    )

    val_manifest = data_config.get("val_manifest")
    if not val_manifest:
        return train_loader, None

    val_dataset = IFCapsFineTuneDataset(
        manifest_path=val_manifest,
        random_crop=False,
        seed=data_config.get("seed", 0) + 100000,
        sample_text_prompt_candidates=data_config.get("sample_text_prompt_candidates_for_eval", False),
        **shared_dataset_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get("eval_batch_size", data_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audiox_batch,
        drop_last=False,
        pin_memory=data_config.get("pin_memory", False),
        persistent_workers=num_workers > 0 and data_config.get("persistent_workers", False),
    )
    return train_loader, val_loader


def create_trainer(
    trainer_config: tp.Dict[str, tp.Any],
    checkpoint_config: tp.Dict[str, tp.Any],
    wandb_config: tp.Optional[tp.Dict[str, tp.Any]],
    output_dir: tp.Union[str, Path],
) -> pl.Trainer:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [LearningRateMonitor(logging_interval="step")]
    checkpoint_enabled = checkpoint_config.get("enabled", True)
    periodic_lora_epochs = int(checkpoint_config.get("every_n_epochs", 0) or 0)
    use_periodic_lora_checkpoints = checkpoint_enabled and checkpoint_config.get("save_lora_only", False) and periodic_lora_epochs > 0
    save_resume_checkpoints = checkpoint_enabled and checkpoint_config.get("save_resume_checkpoints", False)
    if checkpoint_enabled and (not use_periodic_lora_checkpoints or save_resume_checkpoints):
        checkpoint_filename = checkpoint_config.get("filename", "step={step}")
        resume_checkpoint_filename = checkpoint_config.get("resume_checkpoint_filename", checkpoint_filename)
        checkpoint_kwargs: tp.Dict[str, tp.Any] = {
            "dirpath": checkpoint_config.get("resume_checkpoint_dirpath", checkpoint_config.get("dirpath", str(output_dir / "checkpoints"))),
            "filename": resume_checkpoint_filename if save_resume_checkpoints else checkpoint_filename,
            "save_last": checkpoint_config.get("save_last", True),
            "save_weights_only": checkpoint_config.get(
                "resume_save_weights_only" if save_resume_checkpoints else "save_weights_only",
                False,
            ),
        }

        if checkpoint_config.get("monitor"):
            checkpoint_kwargs.update(
                {
                    "monitor": checkpoint_config["monitor"],
                    "mode": checkpoint_config.get("mode", "min"),
                    "save_top_k": checkpoint_config.get("save_top_k", 2),
                }
            )
        else:
            checkpoint_kwargs.update(
                {
                    "save_top_k": checkpoint_config.get("save_top_k", -1),
                }
            )
            if "every_n_epochs" in checkpoint_config:
                checkpoint_kwargs["every_n_epochs"] = checkpoint_config.get(
                    "resume_every_n_epochs" if save_resume_checkpoints else "every_n_epochs",
                    checkpoint_config["every_n_epochs"],
                )
                checkpoint_kwargs["save_on_train_epoch_end"] = checkpoint_config.get("save_on_train_epoch_end", True)
            else:
                checkpoint_kwargs["every_n_train_steps"] = checkpoint_config.get(
                    "resume_every_n_train_steps" if save_resume_checkpoints else "every_n_train_steps",
                    checkpoint_config.get("every_n_train_steps", 1000),
                )
        callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

    loggers: tp.List[tp.Any] = [CSVLogger(save_dir=str(output_dir), name=trainer_config.get("logger_name", "logs"))]
    wandb_config = wandb_config or {}
    if wandb_config.get("enabled", False):
        api_key = wandb_config.get("api_key")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        for env_key in ("project", "entity", "name", "notes", "group", "job_type", "tags"):
            env_value = wandb_config.get(env_key)
            if env_value is None:
                continue
            if isinstance(env_value, list):
                env_value = ",".join(str(item) for item in env_value)
            os.environ.setdefault(f"WANDB_{env_key.upper()}", str(env_value))

        loggers.append(
            WandbLogger(
                project=wandb_config.get("project", "audiox-finetune"),
                entity=wandb_config.get("entity"),
                name=wandb_config.get("name"),
                save_dir=str(output_dir),
                offline=wandb_config.get("offline", False),
                log_model=wandb_config.get("log_model", False),
                tags=wandb_config.get("tags"),
                notes=wandb_config.get("notes"),
                group=wandb_config.get("group"),
                job_type=wandb_config.get("job_type"),
                config=wandb_config.get("config"),
            )
        )

    trainer_kwargs = {
        "accelerator": trainer_config.get("accelerator", "auto"),
        "devices": trainer_config.get("devices", "auto"),
        "default_root_dir": str(output_dir),
        "precision": trainer_config.get("precision", "16-mixed"),
        "max_epochs": trainer_config.get("max_epochs", 1),
        "max_steps": trainer_config.get("max_steps", -1),
        "accumulate_grad_batches": trainer_config.get("accumulate_grad_batches", 1),
        "gradient_clip_val": trainer_config.get("gradient_clip_val", 1.0),
        "log_every_n_steps": trainer_config.get("log_every_n_steps", 10),
        "callbacks": callbacks,
        "logger": loggers[0] if len(loggers) == 1 else loggers,
        "num_sanity_val_steps": trainer_config.get("num_sanity_val_steps", 0),
        "enable_checkpointing": checkpoint_enabled,
    }

    for optional_key in ("limit_train_batches", "limit_val_batches", "val_check_interval"):
        if optional_key in trainer_config:
            trainer_kwargs[optional_key] = trainer_config[optional_key]

    return pl.Trainer(**trainer_kwargs)


def _resolve_hf_token(upload_config: tp.Dict[str, tp.Any]) -> tp.Optional[str]:
    explicit_token = upload_config.get("token")
    if explicit_token:
        return explicit_token

    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
    return None


def _join_repo_path(prefix: str, filename: str) -> str:
    parts = [part.strip("/") for part in (prefix, filename) if part]
    return "/".join(parts)


def _build_lora_artifact_payload(
    training_wrapper: nn.Module,
    *,
    global_step: int,
    epoch: tp.Optional[int],
    lora_info: tp.Dict[str, tp.Any],
    lora_config: tp.Optional[tp.Dict[str, tp.Any]],
    trainable_scope_info: tp.Optional[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    trainable_scope_state_dict = {}
    if trainable_scope_info:
        trainable_scope_state_dict = extract_parameter_state_dict(
            training_wrapper.diffusion,
            trainable_scope_info.get("non_lora_parameter_names", []),
        )
    payload = {
        "global_step": global_step,
        "lora_info": lora_info,
        "lora_config": lora_config,
        "trainable_scope_info": trainable_scope_info,
        "trainable_scope_state_dict": trainable_scope_state_dict,
        "lora_state_dict": extract_lora_state_dict(training_wrapper.diffusion),
    }
    if epoch is not None:
        payload["epoch"] = epoch
    return payload


class EpochLoRACheckpointCallback(pl.Callback):
    def __init__(
        self,
        *,
        checkpoint_config: tp.Dict[str, tp.Any],
        output_dir: tp.Union[str, Path],
        run_config: tp.Dict[str, tp.Any],
        lora_info: tp.Dict[str, tp.Any],
        trainable_scope_info: tp.Optional[tp.Dict[str, tp.Any]],
    ) -> None:
        super().__init__()
        self.dirpath = Path(checkpoint_config.get("dirpath", Path(output_dir) / "checkpoints"))
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.resume_dirpath = Path(
            checkpoint_config.get("resume_checkpoint_dirpath", checkpoint_config.get("dirpath", Path(output_dir) / "checkpoints"))
        )
        self.resume_dirpath.mkdir(parents=True, exist_ok=True)
        self.filename = checkpoint_config.get("filename", "epoch={epoch}-step={step}")
        self.save_last = checkpoint_config.get("save_last", True)
        self.every_n_epochs = max(int(checkpoint_config.get("every_n_epochs", 1) or 1), 1)
        self.save_on_train_epoch_end = checkpoint_config.get("save_on_train_epoch_end", True)
        self.run_config = run_config
        self.lora_info = lora_info
        self.trainable_scope_info = trainable_scope_info
        self.upload_config = run_config.get("huggingface") or run_config.get("hf_upload") or {}
        self.upload_enabled = bool(
            self.upload_config.get("enabled", False) and self.upload_config.get("upload_checkpoint_dir", False)
        )
        self.repo_id = self.upload_config.get("repo_id")
        self.repo_type = self.upload_config.get("repo_type", "model")
        self.path_prefix = self.upload_config.get("path_prefix", "").strip("/")
        self.remote_checkpoint_dir = self.upload_config.get("checkpoint_dir_path_in_repo", "checkpoints").strip("/")
        self.commit_message = self.upload_config.get("commit_message", "Upload AudioX fine-tune artifacts")
        self._hf_api: tp.Optional[HfApi] = None

    def _format_name(self, trainer: pl.Trainer) -> str:
        name = self.filename.format(epoch=trainer.current_epoch + 1, step=trainer.global_step)
        if Path(name).suffix:
            return name
        return f"{name}.pt"

    def _upload_checkpoint(self, checkpoint_path: Path) -> None:
        if not self.upload_enabled or not self.repo_id:
            return
        if self._hf_api is None:
            token = _resolve_hf_token(self.upload_config)
            self._hf_api = HfApi(token=token)
            self._hf_api.create_repo(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                private=self.upload_config.get("private", True),
                exist_ok=True,
            )
        remote_prefix = _join_repo_path(self.path_prefix, self.remote_checkpoint_dir)
        self._hf_api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=_join_repo_path(remote_prefix, checkpoint_path.name),
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=self.commit_message,
        )

    def _upload_resume_checkpoints_for_epoch(self, epoch_number: int) -> None:
        if not self.upload_enabled or not self.repo_id:
            return

        uploaded_paths: tp.Set[Path] = set()
        for checkpoint_dir in {self.dirpath, self.resume_dirpath}:
            if not checkpoint_dir.exists():
                continue
            for child in sorted(checkpoint_dir.glob("*.ckpt")):
                if f"epoch={epoch_number}" in child.name or child.name == "last.ckpt":
                    if child in uploaded_paths:
                        continue
                    self._upload_checkpoint(child)
                    uploaded_paths.add(child)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.save_on_train_epoch_end:
            return
        epoch_number = trainer.current_epoch + 1
        if epoch_number % self.every_n_epochs != 0:
            return

        checkpoint_path = self.dirpath / self._format_name(trainer)
        payload = _build_lora_artifact_payload(
            pl_module,
            global_step=trainer.global_step,
            epoch=epoch_number,
            lora_info=self.lora_info,
            lora_config=self.run_config.get("lora"),
            trainable_scope_info=self.trainable_scope_info,
        )
        torch.save(payload, checkpoint_path)
        self._upload_checkpoint(checkpoint_path)

        if self.save_last:
            last_path = self.dirpath / "last-lora-state.pt"
            torch.save(payload, last_path)
            self._upload_checkpoint(last_path)

        self._upload_resume_checkpoints_for_epoch(epoch_number)


def maybe_upload_huggingface_artifacts(
    run_config: tp.Dict[str, tp.Any],
    *,
    source_config_path: tp.Union[str, Path],
    resolved_model_config_path: tp.Union[str, Path],
    final_checkpoint_path: tp.Optional[tp.Union[str, Path]],
) -> tp.Optional[tp.Dict[str, tp.Any]]:
    upload_config = run_config.get("huggingface") or run_config.get("hf_upload") or {}
    if not upload_config.get("enabled", False):
        return None

    repo_id = upload_config.get("repo_id")
    if not repo_id:
        raise ValueError("huggingface.repo_id must be provided when Hugging Face upload is enabled.")

    repo_type = upload_config.get("repo_type", "model")
    private = upload_config.get("private", True)
    token = _resolve_hf_token(upload_config)
    path_prefix = upload_config.get("path_prefix", "").strip("/")
    commit_message = upload_config.get("commit_message", "Upload AudioX fine-tune artifacts")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    uploaded_files: tp.List[tp.Dict[str, str]] = []

    def upload_if_exists(
        local_path: tp.Optional[tp.Union[str, Path]],
        *,
        enabled: bool,
        remote_name: tp.Optional[str] = None,
        required: bool = False,
    ) -> None:
        if not enabled:
            return
        if local_path is None:
            if required:
                raise ValueError(
                    f"Hugging Face upload expected an artifact for {remote_name or 'final checkpoint'}, "
                    "but it was not produced."
                )
            return

        resolved_path = Path(local_path)
        if not resolved_path.exists():
            if required:
                raise FileNotFoundError(f"Hugging Face upload artifact is missing: {resolved_path}")
            return

        path_in_repo = _join_repo_path(path_prefix, remote_name or resolved_path.name)
        api.upload_file(
            path_or_fileobj=str(resolved_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        uploaded_files.append(
            {
                "local_path": str(resolved_path),
                "path_in_repo": path_in_repo,
            }
        )

    def upload_dir_if_exists(
        local_dir: tp.Optional[tp.Union[str, Path]],
        *,
        enabled: bool,
        remote_prefix: tp.Optional[str] = None,
    ) -> None:
        if not enabled or local_dir is None:
            return
        resolved_dir = Path(local_dir)
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            return
        base_prefix = _join_repo_path(path_prefix, remote_prefix or resolved_dir.name)
        for child in sorted(path for path in resolved_dir.rglob("*") if path.is_file()):
            relative_path = child.relative_to(resolved_dir).as_posix()
            path_in_repo = _join_repo_path(base_prefix, relative_path)
            api.upload_file(
                path_or_fileobj=str(child),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            uploaded_files.append(
                {
                    "local_path": str(child),
                    "path_in_repo": path_in_repo,
                }
            )

    source_config_path = Path(source_config_path)
    manifest_summary_path = source_config_path.parent / "manifest_summary.json"

    upload_if_exists(
        final_checkpoint_path,
        enabled=upload_config.get("upload_final_checkpoint", True),
        remote_name=upload_config.get("final_checkpoint_path_in_repo"),
        required=upload_config.get("upload_final_checkpoint", True),
    )
    upload_if_exists(
        resolved_model_config_path,
        enabled=upload_config.get("upload_resolved_config", True),
        remote_name=upload_config.get("resolved_model_config_path_in_repo", "resolved_model_config.json"),
    )
    upload_if_exists(
        source_config_path,
        enabled=upload_config.get("upload_run_config", True),
        remote_name=upload_config.get("run_config_path_in_repo", "run_config.json"),
    )
    upload_if_exists(
        manifest_summary_path,
        enabled=upload_config.get("upload_manifest_summary", True),
        remote_name=upload_config.get("manifest_summary_path_in_repo", "manifest_summary.json"),
    )
    upload_if_exists(
        upload_config.get("train_log_path"),
        enabled=upload_config.get("upload_train_log", False),
        remote_name=upload_config.get("train_log_path_in_repo"),
    )
    upload_dir_if_exists(
        upload_config.get("checkpoint_dir") or run_config.get("checkpointing", {}).get("dirpath"),
        enabled=upload_config.get("upload_checkpoint_dir", False),
        remote_prefix=upload_config.get("checkpoint_dir_path_in_repo", "checkpoints"),
    )

    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "uploaded_files": uploaded_files,
    }


def run_finetune(config_path: tp.Union[str, Path]) -> tp.Dict[str, tp.Any]:
    run_config = _load_json(config_path)
    output_dir = Path(run_config.get("output_dir", "./outputs/audiox_finetune")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_config = run_config.get("checkpointing", {})

    model, model_config, artifact_paths = load_model_and_config(run_config)
    lora_info = maybe_apply_lora(model, run_config)
    trainable_scope_info = apply_trainable_scope(model, model_config, run_config)
    trainable_params, total_params = count_parameters(model)
    if lora_info is not None:
        lora_info["trainable_params"] = trainable_params
        lora_info["total_params"] = total_params
        lora_info["trainable_scope"] = trainable_scope_info

    resolved_config_path = output_dir / "resolved_model_config.json"
    with resolved_config_path.open("w") as handle:
        json.dump(model_config, handle, indent=2)

    training_wrapper = create_training_wrapper_from_config(model_config, model)
    train_loader, val_loader = create_finetune_dataloaders(
        run_config.get("data", {}),
        model_config,
        artifact_paths,
        pretrained_name=run_config.get("pretrained_name"),
    )
    trainer = create_trainer(
        run_config.get("trainer", {}),
        checkpoint_config,
        run_config.get("wandb"),
        output_dir,
    )
    if checkpoint_config.get("enabled", True) and checkpoint_config.get("save_lora_only", False):
        periodic_lora_epochs = int(checkpoint_config.get("every_n_epochs", 0) or 0)
        if periodic_lora_epochs > 0:
            if lora_info is None:
                raise ValueError("checkpointing.save_lora_only requires LoRA to be enabled.")
            trainer.callbacks.append(
                EpochLoRACheckpointCallback(
                    checkpoint_config=checkpoint_config,
                    output_dir=output_dir,
                    run_config=run_config,
                    lora_info=lora_info,
                    trainable_scope_info=trainable_scope_info,
                )
            )

    resume_from_checkpoint = run_config.get("resume_from_checkpoint")
    if val_loader is None:
        trainer.fit(training_wrapper, train_dataloaders=train_loader, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(
            training_wrapper,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_from_checkpoint,
        )

    final_checkpoint_path = None
    final_resume_checkpoint_path = None
    save_final_checkpoint = checkpoint_config.get("save_final_checkpoint", checkpoint_config.get("enabled", True))
    if trainer.global_step > 0 and save_final_checkpoint:
        final_checkpoint_dir = Path(checkpoint_config.get("dirpath", output_dir / "checkpoints"))
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_config.get("save_lora_only", False):
            if lora_info is None:
                raise ValueError("checkpointing.save_lora_only requires LoRA to be enabled.")
            final_checkpoint_path = final_checkpoint_dir / run_config.get(
                "final_checkpoint_name", "final-lora-state.pt"
            )
            torch.save(
                _build_lora_artifact_payload(
                    training_wrapper,
                    global_step=trainer.global_step,
                    epoch=trainer.current_epoch + 1 if trainer.current_epoch is not None else None,
                    lora_info=lora_info,
                    lora_config=run_config.get("lora"),
                    trainable_scope_info=trainable_scope_info,
                ),
                final_checkpoint_path,
            )
            if checkpoint_config.get("save_resume_checkpoints", False):
                final_resume_checkpoint_dir = Path(
                    checkpoint_config.get("resume_checkpoint_dirpath", checkpoint_config.get("dirpath", output_dir / "checkpoints"))
                )
                final_resume_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                final_resume_checkpoint_path = final_resume_checkpoint_dir / checkpoint_config.get(
                    "final_resume_checkpoint_name",
                    "final-resume.ckpt",
                )
                trainer.save_checkpoint(
                    str(final_resume_checkpoint_path),
                    weights_only=checkpoint_config.get("resume_save_weights_only", False),
                )
        else:
            final_checkpoint_path = final_checkpoint_dir / run_config.get("final_checkpoint_name", "final-step.ckpt")
            trainer.save_checkpoint(
                str(final_checkpoint_path),
                weights_only=checkpoint_config.get("save_weights_only", False),
            )

    huggingface_upload_info = maybe_upload_huggingface_artifacts(
        run_config,
        source_config_path=config_path,
        resolved_model_config_path=resolved_config_path,
        final_checkpoint_path=final_checkpoint_path,
    )

    return {
        "artifact_paths": artifact_paths,
        "model_config": model_config,
        "lora_info": lora_info,
        "trainable_scope_info": trainable_scope_info,
        "output_dir": str(output_dir),
        "resolved_model_config_path": str(resolved_config_path),
        "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path else None,
        "final_resume_checkpoint_path": str(final_resume_checkpoint_path) if final_resume_checkpoint_path else None,
        "huggingface_upload_info": huggingface_upload_info,
        "trainer": trainer,
        "training_wrapper": training_wrapper,
    }
