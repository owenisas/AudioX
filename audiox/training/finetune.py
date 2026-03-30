import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from huggingface_hub import hf_hub_download
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from ..data.ifcaps import IFCapsFineTuneDataset, collate_finetune_batch
from ..models.factory import create_model_from_config
from ..models.utils import load_ckpt_state_dict
from .factory import create_training_wrapper_from_config
from .utils import copy_state_dict


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _download_pretrained_checkpoint(repo_id: str) -> Tuple[str, str]:
    config_path = hf_hub_download(repo_id, filename="config.json", repo_type="model")
    try:
        ckpt_path = hf_hub_download(repo_id, filename="model.safetensors", repo_type="model")
    except Exception:
        ckpt_path = hf_hub_download(repo_id, filename="model.ckpt", repo_type="model")
    return config_path, ckpt_path


def apply_finetune_defaults(
    model_config: Dict[str, Any],
    *,
    text_max_length: int = 256,
) -> Dict[str, Any]:
    patched = copy.deepcopy(model_config)

    conditioning = patched.setdefault("model", {}).setdefault("conditioning", {})
    default_keys = dict(conditioning.get("default_keys", {}))
    default_keys.setdefault("text_prompt", "prompt")
    conditioning["default_keys"] = default_keys

    for conditioner in conditioning.get("configs", []):
        if conditioner.get("id") == "text_prompt" and conditioner.get("type") == "t5":
            conditioner.setdefault("config", {})
            conditioner["config"]["max_length"] = max(
                int(conditioner["config"].get("max_length", 0)),
                int(text_max_length),
            )

    return patched


def _ensure_training_config(model_config: Dict[str, Any], finetune_config: Dict[str, Any]) -> Dict[str, Any]:
    patched = copy.deepcopy(model_config)
    training_cfg = patched.setdefault("training", {})

    training_cfg.setdefault("learning_rate", finetune_config.get("learning_rate", 1e-5))
    training_cfg.setdefault("use_ema", finetune_config.get("use_ema", True))
    training_cfg.setdefault("mask_padding", finetune_config.get("mask_padding", True))
    training_cfg.setdefault("mask_padding_dropout", finetune_config.get("mask_padding_dropout", 0.0))
    training_cfg.setdefault("cfg_dropout_prob", finetune_config.get("cfg_dropout_prob", 0.1))
    training_cfg.setdefault("timestep_sampler", finetune_config.get("timestep_sampler", "uniform"))
    training_cfg.setdefault("pre_encoded", finetune_config.get("pre_encoded", False))
    if "optimizer_configs" in finetune_config:
        training_cfg["optimizer_configs"] = finetune_config["optimizer_configs"]

    return patched


def create_finetune_dataloaders(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader]]:
    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])
    video_fps = int(model_config.get("video_fps", 5))

    dataset_kwargs = {
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "video_fps": video_fps,
        "prompt_format": data_config.get("prompt_format", "mixed"),
        "xml_compact": data_config.get("xml_compact", True),
        "xml_include_caption": data_config.get("xml_include_caption", True),
        "xml_max_events": int(data_config.get("xml_max_events", 8)),
        "prompt_seed": int(data_config.get("prompt_seed", 0)),
        "random_crop": bool(data_config.get("random_crop", True)),
    }

    train_dataset = IFCapsFineTuneDataset(
        manifest_path=data_config["train_manifest"],
        **dataset_kwargs,
    )

    loader_kwargs = {
        "batch_size": int(data_config.get("batch_size", 1)),
        "num_workers": int(data_config.get("num_workers", 0)),
        "pin_memory": bool(data_config.get("pin_memory", False)),
        "collate_fn": collate_finetune_batch,
        "persistent_workers": int(data_config.get("num_workers", 0)) > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(data_config.get("shuffle", True)),
        drop_last=bool(data_config.get("drop_last", True)),
        **loader_kwargs,
    )

    val_manifest = data_config.get("val_manifest")
    if not val_manifest:
        return train_loader, None

    val_dataset = IFCapsFineTuneDataset(
        manifest_path=val_manifest,
        **dataset_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def create_trainer(trainer_config: Dict[str, Any], output_dir: str) -> pl.Trainer:
    os.makedirs(output_dir, exist_ok=True)

    logger = CSVLogger(save_dir=output_dir, name="logs")
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{step:08d}-{train_loss:.4f}",
        save_top_k=int(trainer_config.get("save_top_k", 3)),
        monitor=trainer_config.get("monitor", "train/loss"),
        mode=trainer_config.get("monitor_mode", "min"),
        every_n_train_steps=int(trainer_config.get("checkpoint_every_n_train_steps", 500)),
        save_last=True,
    )

    trainer_kwargs = {
        "accelerator": trainer_config.get("accelerator", "auto"),
        "devices": trainer_config.get("devices", "auto"),
        "precision": trainer_config.get("precision", "16-mixed"),
        "default_root_dir": output_dir,
        "max_steps": int(trainer_config.get("max_steps", 1000)),
        "accumulate_grad_batches": int(trainer_config.get("accumulate_grad_batches", 1)),
        "gradient_clip_val": float(trainer_config.get("gradient_clip_val", 0.0)),
        "log_every_n_steps": int(trainer_config.get("log_every_n_steps", 10)),
        "val_check_interval": trainer_config.get("val_check_interval", 250),
        "limit_val_batches": trainer_config.get("limit_val_batches", 1.0),
        "callbacks": [checkpoint],
        "logger": logger,
        "num_sanity_val_steps": int(trainer_config.get("num_sanity_val_steps", 0)),
    }
    return pl.Trainer(**trainer_kwargs)


def load_model_and_config(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    model_source = config.get("model", {})
    resume_from_checkpoint = model_source.get("resume_from_checkpoint")

    if model_source.get("pretrained_name"):
        downloaded_config, downloaded_ckpt = _download_pretrained_checkpoint(model_source["pretrained_name"])
        raw_model_config = _load_json(downloaded_config)
        checkpoint_path = model_source.get("checkpoint_path") or downloaded_ckpt
    else:
        raw_model_config = _load_json(model_source["model_config_path"])
        checkpoint_path = model_source.get("checkpoint_path")

    patched_model_config = apply_finetune_defaults(
        raw_model_config,
        text_max_length=int(model_source.get("text_max_length", 256)),
    )
    patched_model_config = _ensure_training_config(
        patched_model_config,
        finetune_config=config.get("training", {}),
    )

    model = create_model_from_config(patched_model_config)
    if checkpoint_path:
        copy_state_dict(model, load_ckpt_state_dict(checkpoint_path))

    return model, patched_model_config, resume_from_checkpoint


def run_finetune(config_path: str) -> None:
    config = _load_json(config_path)
    seed_everything(int(config.get("seed", 42)), workers=True)

    model, model_config, resume_from_checkpoint = load_model_and_config(config)
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    train_loader, val_loader = create_finetune_dataloaders(config["data"], model_config)

    output_dir = config.get("output_dir", os.path.join(os.getcwd(), "finetune_runs"))
    trainer = create_trainer(config.get("trainer", {}), output_dir=output_dir)

    trainer.fit(
        training_wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint,
    )
