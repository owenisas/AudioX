import argparse
import json
from pathlib import Path


def _extract_categories(label: dict) -> list[str]:
    category = label.get("category") or label.get("categories") or []
    if isinstance(category, dict):
        return [key for key in category.keys() if key]
    if isinstance(category, list):
        return [str(item) for item in category if str(item).strip()]
    if isinstance(category, str) and category.strip():
        return [category.strip()]
    return []


def _join_optional_list(value):
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) if parts else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_manifest_row(label: dict, audio_path: str) -> dict:
    music = label.get("music") or {}
    return {
        "audio_path": audio_path,
        "caption": label.get("caption"),
        "category": _extract_categories(label),
        "sed": label.get("sed") or label.get("SED"),
        "time_relation": label.get("time_relation"),
        "genre": music.get("genre"),
        "mood": _join_optional_list(music.get("mood")),
        "instrument": _join_optional_list(music.get("instrumentation") or music.get("instrument")),
        "tempo": music.get("tempo"),
    }


def build_run_config(
    output_dir: str,
    train_manifest: str,
    cache_dir: str,
    pretrained_name: str,
    run_name: str,
    wandb_project: str,
    wandb_entity: str | None = None,
    wandb_offline: bool = False,
) -> dict:
    return {
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "pretrained_name": pretrained_name,
        "data": {
            "train_manifest": train_manifest,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
            "random_crop": False,
            "seed": 0,
            "prompt_format": "xml",
            "xml_compact": True,
            "xml_include_caption": True,
            "xml_max_events": 8,
            "include_video_conditioning": False,
            "include_audio_conditioning": False,
            "compute_video_sync_on_the_fly": False,
        },
        "training": {
            "text_max_length": 256,
            "learning_rate": 1e-4,
            "use_ema": False,
            "mask_padding": True,
            "cfg_dropout_prob": 0.1,
            "timestep_sampler": "uniform",
            "log_loss_info": False,
        },
        "trainer": {
            "accelerator": "cuda",
            "devices": 1,
            "precision": "16-mixed",
            "max_epochs": 1,
            "max_steps": 1,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 0.0,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "limit_train_batches": 1,
        },
        "checkpointing": {
            "dirpath": f"{output_dir}/checkpoints",
            "enabled": False,
            "save_last": False,
            "save_top_k": 0,
            "every_n_train_steps": 0,
            "save_final_checkpoint": False,
        },
        "lora": {
            "enabled": True,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
            "freeze_non_lora": True,
            "target_patterns": [
                "to_q",
                "to_kv",
                "to_qkv",
                "to_out",
                ".qkv",
                "proj_mm_tokens",
                "proj_mm_seq_len",
                "gating_network.0",
                "gating_network.2",
            ],
        },
        "wandb": {
            "enabled": True,
            "project": wandb_project,
            "entity": wandb_entity,
            "name": run_name,
            "offline": wandb_offline,
            "log_model": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a one-sample IFCaps LoRA smoke run for AudioX.")
    parser.add_argument("--label", required=True, help="Path to the IFCaps label JSON.")
    parser.add_argument("--audio", required=True, help="Path to the target audio file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write train.jsonl, config_cuda_lora.json, and outputs/ into.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".hf_home",
        help="Directory used by huggingface_hub for model downloads.",
    )
    parser.add_argument(
        "--pretrained-name",
        default="HKUSTAudio/AudioX-MAF-MMDiT",
        help="Hugging Face model identifier to fine-tune from.",
    )
    parser.add_argument(
        "--wandb-project",
        default="audiox-finetune",
        help="Weights & Biases project name for the smoke run.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Enable offline Weights & Biases logging.",
    )
    args = parser.parse_args()

    label_path = Path(args.label)
    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)

    label = json.loads(label_path.read_text())
    manifest_row = build_manifest_row(label, str(audio_path))
    train_manifest_path = output_dir / "train.jsonl"
    train_manifest_path.write_text(json.dumps(manifest_row) + "\n")

    run_config = build_run_config(
        output_dir=str(output_dir / "outputs"),
        train_manifest=str(train_manifest_path),
        cache_dir=args.cache_dir,
        pretrained_name=args.pretrained_name,
        run_name=f"ifcaps-lora-smoke-{label.get('clip_id', label_path.stem)}",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_offline=args.wandb_offline,
    )
    config_path = output_dir / "config_cuda_lora.json"
    config_path.write_text(json.dumps(run_config, indent=2))

    print(train_manifest_path)
    print(config_path)


if __name__ == "__main__":
    main()
