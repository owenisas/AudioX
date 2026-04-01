import argparse
import json
from pathlib import Path

from audiox.data.asmr import (
    ASMR_SAMPLE_RATE,
    ASMR_SAMPLE_SIZE,
    ASMR_SECONDS_TOTAL,
    prepare_asmr_continuation_manifests,
)
from audiox.models.lora import DEFAULT_LORA_TARGET_PATTERNS


def build_run_config(
    *,
    output_dir: str,
    train_manifest: str,
    val_manifest: str | None,
    cache_dir: str,
    pretrained_name: str,
    run_name: str,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_offline: bool,
    include_video_conditioning: bool,
    standalone_ratio: float,
    continuation_ratio: float,
) -> dict:
    return {
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "pretrained_name": pretrained_name,
        "data": {
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "batch_size": 2,
            "eval_batch_size": 1,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": False,
            "drop_last": False,
            "random_crop": False,
            "seed": 0,
            "prompt_format": "natural",
            "include_video_conditioning": include_video_conditioning,
            "include_audio_conditioning": True,
            "compute_video_sync_on_the_fly": False,
            "sample_strategy": "weighted",
            "standalone_ratio": standalone_ratio,
            "continuation_ratio": continuation_ratio,
            "sample_text_prompt_candidates": True,
            "sample_text_prompt_candidates_for_eval": False,
            "video_duration_seconds": ASMR_SECONDS_TOTAL,
        },
        "training": {
            "text_max_length": 256,
            "learning_rate": 1e-5,
            "lora_learning_rate": 1e-4,
            "use_ema": True,
            "mask_padding": True,
            "cfg_dropout_prob": 0.1,
            "timestep_sampler": "uniform",
            "log_loss_info": False,
            "trainable_scope": "asmr_continuation_lora",
        },
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "precision": "16-mixed",
            "max_epochs": 1,
            "max_steps": 6000,
            "accumulate_grad_batches": 4,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "num_sanity_val_steps": 0,
            "val_check_interval": 500,
        },
        "checkpointing": {
            "dirpath": f"{output_dir}/checkpoints",
            "filename": "step={step}",
            "enabled": True,
            "save_last": True,
            "monitor": "val/loss",
            "mode": "min",
            "save_top_k": 3,
            "every_n_train_steps": 500,
            "save_final_checkpoint": True,
            "save_lora_only": True,
            "final_checkpoint_name": "final-lora-state.pt",
        },
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32.0,
            "dropout": 0.05,
            "freeze_non_lora": True,
            "target_patterns": list(DEFAULT_LORA_TARGET_PATTERNS),
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
    parser = argparse.ArgumentParser(description="Prepare an ASMR continuation fine-tune run for AudioX.")
    parser.add_argument("--chunks-manifest", required=True, help="Path to ASMR chunk metadata JSON/JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory to write manifests, config, and outputs.")
    parser.add_argument("--cache-dir", default=".hf_home", help="Hugging Face cache directory.")
    parser.add_argument(
        "--pretrained-name",
        default="HKUSTAudio/AudioX-MAF-MMDiT",
        help="Hugging Face model identifier to fine-tune from.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of sequences held out for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for the sequence split.")
    parser.add_argument(
        "--standalone-ratio",
        type=float,
        default=0.3,
        help="Target standalone sampling ratio during training.",
    )
    parser.add_argument(
        "--continuation-ratio",
        type=float,
        default=0.7,
        help="Target continuation sampling ratio during training.",
    )
    parser.add_argument("--wandb-project", default="audiox-finetune", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity.")
    parser.add_argument("--wandb-offline", action="store_true", help="Enable offline W&B logging.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to the chunk-manifest stem with an ASMR prefix.",
    )
    parser.add_argument(
        "--include-video-conditioning",
        action="store_true",
        help="Enable video conditioning in the generated config.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)

    manifest_info = prepare_asmr_continuation_manifests(
        args.chunks_manifest,
        output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        seconds_total=ASMR_SECONDS_TOTAL,
    )

    run_name = args.run_name or f"asmr-continuation-{Path(args.chunks_manifest).stem}"
    run_config = build_run_config(
        output_dir=str(output_dir / "outputs"),
        train_manifest=manifest_info["train_manifest_path"],
        val_manifest=manifest_info["val_manifest_path"],
        cache_dir=args.cache_dir,
        pretrained_name=args.pretrained_name,
        run_name=run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_offline=args.wandb_offline,
        include_video_conditioning=args.include_video_conditioning,
        standalone_ratio=args.standalone_ratio,
        continuation_ratio=args.continuation_ratio,
    )

    config_path = output_dir / "config_asmr_continuation.json"
    config_path.write_text(json.dumps(run_config, indent=2))
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(manifest_info, indent=2))

    print(manifest_info["train_manifest_path"])
    print(manifest_info["val_manifest_path"] or "")
    print(config_path)


if __name__ == "__main__":
    main()
