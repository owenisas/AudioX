import argparse
import json
from pathlib import Path

from audiox.data.mixed_preference import (
    DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES,
    prepare_mixed_preference_manifests,
)
from audiox.models.lora import DEFAULT_LORA_TARGET_PATTERNS


def build_run_config(
    *,
    output_dir: str,
    train_manifest: str,
    val_manifest: str | None,
    test_manifest: str | None,
    cache_dir: str,
    pretrained_name: str,
    run_name: str,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_offline: bool,
    huggingface_repo_id: str | None,
    huggingface_private: bool,
    include_video_conditioning: bool,
    standalone_ratio: float,
    continuation_ratio: float,
) -> dict:
    run_config = {
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "pretrained_name": pretrained_name,
        "data": {
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "batch_size": 1,
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
        },
        "evaluation": {
            "test_manifest": test_manifest,
        },
        "training": {
            "text_max_length": 256,
            "learning_rate": 1e-5,
            "use_ema": False,
            "mask_padding": True,
            "cfg_dropout_prob": 0.1,
            "timestep_sampler": "uniform",
            "log_loss_info": False,
            "trainable_scope": "multimodal_continuation_lora",
        },
        "trainer": {
            "accelerator": "cuda",
            "devices": 1,
            "precision": "16-mixed",
            "max_epochs": 1,
            "max_steps": 3000,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "num_sanity_val_steps": 0,
        },
        "checkpointing": {
            "dirpath": f"{output_dir}/checkpoints",
            "filename": "step={step}",
            "enabled": True,
            "save_last": True,
            "save_top_k": -1,
            "every_n_train_steps": 500,
            "save_final_checkpoint": True,
            "save_lora_only": True,
            "final_checkpoint_name": "final-lora-state.pt",
        },
        "lora": {
            "enabled": True,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.0,
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
    if huggingface_repo_id:
        run_config["huggingface"] = {
            "enabled": True,
            "repo_id": huggingface_repo_id,
            "repo_type": "model",
            "private": huggingface_private,
            "upload_final_checkpoint": True,
            "upload_resolved_config": True,
            "upload_run_config": True,
            "upload_manifest_summary": True,
            "upload_train_log": False,
        }
    return run_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a mixed-preference AudioX fine-tune run from audio_manifest_split.jsonl."
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Dataset root containing audio/audio_manifest_split.jsonl.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional path to audio_manifest_split.jsonl. Overrides --dataset-root.",
    )
    parser.add_argument(
        "--media-root",
        default=None,
        help="Optional root used to rewrite audio/video paths for remote training hosts.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write manifests, config, and outputs.")
    parser.add_argument("--cache-dir", default=".hf_home", help="Hugging Face cache directory.")
    parser.add_argument(
        "--pretrained-name",
        default="HKUSTAudio/AudioX-MAF-MMDiT",
        help="Hugging Face model identifier to fine-tune from.",
    )
    parser.add_argument(
        "--caption-field",
        default="tagged_training_caption",
        help="Caption field to use for text_prompt generation.",
    )
    parser.add_argument(
        "--source-families",
        default=",".join(DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES),
        help="Comma-separated source families to keep.",
    )
    parser.add_argument(
        "--disable-video-conditioning",
        action="store_true",
        help="Disable video conditioning in the generated manifests and config.",
    )
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
    parser.add_argument(
        "--adjacency-tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed gap between prev end and next start when generating continuation pairs.",
    )
    parser.add_argument("--wandb-project", default="audiox-finetune", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity.")
    parser.add_argument("--wandb-offline", action="store_true", help="Enable offline W&B logging.")
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Optional Hugging Face model repo to upload final artifacts to.",
    )
    parser.add_argument(
        "--hf-public",
        action="store_true",
        help="Create/update the Hugging Face repo as public instead of private.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to a mixed-preference prefix with the dataset stem.",
    )
    args = parser.parse_args()

    dataset_root_or_manifest = args.manifest_path or args.dataset_root
    if not dataset_root_or_manifest:
        raise SystemExit("Provide either --dataset-root or --manifest-path.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)

    source_families = [family.strip() for family in args.source_families.split(",") if family.strip()]
    manifest_info = prepare_mixed_preference_manifests(
        dataset_root_or_manifest,
        output_dir / "manifests",
        media_root=args.media_root,
        source_families=source_families,
        caption_field=args.caption_field,
        include_video=not args.disable_video_conditioning,
        adjacency_tolerance=args.adjacency_tolerance,
    )

    run_name = args.run_name or f"mixed-preference-{Path(dataset_root_or_manifest).stem}"
    run_config = build_run_config(
        output_dir=str(output_dir / "outputs"),
        train_manifest=manifest_info["train_manifest_path"],
        val_manifest=manifest_info["val_manifest_path"],
        test_manifest=manifest_info["test_manifest_path"],
        cache_dir=args.cache_dir,
        pretrained_name=args.pretrained_name,
        run_name=run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_offline=args.wandb_offline,
        huggingface_repo_id=args.hf_repo_id,
        huggingface_private=not args.hf_public,
        include_video_conditioning=not args.disable_video_conditioning,
        standalone_ratio=args.standalone_ratio,
        continuation_ratio=args.continuation_ratio,
    )

    config_path = output_dir / "config_mixed_preference.json"
    config_path.write_text(json.dumps(run_config, indent=2))
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(manifest_info, indent=2))

    print(manifest_info["train_manifest_path"])
    print(manifest_info["val_manifest_path"])
    print(manifest_info["test_manifest_path"])
    print(config_path)


if __name__ == "__main__":
    main()
