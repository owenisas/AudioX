import argparse

from audiox.inference.asmr import run_soundfx_eval_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standalone SoundFX evaluation clips from a prompt manifest using a base AudioX model plus LoRA."
    )
    parser.add_argument("--prompt-manifest", required=True, help="Path to a JSON/JSONL prompt manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to write WAVs and eval metadata.")
    parser.add_argument(
        "--pretrained-name",
        default="HKUSTAudio/AudioX-MAF-MMDiT",
        help="Base AudioX checkpoint to load.",
    )
    parser.add_argument("--lora-path", default=None, help="Optional LoRA checkpoint exported by train_finetune.py.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--device", default=None, help="Override device (default: cuda if available).")
    parser.add_argument("--steps", type=int, default=250, help="Number of diffusion steps per clip.")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma-min", type=float, default=0.3, help="Minimum sigma for sampling.")
    parser.add_argument("--sigma-max", type=float, default=500.0, help="Maximum sigma for sampling.")
    parser.add_argument("--sampler-type", default="dpmpp-3m-sde", help="Sampler type for generation.")
    parser.add_argument("--seed-base", type=int, default=0, help="Deterministic seed base. Row i uses seed_base + i.")
    args = parser.parse_args()

    run_soundfx_eval_batch(
        args.prompt_manifest,
        output_dir=args.output_dir,
        pretrained_name=args.pretrained_name,
        lora_path=args.lora_path,
        cache_dir=args.cache_dir,
        device=args.device,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sampler_type=args.sampler_type,
        seed_base=args.seed_base,
    )


if __name__ == "__main__":
    main()
