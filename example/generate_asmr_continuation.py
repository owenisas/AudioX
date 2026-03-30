import argparse

from audiox.inference.asmr import load_prompt_list, run_asmr_continuation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stitched ASMR continuations from an ordered prompt list.")
    parser.add_argument("--prompts", required=True, help="Path to a newline-delimited or JSON prompt list.")
    parser.add_argument("--output-dir", required=True, help="Directory to write chunk WAVs and the stitched output.")
    parser.add_argument(
        "--pretrained-name",
        default="HKUSTAudio/AudioX-MAF-MMDiT",
        help="Base AudioX checkpoint to load.",
    )
    parser.add_argument("--lora-path", default=None, help="Optional LoRA checkpoint exported by train_finetune.py.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--device", default=None, help="Override device (default: cuda if available).")
    parser.add_argument("--steps", type=int, default=250, help="Number of diffusion steps per chunk.")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="Classifier-free guidance scale.")
    parser.add_argument("--sigma-min", type=float, default=0.3, help="Minimum sigma for sampling.")
    parser.add_argument("--sigma-max", type=float, default=500.0, help="Maximum sigma for sampling.")
    parser.add_argument("--sampler-type", default="dpmpp-3m-sde", help="Sampler type for generation.")
    parser.add_argument("--seed", type=int, default=-1, help="Base seed. Each chunk uses seed + index.")
    parser.add_argument(
        "--crossfade-overlap-seconds",
        type=float,
        default=0.5,
        help="Cosmetic crossfade overlap applied when stitching chunks.",
    )
    args = parser.parse_args()

    prompts = load_prompt_list(args.prompts)
    run_asmr_continuation(
        prompts,
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
        seed=args.seed,
        crossfade_overlap_seconds=args.crossfade_overlap_seconds,
    )


if __name__ == "__main__":
    main()
