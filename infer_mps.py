"""
Standalone AudioX inference optimized for Apple Silicon (MPS).

Loads HKUSTAudio/AudioX-MAF-MMDiT from HF cache, applies LoRA weights,
merges them into the base model, converts to float16, and runs on MPS.

Usage:
    python infer_mps.py
    python infer_mps.py --prompt "your custom prompt"
    python infer_mps.py --steps 50 --seed 123
    python infer_mps.py --no-lora --dtype float32
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Allow MPS fallback for ops not yet natively supported
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
PRETRAINED_NAME = "HKUSTAudio/AudioX-MAF-MMDiT"
LORA_CHECKPOINT = Path(__file__).parent.parent / "last-lora-state.pt"
OUTPUT_DIR = Path(__file__).parent / "mps_inference_output"

TEST_PROMPT = (
    "A quiet tropical rainforest at dawn, dense green foliage dripping with "
    "moisture, soft ambient sounds of distant birds calling and gentle insect "
    "chirps surrounding you, warm humid air, occasional rustling leaves, "
    "subtle breeze moving through the canopy, hig"
)


def parse_args():
    p = argparse.ArgumentParser(description="AudioX MPS inference")
    p.add_argument("--prompt", type=str, default=TEST_PROMPT, help="Text prompt")
    p.add_argument("--steps", type=int, default=300, help="Diffusion steps (300 matches Colab quality)")
    p.add_argument("--cfg-scale", type=float, default=7.0, help="CFG guidance scale")
    p.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    p.add_argument("--sampler", type=str, default="dpmpp-3m-sde",
                    choices=["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-lms"],
                    help="All use MPS-safe CPU noise (no torchsde)")
    p.add_argument("--eta", type=float, default=1.0,
                    help="SDE noise level (0=deterministic, 1=full SDE). Only for dpmpp-*m-sde samplers")
    p.add_argument("--sigma-min", type=float, default=0.3)
    p.add_argument("--sigma-max", type=float, default=500.0)
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"],
                    help="Model dtype (float32 stable on MPS; float16 faster but may produce NaN)")
    p.add_argument("--lora", type=str, default=str(LORA_CHECKPOINT),
                    help="Path to LoRA checkpoint (empty string to skip)")
    p.add_argument("--no-lora", action="store_true", help="Skip LoRA loading")
    p.add_argument("--no-merge", action="store_true", help="Skip LoRA weight merging")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default="output.wav")
    p.add_argument("--device", type=str, default="auto",
                    choices=["auto", "mps", "cuda", "cpu"])
    return p.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def empty_cache(device: torch.device):
    """Device-agnostic cache clearing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def main():
    args = parse_args()
    device = select_device(args.device)
    model_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:  {device}")
    print(f"Dtype:   {model_dtype}")
    print(f"Steps:   {args.steps}")
    print(f"Seed:    {args.seed}")
    print(f"Sampler: {args.sampler}")
    print(f"Prompt:  {args.prompt[:80]}...")
    print()

    # ------------------------------------------------------------------
    # 1. Load base model from HuggingFace cache
    # ------------------------------------------------------------------
    print("Loading base model (HKUSTAudio/AudioX-MAF-MMDiT)...")
    t0 = time.perf_counter()

    from audiox.models.pretrained import get_pretrained_model
    model, model_config = get_pretrained_model(PRETRAINED_NAME)

    t_load = time.perf_counter() - t0
    print(f"  Base model loaded in {t_load:.1f}s")

    # ------------------------------------------------------------------
    # 2. Load + merge LoRA weights
    # ------------------------------------------------------------------
    lora_path = None if args.no_lora else args.lora
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA checkpoint: {lora_path}")
        from audiox.models.lora import load_lora_checkpoint, merge_lora_weights

        t0 = time.perf_counter()
        load_lora_checkpoint(model, lora_path)
        t_lora = time.perf_counter() - t0
        print(f"  LoRA injected in {t_lora:.1f}s")

        if not args.no_merge:
            t0 = time.perf_counter()
            merged_count = merge_lora_weights(model)
            t_merge = time.perf_counter() - t0
            print(f"  Merged {merged_count} LoRA layers in {t_merge:.2f}s")
        else:
            print("  Skipping LoRA merge (--no-merge)")
    elif lora_path:
        print(f"  WARNING: LoRA checkpoint not found at {lora_path}, running base model only")
    else:
        print("  Skipping LoRA (--no-lora)")

    # ------------------------------------------------------------------
    # 3. Compute conditioning on CPU (audio_prompt VAE has MPS precision bug)
    # ------------------------------------------------------------------
    sample_rate = model_config["sample_rate"]  # 44100
    sample_size = model_config["sample_size"]  # 485100
    video_fps = model_config.get("video_fps", 5)
    seconds_total = sample_size / sample_rate  # ~11s
    cond_duration = 10  # Conditioners hardcode 10s
    num_video_frames = int(video_fps * cond_duration)

    print(f"Audio: {sample_rate}Hz, {sample_size} samples ({seconds_total:.1f}s)")

    conditioning = [{
        "video_prompt": {
            "video_tensors": torch.zeros(1, num_video_frames, 3, 224, 224),
            "video_sync_frames": torch.zeros(1, 240, 768),
        },
        "text_prompt": args.prompt,
        "audio_prompt": torch.zeros(1, 2, int(sample_rate * cond_duration)),
        "seconds_start": 0,
        "seconds_total": seconds_total,
    }]

    # CRITICAL: Conditioning (especially audio_prompt VAE) must run on CPU.
    # The audio autoencoder produces 136% different embeddings on MPS vs CPU,
    # which corrupts the entire cross-attention signal.
    print("Computing conditioning + MAF on CPU...")
    model.eval().requires_grad_(False)
    t0 = time.perf_counter()
    with torch.inference_mode():
        cond_tensors = model.conditioner(conditioning, "cpu")
        cond_inputs_cpu = model.get_conditioning_inputs(cond_tensors)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Move model to target device for fast sampling
    print(f"Moving model to {device}...")
    t0 = time.perf_counter()
    model = model.to(device=device)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Generate audio using pre-computed conditioning
    # ------------------------------------------------------------------
    cond_inputs = {k: v.to(device) if v is not None else v for k, v in cond_inputs_cpu.items()}

    import numpy as np
    import k_diffusion as K
    from tqdm import tqdm

    # ---------------------------------------------------------------
    # MPS-safe samplers: all SDE noise generated on CPU to bypass
    # torchsde MPS bugs, then moved to device for math.
    # ---------------------------------------------------------------
    def _sde_noise_inject(x, sigmas, i, eta):
        """Add SDE noise (CPU-generated) after a deterministic step."""
        if sigmas[i + 1] > 0 and eta > 0:
            noise = torch.randn(x.shape, device="cpu", dtype=x.dtype).to(x.device)
            sigma_up = min(
                sigmas[i + 1],
                eta * (sigmas[i + 1] ** 2
                       * (sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                       / sigmas[i] ** 2).sqrt()
            )
            x = x + noise * sigma_up
        return x

    @torch.no_grad()
    def sample_dpmpp_2m_sde_safe(denoiser, x, sigmas, eta=1.0,
                                  extra_args=None, disable=False):
        """DPM-Solver++(2M) SDE — MPS safe."""
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        old_denoised = None
        h_last = None
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = denoiser(x, sigmas[i] * s_in, **extra_args)
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised
            else:
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised_d
            x = _sde_noise_inject(x, sigmas, i, eta)
            old_denoised = denoised
            h_last = h
        return x

    @torch.no_grad()
    def sample_dpmpp_3m_sde_safe(denoiser, x, sigmas, eta=1.0,
                                  extra_args=None, disable=False):
        """DPM-Solver++(3M) SDE — MPS safe. Matches Colab's default sampler."""
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        denoised_1, denoised_2 = None, None
        h_1, h_2 = None, None
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = denoiser(x, sigmas[i] * s_in, **extra_args)
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            if denoised_1 is None or sigmas[i + 1] == 0:
                x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised
            elif denoised_2 is None:
                r = h_1 / h
                d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * denoised_1
                x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * d
            else:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (1 + 1 / (2 * r0)) * denoised - (1 / (2 * r0)) * denoised_1
                d1_1 = (1 + 1 / (2 * r1)) * denoised_1 - (1 / (2 * r1)) * denoised_2
                d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
                d2 = (1 / (2 * r0)) * (d1_0 - d1_1)
                phi_2 = h.expm1() / h + 1
                phi_3 = phi_2 / h - 0.5
                x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * d1 - phi_2 * d2 - phi_3 * d2
            x = _sde_noise_inject(x, sigmas, i, eta)
            denoised_1, denoised_2 = denoised, denoised_1
            h_1, h_2 = h, h_1
        return x

    print(f"\nGenerating ({args.steps} steps, cfg={args.cfg_scale}, "
          f"sampler={args.sampler}, eta={args.eta})...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        seed = args.seed if args.seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
        print(f"  Seed: {seed}")
        torch.manual_seed(seed)

        latent_size = sample_size // model.pretransform.downsampling_ratio
        sigmas = K.sampling.get_sigmas_polyexponential(
            args.steps, args.sigma_min, args.sigma_max, 1.0, device=device)
        noise = torch.randn([1, model.io_channels, latent_size], device=device) * sigmas[0]

        denoiser = K.external.VDenoiser(model.model)
        extra = {**cond_inputs, "cfg_scale": args.cfg_scale,
                 "batch_cfg": True, "rescale_cfg": True}

        if args.sampler == "dpmpp-3m-sde":
            sampled = sample_dpmpp_3m_sde_safe(
                denoiser, noise, sigmas, eta=args.eta, extra_args=extra)
        elif args.sampler == "dpmpp-2m-sde":
            sampled = sample_dpmpp_2m_sde_safe(
                denoiser, noise, sigmas, eta=args.eta, extra_args=extra)
        elif args.sampler in ("k-heun", "k-lms"):
            sampled = K.sampling.sample_heun(denoiser, noise, sigmas,
                extra_args=extra) if args.sampler == "k-heun" else \
                K.sampling.sample_lms(denoiser, noise, sigmas, extra_args=extra)

        # Decode latents to audio (float32 for numerical stability)
        if model.pretransform is not None:
            model.pretransform = model.pretransform.to(dtype=torch.float32).eval()
            _dt = sampled.device.type
            with torch.amp.autocast(device_type=_dt, enabled=False):
                audio = model.pretransform.decode(sampled.to(dtype=torch.float32))
        else:
            audio = sampled

    t_gen = time.perf_counter() - t0
    steps_per_sec = args.steps / t_gen
    print(f"  Generation complete: {t_gen:.1f}s ({steps_per_sec:.2f} steps/s)")

    empty_cache(device)

    # ------------------------------------------------------------------
    # 6. Post-process and save
    # ------------------------------------------------------------------
    # audio shape: (1, channels, samples) -> (channels, samples)
    audio = audio.squeeze(0)
    audio = audio.to(torch.float32).cpu()

    # Normalize to prevent clipping
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak

    # Convert to 16-bit PCM range
    audio_int16 = (audio * 32767).clamp(-32768, 32767).to(torch.int16)

    output_path = output_dir / args.output_name

    # Save WAV using scipy (avoids torchaudio/torchcodec version issues)
    import scipy.io.wavfile as wavfile
    # scipy expects (samples, channels) for multichannel
    wav_data = audio.numpy().T  # (channels, samples) -> (samples, channels)
    wavfile.write(str(output_path), sample_rate, wav_data)

    print(f"\nSaved: {output_path}")
    print(f"  Duration: {audio.shape[-1] / sample_rate:.1f}s")
    print(f"  Channels: {audio.shape[0]}")
    print(f"  Sample rate: {sample_rate}Hz")

    print("\nDone!")


if __name__ == "__main__":
    main()
