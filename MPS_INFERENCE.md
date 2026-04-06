# AudioX on Apple Silicon (MPS)

Run AudioX inference on Mac with Apple Silicon GPU acceleration.

## Quick Start

```bash
# Default: 300 steps, dpmpp-3m-sde, with LoRA
python infer_mps.py

# Custom prompt
python infer_mps.py --prompt "gentle ocean waves on a sandy beach, seagulls calling overhead"

# Random seed (try several to find good ones)
python infer_mps.py --seed -1

# Output saved to mps_inference_output/output.wav
```

## Samplers

| Sampler | Flag | Notes |
|---------|------|-------|
| `dpmpp-3m-sde` | `--sampler dpmpp-3m-sde` | **Default.** 3rd order, matches Colab. Best quality. |
| `dpmpp-2m-sde` | `--sampler dpmpp-2m-sde` | 2nd order, slightly faster. |
| `k-heun` | `--sampler k-heun --steps 150` | Deterministic, needs fewer steps but slower per step. |

### SDE Noise Control (`--eta`)

The `--eta` flag controls stochastic noise injection in SDE samplers:

```bash
python infer_mps.py --eta 0      # Deterministic (cleanest, quieter)
python infer_mps.py --eta 0.3    # Mild noise (good balance)
python infer_mps.py --eta 1.0    # Full SDE (default, more texture)
```

## All Options

```
--prompt TEXT         Text prompt for generation
--steps N             Diffusion steps (default: 300)
--cfg-scale F         Classifier-free guidance scale (default: 7.0)
--seed N              Random seed, -1 for random (default: 42)
--sampler S           dpmpp-3m-sde | dpmpp-2m-sde | k-heun | k-lms
--eta F               SDE noise level 0-1 (default: 1.0)
--sigma-min F         Min sigma (default: 0.3)
--sigma-max F         Max sigma (default: 500.0)
--lora PATH           LoRA checkpoint path
--no-lora             Run base model without LoRA
--output-dir DIR      Output directory
--output-name FILE    Output filename (default: output.wav)
--device DEVICE       auto | mps | cuda | cpu
```

## How It Works

The script handles three MPS-specific issues automatically:

1. **Conditioning on CPU**: The audio autoencoder conditioner produces incorrect
   embeddings on MPS (136% error vs CPU). All conditioning (CLIP, T5, audio VAE,
   MAF fusion) runs on CPU, then results are moved to MPS for sampling.

2. **CPU-generated SDE noise**: The `torchsde` library's Brownian motion has MPS
   bugs that corrupt SDE samplers. The script generates all stochastic noise on
   CPU and transfers it to MPS.

3. **Device-agnostic patches**: All `torch.cuda.*` calls in the AudioX library
   are patched to work on MPS (autocast, empty_cache, sdp_kernel, etc.).

## Performance

On Apple Silicon with 64GB RAM:

| Config | Speed | Time (300 steps) |
|--------|-------|-----------------|
| MPS float32 | ~1.5 steps/s | ~3.5 min |
| CPU float32 | ~0.7 steps/s | ~7 min |

## Library Changes (Device-Agnostic Patches)

These files were patched to replace CUDA-specific code with device-agnostic equivalents:

- `audiox/models/transformer.py` — SDPA attention on MPS, autocast decorator
- `audiox/inference/sampling.py` — `torch.amp.autocast` replacements
- `audiox/inference/generation.py` — CUDA backend guards, empty_cache
- `audiox/models/conditioners.py` — empty_cache, autocast fixes
- `audiox/models/pretransforms.py` — autocast fixes
- `audiox/models/lora.py` — added `merge_lora_weights()` for inference optimization
- `audiox/interface/gradio.py` — empty_cache fix

## Requirements

```
torch >= 2.0 (tested with 2.9.0)
macOS with Apple Silicon (M1/M2/M3/M4)
64GB+ RAM recommended
```

Install AudioX dependencies:
```bash
pip install --no-deps -e .
pip install einops_exts aeiou alias-free-torch auraloss ema-pytorch encodec \
    k-diffusion laion-clap local-attention pedalboard safetensors \
    sentencepiece v-diffusion-pytorch vector-quantize-pytorch x-transformers \
    descript-audio-codec 'transformers>=4.40,<4.50'
```
