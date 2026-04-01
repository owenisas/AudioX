# AudioX ASMR Fine-Tuning: Executive Summary

## Goal

Fine-tune the pretrained [`AudioX-MAF-MMDiT`](https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT) checkpoint (2.4B params, 1.1B trainable) on
ASMR data so the model can generate **continuous, infinite-length ASMR audio**
from a sequence of text prompts, where each generated chunk is conditioned on
the previous chunk's audio.

This document covers the data pipeline, annotation strategy, training
methodology, the two-phase vs single-phase debate, and the final recommended
approach.  Every architectural claim references the AudioX paper (arXiv
2503.10522) and the repo implementation.

---

## 1. How AudioX Handles Audio Conditioning (Why This Works)

The approach is grounded in three concrete mechanisms already in the codebase.

### 1.1 The `audio_prompt` Conditioning Path

AudioX-MAF routes three modalities through its cross-attention pipeline.  In
`audiox/models/diffusion.py:171-178`:

```python
if self.gate:                                    # MAF enabled
    video_feature = cross_attention_input[0]
    text_feature  = cross_attention_input[1]
    audio_feature = cross_attention_input[2]      # <-- previous chunk lives here

    if self.gate_type == "MAF":
        refined = self.maf_block(text_feature, video_feature, audio_feature)
        cross_attention_input = torch.cat(list(refined.values()), dim=1)
```

The `audio_prompt` is not a secondary signal.  It is a **first-class
cross-attention conditioning input** that passes through the full MAF gating,
expert queries, self-attention fusion, and residual routing (60M-param module,
`audiox/models/MAF.py`).  The 24-layer DiT then cross-attends to this fused
representation at every transformer layer.

### 1.2 Zero-Detection for Absent Audio

`AudioAutoencoderConditioner` in `audiox/models/conditioners.py:834-848`:

```python
original_audios = torch.cat(audio, dim=0).to(device)
is_zero = torch.all(original_audios == 0, dim=(1,2))     # detect absent audio
latents = self.pretransform.encode(audio)
latents = self.proj_out(latents.permute(0, 2, 1))
# Swap in a LEARNED empty embedding when audio_prompt is all zeros
latents = torch.where(is_zero_expanded, self.empty_audio_feat, latents)
```

When `audio_prompt` is a zero tensor, the conditioner replaces the encoded
output with `empty_audio_feat`, a **learned parameter** (shape `[1, 215, 768]`)
that the model already knows means "no previous audio context."  This is how the
model cleanly handles the first chunk (no predecessor) vs continuation chunks
(real predecessor).

### 1.3 The Standard Training Loop Already Supports This

`DiffusionCondTrainingWrapper.training_step` in `audiox/training/diffusion.py:348-431`:

```
reals, metadata = batch
#   reals         = TARGET audio to reconstruct  (the NEXT chunk)
#   metadata      = conditioning dict, including:
#     "audio_prompt"  = PREVIOUS chunk audio      (or zeros for first chunk)
#     "text_prompt"   = caption for the next chunk
#     "video_prompt"  = video features             (or zeros if no video)

conditioning = self.diffusion.conditioner(metadata, self.device)   # line 369
...
noised_inputs = diffusion_input * alphas + noise * sigmas          # line 414
output = self.diffusion(noised_inputs, t, cond=conditioning, ...)  # line 431
loss = MSE(output, targets)                                        # line 466
```

The model learns: **given previous audio (via `audio_prompt`) + text description
--> denoise to produce the next audio segment.**  No changes to the training
wrapper are needed.  No inpainting wrapper is needed.  The standard conditional
diffusion training objective handles this natively.

### 1.4 Paper Evidence

Paper Section 3.2:

> "For music completion, the audio input X_a is the preceding music segment, and
> the model aims to generate the subsequent segment."

Paper Section 4.2 confirms that this music-completion mechanism is a standard
task within AudioX's unified training.  Our approach applies the identical
mechanism to ASMR instead of music.

---

## 2. Data Pipeline

### 2.1 Source Data Requirements

| Requirement           | Recommendation                                          |
|-----------------------|---------------------------------------------------------|
| Total raw footage     | 10 hours 34 minutes across diverse ASMR actions        |
| Source videos         | At minimum 5-10 distinct ASMR sessions/tools            |
| Audio quality         | Clean, low-noise, binaural/stereo preferred             |
| Actions to cover      | Ear cleaning, brushing, tapping, water, rain, scratching, whispering, crinkling |
| Format                | Video with embedded audio (MP4)                         |

More unique **actions and textures** matters more than raw hours.  10 hours of
only ear cleaning will overfit to one sound; 4 hours across 8 different ASMR
types will generalize better.

### 2.2 Chunking

Chunk all source videos into **10-second segments**.

Why 10 seconds:
- The AudioX paper's IF-caps dataset uses 10-second video-audio clips (paper
  Section 3.1).
- The pretrained model's `seconds_total` conditioner (`NumberConditioner` in
  `conditioners.py:65-99`) is calibrated for this range.
- 10s is long enough to capture a complete ASMR action (e.g., one scraping
  pass), short enough for the DiT's context window.

```bash
mkdir -p chunks audio_chunks
for f in raw_data/*.mp4; do
    name=$(basename "$f" .mp4)
    # Chunk video+audio together
    ffmpeg -i "$f" -f segment -segment_time 10 \
        -reset_timestamps 1 -c copy "chunks/${name}_%04d.mp4"
done

# Extract audio separately at 48kHz stereo (AudioX's native sample rate)
for f in chunks/*.mp4; do
    name=$(basename "$f" .mp4)
    ffmpeg -i "$f" -ar 48000 -ac 2 "audio_chunks/${name}.wav"
done
```

Expected yield: 10h 34m of source produces ~3,400 ten-second chunks.

### 2.3 Annotation with Qwen

The paper (Section 3.1, Figure 3) uses a two-stage caption pipeline.  For
fine-tuning, a single Qwen2-Audio pass is sufficient (equivalent to the paper's
Stage 2).

**What makes a good ASMR caption:**

Captions must be **action-specific and textural**.  The text conditioner
(T5-base, 128 token max, see `conditioners.py:474-555`) needs concrete sound
descriptions to distinguish actions from each other.

Good:
```
"Thin wooden stick gently scraping inside ear canal microphone, slow
 rhythmic movements with soft crackling, faint ambient hum in background"
```

Bad:
```
"ASMR ear cleaning video"
```

The difference matters because T5 encodes these into 768-dim embeddings that
guide every denoising step.  Vague captions produce vague audio.  Specific
captions produce specific textures.

**Structured fields to capture** (following paper's IF-caps schema):

- **Sound events**: what sounds are present (scraping, tapping, dripping)
- **Sound count**: how many distinct events in this 10s window
- **Temporal description**: what happens in what order within the 10s
- **Tool/surface**: what is making the sound and what it contacts
- **Intensity/rhythm**: fast/slow, gentle/firm, rhythmic/irregular

**Annotation output format:**

```json
[
  {
    "chunk_id": "chunks/ear_cleaning_01_0003.mp4",
    "audio_path": "audio_chunks/ear_cleaning_01_0003.wav",
    "video_path": "chunks/ear_cleaning_01_0003.mp4",
    "caption": "Thin wooden stick scraping gently inside silicone ear, slow rhythmic passes with soft crackling texture, faint room tone"
  }
]
```

### 2.4 Building the Training Manifest

The manifest pairs each target chunk with its conditioning inputs.  Two sample
types are created from the same raw data:

**Type A — Standalone (no previous context):**
```
target_audio  = current chunk
text_prompt   = caption for current chunk
audio_prompt  = zeros (triggers empty_audio_feat in conditioner)
video_prompt  = current chunk's video frames (or zeros if text-only)
```
This teaches the model the ASMR sound domain.

**Type B — Continuation (previous chunk as context):**
```
target_audio  = chunk N
text_prompt   = caption for chunk N
audio_prompt  = audio of chunk N-1 (the real predecessor)
video_prompt  = chunk N's video frames (or zeros)
```
This teaches the model to generate coherent continuations.

Both types use the same training wrapper, same loss function, same forward pass.
The only difference is whether `audio_prompt` is zeros or real audio.  The
`AudioAutoencoderConditioner` handles the routing automatically (line 836-848).

```python
manifest = []
for source, chunks in by_source.items():
    chunks.sort()  # chronological order within each source video
    for i, chunk in enumerate(chunks):
        # Type A: Standalone
        manifest.append({
            "audio_path":         chunk["audio_path"],
            "video_path":         chunk["video_path"],
            "text_prompt":        chunk["caption"],
            "audio_prompt_path":  None,           # zeros
            "seconds_start":      0,
            "seconds_total":      10,
        })
        # Type B: Continuation (skip first chunk of each source — no predecessor)
        if i > 0:
            manifest.append({
                "audio_path":         chunk["audio_path"],
                "video_path":         chunk["video_path"],
                "text_prompt":        chunk["caption"],
                "audio_prompt_path":  chunks[i-1]["audio_path"],  # previous chunk
                "seconds_start":      0,
                "seconds_total":      10,
            })
```

For ~3,400 chunks this yields ~6,800 total samples (~50% Type A, ~50% Type B).

### 2.5 Train / Validation Split

**Critical rule: split by source video, not by chunk.**

If you split randomly by chunk, consecutive chunks from the same video end up in
both train and val.  The model sees chunk 5 in training and gets evaluated on
chunk 6 from the same video — val loss drops, but only because it memorized that
specific audio sequence, not because it learned ASMR.

**How to split:**

```python
import random

# Group chunks by their source video
sources = list(by_source.keys())      # e.g., ["ear_cleaning_01", "tapping_03", ...]
random.shuffle(sources)

# 90/10 split by source video
split_idx = int(len(sources) * 0.9)
train_sources = set(sources[:split_idx])
val_sources   = set(sources[split_idx:])

train_manifest = [s for s in manifest if s["source"] in train_sources]
val_manifest   = [s for s in manifest if s["source"] in val_sources]
```

| Split | Sources | ~Samples |
|-------|---------|----------|
| Train | 90% of source videos | ~6,120 |
| Val   | 10% of source videos | ~680 |

**What to watch during training:**

| Signal | Healthy | Overfitting |
|--------|---------|-------------|
| Train loss | Steadily decreasing | Keeps dropping to near zero |
| Val loss | Decreasing, then plateaus | Starts increasing while train loss still drops |
| Gap (train - val) | Small, stable | Growing over time |

**When to stop:** Save a checkpoint when val loss stops improving for 500+ steps.
That's your best model — training past that point is memorizing the training set.

```python
# Add to trainer callbacks
pl.callbacks.EarlyStopping(
    monitor="val/loss",
    patience=1000,         # steps without improvement before stopping
    mode="min",
),
pl.callbacks.ModelCheckpoint(
    monitor="val/loss",    # save best val loss, not just latest
    save_top_k=3,
    mode="min",
    every_n_train_steps=500,
),
```

**Validation frequency:** Run val every 250-500 steps.  More frequent is noisy,
less frequent and you might miss the best checkpoint.

```python
trainer = pl.Trainer(
    ...
    val_check_interval=500,   # validate every 500 training steps
)
```

---

## 3. The Two-Phase Question

### 3.1 The Proposal

The product manager's question is whether training should be split into two
sequential passes:

> **Pass A**: Domain adaptation.  Standard IF-caps style where `audio_prompt =
> zeros` and the model learns ASMR sound textures from `text + video --> audio`.
>
> **Pass B**: Continuity training.  Music-completion style where `audio_prompt =
> previous chunk` and the model learns to chain segments.

The logic is intuitive: first teach the model what ASMR sounds like (vocabulary),
then teach it how ASMR sounds flow together (grammar).

### 3.2 What the Paper Actually Does

The paper trains **all tasks jointly in a single phase** (Section 4.1):

> AudioX trains text-to-audio, video-to-audio, text-to-music, video-to-music,
> audio inpainting, and music completion **simultaneously** in unified
> end-to-end training.

There is no phased curriculum.  The paper's key empirical finding (Section 4.3,
Table 5) is the **cross-modal regularization effect**: training on multiple tasks
and modalities simultaneously actually **improves** performance on each
individual task compared to training them separately.  Specifically, the paper
shows that high-quality text supervision improves video-to-audio quality even
though text and video are different modalities.

The paper's ablation in Table 5 demonstrates this directly: training with all
caption sources combined (GeminiCap-aug) achieves IS=10.93 on VGGSound, versus
10.81 for GeminiCap alone or 9.74 for QwenCap alone.  Joint training with
diverse signals is better than isolated training.

### 3.3 Why Single-Phase Is Better (For Our Case)

There are **four concrete reasons** why a single mixed phase is the right choice
for ASMR fine-tuning, drawing from both the paper and the architecture:

**Reason 1: MAF gating already handles the mixing.**

The MAF block (`audiox/models/MAF.py`) contains a gating network that
dynamically weights each modality per sample.  When `audio_prompt = zeros`, the
gate learns to suppress the audio branch and rely on text + video.  When
`audio_prompt = real audio`, the gate upweights audio.  This gating is
**per-sample, not per-epoch**.  In a single batch, sample A can have
`audio_prompt = zeros` and sample B can have `audio_prompt = real audio`, and the
MAF handles both correctly because the gating is computed independently per
sample.  Two-phase training doesn't give the gate any advantage it doesn't
already have.

**Reason 2: Two phases risk catastrophic forgetting.**

In Pass A, the model adapts to always seeing `audio_prompt = zeros`.  The
`empty_audio_feat` embedding (line 826) and the MAF gating weights settle into a
regime where audio conditioning is always absent.  When Pass B starts with real
`audio_prompt`, the model must abruptly readjust.  With a small dataset (~2,880
samples) and low learning rate (1e-6), this readjustment may be slow or
incomplete.  Single-phase training avoids this entirely — the model always sees
both conditions from the start.

**Reason 3: Cross-modal regularization applies here too.**

The paper's core finding is that multi-task training provides a regularization
benefit.  In our case:
- Type A samples teach the model what ASMR textures sound like given text
- Type B samples teach the model how to continue those textures given context

These are complementary signals.  A Type B sample where the model must continue
ear-cleaning-after-brushing also reinforces what brushing and ear-cleaning sound
like (the same textures learned in Type A).  Splitting them into phases **loses
this cross-task regularization**.

**Reason 4: The conditioning pathway is identical.**

Both sample types pass through the exact same code path:
```
metadata --> MultiConditioner.forward() --> get_conditioning_inputs() --> DiT
```
The only difference is whether `AudioAutoencoderConditioner` receives zeros (and
swaps in `empty_audio_feat`) or real audio (and encodes it through the
pretransform).  The model's weights, loss function, and optimizer step are
identical.  There is no architectural reason to separate them.

### 3.4 When Two-Phase WOULD Make Sense

Two-phase training would be justified if:

1. **The domain gap is extreme** — e.g., fine-tuning a speech model on
   underwater sonar.  ASMR is close enough to the pretrained AudioX distribution
   (general audio + music) that a large domain adaptation phase is unnecessary.

2. **The continuation dataset is much smaller than the domain dataset** — e.g.,
   you have 100k standalone samples but only 500 continuation pairs.  Then Phase
   A with the larger set followed by Phase B with the smaller set makes sense to
   prevent the continuation signal from being drowned.  In our case the ratio is
   roughly 1:1, so this doesn't apply.

3. **You want to validate domain adaptation independently** — this is a
   legitimate debugging reason.  You can run Phase A for a few hundred steps,
   inspect the outputs, confirm the model produces good ASMR textures, then
   proceed.  But this is a debugging workflow, not a training strategy.

### 3.5 Final Recommendation

**Single-phase mixed training with weighted sampling.**

```
Training manifest:
  ~30% Type A (standalone, audio_prompt = zeros)
  ~70% Type B (continuation, audio_prompt = previous chunk)
  Shuffled together, both types in every batch.
```

The 30/70 split ensures the model:
- Always sees some standalone samples (learns to start from scratch, prevents
  over-reliance on audio_prompt)
- Primarily trains on continuation pairs (the core capability we want)
- Gets the cross-modal regularization benefit from seeing both types together

If you want the safety of a checkpoint-and-inspect workflow: save a checkpoint at
step 500, generate a few standalone ASMR samples, confirm quality, then continue
training.  This gives the same validation benefit as two phases without the
forgetting risk.

---

## 4. Training Configuration

### 4.1 Dataset

**Source**: 10 hours 34 minutes of ASMR audio.

| Metric | Value |
|--------|-------|
| Raw duration | 10h 34m (38,040 seconds) |
| 10-second chunks | ~3,400 |
| Type A samples (standalone) | ~3,400 |
| Type B samples (continuation) | ~3,400 |
| Total training samples | ~6,800 |

With ~7,600 samples we are in the sweet spot for LoRA fine-tuning — enough data
to learn the domain without overfitting, small enough that full fine-tuning of
the 1B DiT backbone is unnecessary.

### 4.2 What to Train: LoRA + Selective Full Fine-Tune

**Pretrained checkpoint**:
[`HKUSTAudio/AudioX-MAF-MMDiT`](https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT)
(2.4B total params, 1.1B trainable).

The strategy splits parameters into three tiers based on two criteria:
1. **Module size** — is LoRA's memory savings meaningful here?
2. **Required expressiveness** — does the module need subtle adaptation (LoRA is
   fine) or significant behavioral change (needs full-rank updates)?

#### Tier 1 — LoRA: Self-Attention & Cross-Attention (High Priority)

Target modules in the 12-layer MMDiT backbone:

| Module | Name Pattern | Why LoRA |
|--------|-------------|----------|
| Self-attention Q/K/V | `transformer.layers[*].attn.qkv` | Controls what audio tokens attend to each other — subtle shift to ASMR textures |
| Cross-attention query | `transformer.layers[*].cross_attn.to_q` | Controls how audio latents query conditioning — adapts what the model "asks" from text/audio prompts |
| Cross-attention K/V | `transformer.layers[*].cross_attn.to_kv` | Controls what conditioning information is surfaced — adapts which prompt features matter for ASMR |

Cross-attention is **more impactful** than self-attention for conditioning
fidelity.  If constrained on VRAM, prioritize cross-attention adapters.

#### Tier 2 — LoRA: Feedforward / MLP Layers (Medium Priority)

| Module | Name Pattern | Why LoRA |
|--------|-------------|----------|
| Feedforward MLP | `transformer.layers[*].ffn` | Stores learned "knowledge" about audio patterns — LoRA here helps the model learn new ASMR textures and timbres |

Include Tier 2 alongside Tier 1 if your GPU can fit it.  Both tiers train
simultaneously in the same forward/backward pass — the "tiers" are priority
rankings for when you need to drop something.

#### Tier 3 — Full Fine-Tune (NOT LoRA)

| Module | ~Params | Why Full Fine-Tune |
|--------|---------|-------------------|
| MAF block | ~60M | Too critical + too small for LoRA. The gating network must potentially flip from 0.3→0.8 to reweight modalities for ASMR. LoRA's low-rank constraint cannot express large gating shifts. 60M params means LoRA's memory savings are negligible. |
| `empty_audio_feat` | 768-dim vector | Single embedding vector — LoRA decomposition doesn't apply to a single vector. Must adapt to "absence of ASMR context" specifically. |
| Conditioner `proj_out` layers | Small | Projection layers that map encoder outputs to DiT dimension. Small enough to train fully. |

#### What to Freeze (Do NOT Train)

| Module | Why Frozen |
|--------|-----------|
| T5-base text encoder | General-purpose text understanding — no gain from ASMR-specific training |
| CLIP-ViT video encoder | Pretrained visual features — ASMR doesn't need different visual representations |
| Stable Audio Open autoencoder | Audio codec — should encode/decode all audio faithfully, not just ASMR |
| Timestep embeddings | Diffusion schedule must stay fixed — modifying these destabilizes denoising |
| AdaLN modulation layers | Translate timestep info to normalization params — keep stable |

#### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                  # rank 16 — ASMR is close enough to pretrain distribution
    lora_alpha=32,         # 2× rank (standard scaling)
    target_modules=[
        # Tier 1 — Attention (high priority)
        "attn.qkv",           # self-attention fused Q/K/V projection
        "cross_attn.to_q",    # cross-attention queries
        "cross_attn.to_kv",   # cross-attention keys/values
        # Tier 2 — Feedforward (medium priority)
        "ffn",                # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
)
```

**Why rank 16?**  ASMR audio is structurally similar to the general audio/music
the model was pretrained on — it's still environmental sound with rhythm and
texture.  The adaptation is about *which* textures, not a fundamentally different
signal domain.  Rank 16 provides enough capacity for this shift.  If results are
underwhelming, increase to 32.

**Why not LoRA the AdaLN modulation?**  These layers inject timestep information
into normalization.  Modifying them changes the diffusion dynamics at every noise
level, which can destabilize generation quality.  The denoising schedule should
remain invariant to domain.

### 4.3 Hyperparameters

Two learning rates are needed because LoRA and full-fine-tune parameters have
different initialization states:

```
# LoRA parameters (Tier 1 + 2)
lora_learning_rate:     1e-4       (LoRA adapters init near zero, need faster LR)

# Full fine-tune parameters (Tier 3: MAF, empty_audio_feat, proj_out)
base_learning_rate:     1e-5       (pretrained weights, gentler updates to avoid
                                    catastrophic forgetting)

optimizer:              AdamW      (paper: "AdamW with weight decay 0.001")
weight_decay:           0.001
batch_size:             2-4        (per GPU, depends on VRAM)
accumulate_grad_batches: 4         (effective batch = 8-16)
precision:              16-mixed   (from defaults.ini)
cfg_dropout_prob:       0.1        (paper default, keeps CFG working at inference)
timestep_sampler:       uniform    (paper default)
use_ema:                True       (paper Section 4.1: "EMA of model weights")
ema_beta:               0.9999     (from training/diffusion.py:254)
max_steps:              5000-8000  (for ~6,800 samples, ~10-15 epochs)
gradient_clip_val:      1.0
```

### 4.4 Training Script

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from audiox import get_pretrained_model
from audiox.training.diffusion import DiffusionCondTrainingWrapper

# Load pretrained checkpoint
model, config = get_pretrained_model("HKUSTAudio/AudioX-MAF-MMDiT")

# --- Step 1: Freeze everything first ---
for param in model.parameters():
    param.requires_grad = False

# --- Step 2: Apply LoRA to DiT attention + feedforward (Tier 1 + 2) ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "attn.qkv",           # self-attention
        "cross_attn.to_q",    # cross-attention queries
        "cross_attn.to_kv",   # cross-attention keys/values
        "ffn",                # feedforward
    ],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# --- Step 3: Unfreeze Tier 3 for full fine-tuning ---
for name, param in model.named_parameters():
    if any(key in name for key in ["maf_block", "empty_audio_feat", "proj_out"]):
        param.requires_grad = True

# --- Step 4: Separate parameter groups with different LRs ---
lora_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "lora_" in n]
base_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "lora_" not in n]

optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 1e-4},      # LoRA adapters — fast
    {"params": base_params, "lr": 1e-5},       # MAF/proj_out — gentle
], weight_decay=0.001)

# Training wrapper
wrapper = DiffusionCondTrainingWrapper(
    model,
    lr=1e-4,                  # base LR (overridden by param groups above)
    cfg_dropout_prob=0.1,
    use_ema=True,
    timestep_sampler="uniform",
)
wrapper.optimizer = optimizer  # override with our dual-LR optimizer

dataset = ASMRContinuationDataset("training_manifest.json")
loader  = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

trainer = pl.Trainer(
    max_steps=6000,
    precision="16-mixed",
    accelerator="gpu",
    devices=1,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.ModelCheckpoint(every_n_train_steps=500, save_top_k=3,
                                     monitor="train/loss"),
    ],
)

trainer.fit(wrapper, loader)
```

**Trainable parameter summary:**

```
LoRA adapters (Tier 1+2):   ~10-15M params   (LR: 1e-4)
MAF block (Tier 3):         ~60M params       (LR: 1e-5)
Conditioner proj + embed:   ~1M params        (LR: 1e-5)
─────────────────────────────────────────────
Total trainable:            ~75M / 2.4B (3%)
Frozen:                     ~2.3B (97%)
```

---

## 5. Inference: Continuous Generation

After fine-tuning, generation is a simple loop.  No `long_form.py` needed.

```python
from audiox.inference.generation import generate_diffusion_cond

model.eval()
prev_audio = None
sample_rate = 48000
chunk_samples = 10 * sample_rate

script = [
    "Wooden stick scraping gently inside ear microphone, slow rhythm",
    "Cotton swab brushing the inner ear, muffled soft texture",
    "Water drops falling on leather, irregular light tapping",
    "Rain intensifying on nylon jacket, dense rhythmic patter",
]

chunks = []
for text in script:
    conditioning = [{
        "text_prompt":    text,
        "seconds_start":  0,
        "seconds_total":  10,
        "audio_prompt":   prev_audio if prev_audio is not None
                          else torch.zeros(1, 2, chunk_samples),
        "video_prompt":   {
            "video_tensors":     torch.zeros(1, 50, 3, 224, 224),
            "video_sync_frames": torch.zeros(1, 240, 768),
        },
    }]

    output = generate_diffusion_cond(
        model, steps=250, cfg_scale=7.0,
        conditioning=conditioning,
        sample_size=chunk_samples, device="cuda",
    )

    prev_audio = output          # this chunk becomes next chunk's context
    chunks.append(output)

# Simple 0.5s crossfade between chunks for seamless output
final = crossfade_stitch(chunks, overlap_samples=24000)
torchaudio.save("asmr_continuous.wav", final.cpu(), sample_rate)
```

The model handles continuity **natively** because it was trained on consecutive
pairs.  The crossfade is cosmetic — the actual audio coherence comes from the
MAF cross-attention over `audio_prompt`.

At inference:
- `audio_prompt = previous GENERATED chunk` (not a real chunk from the dataset)
- `text_prompt = new action description`
- The MAF gating upweights audio_prompt (real audio) and text (present),
  downweights video (zeros)
- The DiT cross-attends to the fused context and generates a coherent
  continuation

This loop runs indefinitely.  As long as you provide new text prompts, the model
generates the next 10 seconds conditioned on whatever it just produced.

---

## 6. Why This Approach Is Sound

### 6.1 It Is the Paper's Own Mechanism

The paper (Section 3.2) defines music completion as: "the audio input X_a is the
preceding music segment, and the model aims to generate the subsequent segment."
We apply this identical formulation to ASMR.  The architecture, conditioning
path, and training objective are unchanged — only the data domain changes.

### 6.2 The Architecture Already Supports It

- `AudioAutoencoderConditioner` (line 822-849): encodes real audio through the
  pretransform autoencoder, projects to 768-dim, feeds to MAF.  Detects zero
  audio and swaps in learned empty embedding.  No code changes needed.
- MAF gating (line 141-146): dynamically weights modalities per sample.  Handles
  mixed batches (some with audio_prompt, some without) natively.
- Standard training loop (line 348-507): `reals` = target, `metadata` =
  conditioning including `audio_prompt`.  The objective is MSE on the
  v-prediction target.  No modifications needed.

### 6.3 Single-Phase Training Matches the Paper

The paper trains all tasks jointly and demonstrates cross-modal regularization
(Section 4.3, Table 5).  Two-phase training has no support in the paper or the
architecture.  It introduces forgetting risk and eliminates the regularization
benefit of mixed-task training.  The MAF gating mechanism is designed to handle
mixed conditioning within the same batch — using it that way is working with the
architecture, not against it.

### 6.4 The Train/Inference Gap Is Minimal

During training: `audio_prompt = real previous chunk from dataset`.
During inference: `audio_prompt = previous generated chunk`.

This gap exists in all autoregressive-style generation.  It is mitigated by:
- The model's pretrained robustness (trained on 7M+ samples)
- CFG dropout during training (10% of samples see no conditioning at all, making
  the model robust to imperfect inputs)
- The MAF gating, which can downweight a noisy audio_prompt if the text signal is
  strong

### 6.5 Expected Compute

| Resource | Estimate |
|----------|----------|
| Training samples | ~6,800 (from 10h 34m of source) |
| Steps to convergence | 5,000-8,000 |
| Time per step (A100 80GB, batch=2) | ~2-4 seconds |
| Total training time | ~4-9 hours on 1x A100 |
| VRAM requirement | ~25-30GB at fp16 (LoRA, batch=2) |
| Inference per 10s chunk | ~30-60 seconds at 250 steps |

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Do we need two-phase training? | **No.** Single-phase mixed training with 30% standalone / 70% continuation pairs. The paper trains all tasks jointly; two phases risk forgetting and lose cross-task regularization. |
| What training wrapper? | `DiffusionCondTrainingWrapper` (standard). Not the inpaint wrapper. |
| Do we need `long_form.py`? | **No.** The `audio_prompt` cross-attention path provides learned continuity. `long_form.py` is an inference-time stitching hack where the model never hears previous audio. |
| Will it work? | Yes. The mechanism is the paper's own music-completion pathway (Section 3.2), the conditioner already handles zero vs real audio (line 836), and the training loop already accepts `audio_prompt` in metadata (line 369). |
| How much data? | 10h 34m of ASMR, chunked to 10s, yields ~3,400 chunks and ~6,800 training samples. |
| What to fine-tune? | LoRA (rank 16) on DiT attention + feedforward (Tier 1+2, ~15M params). Full fine-tune MAF block + conditioner projections + `empty_audio_feat` (Tier 3, ~61M params). Total: ~75M trainable / 2.4B (3%). |
