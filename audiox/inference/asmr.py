import json
import re
import typing as tp
from pathlib import Path

import numpy as np
import torch

from .generation import generate_diffusion_cond
from ..models.pretrained import get_pretrained_model
from ..training.finetune import extract_audio_prompt_num_samples


def load_prompt_list(path: tp.Union[str, Path]) -> tp.List[str]:
    prompt_path = Path(path)
    if prompt_path.suffix == ".json":
        payload = json.loads(prompt_path.read_text())
        if isinstance(payload, list):
            prompts = payload
        elif isinstance(payload, dict):
            prompts = payload.get("prompts", [])
        else:
            prompts = []
        return [str(prompt).strip() for prompt in prompts if str(prompt).strip()]

    return [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]


def load_prompt_manifest(path: tp.Union[str, Path]) -> tp.List[tp.Dict[str, str]]:
    prompt_path = Path(path)
    suffix = prompt_path.suffix.lower()

    if suffix == ".jsonl":
        records = []
        for line in prompt_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Prompt manifest row must be an object: {prompt_path}")
            records.append(payload)
    elif suffix == ".json":
        payload = json.loads(prompt_path.read_text())
        if isinstance(payload, dict):
            records = payload.get("prompts", [])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported JSON prompt manifest shape in {prompt_path}")
    else:
        records = [{"clip_id": f"prompt_{idx:03d}", "text_prompt": prompt} for idx, prompt in enumerate(load_prompt_list(prompt_path))]

    normalized_records: tp.List[tp.Dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Prompt manifest entry {index} must be an object.")
        text_prompt = str(record.get("text_prompt") or record.get("prompt") or "").strip()
        if not text_prompt:
            raise ValueError(f"Prompt manifest entry {index} is missing text_prompt/prompt.")
        clip_id = str(record.get("clip_id") or f"prompt_{index:03d}").strip()
        normalized_records.append({"clip_id": clip_id, "text_prompt": text_prompt})
    return normalized_records


def build_zero_video_prompt(
    *,
    seconds_total: float,
    video_fps: int,
    frame_count: tp.Optional[int] = None,
    device: tp.Union[str, torch.device] = "cpu",
) -> tp.Dict[str, torch.Tensor]:
    resolved_frame_count = frame_count if frame_count is not None else int(round(seconds_total * video_fps))
    resolved_frame_count = max(1, resolved_frame_count)
    return {
        "video_tensors": torch.zeros(1, resolved_frame_count, 3, 224, 224, device=device),
        "video_sync_frames": torch.zeros(1, 240, 768, device=device),
    }


def prepare_audio_prompt(
    audio_prompt: tp.Optional[torch.Tensor],
    *,
    audio_prompt_num_samples: int,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    if audio_prompt is None:
        return torch.zeros(1, 2, audio_prompt_num_samples, device=device)

    prepared = audio_prompt.detach().to(device)
    if prepared.ndim == 2:
        prepared = prepared.unsqueeze(0)
    if prepared.shape[-1] > audio_prompt_num_samples:
        prepared = prepared[..., :audio_prompt_num_samples]
    elif prepared.shape[-1] < audio_prompt_num_samples:
        prepared = torch.nn.functional.pad(prepared, (0, audio_prompt_num_samples - prepared.shape[-1]))
    return prepared


def infer_zero_video_frame_count(model: tp.Any, *, sample_rate: int, sample_size: int, video_fps: int) -> int:
    default_frame_count = max(1, int(round(sample_size / sample_rate * video_fps)))
    conditioners = getattr(getattr(model, "conditioner", None), "conditioners", None)
    conditioner = None
    if conditioners is not None and "video_prompt" in conditioners:
        conditioner = conditioners["video_prompt"]
    in_features = getattr(conditioner, "in_features", None)
    if in_features is None:
        return default_frame_count
    if in_features % 50 != 0:
        return default_frame_count
    inferred_frame_count = int(in_features // 50)
    return max(1, inferred_frame_count)


def build_continuation_conditioning(
    text_prompt: str,
    *,
    sample_rate: int,
    sample_size: int,
    video_fps: int,
    audio_prompt_num_samples: int,
    previous_audio: tp.Optional[torch.Tensor] = None,
    zero_video_frame_count: tp.Optional[int] = None,
    device: tp.Union[str, torch.device] = "cpu",
) -> tp.List[tp.Dict[str, tp.Any]]:
    seconds_total = sample_size / sample_rate
    return [
        {
            "text_prompt": text_prompt,
            "audio_prompt": prepare_audio_prompt(
                previous_audio,
                audio_prompt_num_samples=audio_prompt_num_samples,
                device=device,
            ),
            "video_prompt": build_zero_video_prompt(
                seconds_total=seconds_total,
                video_fps=video_fps,
                frame_count=zero_video_frame_count,
                device=device,
            ),
            "seconds_start": 0,
            "seconds_total": seconds_total,
        }
    ]


def build_standalone_conditioning(
    text_prompt: str,
    *,
    sample_rate: int,
    sample_size: int,
    video_fps: int,
    audio_prompt_num_samples: int,
    zero_video_frame_count: tp.Optional[int] = None,
    device: tp.Union[str, torch.device] = "cpu",
) -> tp.List[tp.Dict[str, tp.Any]]:
    return build_continuation_conditioning(
        text_prompt,
        sample_rate=sample_rate,
        sample_size=sample_size,
        video_fps=video_fps,
        audio_prompt_num_samples=audio_prompt_num_samples,
        previous_audio=None,
        zero_video_frame_count=zero_video_frame_count,
        device=device,
    )


def crossfade_stitch(
    chunks: tp.Sequence[torch.Tensor], overlap_samples: int
) -> torch.Tensor:
    if not chunks:
        raise ValueError("crossfade_stitch requires at least one chunk.")

    normalized_chunks = []
    for chunk in chunks:
        if chunk.ndim == 3:
            normalized_chunks.append(chunk.squeeze(0))
        else:
            normalized_chunks.append(chunk)

    if len(normalized_chunks) == 1 or overlap_samples <= 0:
        return torch.cat(normalized_chunks, dim=-1)

    stitched = normalized_chunks[0].clone()
    for chunk in normalized_chunks[1:]:
        current_overlap = min(overlap_samples, stitched.shape[-1], chunk.shape[-1])
        fade_out = torch.linspace(1.0, 0.0, current_overlap, device=stitched.device, dtype=stitched.dtype)
        fade_in = torch.linspace(0.0, 1.0, current_overlap, device=chunk.device, dtype=chunk.dtype)
        blended = stitched[:, -current_overlap:] * fade_out + chunk[:, :current_overlap] * fade_in
        stitched = torch.cat([stitched[:, :-current_overlap], blended, chunk[:, current_overlap:]], dim=-1)
    return stitched


def run_asmr_continuation(
    prompts: tp.Sequence[str],
    *,
    output_dir: tp.Union[str, Path],
    pretrained_name: str = "HKUSTAudio/AudioX-MAF-MMDiT",
    lora_path: tp.Optional[tp.Union[str, Path]] = None,
    cache_dir: tp.Optional[tp.Union[str, Path]] = None,
    device: tp.Optional[str] = None,
    steps: int = 250,
    cfg_scale: float = 7.0,
    sigma_min: float = 0.3,
    sigma_max: float = 500.0,
    sampler_type: str = "dpmpp-3m-sde",
    seed: int = -1,
    crossfade_overlap_seconds: float = 0.5,
) -> tp.Dict[str, tp.Any]:
    if not prompts:
        raise ValueError("run_asmr_continuation requires at least one prompt.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = get_pretrained_model(pretrained_name, cache_dir=cache_dir, lora_path=lora_path)
    model = model.to(resolved_device).eval()

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    video_fps = model_config.get("video_fps", 5)
    audio_prompt_num_samples = extract_audio_prompt_num_samples(model_config, sample_size)
    zero_video_frame_count = infer_zero_video_frame_count(
        model,
        sample_rate=sample_rate,
        sample_size=sample_size,
        video_fps=video_fps,
    )
    overlap_samples = int(round(crossfade_overlap_seconds * sample_rate))

    output_dir = Path(output_dir)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    chunk_seeds = []
    generated_chunks = []
    previous_audio = None

    for index, text_prompt in enumerate(prompts):
        chunk_seed = seed + index if seed >= 0 else -1
        conditioning = build_continuation_conditioning(
            text_prompt,
            sample_rate=sample_rate,
            sample_size=sample_size,
            video_fps=video_fps,
            audio_prompt_num_samples=audio_prompt_num_samples,
            previous_audio=previous_audio,
            zero_video_frame_count=zero_video_frame_count,
            device=resolved_device,
        )
        generated = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            seed=chunk_seed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampler_type=sampler_type,
            device=resolved_device,
        ).detach().cpu()
        previous_audio = generated
        generated_chunks.append(generated)

        chunk_path = chunk_dir / f"chunk_{index:03d}.wav"
        save_wav_file(chunk_path, generated.squeeze(0), sample_rate)
        chunk_paths.append(str(chunk_path))
        chunk_seeds.append(chunk_seed)

    stitched = crossfade_stitch(generated_chunks, overlap_samples=overlap_samples).to(torch.float32)
    stitched_path = output_dir / "stitched.wav"
    save_wav_file(stitched_path, stitched, sample_rate)

    sidecar_path = output_dir / "run.json"
    sidecar = {
        "prompts": list(prompts),
        "chunk_paths": chunk_paths,
        "chunk_count": len(prompts),
        "chunk_seeds": chunk_seeds,
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "seconds_total": sample_size / sample_rate,
        "audio_prompt_num_samples": audio_prompt_num_samples,
        "crossfade_overlap_seconds": crossfade_overlap_seconds,
        "crossfade_overlap_samples": overlap_samples,
        "pretrained_name": pretrained_name,
        "lora_path": str(lora_path) if lora_path is not None else None,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return {
        "chunk_paths": chunk_paths,
        "stitched_path": str(stitched_path),
        "sidecar_path": str(sidecar_path),
        "sample_rate": sample_rate,
        "sample_size": sample_size,
    }


def _sanitize_clip_id(value: str, index: int) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if sanitized:
        return sanitized
    return f"prompt_{index:03d}"


def save_wav_file(path: tp.Union[str, Path], audio: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    waveform = audio.detach().cpu().to(torch.float32)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    try:
        import torchaudio

        torchaudio.save(str(path), waveform, sample_rate)
        return
    except Exception:
        pass

    try:
        import soundfile as sf

        sf.write(str(path), waveform.transpose(0, 1).numpy(), sample_rate)
        return
    except Exception:
        pass

    from scipy.io import wavfile

    clipped = waveform.clamp(-1.0, 1.0).transpose(0, 1).numpy()
    pcm16 = np.int16(clipped * 32767.0)
    wavfile.write(str(path), sample_rate, pcm16)


def run_soundfx_eval_batch(
    prompt_manifest: tp.Union[str, Path],
    *,
    output_dir: tp.Union[str, Path],
    pretrained_name: str = "HKUSTAudio/AudioX-MAF-MMDiT",
    lora_path: tp.Optional[tp.Union[str, Path]] = None,
    cache_dir: tp.Optional[tp.Union[str, Path]] = None,
    device: tp.Optional[str] = None,
    steps: int = 250,
    cfg_scale: float = 7.0,
    sigma_min: float = 0.3,
    sigma_max: float = 500.0,
    sampler_type: str = "dpmpp-3m-sde",
    seed_base: int = 0,
) -> tp.Dict[str, tp.Any]:
    records = load_prompt_manifest(prompt_manifest)
    if not records:
        raise ValueError("run_soundfx_eval_batch requires at least one prompt.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = get_pretrained_model(pretrained_name, cache_dir=cache_dir, lora_path=lora_path)
    model = model.to(resolved_device).eval()

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    video_fps = model_config.get("video_fps", 5)
    audio_prompt_num_samples = extract_audio_prompt_num_samples(model_config, sample_size)
    zero_video_frame_count = infer_zero_video_frame_count(
        model,
        sample_rate=sample_rate,
        sample_size=sample_size,
        video_fps=video_fps,
    )

    output_dir = Path(output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    outputs_path = output_dir / "outputs.jsonl"
    prompt_copy_path = output_dir / "prompt_manifest.json"
    generation_config_path = output_dir / "generation_config.json"

    prompt_copy_path.write_text(json.dumps(records, indent=2))
    generation_config_path.write_text(
        json.dumps(
            {
                "pretrained_name": pretrained_name,
                "lora_path": str(lora_path) if lora_path is not None else None,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "sampler_type": sampler_type,
                "seed_base": seed_base,
                "sample_rate": sample_rate,
                "sample_size": sample_size,
                "audio_prompt_num_samples": audio_prompt_num_samples,
                "video_fps": video_fps,
            },
            indent=2,
        )
    )

    output_rows: tp.List[tp.Dict[str, tp.Any]] = []
    with outputs_path.open("w") as handle:
        for index, record in enumerate(records):
            clip_id = _sanitize_clip_id(record["clip_id"], index)
            seed = seed_base + index
            conditioning = build_standalone_conditioning(
                record["text_prompt"],
                sample_rate=sample_rate,
                sample_size=sample_size,
                video_fps=video_fps,
                audio_prompt_num_samples=audio_prompt_num_samples,
                zero_video_frame_count=zero_video_frame_count,
                device=resolved_device,
            )
            generated = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                seed=seed,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=resolved_device,
            ).detach().cpu()

            wav_path = wav_dir / f"{clip_id}.wav"
            save_wav_file(wav_path, generated.squeeze(0), sample_rate)

            output_row = {
                "clip_id": clip_id,
                "text_prompt": record["text_prompt"],
                "seed": seed,
                "wav_path": str(wav_path),
            }
            handle.write(json.dumps(output_row) + "\n")
            output_rows.append(output_row)

    return {
        "prompt_manifest_path": str(prompt_copy_path),
        "generation_config_path": str(generation_config_path),
        "outputs_path": str(outputs_path),
        "wav_dir": str(wav_dir),
        "row_count": len(output_rows),
    }
