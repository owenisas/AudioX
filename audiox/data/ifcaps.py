import json
import math
import random
import typing as tp
import warnings
import wave
from html import escape
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .utils import (
    PadCrop_Normalized_T,
    Stereo,
    encode_video_with_synchformer,
    load_and_process_audio,
    read_video,
)


def _first_nonempty(*values: tp.Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        elif isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return ", ".join(items)
        else:
            text = str(value).strip()
            if text:
                return text
    return ""


def _coerce_jsonish(value: tp.Any) -> tp.Any:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_sequence(value: tp.Any) -> tp.List[tp.Any]:
    value = _coerce_jsonish(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_float(value: tp.Any) -> tp.Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: tp.Any) -> tp.Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        float_value = _to_float(value)
        if float_value is None:
            return None
        return int(float_value)


def _format_timestamp(value: tp.Any) -> tp.Optional[str]:
    float_value = _to_float(value)
    if float_value is None:
        return None
    return f"{float_value:.1f}"


def _event_name(entry: tp.Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0]).strip()
    if isinstance(entry, dict):
        return _first_nonempty(
            entry.get("name"),
            entry.get("event"),
            entry.get("label"),
            entry.get("category"),
            entry.get("caption"),
            entry.get("text"),
        )
    return ""


def _extract_events(record: tp.Dict[str, tp.Any], xml_max_events: int) -> tp.List[tp.Dict[str, tp.Any]]:
    raw_events = _coerce_sequence(
        record.get("SED", record.get("sed", record.get("events")))
    )
    categories = _coerce_sequence(record.get("category", record.get("categories")))

    parsed_events: tp.List[tp.Dict[str, tp.Any]] = []

    for entry in raw_events:
        if isinstance(entry, dict):
            event = {
                "name": _event_name(entry),
                "start": _to_float(
                    _first_nonempty(
                        entry.get("start"),
                        entry.get("start_time"),
                        entry.get("onset"),
                    )
                ),
                "end": _to_float(
                    _first_nonempty(
                        entry.get("end"),
                        entry.get("end_time"),
                        entry.get("offset"),
                    )
                ),
                "count": _to_int(
                    _first_nonempty(entry.get("count"), entry.get("num"), entry.get("n"))
                ),
            }
        elif isinstance(entry, (list, tuple)):
            event = {
                "name": _event_name(entry),
                "start": _to_float(entry[1]) if len(entry) > 1 else None,
                "end": _to_float(entry[2]) if len(entry) > 2 else None,
                "count": _to_int(entry[3]) if len(entry) > 3 else None,
            }
        else:
            event = {"name": _event_name(entry), "start": None, "end": None, "count": None}

        if event["name"]:
            parsed_events.append(event)

    if not parsed_events:
        for category in categories:
            name = str(category).strip()
            if name:
                parsed_events.append({"name": name, "start": None, "end": None, "count": None})

    collapsed: tp.Dict[tp.Tuple[str, tp.Optional[float], tp.Optional[float]], tp.Dict[str, tp.Any]] = {}
    ordered_keys: tp.List[tp.Tuple[str, tp.Optional[float], tp.Optional[float]]] = []

    for event in parsed_events:
        key = (event["name"], event["start"], event["end"])
        if key not in collapsed:
            collapsed[key] = dict(event)
            ordered_keys.append(key)
            continue

        existing_count = collapsed[key].get("count")
        incoming_count = event.get("count")
        if existing_count is None and incoming_count is None:
            collapsed[key]["count"] = 2
        elif existing_count is None:
            collapsed[key]["count"] = incoming_count
        elif incoming_count is None:
            collapsed[key]["count"] = existing_count + 1
        else:
            collapsed[key]["count"] = existing_count + incoming_count

    collapsed_events = [collapsed[key] for key in ordered_keys]
    return collapsed_events[:xml_max_events]


def _format_time_relation(value: tp.Any) -> str:
    value = _coerce_jsonish(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        left = _first_nonempty(value.get("first"), value.get("left"), value.get("source"))
        relation = _first_nonempty(value.get("relation"), value.get("type"))
        right = _first_nonempty(value.get("second"), value.get("right"), value.get("target"))
        if left and relation and right:
            return f"{left} {relation} {right}"
        return _first_nonempty(*value.values())
    if isinstance(value, (list, tuple)):
        parts = [_format_time_relation(item) for item in value]
        parts = [part for part in parts if part]
        return "; ".join(parts)
    return str(value).strip()


def build_natural_prompt(record: tp.Dict[str, tp.Any]) -> str:
    caption = _first_nonempty(
        record.get("text_prompt"),
        record.get("prompt"),
        record.get("caption"),
        record.get("instruction"),
    )
    if caption:
        return caption

    events = _extract_events(record, xml_max_events=8)
    event_names = [event["name"] for event in events if event["name"]]
    order = _format_time_relation(record.get("time_relation"))

    music_parts = [
        _first_nonempty(record.get("genre")),
        _first_nonempty(record.get("mood")),
        _first_nonempty(record.get("instrument")),
        _first_nonempty(record.get("tempo")),
    ]
    music_parts = [part for part in music_parts if part]

    parts = []
    if event_names:
        parts.append("Audio with " + ", ".join(event_names))
    if order:
        parts.append(order)
    if music_parts:
        parts.append("Music tags: " + ", ".join(music_parts))

    if parts:
        return ". ".join(parts)

    return "Generate audio."


def serialize_ifcaps_to_xml(
    record: tp.Dict[str, tp.Any],
    compact: bool = True,
    include_caption: bool = True,
    xml_max_events: int = 8,
) -> str:
    caption = build_natural_prompt(record)
    events = _extract_events(record, xml_max_events=xml_max_events)
    order = _format_time_relation(record.get("time_relation"))

    music_fields = {
        "genre": _first_nonempty(record.get("genre")),
        "mood": _first_nonempty(record.get("mood")),
        "instrument": _first_nonempty(record.get("instrument")),
        "tempo": _first_nonempty(record.get("tempo")),
    }
    music_fields = {key: value for key, value in music_fields.items() if value}

    lines = ["<audio>"]
    if include_caption and caption:
        lines.append(f"  <caption>{escape(caption)}</caption>")

    if events:
        lines.append("  <events>")
        for event in events:
            attributes = [f'name="{escape(event["name"])}"']
            start = _format_timestamp(event.get("start"))
            end = _format_timestamp(event.get("end"))
            count = event.get("count")
            if start is not None:
                attributes.append(f'start="{start}"')
            if end is not None:
                attributes.append(f'end="{end}"')
            if count is not None:
                attributes.append(f'count="{count}"')
            lines.append("    <event " + " ".join(attributes) + "/>")
        lines.append("  </events>")

    if order:
        lines.append(f"  <order>{escape(order)}</order>")

    if music_fields:
        lines.append("  <music>")
        for key, value in music_fields.items():
            lines.append(f"    <{key}>{escape(value)}</{key}>")
        lines.append("  </music>")

    lines.append("</audio>")

    if compact:
        return "".join(line.strip() for line in lines)
    return "\n".join(lines)


def select_prompt_variant(index: int, seed: int = 0) -> str:
    slot = (index + seed) % 4
    if slot in (0, 1):
        return "natural"
    if slot == 2:
        return "xml"
    return "xml_natural"


def build_text_prompt(
    record: tp.Dict[str, tp.Any],
    prompt_format: str = "mixed",
    *,
    xml_compact: bool = True,
    xml_include_caption: bool = True,
    xml_max_events: int = 8,
    mixed_variant: tp.Optional[str] = None,
) -> str:
    natural_prompt = build_natural_prompt(record)
    xml_prompt = serialize_ifcaps_to_xml(
        record,
        compact=xml_compact,
        include_caption=xml_include_caption,
        xml_max_events=xml_max_events,
    )

    if prompt_format == "natural":
        return natural_prompt
    if prompt_format == "xml":
        return xml_prompt
    if prompt_format != "mixed":
        raise ValueError(f"Unsupported prompt_format: {prompt_format}")

    variant = mixed_variant or "natural"
    if variant == "natural":
        return natural_prompt
    if variant == "xml":
        return xml_prompt
    if variant == "xml_natural":
        return f"{xml_prompt}\n{natural_prompt}" if natural_prompt else xml_prompt
    raise ValueError(f"Unsupported mixed prompt variant: {variant}")


def normalize_training_metadata(metadata: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    normalized = dict(metadata)
    if "text_prompt" not in normalized and "prompt" in normalized:
        normalized["text_prompt"] = normalized["prompt"]
    return normalized


def _coerce_text_prompt_candidates(value: tp.Any) -> tp.List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        candidates: tp.List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            else:
                text = str(item).strip()
            if text:
                candidates.append(text)
        return candidates
    text = str(value).strip()
    return [text] if text else []


def resolve_text_prompt_record(
    record: tp.Dict[str, tp.Any],
    *,
    sample_text_prompt_candidates: bool = False,
) -> tp.Dict[str, tp.Any]:
    resolved = dict(record)
    candidates = _coerce_text_prompt_candidates(resolved.get("text_prompt_candidates"))
    if not candidates:
        return resolved

    if sample_text_prompt_candidates and len(candidates) > 1:
        candidate_index = int(torch.randint(len(candidates), (1,)).item())
        resolved["text_prompt"] = candidates[candidate_index]
    else:
        resolved["text_prompt"] = candidates[0]
    resolved["text_prompt_candidates"] = candidates
    return resolved


def _resample_audio(audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return audio
    try:
        import torchaudio

        return torchaudio.functional.resample(audio, source_sr, target_sr)
    except Exception:
        new_length = int(round(audio.shape[-1] * target_sr / source_sr))
        return F.interpolate(audio.unsqueeze(0), size=new_length, mode="linear", align_corners=False).squeeze(0)


def _load_wav_with_wave(audio_path: Path) -> tp.Tuple[torch.Tensor, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    waveform = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.float32)
    if num_channels <= 0:
        raise ValueError(f"Invalid WAV channel count: {num_channels}")
    remainder = waveform.numel() % num_channels
    if remainder:
        valid_values = waveform.numel() - remainder
        if valid_values <= 0:
            raise ValueError(
                f"WAV payload in {audio_path} does not contain a complete frame for {num_channels} channels"
            )
        warnings.warn(
            f"Truncating {remainder} trailing PCM sample(s) from malformed WAV {audio_path}",
            RuntimeWarning,
        )
        waveform = waveform[:valid_values]
    waveform = waveform.view(-1, num_channels).t().contiguous() / 32768.0
    return waveform, sample_rate


def _load_audio_waveform(audio_path: Path, target_sample_rate: int) -> torch.Tensor:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception:
        waveform, sample_rate = _load_wav_with_wave(audio_path)

    if sample_rate != target_sample_rate:
        waveform = _resample_audio(waveform, sample_rate, target_sample_rate)

    return waveform


class IFCapsFineTuneDataset(Dataset):
    def __init__(
        self,
        manifest_path: tp.Union[str, Path],
        sample_rate: int,
        sample_size: int,
        *,
        prompt_format: str = "mixed",
        xml_compact: bool = True,
        xml_include_caption: bool = True,
        xml_max_events: int = 8,
        model_name: str = "HKUSTAudio/AudioX-MAF-MMDiT",
        include_video_conditioning: bool = False,
        include_audio_conditioning: bool = False,
        video_fps: int = 5,
        video_duration_seconds: float = 10.0,
        audio_prompt_num_samples: tp.Optional[int] = None,
        random_crop: bool = True,
        seed: int = 0,
        sample_text_prompt_candidates: bool = False,
        synchformer_ckpt_path: tp.Optional[tp.Union[str, Path]] = None,
        compute_video_sync_on_the_fly: bool = False,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.prompt_format = prompt_format
        self.xml_compact = xml_compact
        self.xml_include_caption = xml_include_caption
        self.xml_max_events = xml_max_events
        self.model_name = model_name
        self.include_video_conditioning = include_video_conditioning
        self.include_audio_conditioning = include_audio_conditioning
        self.video_fps = video_fps
        self.video_duration_seconds = video_duration_seconds
        self.audio_prompt_num_samples = audio_prompt_num_samples or sample_size
        self.seed = seed
        self.sample_text_prompt_candidates = sample_text_prompt_candidates
        self.synchformer_ckpt_path = str(synchformer_ckpt_path) if synchformer_ckpt_path else None
        self.compute_video_sync_on_the_fly = compute_video_sync_on_the_fly

        self.records = self._load_manifest(self.manifest_path)
        self.stereo = Stereo()
        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

    def _load_manifest(self, manifest_path: Path) -> tp.List[tp.Dict[str, tp.Any]]:
        if manifest_path.suffix == ".jsonl":
            records = []
            with manifest_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        with manifest_path.open() as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            records = payload.get("records") or payload.get("data")
            if isinstance(records, list):
                return records
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, value: tp.Any) -> tp.Optional[Path]:
        path_value = _first_nonempty(value)
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        return path

    def _zero_video_prompt(self, clip_seconds: float) -> tp.Dict[str, torch.Tensor]:
        frame_count = max(1, int(round(clip_seconds * self.video_fps)))
        return {
            "video_tensors": torch.zeros(1, frame_count, 3, 224, 224),
            "video_sync_frames": torch.zeros(1, 240, 768),
        }

    def _resolve_video_duration(self, record: tp.Dict[str, tp.Any]) -> float:
        explicit_duration = _to_float(record.get("video_duration_seconds"))
        if explicit_duration is not None:
            return explicit_duration

        start = _to_float(record.get("start_s"))
        end = _to_float(record.get("end_s"))
        if start is not None and end is not None and end > start:
            return end - start

        return self.video_duration_seconds

    def _load_video_prompt(
        self,
        record: tp.Dict[str, tp.Any],
        seconds_start: int,
        clip_seconds: float,
    ) -> tp.Dict[str, torch.Tensor]:
        if not self.include_video_conditioning:
            return self._zero_video_prompt(clip_seconds)

        video_path = self._resolve_path(record.get("video_path") or record.get("video"))
        if video_path is None:
            return self._zero_video_prompt(clip_seconds)

        video_tensor = read_video(
            str(video_path),
            seek_time=seconds_start,
            duration=clip_seconds,
            target_fps=self.video_fps,
        )
        if video_tensor.shape[0] == 0:
            return self._zero_video_prompt(clip_seconds)
        video_tensor = video_tensor.unsqueeze(0)

        sync_path = self._resolve_path(record.get("video_sync_frames_path"))
        if sync_path is not None and sync_path.exists():
            video_sync_frames = torch.load(sync_path, map_location="cpu")
            if video_sync_frames.ndim == 2:
                video_sync_frames = video_sync_frames.unsqueeze(0)
        else:
            inline_sync = record.get("video_sync_frames")
            if inline_sync is not None:
                video_sync_frames = torch.as_tensor(inline_sync).float()
                if video_sync_frames.ndim == 2:
                    video_sync_frames = video_sync_frames.unsqueeze(0)
            else:
                if self.compute_video_sync_on_the_fly:
                    video_sync_frames = encode_video_with_synchformer(
                        str(video_path),
                        self.model_name,
                        seconds_start=seconds_start,
                        seconds_total=clip_seconds,
                        device="cpu",
                        synchformer_ckpt_path=self.synchformer_ckpt_path,
                    ).cpu()
                else:
                    video_sync_frames = torch.zeros(1, 240, 768)

        return {
            "video_tensors": video_tensor,
            "video_sync_frames": video_sync_frames,
        }

    def _load_audio_prompt(self, record: tp.Dict[str, tp.Any], seconds_start: int, clip_seconds: float) -> torch.Tensor:
        if not self.include_audio_conditioning:
            return torch.zeros(1, 2, self.audio_prompt_num_samples)

        explicit_audio_prompt_key = None
        explicit_audio_prompt_value = None
        for candidate_key in ("audio_prompt_path", "audio_conditioning_path"):
            if candidate_key in record:
                explicit_audio_prompt_key = candidate_key
                explicit_audio_prompt_value = record.get(candidate_key)
                break

        if explicit_audio_prompt_key is not None:
            if explicit_audio_prompt_value is None:
                return torch.zeros(1, 2, self.audio_prompt_num_samples)
            if isinstance(explicit_audio_prompt_value, str) and not explicit_audio_prompt_value.strip():
                return torch.zeros(1, 2, self.audio_prompt_num_samples)
            audio_prompt_path = self._resolve_path(explicit_audio_prompt_value)
        else:
            audio_prompt_path = self._resolve_path(record.get("audio_path"))

        if audio_prompt_path is None:
            return torch.zeros(1, 2, self.audio_prompt_num_samples)

        audio_prompt_seconds = _to_float(record.get("audio_prompt_duration_seconds"))
        if audio_prompt_seconds is None:
            audio_prompt_samples = _to_int(record.get("audio_prompt_num_samples"))
            if audio_prompt_samples is None:
                audio_prompt_samples = self.audio_prompt_num_samples
            audio_prompt_seconds = audio_prompt_samples / self.sample_rate

        try:
            audio_tensor = load_and_process_audio(
                str(audio_prompt_path),
                self.sample_rate,
                seconds_start,
                audio_prompt_seconds,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to load audio conditioning from {audio_prompt_path}: {exc}. "
                "Using zero audio conditioning instead.",
                RuntimeWarning,
            )
            return torch.zeros(1, 2, self.audio_prompt_num_samples)
        if audio_tensor.shape[-1] > self.audio_prompt_num_samples:
            audio_tensor = audio_tensor[..., : self.audio_prompt_num_samples]
        elif audio_tensor.shape[-1] < self.audio_prompt_num_samples:
            audio_tensor = F.pad(audio_tensor, (0, self.audio_prompt_num_samples - audio_tensor.shape[-1]))
        return audio_tensor.unsqueeze(0)

    def __getitem__(self, index: int) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any]]:
        record = resolve_text_prompt_record(
            self.records[index],
            sample_text_prompt_candidates=self.sample_text_prompt_candidates,
        )
        audio_path = self._resolve_path(record.get("audio_path") or record.get("path"))
        if audio_path is None:
            raise ValueError(f"Record {index} is missing audio_path/path")

        waveform = _load_audio_waveform(audio_path, self.sample_rate)
        waveform = self.stereo(waveform)
        chunk, _, _, seconds_start, seconds_total, padding_mask = self.pad_crop(waveform)
        clip_seconds = self.sample_size / self.sample_rate
        video_seconds = self.video_duration_seconds

        variant = None
        if self.prompt_format == "mixed":
            variant = select_prompt_variant(index, self.seed)

        metadata = normalize_training_metadata(
            {
                "path": str(audio_path),
                "relpath": str(audio_path.relative_to(self.manifest_path.parent)) if audio_path.is_relative_to(self.manifest_path.parent) else audio_path.name,
                "text_prompt": build_text_prompt(
                    record,
                    prompt_format=self.prompt_format,
                    xml_compact=self.xml_compact,
                    xml_include_caption=self.xml_include_caption,
                    xml_max_events=self.xml_max_events,
                    mixed_variant=variant,
                ),
                "video_prompt": self._load_video_prompt(record, seconds_start, video_seconds),
                "audio_prompt": self._load_audio_prompt(record, seconds_start, clip_seconds),
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
                "padding_mask": padding_mask,
            }
        )

        return chunk, metadata


def collate_audiox_batch(
    batch: tp.Sequence[tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any]]]
) -> tp.Tuple[torch.Tensor, tp.List[tp.Dict[str, tp.Any]]]:
    audio = torch.stack([item[0] for item in batch], dim=0)
    metadata = [item[1] for item in batch]
    return audio, metadata
