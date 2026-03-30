import hashlib
import json
import math
import os
import wave
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import PadCrop_Normalized_T, Stereo, read_video


def _stable_bucket(seed_text: str) -> int:
    digest = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


def _normalize_name(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _normalize_lookup(value: Any) -> str:
    return _normalize_name(value).lower()


def _coerce_count(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _format_seconds(value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return None


def _parse_timestamp_range(value: str) -> Tuple[Optional[str], Optional[str]]:
    if not value or "-" not in value:
        return None, None
    start_raw, end_raw = value.split("-", 1)
    return _format_timestamp_token(start_raw), _format_timestamp_token(end_raw)


def _format_timestamp_token(token: str) -> Optional[str]:
    token = str(token).strip()
    if not token:
        return None
    try:
        if ":" in token:
            parts = [float(part) for part in token.split(":")]
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60 + part
            return f"{seconds:.1f}"
        return f"{float(token):.1f}"
    except ValueError:
        return None


def _dedupe_events(events: Iterable[Dict[str, Any]], max_events: int) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for event in events:
        key = (
            _normalize_lookup(event.get("name")),
            event.get("start"),
            event.get("end"),
            event.get("count"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
        if len(deduped) >= max_events:
            break
    return deduped


def _build_event_attrs(event: Dict[str, Any], compact: bool) -> str:
    attrs = [f'name="{escape(event["name"])}"']
    start = event.get("start")
    end = event.get("end")
    count = event.get("count")
    if start is not None:
        attrs.append(f'start="{start}"')
    if end is not None:
        attrs.append(f'end="{end}"')
    if count is not None:
        attrs.append(f'count="{count}"')
    elif not compact:
        attrs.append('count="null"')
    return " ".join(attrs)


def _extract_events(record: Dict[str, Any], max_events: int) -> List[Dict[str, Any]]:
    category = record.get("category") or {}
    sed_entries = record.get("SED") or record.get("sed") or []

    counts_by_name: Dict[str, Optional[int]] = {}
    if isinstance(category, dict):
        for name, count in category.items():
            counts_by_name[_normalize_lookup(name)] = _coerce_count(count)

    events: List[Dict[str, Any]] = []
    for entry in sed_entries:
        if isinstance(entry, dict):
            if {"name", "start", "end"} <= set(entry.keys()):
                name = _normalize_name(entry.get("name"))
                start = _format_seconds(entry.get("start"))
                end = _format_seconds(entry.get("end"))
                count = _coerce_count(entry.get("count"))
            elif len(entry) == 1:
                timestamp, description = next(iter(entry.items()))
                start, end = _parse_timestamp_range(str(timestamp))
                name = _normalize_name(description)
                count = None
            else:
                name = _normalize_name(entry.get("label") or entry.get("event") or entry.get("description"))
                start = _format_seconds(entry.get("start"))
                end = _format_seconds(entry.get("end"))
                count = _coerce_count(entry.get("count"))
            if name:
                inferred_count = counts_by_name.get(_normalize_lookup(name))
                events.append(
                    {
                        "name": name,
                        "start": start,
                        "end": end,
                        "count": count if count is not None else inferred_count,
                    }
                )

    for raw_name, raw_count in category.items() if isinstance(category, dict) else []:
        name = _normalize_name(raw_name)
        if not name:
            continue
        events.append({"name": name, "start": None, "end": None, "count": _coerce_count(raw_count)})

    return _dedupe_events(events, max_events=max_events)


def build_natural_prompt(record: Dict[str, Any]) -> str:
    for key in ("text_prompt", "prompt", "caption", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    categories = record.get("category") or {}
    if isinstance(categories, dict) and categories:
        parts = []
        for name, count in categories.items():
            normalized = _normalize_name(name)
            if not normalized:
                continue
            normalized_count = _coerce_count(count)
            if normalized_count is None:
                parts.append(normalized)
            else:
                parts.append(f"{normalized} x{normalized_count}")
        if parts:
            return "Audio containing " + ", ".join(parts) + "."

    music_parts = []
    for key in ("genre", "mood", "instrument", "tempo"):
        value = record.get(key)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if str(item).strip())
        if value:
            music_parts.append(f"{key}: {value}")
    if music_parts:
        return "Music attributes: " + "; ".join(music_parts) + "."

    return "Generate audio."


def serialize_ifcaps_to_xml(
    record: Dict[str, Any],
    compact: bool = True,
    include_caption: bool = True,
    max_events: int = 8,
) -> str:
    events = _extract_events(record, max_events=max_events)
    category = record.get("category") or {}
    time_relation = _normalize_name(record.get("time_relation"))

    lines: List[str] = ["<audio>"]
    caption = build_natural_prompt(record)
    if include_caption and caption:
        lines.append(f'  <caption>{escape(caption)}</caption>')

    if events:
        lines.append("  <events>")
        for event in events:
            lines.append(f"    <event {_build_event_attrs(event, compact=compact)}/>")
        lines.append("  </events>")
    elif isinstance(category, dict) and category:
        lines.append("  <events/>")

    if time_relation:
        lines.append(f'  <order>{escape(time_relation)}</order>')

    music_attrs = {}
    for key in ("genre", "mood", "tempo"):
        value = record.get(key)
        if value:
            music_attrs[key] = value
    instruments = record.get("instrument")
    if instruments:
        if isinstance(instruments, list):
            music_attrs["instrument"] = "|".join(str(item) for item in instruments if str(item).strip())
        else:
            music_attrs["instrument"] = str(instruments)
    if music_attrs:
        attrs = " ".join(f'{key}="{escape(str(value))}"' for key, value in music_attrs.items())
        lines.append(f"  <music {attrs}/>")

    lines.append("</audio>")
    if compact:
        return "".join(line.strip() for line in lines)
    return "\n".join(lines)


def build_text_prompt(
    record: Dict[str, Any],
    prompt_format: str = "mixed",
    *,
    compact_xml: bool = True,
    include_caption_in_xml: bool = True,
    max_events: int = 8,
    selector: Optional[int] = None,
) -> Tuple[str, str]:
    prompt_format = (prompt_format or "mixed").lower()
    natural_prompt = build_natural_prompt(record)
    xml_prompt = serialize_ifcaps_to_xml(
        record,
        compact=compact_xml,
        include_caption=include_caption_in_xml,
        max_events=max_events,
    )

    if prompt_format == "natural":
        return natural_prompt, "natural"
    if prompt_format == "xml":
        return xml_prompt, "xml"
    if prompt_format != "mixed":
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    if selector is None:
        selector = 0
    bucket = selector % 4
    if bucket in (0, 1):
        return natural_prompt, "natural"
    if bucket == 2:
        return xml_prompt, "xml"
    return f"{xml_prompt}\n{natural_prompt}", "mixed"


def normalize_training_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(metadata)
    if "text_prompt" not in normalized or not str(normalized.get("text_prompt") or "").strip():
        legacy_prompt = normalized.get("prompt")
        if isinstance(legacy_prompt, str) and legacy_prompt.strip():
            normalized["text_prompt"] = legacy_prompt.strip()
    return normalized


def _load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"Unsupported manifest format in {path}")


def load_manifest_records(path: str) -> List[Dict[str, Any]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        return _load_json_records(path)

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def _load_audio_segment(
    path: Optional[str],
    sample_rate: int,
    sample_size: int,
    seconds_start: float,
    seconds_total: float,
) -> torch.Tensor:
    if not path:
        return torch.zeros((2, sample_size), dtype=torch.float32)
    audio, source_sample_rate = _load_audio_waveform(path)
    audio = Stereo()(audio).to(torch.float32)
    if source_sample_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_sample_rate, sample_rate)
    start_index = int(sample_rate * seconds_start)
    target_length = int(sample_rate * seconds_total)
    end_index = start_index + target_length
    audio = audio[:, start_index:end_index]
    if audio.shape[-1] < target_length:
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))
    audio = Stereo()(audio).to(torch.float32)
    if audio.shape[-1] > sample_size:
        audio = audio[:, :sample_size]
    elif audio.shape[-1] < sample_size:
        audio = torch.nn.functional.pad(audio, (0, sample_size - audio.shape[-1]))
    return audio


def _load_audio_waveform(path: str) -> Tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(path)
    except ImportError:
        with wave.open(path, "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV fallback is supported, got sample width {sample_width}")

        audio = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32).div(32767.0)
        audio = audio.view(-1, channels).transpose(0, 1).contiguous()
        return audio, sample_rate


class IFCapsFineTuneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        sample_size: int,
        *,
        video_fps: int = 5,
        prompt_format: str = "mixed",
        xml_compact: bool = True,
        xml_include_caption: bool = True,
        xml_max_events: int = 8,
        prompt_seed: int = 0,
        random_crop: bool = True,
    ):
        super().__init__()
        self.records = load_manifest_records(manifest_path)
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.video_fps = video_fps
        self.prompt_format = prompt_format
        self.xml_compact = xml_compact
        self.xml_include_caption = xml_include_caption
        self.xml_max_events = xml_max_events
        self.prompt_seed = prompt_seed
        self.clip_seconds = sample_size / sample_rate
        self.audio_transform = PadCrop_Normalized_T(
            n_samples=sample_size,
            sample_rate=sample_rate,
            randomize=random_crop,
        )
        self.stereo = Stereo()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        record = dict(self.records[index])
        audio_path = record.get("audio_path") or record.get("path") or record.get("waveform_path")
        if not audio_path:
            raise ValueError(f"Record at index {index} is missing audio_path/path")

        waveform, sample_rate = _load_audio_waveform(audio_path)
        waveform = self.stereo(waveform).to(torch.float32)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        audio_chunk, _, _, chunk_seconds_start, _, padding_mask = self.audio_transform(waveform)
        seconds_start = float(record.get("seconds_start", chunk_seconds_start))
        seconds_total = float(record.get("seconds_total", self.clip_seconds))

        selector_source = record.get("id") or record.get("audio_id") or audio_path
        selector = _stable_bucket(f"{self.prompt_seed}:{index}:{selector_source}")
        text_prompt, prompt_variant = build_text_prompt(
            record,
            prompt_format=self.prompt_format,
            compact_xml=self.xml_compact,
            include_caption_in_xml=self.xml_include_caption,
            max_events=self.xml_max_events,
            selector=selector,
        )

        metadata = normalize_training_metadata(record)
        metadata["text_prompt"] = text_prompt
        metadata["prompt_variant"] = prompt_variant
        metadata["seconds_start"] = seconds_start
        metadata["seconds_total"] = seconds_total
        metadata["padding_mask"] = padding_mask
        metadata["path"] = audio_path

        audio_prompt_path = record.get("audio_prompt_path") or record.get("conditioning_audio_path")
        metadata["audio_prompt"] = _load_audio_segment(
            audio_prompt_path,
            self.sample_rate,
            self.sample_size,
            seconds_start,
            seconds_total,
        ).unsqueeze(0)

        video_path = record.get("video_path") or record.get("image_path")
        target_frames = max(int(round(seconds_total * self.video_fps)), 1)
        if video_path:
            video_tensor = read_video(
                video_path,
                seek_time=seconds_start,
                duration=seconds_total,
                target_fps=self.video_fps,
            ).to(torch.float32)
        else:
            video_tensor = torch.zeros((target_frames, 3, 224, 224), dtype=torch.float32)

        sync_feature_path = record.get("video_sync_path")
        if sync_feature_path:
            sync_features = torch.load(sync_feature_path, map_location="cpu")
            if sync_features.ndim == 2:
                sync_features = sync_features.unsqueeze(0)
        else:
            sync_features = torch.zeros((1, 240, 768), dtype=torch.float32)

        metadata["video_prompt"] = {
            "video_tensors": video_tensor.unsqueeze(0),
            "video_sync_frames": sync_features.to(torch.float32),
        }

        return audio_chunk, metadata


def collate_finetune_batch(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    reals = torch.stack([item[0] for item in batch], dim=0)
    metadata = [item[1] for item in batch]
    return reals, metadata
