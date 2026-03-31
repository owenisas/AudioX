import json
import typing as tp
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES = ("asmr", "ambience", "music")


def _load_jsonl(path: Path) -> tp.List[tp.Dict[str, tp.Any]]:
    records: tp.List[tp.Dict[str, tp.Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def _coerce_list(value: tp.Any) -> tp.List[tp.Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _maybe_float(value: tp.Any) -> tp.Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: tp.Any) -> tp.Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        float_value = _maybe_float(value)
        if float_value is None:
            return None
        return int(float_value)


def _resolve_source_manifest(path: tp.Union[str, Path]) -> tp.Tuple[Path, Path]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        manifest_path = candidate / "audio" / "audio_manifest_split.jsonl"
        if manifest_path.exists():
            return candidate, manifest_path
        raise ValueError(
            f"Dataset root {candidate} does not contain audio/audio_manifest_split.jsonl."
        )

    manifest_path = candidate
    if not manifest_path.exists():
        raise ValueError(f"Mixed-preference manifest path does not exist: {manifest_path}")
    if manifest_path.name != "audio_manifest_split.jsonl":
        dataset_root = manifest_path.parent.parent if manifest_path.parent.name == "audio" else manifest_path.parent
    else:
        dataset_root = manifest_path.parent.parent
    return dataset_root.resolve(), manifest_path.resolve()


def _resolve_source_path(value: tp.Any, base_dir: Path) -> tp.Optional[Path]:
    text = _first_nonempty(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _rewrite_media_path(
    source_path: Path,
    *,
    dataset_root: Path,
    media_root: tp.Optional[Path],
) -> str:
    if media_root is None:
        return str(source_path)
    try:
        relative_path = source_path.relative_to(dataset_root)
    except ValueError:
        return str(source_path)
    return str(media_root / relative_path)


def _resolve_text_prompt(record: tp.Dict[str, tp.Any], caption_field: str) -> str:
    candidates = collect_text_prompt_candidates(record, caption_field)
    return candidates[0] if candidates else ""


def _normalize_prompt_key(value: tp.Any) -> str:
    return " ".join(str(value).strip().split())


def _iter_prompt_values(value: tp.Any) -> tp.Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        text = _first_nonempty(value.get("text"), value.get("caption"), value.get("prompt"))
        if text:
            yield text
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_prompt_values(item)
        return

    text = str(value).strip()
    if text:
        yield text


def collect_text_prompt_candidates(record: tp.Dict[str, tp.Any], caption_field: str) -> tp.List[str]:
    ordered_values = [
        record.get(caption_field),
        record.get("text_prompt"),
        record.get("tagged_training_caption"),
        record.get("training_caption"),
        record.get("tagged_caption"),
        record.get("caption"),
        record.get("tagged_alternate_captions"),
        record.get("alternate_captions"),
        record.get("augmented_captions"),
    ]

    candidates: tp.List[str] = []
    seen: tp.Set[str] = set()
    for value in ordered_values:
        for prompt in _iter_prompt_values(value):
            key = _normalize_prompt_key(prompt)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(prompt.strip())
    return candidates


def _resolve_clip_duration(record: tp.Dict[str, tp.Any], default_seconds: float = 10.0) -> float:
    start_s = _maybe_float(record.get("start_s"))
    end_s = _maybe_float(record.get("end_s"))
    if start_s is not None and end_s is not None and end_s > start_s:
        return end_s - start_s
    return default_seconds


def _resolve_split(record: tp.Dict[str, tp.Any]) -> str:
    split = _first_nonempty(record.get("split")).lower()
    if split in {"train", "val", "test"}:
        return split
    return "train"


def _resolve_sequence_id(record: tp.Dict[str, tp.Any]) -> str:
    sequence_id = _first_nonempty(
        record.get("sample_group_id"),
        record.get("sequence_id"),
        record.get("source_video_id"),
        record.get("clip_id"),
    )
    if sequence_id:
        return sequence_id
    raise ValueError("Mixed-preference record is missing sample_group_id/sequence_id/source_video_id/clip_id.")


def build_mixed_preference_manifest_rows(
    records: tp.Sequence[tp.Dict[str, tp.Any]],
    *,
    dataset_root: tp.Union[str, Path],
    media_root: tp.Optional[tp.Union[str, Path]] = None,
    source_families: tp.Sequence[str] = DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES,
    caption_field: str = "tagged_training_caption",
    include_video: bool = True,
    adjacency_tolerance: float = 1e-6,
) -> tp.List[tp.Dict[str, tp.Any]]:
    dataset_root = Path(dataset_root).resolve()
    media_root_path = Path(media_root).expanduser() if media_root else None
    allowed_source_families = {family.strip() for family in source_families if family.strip()}

    grouped: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]] = defaultdict(list)
    for record in records:
        source_family = _first_nonempty(record.get("source_family"))
        if allowed_source_families and source_family not in allowed_source_families:
            continue
        grouped[_resolve_sequence_id(record)].append(dict(record))

    manifest_rows: tp.List[tp.Dict[str, tp.Any]] = []
    for sequence_id, items in grouped.items():
        items.sort(key=lambda row: (_maybe_float(row.get("start_s")) or 0.0, _maybe_int(row.get("clip_index")) or 0))
        previous_row: tp.Optional[tp.Dict[str, tp.Any]] = None

        for row in items:
            audio_source_path = _resolve_source_path(row.get("audio_path"), dataset_root)
            if audio_source_path is None or not audio_source_path.exists():
                raise ValueError(f"Missing audio target for clip {row.get('clip_id')}: {row.get('audio_path')}")

            text_prompt_candidates = collect_text_prompt_candidates(row, caption_field)
            text_prompt = text_prompt_candidates[0] if text_prompt_candidates else ""
            if not text_prompt:
                raise ValueError(f"Missing caption field '{caption_field}' and fallback training caption for clip {row.get('clip_id')}")

            clip_duration = _resolve_clip_duration(row)
            source_family = _first_nonempty(row.get("source_family"))
            preference_tags = [str(tag).strip() for tag in _coerce_list(row.get("preference_tags")) if str(tag).strip()]
            clip_index = _maybe_int(row.get("clip_index"))
            if clip_index is None:
                clip_index = len(manifest_rows)

            output_row = {
                "audio_path": _rewrite_media_path(
                    audio_source_path,
                    dataset_root=dataset_root,
                    media_root=media_root_path,
                ),
                "text_prompt": text_prompt,
                "text_prompt_candidates": text_prompt_candidates,
                "sample_type": "standalone",
                "sequence_id": sequence_id,
                "chunk_index": clip_index,
                "seconds_start": 0.0,
                "seconds_total": clip_duration,
                "video_duration_seconds": clip_duration,
                "source_family": source_family,
                "preference_tags": preference_tags,
                "clip_id": _first_nonempty(row.get("clip_id")),
                "split": _resolve_split(row),
            }

            video_source_path = _resolve_source_path(row.get("video_path"), dataset_root)
            if include_video and video_source_path is not None and video_source_path.exists():
                output_row["video_path"] = _rewrite_media_path(
                    video_source_path,
                    dataset_root=dataset_root,
                    media_root=media_root_path,
                )

            manifest_rows.append(output_row)

            if previous_row is None:
                previous_row = row
                continue

            previous_split = _resolve_split(previous_row)
            current_split = _resolve_split(row)
            previous_end_s = _maybe_float(previous_row.get("end_s"))
            current_start_s = _maybe_float(row.get("start_s"))
            is_adjacent = (
                previous_split == current_split
                and previous_end_s is not None
                and current_start_s is not None
                and abs(current_start_s - previous_end_s) <= adjacency_tolerance
            )

            if is_adjacent:
                previous_audio_source_path = _resolve_source_path(previous_row.get("audio_path"), dataset_root)
                if previous_audio_source_path is None or not previous_audio_source_path.exists():
                    raise ValueError(
                        f"Missing continuation audio prompt for clip {row.get('clip_id')}: {previous_row.get('audio_path')}"
                    )

                continuation_row = dict(output_row)
                continuation_row["sample_type"] = "continuation"
                continuation_row["audio_prompt_path"] = _rewrite_media_path(
                    previous_audio_source_path,
                    dataset_root=dataset_root,
                    media_root=media_root_path,
                )
                manifest_rows.append(continuation_row)

            previous_row = row

    return manifest_rows


def split_rows_by_split(
    rows: tp.Sequence[tp.Dict[str, tp.Any]]
) -> tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]]:
    split_to_rows: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = _resolve_split(row)
        row_copy = dict(row)
        row_copy.pop("split", None)
        split_to_rows.setdefault(split, []).append(row_copy)
    return split_to_rows


def write_jsonl(path: tp.Union[str, Path], rows: tp.Sequence[tp.Dict[str, tp.Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare_mixed_preference_manifests(
    dataset_root_or_manifest: tp.Union[str, Path],
    output_dir: tp.Union[str, Path],
    *,
    media_root: tp.Optional[tp.Union[str, Path]] = None,
    source_families: tp.Sequence[str] = DEFAULT_MIXED_PREFERENCE_SOURCE_FAMILIES,
    caption_field: str = "tagged_training_caption",
    include_video: bool = True,
    adjacency_tolerance: float = 1e-6,
) -> tp.Dict[str, tp.Any]:
    dataset_root, manifest_path = _resolve_source_manifest(dataset_root_or_manifest)
    records = _load_jsonl(manifest_path)
    rows = build_mixed_preference_manifest_rows(
        records,
        dataset_root=dataset_root,
        media_root=media_root,
        source_families=source_families,
        caption_field=caption_field,
        include_video=include_video,
        adjacency_tolerance=adjacency_tolerance,
    )
    split_rows = split_rows_by_split(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: tp.Dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows.get(split, []))
        manifest_paths[f"{split}_manifest_path"] = str(path)

    source_family_counts = Counter()
    sample_type_counts = Counter()
    for row in rows:
        source_family_counts.update([row.get("source_family")])
        sample_type_counts.update([row.get("sample_type")])

    return {
        **manifest_paths,
        "dataset_root": str(dataset_root),
        "source_manifest_path": str(manifest_path),
        "row_count": len(rows),
        "split_counts": {split: len(split_rows.get(split, [])) for split in ("train", "val", "test")},
        "source_family_counts": dict(source_family_counts),
        "sample_type_counts": dict(sample_type_counts),
        "caption_field": caption_field,
        "source_families": list(source_families),
        "include_video": include_video,
        "media_root": str(Path(media_root).expanduser()) if media_root else None,
        "adjacency_tolerance": adjacency_tolerance,
    }
