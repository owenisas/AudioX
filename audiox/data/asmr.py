import json
import random
import typing as tp
from pathlib import Path


ASMR_SAMPLE_RATE = 48_000
ASMR_SAMPLE_SIZE = 480_000
ASMR_SECONDS_TOTAL = ASMR_SAMPLE_SIZE / ASMR_SAMPLE_RATE


def _load_records(path: tp.Union[str, Path]) -> tp.List[tp.Dict[str, tp.Any]]:
    manifest_path = Path(path)
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
        records = payload.get("records") or payload.get("data") or payload.get("chunks")
        if isinstance(records, list):
            return records
    raise ValueError(f"Unsupported ASMR metadata format: {manifest_path}")


def _first_nonempty(*values: tp.Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _maybe_int(value: tp.Any) -> tp.Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _maybe_float(value: tp.Any) -> tp.Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(value: tp.Any, base_dir: Path) -> tp.Optional[str]:
    text = _first_nonempty(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def derive_sequence_id(record: tp.Dict[str, tp.Any]) -> str:
    sequence_id = _first_nonempty(
        record.get("sequence_id"),
        record.get("source_id"),
        record.get("source_video_id"),
        record.get("source_path"),
        record.get("channel_id"),
    )
    if sequence_id:
        return sequence_id
    raise ValueError("ASMR chunk metadata must include sequence_id/source_id/source_video_id/source_path/channel_id.")


def derive_chunk_index(record: tp.Dict[str, tp.Any]) -> tp.Tuple[int, float, str]:
    chunk_index = _maybe_int(record.get("chunk_index"))
    if chunk_index is not None:
        return chunk_index, float(chunk_index), str(chunk_index)

    clip_index = _maybe_int(record.get("clip_index"))
    if clip_index is not None:
        return clip_index, float(clip_index), str(clip_index)

    start_s = _maybe_float(record.get("start_s"))
    if start_s is not None:
        return int(round(start_s * 1000)), start_s, f"{start_s:.3f}"

    clip_id = _first_nonempty(record.get("clip_id"), record.get("clip_path"), record.get("audio_path"))
    if clip_id:
        return 0, 0.0, clip_id

    raise ValueError("ASMR chunk metadata must include chunk_index/clip_index/start_s/clip_id.")


def build_asmr_manifest_rows(
    records: tp.Sequence[tp.Dict[str, tp.Any]],
    *,
    base_dir: tp.Union[str, Path] = ".",
    seconds_total: float = ASMR_SECONDS_TOTAL,
) -> tp.List[tp.Dict[str, tp.Any]]:
    base_dir = Path(base_dir)
    by_sequence: tp.Dict[str, tp.List[tp.Tuple[tp.Tuple[int, float, str], tp.Dict[str, tp.Any]]]] = {}

    for record in records:
        sequence_id = derive_sequence_id(record)
        sort_key = derive_chunk_index(record)
        by_sequence.setdefault(sequence_id, []).append((sort_key, dict(record)))

    manifest_rows: tp.List[tp.Dict[str, tp.Any]] = []

    for sequence_id, items in by_sequence.items():
        items.sort(key=lambda item: item[0])
        for order_index, (_, record) in enumerate(items):
            audio_path = _resolve_path(record.get("audio_path"), base_dir)
            if audio_path is None:
                raise ValueError(f"ASMR chunk in sequence {sequence_id} is missing audio_path.")

            text_prompt = _first_nonempty(record.get("text_prompt"), record.get("caption"), record.get("prompt"))
            if not text_prompt:
                raise ValueError(f"ASMR chunk {audio_path} is missing caption/text_prompt.")

            video_path = _resolve_path(record.get("video_path"), base_dir)
            chunk_index = _maybe_int(record.get("chunk_index"))
            if chunk_index is None:
                chunk_index = _maybe_int(record.get("clip_index"))
            if chunk_index is None:
                chunk_index = order_index

            row_base = {
                "audio_path": audio_path,
                "text_prompt": text_prompt,
                "sequence_id": sequence_id,
                "chunk_index": chunk_index,
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
            }
            if video_path:
                row_base["video_path"] = video_path

            manifest_rows.append(
                {
                    **row_base,
                    "audio_prompt_path": None,
                    "sample_type": "standalone",
                }
            )

            if order_index > 0:
                previous_audio_path = _resolve_path(items[order_index - 1][1].get("audio_path"), base_dir)
                manifest_rows.append(
                    {
                        **row_base,
                        "audio_prompt_path": previous_audio_path,
                        "sample_type": "continuation",
                    }
                )

    return manifest_rows


def split_manifest_rows_by_sequence(
    rows: tp.Sequence[tp.Dict[str, tp.Any]],
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tp.Tuple[tp.List[tp.Dict[str, tp.Any]], tp.List[tp.Dict[str, tp.Any]]]:
    sequence_ids = sorted({row["sequence_id"] for row in rows})
    if not sequence_ids or val_ratio <= 0:
        return list(rows), []

    if len(sequence_ids) == 1:
        return list(rows), []

    rng = random.Random(seed)
    shuffled = list(sequence_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(sequence_ids) * val_ratio)))
    val_sequence_ids = set(shuffled[:val_count])

    train_rows = [row for row in rows if row["sequence_id"] not in val_sequence_ids]
    val_rows = [row for row in rows if row["sequence_id"] in val_sequence_ids]
    return train_rows, val_rows


def write_jsonl(path: tp.Union[str, Path], rows: tp.Sequence[tp.Dict[str, tp.Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare_asmr_continuation_manifests(
    chunk_manifest_path: tp.Union[str, Path],
    output_dir: tp.Union[str, Path],
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
    seconds_total: float = ASMR_SECONDS_TOTAL,
) -> tp.Dict[str, tp.Any]:
    chunk_manifest_path = Path(chunk_manifest_path)
    output_dir = Path(output_dir)
    records = _load_records(chunk_manifest_path)
    rows = build_asmr_manifest_rows(
        records,
        base_dir=chunk_manifest_path.parent,
        seconds_total=seconds_total,
    )
    train_rows, val_rows = split_manifest_rows_by_sequence(rows, val_ratio=val_ratio, seed=seed)

    train_manifest_path = output_dir / "train.jsonl"
    val_manifest_path = output_dir / "val.jsonl"
    write_jsonl(train_manifest_path, train_rows)
    write_jsonl(val_manifest_path, val_rows)

    return {
        "train_manifest_path": str(train_manifest_path),
        "val_manifest_path": str(val_manifest_path) if val_rows else None,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "sequence_ids": sorted({row["sequence_id"] for row in rows}),
        "sample_rate": ASMR_SAMPLE_RATE,
        "sample_size": ASMR_SAMPLE_SIZE,
        "seconds_total": seconds_total,
    }
