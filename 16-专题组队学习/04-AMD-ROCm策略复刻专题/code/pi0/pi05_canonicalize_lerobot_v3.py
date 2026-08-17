#!/usr/bin/env python3

"""Canonicalize LeRobot v3 parquet row order without mutating the source.

LeRobot action chunks are addressed by the global ``index`` column.  Aggregating
datasets can leave every index present exactly once while arranging parquet rows
in a different physical order.  In that state, a sample's first chunk action
may no longer be the action stored on the same row.  This tool writes one
episode per parquet file, sorts rows by frame index, and rebuilds global indices
and episode metadata so physical row position and ``index`` agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
REQUIRED_COLUMNS = {
    "episode_index",
    "frame_index",
    "index",
    "timestamp",
    "task_index",
    "action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a LeRobot v3 dataset into canonical episode/frame/index order."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument(
        "--include-episodes-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of source episode ids to retain. Selected episodes are written "
            "in the listed order and renumbered contiguously from zero."
        ),
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "none"),
        help="Parquet compression for rewritten episode files.",
    )
    return parser.parse_args()


def resolve_roots(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to replace it: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate sibling trees, not ancestors of each other")
    forbidden = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if output in forbidden or output.name in {"", ".", ".."}:
        raise ValueError(f"Unsafe output path: {output}")
    return source, output


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def verify_disk_space(source: Path, output: Path) -> tuple[int, int]:
    source_bytes = directory_size(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    required = int(source_bytes * 1.20) + 512 * 1024**2
    if free_bytes < required:
        raise OSError(
            f"Insufficient free space: need about {required / 1024**3:.1f} GiB, "
            f"have {free_bytes / 1024**3:.1f} GiB"
        )
    return source_bytes, free_bytes


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(name)
    field = table.schema.field(name)
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def discover_episode_files(root: Path) -> tuple[dict[int, Path], dict[str, Any]]:
    paths = sorted(root.glob("data/chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data files under {root / 'data'}")

    mapping: dict[int, Path] = {}
    physical_position = 0
    mismatch_count = 0
    first_mismatches: list[dict[str, int | str]] = []
    total_rows = 0
    for path in paths:
        table = pq.read_table(path, columns=["episode_index", "index"])
        episodes = np.unique(table["episode_index"].to_numpy(zero_copy_only=False))
        for episode in episodes:
            episode = int(episode)
            if episode in mapping:
                raise ValueError(
                    f"Episode {episode} is split across {mapping[episode]} and {path}; "
                    "canonicalization requires whole-episode source records"
                )
            mapping[episode] = path

        indices = table["index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        expected = np.arange(physical_position, physical_position + len(indices), dtype=np.int64)
        bad = np.flatnonzero(indices != expected)
        mismatch_count += int(len(bad))
        for local_position in bad[: max(0, 8 - len(first_mismatches))]:
            first_mismatches.append(
                {
                    "physical_position": int(physical_position + local_position),
                    "stored_index": int(indices[local_position]),
                    "file": str(path.relative_to(root)),
                }
            )
        physical_position += len(indices)
        total_rows += table.num_rows

    episode_ids = sorted(mapping)
    if episode_ids != list(range(len(episode_ids))):
        raise ValueError(f"Episode indices are not contiguous: {episode_ids[:5]} ... {episode_ids[-5:]}")
    return mapping, {
        "data_files": len(paths),
        "rows": total_rows,
        "physical_index_mismatch_rows": mismatch_count,
        "first_physical_index_mismatches": first_mismatches,
    }


def numeric_array(table: pa.Table, name: str) -> np.ndarray:
    values = np.asarray(table[name].to_pylist())
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected scalar/vector feature {name!r}, got shape {values.shape}")
    return values


def feature_stats(values: np.ndarray) -> dict[str, list[int | float]]:
    is_integer = np.issubdtype(values.dtype, np.integer)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    result: dict[str, list[int | float]] = {
        "min": [int(value) for value in minimum] if is_integer else minimum.astype(float).tolist(),
        "max": [int(value) for value in maximum] if is_integer else maximum.astype(float).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
    }
    for key, quantile in QUANTILES.items():
        result[key] = np.quantile(values, quantile, axis=0).astype(float).tolist()
    return result


def update_episode_stats(record: dict[str, Any], table: pa.Table, features: dict[str, Any]) -> None:
    for name, feature in features.items():
        if feature.get("dtype") in {"image", "video", "string"} or name not in table.column_names:
            continue
        stats = feature_stats(numeric_array(table, name))
        for stat_name, value in stats.items():
            column = f"stats/{name}/{stat_name}"
            if column in record:
                record[column] = value


def update_global_stats(
    original: dict[str, Any], numeric_values: dict[str, list[np.ndarray]]
) -> dict[str, Any]:
    updated = dict(original)
    for name, batches in numeric_values.items():
        values = np.concatenate(batches, axis=0)
        updated[name] = feature_stats(values)
    return updated


def probe_digest(table: pa.Table, ignored_columns: frozenset[str] = frozenset({"index"})) -> str:
    """Hash numeric trajectories plus sampled embedded images for a compact fingerprint."""
    digest = hashlib.sha256()
    for name in table.column_names:
        if name.startswith("observation.") and name.endswith("image"):
            for row in sorted({0, table.num_rows // 2, table.num_rows - 1}):
                value = table[name][row].as_py()
                digest.update(value.get("bytes") or b"")
                digest.update((value.get("path") or "").encode("utf-8"))
        elif name not in ignored_columns:
            column = table.select([name])
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, column.schema) as writer:
                writer.write_table(column)
            digest.update(sink.getvalue().to_pybytes())
    return digest.hexdigest()


def tables_equal_except_reindexed(source: pa.Table, output: pa.Table) -> bool:
    names = [name for name in source.column_names if name not in {"index", "episode_index"}]
    return source.select(names).combine_chunks().equals(output.select(names).combine_chunks())


def select_episodes(path: Path | None, available: list[int]) -> list[int]:
    if path is None:
        return available
    selected = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(selected, list) or not all(isinstance(value, int) for value in selected):
        raise ValueError("--include-episodes-json must contain a JSON list of integer episode ids")
    if not selected:
        raise ValueError("--include-episodes-json selected no episodes")
    if len(selected) != len(set(selected)):
        raise ValueError("--include-episodes-json contains duplicate episode ids")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Selected episodes are missing from the source dataset: {missing}")
    return selected


def load_episode_metadata(root: Path) -> tuple[dict[int, dict[str, Any]], pa.Schema]:
    paths = sorted(root.glob("meta/episodes/chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet under {root}")
    tables = [pq.read_table(path) for path in paths]
    schema = tables[0].schema
    for table in tables[1:]:
        if not table.schema.equals(schema, check_metadata=False):
            raise ValueError("Episode metadata parquet schemas differ")
    records: dict[int, dict[str, Any]] = {}
    for record in pa.concat_tables(tables).to_pylist():
        episode = int(record["episode_index"])
        if episode in records:
            raise ValueError(f"Duplicate metadata for episode {episode}")
        records[episode] = record
    return records, schema


def canonicalize(args: argparse.Namespace) -> dict[str, Any]:
    source, output = resolve_roots(args.source_root, args.output_root)
    source_bytes, free_bytes = verify_disk_space(source, output)
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    if info.get("video_path") is not None:
        raise ValueError("Video-backed datasets are not supported by this canonicalizer")

    episode_files, source_layout = discover_episode_files(source)
    episode_metadata, metadata_schema = load_episode_metadata(source)
    if set(episode_files) != set(episode_metadata):
        raise ValueError("Data episodes and metadata episodes differ")
    if info.get("total_episodes") != len(episode_files):
        raise ValueError("info.json total_episodes does not match parquet data")
    selected_episodes = select_episodes(args.include_episodes_json, sorted(episode_files))
    reindex_episodes = selected_episodes != list(range(len(episode_files)))

    temp = output.parent / f".{output.name}.incomplete-{os.getpid()}"
    if temp.exists():
        raise FileExistsError(temp)

    compression = None if args.compression == "none" else args.compression
    numeric_values: dict[str, list[np.ndarray]] = {}
    output_records: list[dict[str, Any]] = []
    source_digests: dict[int, str] = {}
    global_index = 0
    cached_path: Path | None = None
    cached_table: pa.Table | None = None

    try:
        shutil.copytree(source / "meta", temp / "meta")
        shutil.rmtree(temp / "meta" / "episodes")
        (temp / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (temp / "data" / "chunk-000").mkdir(parents=True)

        for output_episode, source_episode in enumerate(selected_episodes):
            source_path = episode_files[source_episode]
            if source_path != cached_path:
                cached_table = pq.read_table(source_path)
                missing = REQUIRED_COLUMNS - set(cached_table.column_names)
                if missing:
                    raise ValueError(f"Missing required columns in {source_path}: {sorted(missing)}")
                cached_path = source_path
            assert cached_table is not None
            mask = pc.equal(cached_table["episode_index"], pa.scalar(source_episode))
            episode_table = cached_table.filter(mask).sort_by([("frame_index", "ascending")])
            frame_indices = episode_table["frame_index"].to_numpy(zero_copy_only=False)
            expected_frames = np.arange(episode_table.num_rows, dtype=frame_indices.dtype)
            if not np.array_equal(frame_indices, expected_frames):
                raise ValueError(f"Episode {source_episode} frame_index is not contiguous from zero")
            episode_values = episode_table["episode_index"].to_numpy(zero_copy_only=False)
            if not np.all(episode_values == source_episode):
                raise ValueError(f"Episode {source_episode} contains mixed episode_index values")

            ignored = frozenset({"index", "episode_index"}) if reindex_episodes else frozenset({"index"})
            source_digests[output_episode] = probe_digest(episode_table, ignored)
            new_indices = np.arange(
                global_index,
                global_index + episode_table.num_rows,
                dtype=episode_table["index"].to_numpy(zero_copy_only=False).dtype,
            )
            canonical = replace_column(
                episode_table,
                "episode_index",
                np.full(episode_table.num_rows, output_episode, dtype=episode_values.dtype),
            )
            canonical = replace_column(canonical, "index", new_indices)

            output_path = temp / f"data/chunk-000/file-{output_episode:03d}.parquet"
            pq.write_table(canonical, output_path, compression=compression, use_dictionary=True)
            readback = pq.read_table(output_path)
            if not tables_equal_except_reindexed(episode_table, readback):
                raise ValueError(f"Content changed while rewriting source episode {source_episode}")
            if not np.array_equal(
                readback["index"].to_numpy(zero_copy_only=False), new_indices
            ):
                raise ValueError(f"Index rewrite failed for output episode {output_episode}")
            if not np.all(
                readback["episode_index"].to_numpy(zero_copy_only=False) == output_episode
            ):
                raise ValueError(f"Episode rewrite failed for output episode {output_episode}")
            if probe_digest(readback, ignored) != source_digests[output_episode]:
                raise ValueError(f"Probe digest changed for source episode {source_episode}")

            record = dict(episode_metadata[source_episode])
            if int(record["length"]) != episode_table.num_rows:
                raise ValueError(f"Episode {source_episode} length metadata is stale")
            record.update(
                {
                    "episode_index": output_episode,
                    "data/chunk_index": 0,
                    "data/file_index": output_episode,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + episode_table.num_rows,
                    "length": episode_table.num_rows,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            update_episode_stats(record, canonical, info["features"])
            output_records.append(record)

            for name, feature in info["features"].items():
                if feature.get("dtype") in {"image", "video", "string"}:
                    continue
                if name in canonical.column_names:
                    numeric_values.setdefault(name, []).append(numeric_array(canonical, name))
            global_index += episode_table.num_rows

        output_metadata = pa.Table.from_pylist(output_records, schema=metadata_schema)
        pq.write_table(
            output_metadata,
            temp / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            compression=compression,
        )

        info["total_episodes"] = len(output_records)
        info["total_frames"] = global_index
        info["splits"] = {"train": f"0:{len(output_records)}"}
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        (temp / "meta" / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
        )

        stats_path = source / "meta" / "stats.json"
        original_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        global_stats = update_global_stats(original_stats, numeric_values)
        (temp / "meta" / "stats.json").write_text(
            json.dumps(global_stats, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
        )

        output_layout = discover_episode_files(temp)[1]
        if output_layout["physical_index_mismatch_rows"] != 0:
            raise ValueError("Canonical output still has physical/index mismatches")
        if output_layout["rows"] != global_index:
            raise ValueError("Canonical output row count mismatch")

        summary = {
            "source_repo_id": args.source_repo_id,
            "source_root": str(source),
            "output_repo_id": args.output_repo_id,
            "output_root": str(output),
            "episodes": len(output_records),
            "frames": global_index,
            "fps": info["fps"],
            "source_size_gib": round(source_bytes / 1024**3, 3),
            "free_before_gib": round(free_bytes / 1024**3, 3),
            "source_layout": source_layout,
            "output_layout": output_layout,
            "one_episode_per_file": True,
            "content_equality": (
                "all non-index/non-episode-index Arrow columns compared after every parquet write"
                if reindex_episodes
                else "all non-index Arrow columns compared after every parquet write"
            ),
            "probe_digest": (
                "SHA-256 over all non-image/non-index/non-episode-index columns plus three frames "
                "per image stream"
                if reindex_episodes
                else "SHA-256 over all non-image/non-index columns plus three frames per image stream"
            ),
            "episode_probe_digests": {str(key): value for key, value in source_digests.items()},
            "selected_source_episodes": selected_episodes,
            "source_to_output_episode": {
                str(source_episode): output_episode
                for output_episode, source_episode in enumerate(selected_episodes)
            },
        }
        (temp / "canonicalization_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp, output)
        return summary
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def main() -> int:
    args = parse_args()
    summary = canonicalize(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
