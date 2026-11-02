"""Generate resumable MERT embeddings for the cleaned music library.

This runner deliberately delegates the actual retrieval, preview decoding, and
embedding work to ``song_embedding.py``. It stores each completed vector
individually, which makes a long-running batch safe to resume after a network
failure or an interrupted session. Once all desired records have been
attempted, it writes a compact NumPy archive plus a row-level manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from song_embedding import embed_song


ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "combined_music_library.csv"
DEFAULT_OUTPUT_DIR = ROOT / "embeddings"


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"isrc", "track_name", "artist_name", "library_label"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Catalog is missing required columns: {', '.join(sorted(missing))}")
    return rows


def load_latest_attempts(path: Path) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as checkpoint:
        for line in checkpoint:
            if line.strip():
                record = json.loads(line)
                latest[str(record["isrc"])] = record
    return latest


def append_checkpoint(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as checkpoint:
        checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint.flush()
        os.fsync(checkpoint.fileno())


def save_vector(path: Path, vector: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.save(destination, vector)
    temporary.replace(path)


def iter_pending(
    rows: list[dict[str, str]],
    vectors_dir: Path,
    attempts: dict[str, dict[str, object]],
    retry_failed: bool,
    retry_errors: bool,
) -> Iterator[dict[str, str]]:
    for row in rows:
        vector_path = vectors_dir / f"{row['isrc']}.npy"
        previous_status = str(attempts.get(row["isrc"], {}).get("status", ""))
        if vector_path.exists():
            continue
        if previous_status == "not_found" and not retry_failed:
            continue
        if previous_status == "error" and not (retry_failed or retry_errors):
            continue
        yield row


def write_outputs(
    rows: list[dict[str, str]],
    vectors_dir: Path,
    attempts: dict[str, dict[str, object]],
    output_dir: Path,
) -> tuple[int, int]:
    manifest_path = output_dir / "song_embedding_manifest.csv"
    archive_path = output_dir / "song_embeddings.npz"
    manifest_fields = [
        "isrc",
        "track_name",
        "artist_name",
        "library_label",
        "embedding_status",
        "vector_file",
        "itunes_track_name",
        "itunes_artist_name",
        "itunes_track_id",
        "error",
    ]
    manifest_rows: list[dict[str, str]] = []
    completed_isrcs: list[str] = []
    vectors: list[np.ndarray] = []

    for row in rows:
        isrc = row["isrc"]
        vector_path = vectors_dir / f"{isrc}.npy"
        attempt = attempts.get(isrc, {})
        if vector_path.exists():
            vector = np.load(vector_path)
            completed_isrcs.append(isrc)
            vectors.append(vector)
            status = "embedded"
        else:
            status = str(attempt.get("status", "pending"))

        manifest_rows.append(
            {
                "isrc": isrc,
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "library_label": row["library_label"],
                "embedding_status": status,
                "vector_file": str(vector_path.relative_to(output_dir)) if vector_path.exists() else "",
                "itunes_track_name": str(attempt.get("itunes_track_name", "")),
                "itunes_artist_name": str(attempt.get("itunes_artist_name", "")),
                "itunes_track_id": str(attempt.get("itunes_track_id", "")),
                "error": str(attempt.get("error", "")),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if vectors:
        np.savez_compressed(
            archive_path,
            isrc=np.array(completed_isrcs),
            embeddings=np.stack(vectors).astype(np.float32),
        )
    elif archive_path.exists():
        archive_path.unlink()

    return len(completed_isrcs), len(rows) - len(completed_isrcs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Only attempt this many pending songs.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previous errors and unmatched songs.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry only previous processing errors.")
    args = parser.parse_args()

    rows = read_catalog(args.input)
    output_dir = args.output_dir
    vectors_dir = output_dir / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "embedding_attempts.jsonl"
    attempts = load_latest_attempts(checkpoint_path)

    pending = list(iter_pending(rows, vectors_dir, attempts, args.retry_failed, args.retry_errors))
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Attempting {len(pending)} songs. Catalog total: {len(rows)}.")

    for index, row in enumerate(pending, start=1):
        isrc = row["isrc"]
        print(f"[{index}/{len(pending)}] {row['track_name']} — {row['artist_name']}", flush=True)
        try:
            vector, match = embed_song(row["track_name"], row["artist_name"])
            vector_path = vectors_dir / f"{isrc}.npy"
            save_vector(vector_path, vector)
            record: dict[str, object] = {
                "isrc": isrc,
                "status": "embedded",
                "itunes_track_name": match.get("trackName", ""),
                "itunes_artist_name": match.get("artistName", ""),
                "itunes_track_id": match.get("trackId", ""),
            }
        except LookupError as error:
            record = {"isrc": isrc, "status": "not_found", "error": str(error)}
        except Exception as error:  # Preserve individual failures without stopping the batch.
            record = {"isrc": isrc, "status": "error", "error": f"{type(error).__name__}: {error}"}
        append_checkpoint(checkpoint_path, record)
        attempts[isrc] = record

    embedded, remaining = write_outputs(rows, vectors_dir, attempts, output_dir)
    print(f"Embedded: {embedded}/{len(rows)}. Not embedded: {remaining}.")


if __name__ == "__main__":
    main()
