"""Freeze compact six-camera samples across runs, using remote HDF5 range reads.

No full run downloads. A pinned dataset revision and a manifest preserve the
exact input for local baseline and example comparisons. Example and evaluation
runs are disjoint; no image or motion selection depends on the model output.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_data_pipeline.annotation_common import (
    CAMERAS, GroundTruth, SamplePayload, atomic_json, input_id, load_sample,
    read_json, speed_profile,
)

REPO = "VLA-uwo-2026/six_cam_1600x900"
EXAMPLE_RUNS = ["run01", "run02", "run03", "run04", "run05", "run43"]
EVAL_RUNS = ["run06", "run07", "run08", "run09", "run10"]


def cached_run_samples(root: Path, cached: dict | None, repo: str, revision: str,
                       run: str, split: str, count: int, width: int) -> list[dict] | None:
    if not cached or (
        cached.get("revision"), cached.get("width"), cached.get("count")
    ) != (revision, width, count):
        return None
    samples = cached.get("samples")
    if not isinstance(samples, list) or len(samples) != count:
        return None
    try:
        for entry in samples:
            if entry.get("run") != run or entry.get("split") != split:
                return None
            sample_path = root / entry["path"]
            metadata = read_json(sample_path)
            if not metadata:
                return None
            source = metadata.get("source") or {}
            if (source.get("repo_id"), source.get("revision"), source.get("run")) != (
                repo, revision, run
            ):
                return None
            payload = load_payload(sample_path)
            if (payload.gt.sample_id != entry.get("sample_id")
                    or input_id(payload) != entry.get("input_id")):
                return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return samples


def select_diverse(f: h5py.File, count: int) -> list[int]:
    """Greedily cover motion/road contexts, then maximize temporal separation."""
    si = f["sample_index"]
    def text(x): return x.decode() if isinstance(x, bytes) else str(x)
    candidates = []
    for i in range(1, len(si["sample_id"])):
        key = int(si["key_index"][i])
        profile = speed_profile(f["trajectory/future_waypoints_ego_frame"][i],
                                float(f.attrs.get("waypoint_period_sec", .5)))
        bucket = ("stop" if "stationary throughout" in profile else
                  "braking" if "decelerating" in profile or "slowing" in profile else
                  "starting" if "pulling away" in profile or "accelerating" in profile else "steady")
        features = {"motion:" + bucket,
                    "action:" + text(f["action/action_label"][i]),
                    "path:" + text(f["trajectory/trajectory_type"][i]),
                    "junction:" + str(int(f["map_context/is_junction"][key]))}
        candidates.append((i, features))
    selected, covered = [], set()
    while candidates and len(selected) < count:
        def rank(item):
            i, features = item
            distance = min((abs(i - j) for j in selected), default=i)
            return (len(features - covered), distance, -i)
        pick = max(candidates, key=rank)
        selected.append(pick[0]); covered.update(pick[1]); candidates.remove(pick)
        # Avoid near-duplicate frames from the same event.
        candidates = [(i, features) for i, features in candidates if abs(i - pick[0]) >= 4]
    return sorted(selected)


def prepare_run(repo: str, revision: str, run: str, split: str, count: int,
                root: Path, width: int) -> list[dict]:
    cached = read_json(root / f"{run}.json")
    samples = cached_run_samples(root, cached, repo, revision, run, split, count, width)
    if samples is not None:
        return samples
    fs = HfFileSystem()
    path = f"datasets/{repo}/runs/{run}.h5"
    print(f"reading {run} ({split})", flush=True)
    entries = []
    with fs.open(path, revision=revision, block_size=1024*1024, cache_type="blockcache") as remote:
        with h5py.File(remote, "r") as f:
            picks = [4, 10] if run == "run43" else select_diverse(f, count)
            for index in picks:
                payload = load_sample(f, index, width)
                sid = payload.gt.sample_id
                sample_dir = root / split / sid
                sample_dir.mkdir(parents=True, exist_ok=True)
                for camera, frame in payload.frames.items():
                    (sample_dir / f"{camera}.jpg").write_bytes(frame)
                meta = {"source": {"repo_id": repo, "revision": revision, "run": run,
                                    "map": str(f.attrs["map"]), "sample_index": index},
                        "ground_truth": payload.gt.record(), "input_id": input_id(payload),
                        "frames": {cam: f"{cam}.jpg" for cam in CAMERAS}}
                atomic_json(sample_dir / "sample.json", meta)
                entries.append({"sample_id": sid, "run": run, "map": str(f.attrs["map"]),
                                "split": split, "path": str((sample_dir / "sample.json").relative_to(root)),
                                "action": payload.gt.action_label,
                                "future_motion": payload.gt.traj_summary(), "input_id": input_id(payload)})
                print(f"  {sid}: {payload.gt.action_label}; {payload.gt.traj_summary()}", flush=True)
    atomic_json(root / f"{run}.json", {"revision": revision, "width": width, "count": count, "samples": entries})
    return entries


def load_payload(path: Path) -> SamplePayload:
    d = json.loads(path.read_text())
    frames = {cam: (path.parent / d["frames"][cam]).read_bytes() for cam in CAMERAS}
    blocks = []
    for cam, data in frames.items():
        blocks.extend([{"type": "text", "text": f"Image from the {cam} camera:"},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}])
    payload = SamplePayload(GroundTruth(**d["ground_truth"]), blocks, frames)
    if input_id(payload) != d["input_id"]:
        raise ValueError(f"input checksum mismatch: {path}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/annotation_eval/inputs"))
    ap.add_argument("--revision", required=True,
                    help="immutable dataset commit used for every frozen sample")
    ap.add_argument("--image-width", type=int, default=800)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    jobs = [(run, "examples", 2) for run in EXAMPLE_RUNS] + [(run, "eval", 4) for run in EVAL_RUNS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(lambda job: prepare_run(REPO, revision, *job, root, args.image_width), jobs))
    samples = [entry for group in groups for entry in group]
    atomic_json(root / "manifest.json", {"repo_id": REPO, "revision": revision,
                                        "image_width": args.image_width, "samples": samples})
    print(f"{root / 'manifest.json'}: {len(samples)} samples", flush=True)


if __name__ == "__main__":
    main()
