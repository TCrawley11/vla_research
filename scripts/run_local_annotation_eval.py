"""Run the same frozen local evaluation with examples disabled and enabled.

Each variant generates its own questions: this measures the full pipeline.
Results and input identities are separate, resumable, and checked for leakage.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_data_pipeline.annotate import Annotator, VllmClient, load_config
from carla_data_pipeline.annotation_common import atomic_json, read_json, digest
from scripts.prepare_annotation_eval import load_payload
from scripts.build_inspection import main as build_inspection


def run_evaluation(cfg, manifest_path: Path, out_dir: Path, variants: list[str],
                   sample_ids: list[str] | None = None) -> dict:
    manifest = read_json(manifest_path)
    if not manifest or not isinstance(manifest.get("samples"), list):
        raise ValueError("invalid input manifest; run prepare_annotation_eval.py first")
    samples = [s for s in manifest["samples"] if s["split"] == "eval"]
    if sample_ids is not None:
        unknown = set(sample_ids) - {s["sample_id"] for s in samples}
        if unknown or not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"sample ids must be unique evaluation samples; unknown: {sorted(unknown)}")
        samples = [s for s in samples if s["sample_id"] in sample_ids]
    if not samples:
        raise ValueError("manifest has no evaluation samples")
    example_runs = {s["run"] for s in manifest["samples"] if s["split"] == "examples"}
    if example_runs & {s["run"] for s in samples}:
        raise ValueError("example and evaluation runs overlap")
    cfg = cfg.model_copy(deep=True)
    cfg.revision = manifest["revision"]
    cfg.repo_id = manifest["repo_id"]
    cfg.generation.image_width = manifest["image_width"]
    client = VllmClient(cfg.inference)
    client.ping(cfg.inference.model)
    cfg.inference.model = client.served_model
    report = {"model": client.served_model, "manifest_id": digest(manifest),
              "config": cfg.model_dump(mode="json"), "server": client.model_metadata,
              "sample_ids": [s["sample_id"] for s in samples],
              "revision": manifest["revision"], "samples_per_variant": len(samples),
              "towns": dict(Counter(s["map"] for s in samples)), "variants": {}}
    report["evaluation_id"] = digest({k: v for k, v in report.items() if k != "variants"})
    previous = read_json(out_dir / "evaluation.json")
    if previous and previous.get("evaluation_id") == report["evaluation_id"]:
        report["variants"] = previous.get("variants", {})
    for variant in variants:
        variant_cfg = cfg.model_copy(deep=True)
        variant_cfg.out_dir = out_dir / variant
        if variant == "baseline":
            variant_cfg.examples = None
        elif variant_cfg.examples is None:
            raise ValueError("examples variant requires an examples configuration")
        worker = Annotator(variant_cfg, client)
        if {ex.source_run for ex in worker.examples} & {s["run"] for s in samples}:
            raise ValueError("example pool leaks an evaluation run")
        frames_dir = variant_cfg.out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        records = []
        question_records = []
        started = time.monotonic()

        def save_progress(processed):
            issues = Counter(issue for d in records for issue in d["meta"].get("quality_issues", []))
            pairs = [p for d in records for p in d["annotation"]["qa_pairs"]]
            report["variants"][variant] = {
                "completed": len(records), "expected": len(samples), "processed": processed,
                "status": ("complete" if len(records) == len(samples) else
                           "incomplete" if processed == len(samples) else "running"),
                "config": variant_cfg.model_dump(mode="json"),
                "elapsed_sec": round(time.monotonic() - started, 1),
                "failures": worker.failures,
                "question_retries": sum(q["meta"]["attempts"] - 1 for q in question_records),
                "answer_retries": sum(d["meta"]["attempts"] - 1 for d in records),
                "review_flags": dict(issues),
                "mean_answer_words": (sum(len(p["answer"].split()) for p in pairs) / len(pairs)
                                      if pairs else None),
            }
            atomic_json(out_dir / "evaluation.json", report)

        save_progress(0)
        for processed, entry in enumerate(samples, 1):
            logging.info("%s %d/%d: %s (%s)", variant, processed, len(samples),
                         entry["sample_id"], entry["map"])
            payload = load_payload(manifest_path.parent / entry["path"])
            if payload.gt.sample_id != entry["sample_id"] or entry["input_id"] != digest_input(payload):
                raise ValueError("manifest/sample identity mismatch")
            worker.process_payload(payload, frames_dir)
            qs = worker.load_question_set(worker.question_path(payload.gt.sample_id), payload)
            if qs:
                question_records.append(qs)
            path = worker.result_path(payload.gt.sample_id)
            if qs and worker.result_is_current(path, qs, payload):
                d = read_json(path)
                d["source"] = {k: entry[k] for k in ("run", "map", "sample_id")}
                atomic_json(path, d)
                records.append(d)
            save_progress(processed)
        if records:
            build_inspection(["--dir", str(variant_cfg.out_dir)])
    return report


def digest_input(payload):
    from carla_data_pipeline.annotation_common import input_id
    return input_id(payload)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/annotation/local.yaml"))
    ap.add_argument("--manifest", type=Path, default=Path("data/annotation_eval/inputs/manifest.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/annotation_eval/results"))
    ap.add_argument("--variant", choices=["baseline", "examples", "both"], default="both")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--model-revision", help="weight revision or checksum for reproducible cache identity")
    ap.add_argument("--sample-ids", help="comma-separated evaluation sample ids; default: all 20")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.base_url: cfg.inference.base_url = args.base_url.rstrip("/")
    if args.model: cfg.inference.model = args.model
    if args.model_revision: cfg.inference.model_revision = args.model_revision
    report = run_evaluation(cfg, args.manifest, args.out_dir,
                            ["baseline", "examples"] if args.variant == "both" else [args.variant],
                            args.sample_ids.split(",") if args.sample_ids else None)
    print(args.out_dir / "evaluation.json")
    return int(any(d["status"] != "complete" for d in report["variants"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
