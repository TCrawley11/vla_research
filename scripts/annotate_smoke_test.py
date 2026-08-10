"""Smoke test: annotate dataset samples with a VLM via OpenRouter.

Downloads one run .h5 from the team HF dataset repo (or uses a local one),
picks random samples (never the first), sends the key-frame camera images plus
a ground-truth block to the model, and writes JSON annotations following the
team schema: caption_short / caption_detailed / qa_pairs typed
perception | prediction | planning | behaviour, plus a code-generated action
block. Output shape is enforced with response_format plus code-side
validation and retries. Purely a dataset/pipeline robustness test - not the
production annotate stage.

Usage:
  OPENROUTER_API_KEY=... uv run python scripts/annotate_smoke_test.py
  uv run python scripts/annotate_smoke_test.py --h5 data/runs/run43.h5 \
      --model qwen/qwen2.5-vl-72b-instruct --indices 5,8,12,18,19
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

# scripts/ is sys.path[0] when run as a file; the package lives one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_data_pipeline.build_samples import action_label_from_velocity

REPO_ID = "VLA-uwo-2026/six_cam_1600x900"
PATH_PREFIX = "runs"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.5-27b"
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK"]
IMAGE_WIDTH = 800  # downscale before upload to keep vision tokens sane
MAX_ATTEMPTS = 3
PROMPT_VERSION = "team-schema-v1"

QA_TYPES = ["perception", "prediction", "planning", "behaviour"]
QA_PER_TYPE = 3

# Human-readable action_text per pipeline action label (matches the team's
# example annotation phrasing).
ACTION_TEXT = {
    "STOP": "Stop and wait before continuing.",
    "LEFT_TURN": "Turn left while continuing along the route.",
    "RIGHT_TURN": "Turn right while continuing along the route.",
    "SLOW_FORWARD": "Continue forward slowly.",
    "FORWARD": "Continue driving forward.",
    "UNKNOWN": "Continue along the planned route.",
}

SYSTEM_PROMPT = """\
You are annotating samples for an autonomous driving dataset collected in the
CARLA simulator. Each sample has camera images from the ego vehicle and a
ground-truth record from the simulator API. Your job is to write captions and
question-answer pairs used to train a driving vision-language model.

Hard rules:
1. GROUND TRUTH IS AUTHORITATIVE. The ground-truth block in the user message
   is exact. Any answer that involves speed, motion state, the driving action,
   or the future trajectory must agree with it. Copy numeric values verbatim,
   never estimate them from the images.
2. PERCEPTION ANSWERS DESCRIBE ONLY WHAT IS VISIBLE in the images: road
   layout, lane markings, traffic lights and signs, other road users, weather
   and lighting. Do not mention ground-truth facts that cannot be seen. If
   ground truth and your visual reading conflict, describe what is visible
   and do not invent agreement.
3. No speculation about objects, agents, or signals that are neither visible
   nor in the ground truth.
4. Vary question phrasing; do not copy the same questions across samples.

Output a single JSON object, no markdown fences, no commentary, exactly this
shape:
{
  "caption_short": one sentence stating the ego vehicle's current situation,
  "caption_detailed": 2-4 sentences describing the visible scene and anything
                      relevant to driving,
  "qa_pairs": exactly 12 items, exactly 3 of each type, each item
              {"type": ..., "question": ..., "answer": ...} where type is one
              of "perception", "prediction", "planning", "behaviour":
    perception - what is visible in the scene,
    prediction - what the recorded future trajectory shows, phrased as what
                 the ego vehicle is expected to do next,
    planning   - what the ego vehicle should do next and how, consistent with
                 the ground-truth action,
    behaviour  - the ego vehicle's current motion state and maneuver
}
"""

USER_PROMPT = """\
Ground truth for this frame (simulator API, exact):
- map: {map_name}
- current driving action label: {action_label} ({action_text})
- ego forward velocity: {v:.2f} m/s
- ego angular velocity: {w:.3f} rad/s (positive = left turn)
- ego behavior over the past {clip_sec:.0f} s: {his_action}
- recorded future trajectory (next {horizon_sec:.0f} s): {traj_summary}

The images above are the current key frame from the FRONT, FRONT_LEFT,
FRONT_RIGHT and BACK cameras. Write the annotation JSON now.
"""

# response_format json_schema (strict) for providers that support it; the
# code-side validator below stays the source of truth either way.
ANNOTATION_JSON_SCHEMA = {
    "name": "annotation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["caption_short", "caption_detailed", "qa_pairs"],
        "properties": {
            "caption_short": {"type": "string"},
            "caption_detailed": {"type": "string"},
            "qa_pairs": {
                "type": "array",
                "minItems": 12,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "question", "answer"],
                    "properties": {
                        "type": {"type": "string", "enum": QA_TYPES},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                },
            },
        },
    },
}


def pick_run(api: HfApi, rng: random.Random, run_id: str | None) -> str:
    """Return the repo path of the run .h5 to test on."""
    files = [f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
             if f.startswith(f"{PATH_PREFIX}/") and f.endswith(".h5")]
    if not files:
        sys.exit(f"no .h5 runs found in {REPO_ID}/{PATH_PREFIX}")
    if run_id:
        path = f"{PATH_PREFIX}/{run_id}.h5"
        if path not in files:
            sys.exit(f"{path} not in repo; available: {sorted(files)}")
        return path
    return rng.choice(files)


def encode_jpeg(rgb: np.ndarray, width: int) -> bytes:
    img = Image.fromarray(rgb)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def summarize_trajectory(waypoints: np.ndarray, traj_type: str,
                         horizon_sec: float) -> str:
    """Turn ego-frame future waypoints (T, 2) into a short factual phrase."""
    fwd, lat = float(waypoints[-1][0]), float(waypoints[-1][1])
    if traj_type == "STOPPING":
        return (f"the vehicle stays essentially stationary "
                f"({fwd:.1f} m forward displacement over {horizon_sec:.0f} s)")
    side = "left" if lat > 0 else "right"
    curve = {"STRAIGHT": "in a straight line",
             "LEFT_CURVE": "curving left",
             "RIGHT_CURVE": "curving right"}.get(traj_type, "")
    return (f"the vehicle moves {fwd:.1f} m forward {curve}, ending "
            f"{abs(lat):.1f} m to the {side} of its current heading")


def build_sample_payload(f: h5py.File, sample: int, width: int) -> tuple[list, dict, dict]:
    """Extract key-frame images + GT for one sample index; return
    (message content blocks, ground-truth record, camera jpeg bytes)."""
    si = f["sample_index"]
    key = int(si["key_index"][sample])
    clip = si["clip_frame_indices"][sample]
    cols = [c.decode() if isinstance(c, bytes) else c
            for c in f["telemetry/data"].attrs["columns"]]
    tel = {c: f["telemetry/data"][:, i] for i, c in enumerate(cols)}

    def as_str(x):
        return x.decode() if isinstance(x, bytes) else str(x)

    v, w = float(tel["v"][key]), float(tel["w"][key])
    action = as_str(f["action/action_label"][sample])
    his_labels = [action_label_from_velocity(float(tel["v"][i]), float(tel["w"][i]))
                  for i in clip]
    his_action = max(set(his_labels), key=his_labels.count)
    waypoints = f["trajectory/future_waypoints_ego_frame"][sample]
    traj_type = as_str(f["trajectory/trajectory_type"][sample])
    horizon_sec = float(f.attrs.get("horizon_sec", 3.0))
    clip_sec = float(f.attrs.get("clip_sec", 3.0))
    map_name = as_str(f.attrs.get("map", "unknown town"))

    frames = {cam: encode_jpeg(f["images"][cam][key], width) for cam in CAMERAS}
    content = []
    for cam in CAMERAS:
        content.append({"type": "text", "text": f"Image from the {cam} camera:"})
        b64 = base64.b64encode(frames[cam]).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": USER_PROMPT.format(
        map_name=map_name, action_label=action,
        action_text=ACTION_TEXT.get(action, ACTION_TEXT["UNKNOWN"]),
        v=v, w=w, clip_sec=clip_sec, his_action=his_action,
        horizon_sec=horizon_sec,
        traj_summary=summarize_trajectory(waypoints, traj_type, horizon_sec))})

    sample_id = as_str(si["sample_id"][sample])
    gt = {
        "sample_id": sample_id,
        "sample_index": sample,
        "key_frame_id": int(si["key_frame_id"][sample]),
        "map_name": map_name,
        "action_label": action,
        "his_action": his_action,
        "trajectory_type": traj_type,
        "v": v, "w": w,
        "future_waypoints_ego_frame": waypoints.tolist(),
    }
    return content, gt, frames


def validate_annotation(obj) -> list[str]:
    """Return a list of schema violations (empty = valid)."""
    errors = []
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    for key in ("caption_short", "caption_detailed"):
        if not isinstance(obj.get(key), str) or not obj.get(key, "").strip():
            errors.append(f"'{key}' missing or not a non-empty string")
    pairs = obj.get("qa_pairs")
    if not isinstance(pairs, list):
        return errors + ["'qa_pairs' missing or not a list"]
    counts = dict.fromkeys(QA_TYPES, 0)
    for i, p in enumerate(pairs):
        if not isinstance(p, dict):
            errors.append(f"qa_pairs[{i}] is not an object")
            continue
        t = p.get("type")
        if t not in QA_TYPES:
            errors.append(f"qa_pairs[{i}].type '{t}' not in {QA_TYPES}")
        else:
            counts[t] += 1
        for key in ("question", "answer"):
            if not isinstance(p.get(key), str) or not p.get(key, "").strip():
                errors.append(f"qa_pairs[{i}].{key} missing or empty")
    for t, c in counts.items():
        if c != QA_PER_TYPE:
            errors.append(f"{c} '{t}' pairs (need exactly {QA_PER_TYPE})")
    return errors


def parse_json_content(text: str):
    """Parse model output, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def _post(api_key: str, body: dict) -> dict:
    req = urllib.request.Request(
        OPENROUTER_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def annotate(api_key: str, model: str, content: list) -> tuple[dict, dict]:
    """Call the model with schema enforcement; validate; retry with feedback.

    Returns (annotation, meta) where meta records attempts and token usage.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    response_format = {"type": "json_schema",
                       "json_schema": ANNOTATION_JSON_SCHEMA}
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    errors_seen = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        body = {"model": model, "messages": messages, "max_tokens": 2500,
                "temperature": 0.3, "response_format": response_format}
        try:
            resp = _post(api_key, body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            # provider rejected json_schema enforcement -> degrade once
            if exc.code == 400 and response_format.get("type") == "json_schema":
                print(f"  json_schema rejected ({detail[:120]}); "
                      "falling back to json_object")
                response_format = {"type": "json_object"}
                continue
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc

        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k, 0)
        if not resp.get("choices"):
            # OpenRouter reports provider errors as 200s with an error object
            err = str(resp.get("error", resp))[:300]
            errors_seen.append([f"no choices in response: {err}"])
            print(f"  attempt {attempt} provider error: {err}")
            time.sleep(10)
            continue
        raw = resp["choices"][0]["message"]["content"]
        try:
            obj = parse_json_content(raw)
            errors = validate_annotation(obj)
        except (json.JSONDecodeError, IndexError) as exc:
            errors = [f"output is not valid JSON: {exc}"]
        if not errors:
            return obj, {"attempts": attempt, "usage": usage,
                         "errors_seen": errors_seen,
                         "response_format": response_format["type"]}

        errors_seen.append(errors)
        print(f"  attempt {attempt} invalid: {'; '.join(errors[:4])}")
        messages = messages[:2] + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content":
                "Your output violated the required schema: "
                + "; ".join(errors)
                + ". Reply again with ONLY the corrected JSON object."}]

    raise RuntimeError(f"no valid annotation after {MAX_ATTEMPTS} attempts: "
                       f"{errors_seen[-1]}")


def action_block(gt: dict) -> dict:
    """Deterministic action annotation from ground truth (never the model)."""
    stopped = gt["action_label"] == "STOP"
    return {
        "action_text": ACTION_TEXT.get(gt["action_label"], ACTION_TEXT["UNKNOWN"]),
        "action_label": gt["action_label"],
        "linear_velocity_target": 0.0 if stopped else round(gt["v"], 2),
        "angular_velocity_target": 0.0 if stopped else round(gt["w"], 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run", help="run id, e.g. run03 (default: random run)")
    ap.add_argument("--h5", type=Path,
                    help="local run .h5; skips the HF download (runs are ~15 GB)")
    ap.add_argument("--indices", help="comma-separated sample indices to "
                                      "annotate (overrides random choice)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the run/sample choice for reproducibility")
    ap.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    ap.add_argument("--out", type=Path, default=Path("data/annotation_test"))
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("set OPENROUTER_API_KEY")

    rng = random.Random(args.seed)
    if args.h5:
        h5_path = args.h5
        run_path = str(args.h5)
    else:
        run_path = pick_run(HfApi(), rng, args.run)
        print(f"downloading {REPO_ID}/{run_path} ...")
        h5_path = hf_hub_download(REPO_ID, run_path, repo_type="dataset")

    model_slug = args.model.split("/")[-1]
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        n = len(f["sample_index/sample_id"])
        if args.indices:
            picks = [int(i) for i in args.indices.split(",")]
            bad = [i for i in picks if not 0 < i < n]
            if bad:
                sys.exit(f"indices {bad} out of range 1..{n - 1}")
        else:
            if n <= 1:
                sys.exit(f"{run_path} has {n} samples; nothing beyond the first")
            # never the first sample; it is degenerate (spawn/warm-up frame)
            picks = rng.sample(range(1, n), min(args.num_samples, n - 1))
        print(f"{run_path}: {n} samples, model {args.model}, "
              f"indices {sorted(picks)}")

        for sample in sorted(picks):
            content, gt, frames = build_sample_payload(f, sample, args.image_width)
            for cam, data in frames.items():
                frame_path = frames_dir / f"{gt['sample_id']}_{cam}.jpg"
                if not frame_path.exists():
                    frame_path.write_bytes(data)
            print(f"annotating {gt['sample_id']} (gt {gt['action_label']}) ...")
            annotation, meta = annotate(api_key, args.model, content)
            annotation["action"] = action_block(gt)
            out = {
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "ground_truth": gt,
                "annotation": annotation,
                "meta": meta,
            }
            out_path = args.out / f"{gt['sample_id']}__{model_slug}.json"
            out_path.write_text(json.dumps(out, indent=2))
            print(f"  -> {out_path} ({meta['attempts']} attempt(s), "
                  f"{meta['usage']['completion_tokens']} out tokens)")


if __name__ == "__main__":
    main()
