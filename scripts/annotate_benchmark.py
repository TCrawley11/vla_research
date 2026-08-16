"""Annotation benchmark: compare VLM annotators on dataset samples via OpenRouter.

Reads a benchmark config (configs/annotation/*.yaml), takes samples from one
run .h5 (downloaded from the team HF dataset repo, or a local file), and for
every sample sends the key-frame camera images plus a ground-truth block to
each candidate model. Every (sample, model) result follows the team schema:
caption_short / caption_detailed / qa_pairs typed
perception | prediction | planning | behaviour, plus a code-generated action
block.

Question modes (questions.mode in the config):
  shared - one question set per sample is written once by questions.model and
           cached under <out_dir>/questions/; every candidate answers exactly
           those questions, so answers compare one-to-one across models.
  own    - every candidate writes its own questions and answers (the shape the
           production annotate stage will produce).

Output shape is enforced with response_format json_schema plus code-side
validation and retries. Results whose model / prompt_version / question set
already match on disk are skipped, so adding a model to the config and
re-running only annotates the missing pairs. Not the production annotate stage.

Usage:
  OPENROUTER_API_KEY=... uv run python scripts/annotate_benchmark.py
  uv run python scripts/annotate_benchmark.py --config configs/annotation/benchmark.yaml \
      --h5 data/runs/run43.h5 --models qwen/qwen3.8-27b
  uv run python scripts/build_inspection.py      # side-by-side HTML of the results
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import yaml
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

# scripts/ is sys.path[0] when run as a file; the package lives one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_data_pipeline.build_samples import action_label_from_velocity

DEFAULT_CONFIG = Path("configs/annotation/benchmark.yaml")
DEFAULT_REPO_ID = "VLA-uwo-2026/six_cam_1600x900"
PATH_PREFIX = "runs"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK"]
QA_TYPES = ["perception", "prediction", "planning", "behaviour"]
HTTP_RETRIES = 5                     # transport / provider-side retries per request
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

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


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplesConfig(StrictModel):
    run: Optional[str] = Field(
        None, description="run id in repo_id, e.g. run43; null picks a random run")
    indices: Optional[list[int]] = Field(
        None, description="explicit sample indices = a fixed eval set; wins over "
                          "num_samples/seed when given")
    num_samples: int = Field(
        10, ge=1, description="random picks per run when indices is null")
    seed: Optional[int] = Field(
        None, description="seed for the random run/sample choice")

    @field_validator("indices")
    @classmethod
    def _indices_ok(cls, v):
        if v is not None:
            if not v:
                raise ValueError("indices must not be empty; use null for random picks")
            if any(i < 1 for i in v):
                raise ValueError("indices must be >= 1 (index 0 is the spawn frame)")
            if len(set(v)) != len(v):
                raise ValueError("indices contains duplicates")
        return v


class QuestionsConfig(StrictModel):
    mode: Literal["shared", "own"] = Field(
        "shared", description="shared: one question set per sample answered by "
                              "every model; own: each model writes its own QA")
    model: Optional[str] = Field(
        None, description="question author in shared mode (OpenRouter model id)")
    per_type: int = Field(3, ge=1, description="questions per QA type")

    @field_validator("model")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


class GenerationConfig(StrictModel):
    """Sampling settings; identical for every candidate so the comparison is fair."""
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(4000, ge=256, description="thinking tokens count "
                            "against this on models that reason by default")
    max_attempts: int = Field(3, ge=1, description="schema-violation retries "
                              "with feedback, per request")
    reasoning: Optional[dict] = Field(
        None, description="OpenRouter `reasoning` object passed through as-is, "
                          "e.g. {enabled: false} or {effort: low}; null = provider default")
    image_width: int = Field(800, ge=64, description="cameras are downscaled "
                             "to this width before upload")


class BenchmarkConfig(StrictModel):
    models: list[str] = Field(..., min_length=1,
                              description="candidate annotators (OpenRouter model ids)")
    samples: SamplesConfig = SamplesConfig()
    questions: QuestionsConfig = QuestionsConfig()
    generation: GenerationConfig = GenerationConfig()
    prompt_version: str = Field("team-schema-v1",
                                description="key into PROMPTS in the script")
    repo_id: str = DEFAULT_REPO_ID
    out_dir: Path = Path("data/annotation_test")

    @field_validator("models")
    @classmethod
    def _models_unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("models contains duplicates")
        return v

    @field_validator("prompt_version")
    @classmethod
    def _known_prompt(cls, v):
        if v not in PROMPTS:
            raise ValueError(f"unknown prompt_version {v!r}; known: {sorted(PROMPTS)}")
        return v

    def check(self) -> "BenchmarkConfig":
        if self.questions.mode == "shared" and not self.questions.model:
            raise ValueError("questions.model is required when questions.mode is shared")
        return self


def load_config(path: Path) -> BenchmarkConfig:
    if not path.is_file():
        sys.exit(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        sys.exit(f"config must be a YAML mapping: {path}")
    return BenchmarkConfig.model_validate(raw).check()


# --------------------------------------------------------------------------
# prompts (versioned; the config selects one by name)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSet:
    system_own: str        # model writes captions + its own QA pairs
    system_questions: str  # question author writes the shared question set
    system_shared: str     # model writes captions + answers to the shared set
    user_gt: str           # ground-truth block, shared by all three


_INTRO = """\
You are annotating samples for an autonomous driving dataset collected in the
CARLA simulator. Each sample has camera images from the ego vehicle and a
ground-truth record from the simulator API. Your job is to write captions and
{task} used to train a driving vision-language model.

"""

_RULES = """\
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
"""

_QA_TYPES_GUIDE = """\
    perception - what is visible in the scene,
    prediction - what the recorded future trajectory shows, phrased as what
                 the ego vehicle is expected to do next,
    planning   - what the ego vehicle should do next and how, consistent with
                 the ground-truth action,
    behaviour  - the ego vehicle's current motion state and maneuver
"""

_SYSTEM_OWN = _INTRO.format(task="question-answer pairs") + _RULES + """\
4. Vary question phrasing; do not copy the same questions across samples.

Output a single JSON object, no markdown fences, no commentary, exactly this
shape:
{{
  "caption_short": one sentence stating the ego vehicle's current situation,
  "caption_detailed": 2-4 sentences describing the visible scene and anything
                      relevant to driving,
  "qa_pairs": exactly {n_total} items, exactly {per_type} of each type, each item
              {{"type": ..., "question": ..., "answer": ...}} where type is one
              of "perception", "prediction", "planning", "behaviour":
""" + _QA_TYPES_GUIDE + "}}\n"

_SYSTEM_QUESTIONS = """\
You are writing the question set for one sample of an autonomous driving
dataset collected in the CARLA simulator. Several vision-language models will
later answer exactly these questions for this sample, using the same camera
images and ground-truth record you see now, so every question must be
answerable from those inputs alone. Write questions only, never answers.

Write exactly {per_type} questions of each type:
    perception - about what is visible in the images (road layout, lane
                 markings, traffic lights and signs, other road users, weather
                 and lighting); ask only about things that are visible, or
                 whose absence is worth stating; never presuppose objects
                 that are not there,
    prediction - about what the recorded future trajectory shows, phrased as
                 what the ego vehicle is expected to do next,
    planning   - about what the ego vehicle should do next and how, given the
                 ground-truth action,
    behaviour  - about the ego vehicle's current motion state and maneuver

Rules:
1. Be specific to this scene; a question that fits any frame is a weak one.
2. Ask for a value instead of confirming it: "What is the ego vehicle's
   current speed?", not "Is the ego vehicle driving at 8.06 m/s?".
3. Each question stands alone (no "the object above"), and no two questions
   ask the same thing.
4. Vary phrasing.

Output a single JSON object, no markdown fences, no commentary:
{{"questions": [{{"type": ..., "question": ...}}, ...]}} with exactly {n_total}
items, exactly {per_type} of each type, type one of "perception",
"prediction", "planning", "behaviour".
"""

_SYSTEM_SHARED = _INTRO.format(task="answers to a fixed list of questions") + _RULES + """\

The user message lists the questions to answer, each with an id and its type:
""" + _QA_TYPES_GUIDE + """\
Answer every question in one to three sentences. If a question presupposes
something that is neither visible nor in the ground truth, say so instead of
inventing it.

Output a single JSON object, no markdown fences, no commentary, exactly this
shape:
{{
  "caption_short": one sentence stating the ego vehicle's current situation,
  "caption_detailed": 2-4 sentences describing the visible scene and anything
                      relevant to driving,
  "answers": one item per listed question, in the listed order ({n_total} items),
             each {{"id": the question id, "answer": ...}}
}}
"""

_USER_GT = """\
Ground truth for this frame (simulator API, exact):
- map: {map_name}
- current driving action label: {action_label} ({action_text})
- ego forward velocity: {v:.2f} m/s
- ego angular velocity: {w:.3f} rad/s (positive = left turn)
- ego behavior over the past {clip_sec:.0f} s: {his_action}
- recorded future trajectory (next {horizon_sec:.0f} s): {traj_summary}

The images above are the current key frame from the FRONT, FRONT_LEFT,
FRONT_RIGHT and BACK cameras.
"""

PROMPTS = {
    "team-schema-v1": PromptSet(system_own=_SYSTEM_OWN,
                                system_questions=_SYSTEM_QUESTIONS,
                                system_shared=_SYSTEM_SHARED,
                                user_gt=_USER_GT),
}


# --------------------------------------------------------------------------
# JSON schemas for response_format (strict); the code-side validators below
# stay the source of truth either way
# --------------------------------------------------------------------------

def _strict(name: str, properties: dict) -> dict:
    return {"name": name, "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "required": list(properties), "properties": properties}}


def _array(n: int, item_props: dict) -> dict:
    return {"type": "array", "minItems": n, "maxItems": n,
            "items": {"type": "object", "additionalProperties": False,
                      "required": list(item_props), "properties": item_props}}


def schema_own(n_total: int) -> dict:
    return _strict("annotation", {
        "caption_short": {"type": "string"},
        "caption_detailed": {"type": "string"},
        "qa_pairs": _array(n_total, {"type": {"type": "string", "enum": QA_TYPES},
                                     "question": {"type": "string"},
                                     "answer": {"type": "string"}})})


def schema_questions(n_total: int) -> dict:
    return _strict("question_set", {
        "questions": _array(n_total, {"type": {"type": "string", "enum": QA_TYPES},
                                      "question": {"type": "string"}})})


def schema_shared(ids: list[str]) -> dict:
    return _strict("annotation", {
        "caption_short": {"type": "string"},
        "caption_detailed": {"type": "string"},
        "answers": _array(len(ids), {"id": {"type": "string", "enum": ids},
                                     "answer": {"type": "string"}})})


# --------------------------------------------------------------------------
# validators: return a list of violations (empty = valid)
# --------------------------------------------------------------------------

def _nonempty_str(obj: dict, key: str, where: str, errors: list[str]) -> None:
    if not isinstance(obj.get(key), str) or not obj[key].strip():
        errors.append(f"{where}'{key}' missing or not a non-empty string")


def _check_typed_items(items, per_type: int, fields: tuple[str, ...],
                       name: str) -> list[str]:
    errors = []
    counts = dict.fromkeys(QA_TYPES, 0)
    for i, p in enumerate(items):
        if not isinstance(p, dict):
            errors.append(f"{name}[{i}] is not an object")
            continue
        t = p.get("type")
        if t not in QA_TYPES:
            errors.append(f"{name}[{i}].type '{t}' not in {QA_TYPES}")
        else:
            counts[t] += 1
        for key in fields:
            _nonempty_str(p, key, f"{name}[{i}].", errors)
    for t, c in counts.items():
        if c != per_type:
            errors.append(f"{c} '{t}' items (need exactly {per_type})")
    return errors


def validate_own(obj, per_type: int) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    errors = []
    for key in ("caption_short", "caption_detailed"):
        _nonempty_str(obj, key, "", errors)
    pairs = obj.get("qa_pairs")
    if not isinstance(pairs, list):
        return errors + ["'qa_pairs' missing or not a list"]
    return errors + _check_typed_items(pairs, per_type, ("question", "answer"), "qa_pairs")


def validate_questions(obj, per_type: int) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    qs = obj.get("questions")
    if not isinstance(qs, list):
        return ["'questions' missing or not a list"]
    errors = _check_typed_items(qs, per_type, ("question",), "questions")
    texts = [q.get("question", "").strip().lower() for q in qs if isinstance(q, dict)]
    if len(set(texts)) != len(texts):
        errors.append("duplicate questions")
    return errors


def validate_answers(obj, ids: list[str]) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    errors = []
    for key in ("caption_short", "caption_detailed"):
        _nonempty_str(obj, key, "", errors)
    answers = obj.get("answers")
    if not isinstance(answers, list):
        return errors + ["'answers' missing or not a list"]
    seen = []
    for i, a in enumerate(answers):
        if not isinstance(a, dict):
            errors.append(f"answers[{i}] is not an object")
            continue
        seen.append(a.get("id"))
        _nonempty_str(a, "answer", f"answers[{i}].", errors)
    missing = [i for i in ids if i not in seen]
    extra = [i for i in seen if i not in ids]
    dupes = sorted({i for i in seen if seen.count(i) > 1})
    if missing:
        errors.append(f"unanswered question ids: {missing}")
    if extra:
        errors.append(f"unknown question ids: {extra}")
    if dupes:
        errors.append(f"question ids answered more than once: {dupes}")
    return errors


# --------------------------------------------------------------------------
# sample payload
# --------------------------------------------------------------------------

def pick_run(api: HfApi, repo_id: str, rng: random.Random, run_id: str | None) -> str:
    """Return the repo path of the run .h5 to benchmark on."""
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith(f"{PATH_PREFIX}/") and f.endswith(".h5")]
    if not files:
        sys.exit(f"no .h5 runs found in {repo_id}/{PATH_PREFIX}")
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


@dataclass
class SamplePayload:
    gt: dict                 # ground-truth record (goes into the result file)
    gt_text: str             # formatted ground-truth block for the user message
    image_blocks: list       # OpenAI-style content blocks with the four cameras
    frames: dict             # camera -> jpeg bytes (saved for inspection)


def build_sample_payload(f: h5py.File, sample: int, width: int,
                         prompts: PromptSet) -> SamplePayload:
    """Extract key-frame images + GT for one sample index."""
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
    image_blocks = []
    for cam in CAMERAS:
        image_blocks.append({"type": "text", "text": f"Image from the {cam} camera:"})
        b64 = base64.b64encode(frames[cam]).decode()
        image_blocks.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    gt_text = prompts.user_gt.format(
        map_name=map_name, action_label=action,
        action_text=ACTION_TEXT.get(action, ACTION_TEXT["UNKNOWN"]),
        v=v, w=w, clip_sec=clip_sec, his_action=his_action,
        horizon_sec=horizon_sec,
        traj_summary=summarize_trajectory(waypoints, traj_type, horizon_sec))

    gt = {
        "sample_id": as_str(si["sample_id"][sample]),
        "sample_index": sample,
        "key_frame_id": int(si["key_frame_id"][sample]),
        "map_name": map_name,
        "action_label": action,
        "his_action": his_action,
        "trajectory_type": traj_type,
        "v": v, "w": w,
        "future_waypoints_ego_frame": waypoints.tolist(),
    }
    return SamplePayload(gt=gt, gt_text=gt_text, image_blocks=image_blocks, frames=frames)


def user_content(payload: SamplePayload, tail: str) -> list:
    return payload.image_blocks + [{"type": "text", "text": payload.gt_text + "\n" + tail}]


# --------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------

def parse_json_content(text: str):
    """Parse model output, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def _post(api_key: str, body: dict) -> dict:
    """POST to OpenRouter, retrying transport errors and 429/5xx with backoff.
    Non-retryable HTTP errors propagate as urllib.error.HTTPError."""
    req_body = json.dumps(body).encode()
    for i in range(1, HTTP_RETRIES + 1):
        req = urllib.request.Request(
            OPENROUTER_URL, data=req_body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or i == HTTP_RETRIES:
                raise
            print(f"  HTTP {exc.code}; retry {i}/{HTTP_RETRIES - 1} in {15 * i}s")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            if i == HTTP_RETRIES:
                raise RuntimeError(f"OpenRouter unreachable: {exc}") from exc
            print(f"  {type(exc).__name__}: {exc}; retry {i}/{HTTP_RETRIES - 1} in {15 * i}s")
        time.sleep(15 * i)


def call_model(api_key: str, model: str, system: str, content: list,
               json_schema: dict, validate, gen: GenerationConfig) -> tuple[dict, dict]:
    """Call the model with schema enforcement; validate; retry with feedback.

    Returns (object, meta) where meta records attempts, token usage, cost and
    the serving provider.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content}]
    response_format = {"type": "json_schema", "json_schema": json_schema}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
             "cost_usd": 0.0}
    errors_seen = []
    provider = None

    for attempt in range(1, gen.max_attempts + 1):
        body = {"model": model, "messages": messages, "max_tokens": gen.max_tokens,
                "temperature": gen.temperature, "response_format": response_format}
        if gen.reasoning is not None:
            body["reasoning"] = gen.reasoning
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

        u = resp.get("usage") or {}
        usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
        usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
        usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0) or 0
        usage["cost_usd"] += float(u.get("cost", 0) or 0)
        provider = resp.get("provider", provider)
        if not resp.get("choices"):
            # OpenRouter reports provider errors as 200s with an error object
            err = str(resp.get("error", resp))[:300]
            errors_seen.append([f"no choices in response: {err}"])
            print(f"  attempt {attempt} provider error: {err}")
            time.sleep(10)
            continue
        choice = resp["choices"][0]
        raw = choice["message"].get("content") or ""
        try:
            obj = parse_json_content(raw)
            errors = validate(obj)
        except (json.JSONDecodeError, IndexError) as exc:
            errors = [f"output is not valid JSON: {exc}"]
            if choice.get("finish_reason") == "length":
                errors.append("output truncated at max_tokens (finish_reason=length); "
                              "raise generation.max_tokens or limit reasoning")
        if not errors:
            usage["cost_usd"] = round(usage["cost_usd"], 6)
            return obj, {"attempts": attempt, "usage": usage, "provider": provider,
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

    raise RuntimeError(f"no valid output after {gen.max_attempts} attempts: "
                       f"{errors_seen[-1] if errors_seen else 'no response'}")


# --------------------------------------------------------------------------
# question sets (shared mode)
# --------------------------------------------------------------------------

def canonical_order(items: list[dict]) -> list[dict]:
    """Stable sort by QA type in the team order."""
    return sorted(items, key=lambda p: QA_TYPES.index(p["type"]))


def question_set_id(questions: list[dict]) -> str:
    blob = json.dumps([[q["type"], q["question"]] for q in questions])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def question_ids(n_total: int) -> list[str]:
    return [f"q{i:02d}" for i in range(1, n_total + 1)]


def load_question_set(path: Path, cfg: BenchmarkConfig) -> dict | None:
    """Return the cached question set if it matches the config, else None."""
    if not path.exists():
        return None
    qs = json.loads(path.read_text())
    n_total = cfg.questions.per_type * len(QA_TYPES)
    if (qs.get("model") == cfg.questions.model
            and qs.get("prompt_version") == cfg.prompt_version
            and len(qs.get("questions", [])) == n_total):
        return qs
    return None


def write_question_set(api_key: str, cfg: BenchmarkConfig, prompts: PromptSet,
                       payload: SamplePayload, path: Path) -> dict:
    per_type = cfg.questions.per_type
    n_total = per_type * len(QA_TYPES)
    system = prompts.system_questions.format(per_type=per_type, n_total=n_total)
    content = user_content(payload, "Write the question set JSON now.")
    obj, meta = call_model(api_key, cfg.questions.model, system, content,
                           schema_questions(n_total),
                           lambda o: validate_questions(o, per_type), cfg.generation)
    questions = [{"id": qid, "type": q["type"], "question": q["question"].strip()}
                 for qid, q in zip(question_ids(n_total), canonical_order(obj["questions"]))]
    qs = {"sample_id": payload.gt["sample_id"], "model": cfg.questions.model,
          "prompt_version": cfg.prompt_version, "id": question_set_id(questions),
          "questions": questions, "meta": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qs, indent=2))
    return qs


# --------------------------------------------------------------------------
# annotation
# --------------------------------------------------------------------------

def action_block(gt: dict) -> dict:
    """Deterministic action annotation from ground truth (never the model)."""
    stopped = gt["action_label"] == "STOP"
    return {
        "action_text": ACTION_TEXT.get(gt["action_label"], ACTION_TEXT["UNKNOWN"]),
        "action_label": gt["action_label"],
        "linear_velocity_target": 0.0 if stopped else round(gt["v"], 2),
        "angular_velocity_target": 0.0 if stopped else round(gt["w"], 3),
    }


def annotate_own(api_key: str, model: str, cfg: BenchmarkConfig, prompts: PromptSet,
                 payload: SamplePayload) -> tuple[dict, dict]:
    per_type = cfg.questions.per_type
    n_total = per_type * len(QA_TYPES)
    system = prompts.system_own.format(per_type=per_type, n_total=n_total)
    content = user_content(payload, "Write the annotation JSON now.")
    obj, meta = call_model(api_key, model, system, content, schema_own(n_total),
                           lambda o: validate_own(o, per_type), cfg.generation)
    annotation = {"caption_short": obj["caption_short"],
                  "caption_detailed": obj["caption_detailed"],
                  "qa_pairs": canonical_order(obj["qa_pairs"])}
    return annotation, meta


def annotate_shared(api_key: str, model: str, cfg: BenchmarkConfig, prompts: PromptSet,
                    payload: SamplePayload, qs: dict) -> tuple[dict, dict]:
    questions = qs["questions"]
    ids = [q["id"] for q in questions]
    n_total = len(ids)
    system = prompts.system_shared.format(n_total=n_total)
    listing = "\n".join(f"{q['id']} [{q['type']}] {q['question']}" for q in questions)
    content = user_content(payload, "Questions to answer:\n" + listing
                           + "\n\nWrite the annotation JSON now.")
    obj, meta = call_model(api_key, model, system, content, schema_shared(ids),
                           lambda o: validate_answers(o, ids), cfg.generation)
    by_id = {a["id"]: a["answer"].strip() for a in obj["answers"]}
    annotation = {"caption_short": obj["caption_short"],
                  "caption_detailed": obj["caption_detailed"],
                  "qa_pairs": [{**q, "answer": by_id[q["id"]]} for q in questions]}
    return annotation, meta


def result_is_current(path: Path, model: str, cfg: BenchmarkConfig,
                      qs: dict | None) -> bool:
    """True when the result on disk was produced with the same model, prompt
    version and question set, so it need not be recomputed."""
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if d.get("model") != model or d.get("prompt_version") != cfg.prompt_version:
        return False
    # smoke-test era files predate question_mode and were own-mode
    if d.get("question_mode", "own") != cfg.questions.mode:
        return False
    if cfg.questions.mode == "shared":
        return (d.get("question_set") or {}).get("id") == qs["id"]
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help=f"benchmark config (default {DEFAULT_CONFIG})")
    ap.add_argument("--h5", type=Path,
                    help="local run .h5; skips the HF download (runs are ~15 GB)")
    ap.add_argument("--models", help="comma-separated subset of the config's "
                                     "models to run (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="recompute results that are already current on disk")
    ap.add_argument("--regenerate-questions", action="store_true",
                    help="rewrite cached shared question sets (invalidates results)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = PROMPTS[cfg.prompt_version]
    models = cfg.models
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models if m not in cfg.models]
        if unknown:
            sys.exit(f"--models {unknown} not in config models {cfg.models}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("set OPENROUTER_API_KEY")

    rng = random.Random(cfg.samples.seed)
    if args.h5:
        h5_path, run_path = args.h5, str(args.h5)
    else:
        run_path = pick_run(HfApi(), cfg.repo_id, rng, cfg.samples.run)
        print(f"downloading {cfg.repo_id}/{run_path} ...")
        h5_path = hf_hub_download(cfg.repo_id, run_path, repo_type="dataset")

    out_dir = cfg.out_dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    total_cost = 0.0
    with h5py.File(h5_path, "r") as f:
        n = len(f["sample_index/sample_id"])
        if cfg.samples.indices:
            picks = sorted(cfg.samples.indices)
            bad = [i for i in picks if not 0 < i < n]
            if bad:
                sys.exit(f"indices {bad} out of range 1..{n - 1}")
        else:
            if n <= 1:
                sys.exit(f"{run_path} has {n} samples; nothing beyond the first")
            # never the first sample; it is degenerate (spawn/warm-up frame)
            picks = sorted(rng.sample(range(1, n), min(cfg.samples.num_samples, n - 1)))
        print(f"{run_path}: {n} samples, questions {cfg.questions.mode}"
              + (f" by {cfg.questions.model}" if cfg.questions.mode == "shared" else "")
              + f", models {models}, indices {picks}")

        for sample in picks:
            payload = build_sample_payload(f, sample, cfg.generation.image_width, prompts)
            sample_id = payload.gt["sample_id"]
            for cam, data in payload.frames.items():
                frame_path = frames_dir / f"{sample_id}_{cam}.jpg"
                if not frame_path.exists():
                    frame_path.write_bytes(data)

            qs = None
            if cfg.questions.mode == "shared":
                q_path = out_dir / "questions" / f"{sample_id}.json"
                qs = None if args.regenerate_questions else load_question_set(q_path, cfg)
                if qs is None:
                    print(f"writing question set for {sample_id} "
                          f"(gt {payload.gt['action_label']}) with {cfg.questions.model} ...")
                    try:
                        qs = write_question_set(api_key, cfg, prompts, payload, q_path)
                    except RuntimeError as exc:
                        print(f"  FAILED: {exc}")
                        failures.append((sample_id, cfg.questions.model, str(exc)))
                        continue
                    total_cost += qs["meta"]["usage"]["cost_usd"]
                    print(f"  -> {q_path} ({qs['meta']['attempts']} attempt(s), "
                          f"set {qs['id']})")

            for model in models:
                out_path = out_dir / f"{sample_id}__{model.split('/')[-1]}.json"
                if not args.force and result_is_current(out_path, model, cfg, qs):
                    print(f"{sample_id} {model}: current, skipping")
                    continue
                print(f"annotating {sample_id} (gt {payload.gt['action_label']}) "
                      f"with {model} ...")
                try:
                    if cfg.questions.mode == "shared":
                        annotation, meta = annotate_shared(api_key, model, cfg, prompts,
                                                           payload, qs)
                    else:
                        annotation, meta = annotate_own(api_key, model, cfg, prompts,
                                                        payload)
                except RuntimeError as exc:
                    print(f"  FAILED: {exc}")
                    failures.append((sample_id, model, str(exc)))
                    continue
                annotation["action"] = action_block(payload.gt)
                out = {
                    "model": model,
                    "prompt_version": cfg.prompt_version,
                    "question_mode": cfg.questions.mode,
                    "question_set": ({"id": qs["id"], "model": qs["model"]}
                                     if qs else None),
                    "ground_truth": payload.gt,
                    "annotation": annotation,
                    "meta": meta,
                }
                out_path.write_text(json.dumps(out, indent=2))
                total_cost += meta["usage"]["cost_usd"]
                print(f"  -> {out_path} ({meta['attempts']} attempt(s), "
                      f"{meta['usage']['completion_tokens']} out tokens incl. "
                      f"{meta['usage']['reasoning_tokens']} reasoning, "
                      f"${meta['usage']['cost_usd']:.4f}, {meta['provider']})")

    print(f"done: {len(picks)} samples x {len(models)} models, "
          f"${total_cost:.4f} spent this run, {len(failures)} failure(s)")
    for sample_id, model, err in failures:
        print(f"  FAILED {sample_id} {model}: {err[:200]}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
