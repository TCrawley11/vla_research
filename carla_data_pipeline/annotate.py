"""Stage 4: annotate dataset samples with a local OpenAI-compatible VLM.

Walks the team Hugging Face dataset sequentially (every `runs/*.h5`, samples
1..n-1; index 0 is the spawn frame). Each sample is six key-frame cameras plus
a ground-truth block. The same local model writes the question set and then
answers it. Output is enforced with `response_format` json_schema, code-side
validators, and retries with feedback.

Serve the model first (`scripts/serve_annotator.sh`); this module is a client.

Usage:
  python -m carla_data_pipeline annotate
  python -m carla_data_pipeline annotate --h5 data/runs/run43.h5 --limit 2
  python -m carla_data_pipeline.annotate --config configs/annotation/local.yaml
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args

import h5py
import numpy as np
import yaml
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from pydantic import (BaseModel, ConfigDict, Field, FiniteFloat,
                      field_validator, model_validator)

from .build_samples import action_label_from_velocity

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/annotation/local.yaml")
DEFAULT_REPO_ID = "VLA-uwo-2026/six_cam_1600x900"
PATH_PREFIX = "runs"
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
QaType = Literal["perception", "prediction", "planning", "behaviour"]
QA_TYPES = list(get_args(QaType))

ActionLabel = Literal["STOP", "LEFT_TURN", "RIGHT_TURN",
                      "SLOW_FORWARD", "FORWARD", "UNKNOWN"]
TrajectoryType = Literal["STOPPING", "LEFT_CURVE", "RIGHT_CURVE", "STRAIGHT"]

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
        None, description="run id in repo_id, e.g. run43; null walks every run")
    indices: Optional[list[int]] = Field(
        None, description="explicit sample indices for one run; requires run or --h5")
    limit: Optional[int] = Field(
        None, ge=1, description="stop after this many samples across the walk")

    @field_validator("indices")
    @classmethod
    def _indices_ok(cls, v):
        if v is not None:
            if not v:
                raise ValueError("indices must not be empty; use null for sequential")
            if any(i < 1 for i in v):
                raise ValueError("indices must be >= 1 (index 0 is the spawn frame)")
            if len(set(v)) != len(v):
                raise ValueError("indices contains duplicates")
        return v



class InferenceConfig(StrictModel):
    base_url: str = Field("http://127.0.0.1:8001/v1")
    model: str = Field("qwen3.5-9b-q6_k", min_length=1)
    concurrency: int = Field(2, ge=1)
    timeout_sec: int = Field(300, ge=30)
    chat_template_kwargs: dict = Field(
        default_factory=lambda: {"enable_thinking": False})

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v):
        return v.rstrip("/")


class GenerationConfig(StrictModel):
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(4000, ge=256)
    max_attempts: int = Field(3, ge=1)
    image_width: int = Field(800, ge=64)


class LimitsConfig(StrictModel):
    answer_max_words: int = Field(30, ge=5)
    caption_short_max_words: int = Field(25, ge=5)
    caption_detailed_min_words: int = Field(30, ge=1)
    caption_detailed_max_words: int = Field(70, ge=5)

    @model_validator(mode="after")
    def _detailed_range(self):
        if self.caption_detailed_min_words > self.caption_detailed_max_words:
            raise ValueError("caption_detailed_min_words exceeds caption_detailed_max_words")
        return self


class QaCounts(StrictModel):
    perception: int = Field(6, ge=0)
    prediction: int = Field(4, ge=0)
    planning: int = Field(4, ge=0)
    behaviour: int = Field(4, ge=0)

    @model_validator(mode="after")
    def _some_questions(self):
        if self.total == 0:
            raise ValueError("questions.counts must request at least one question")
        return self

    def as_dict(self) -> dict[str, int]:
        return {t: getattr(self, t) for t in QA_TYPES}

    @property
    def total(self) -> int:
        return sum(self.as_dict().values())

    def text(self) -> str:
        return ", ".join(f"{n} {t}" for t, n in self.as_dict().items() if n)


class QuestionsConfig(StrictModel):
    counts: QaCounts = Field(default_factory=QaCounts)


class AnnotateConfig(StrictModel):
    samples: SamplesConfig = Field(default_factory=SamplesConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    repo_id: str = DEFAULT_REPO_ID
    out_dir: Path = Path("data/annotations")


def load_config(path: Path) -> AnnotateConfig:
    if not path.is_file():
        sys.exit(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        sys.exit(f"config must be a YAML mapping: {path}")
    return AnnotateConfig.model_validate(raw)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

QUESTION_WRITER_SYSTEM = """\
Write the question set for one sample of an autonomous driving
dataset collected in the CARLA simulator. Several vision-language models will
later answer exactly these questions for this sample, using the same camera
images and ground-truth record you see now, so every question must be
answerable from those inputs alone. Write questions only, never answers.

Write exactly {n_total} questions, {counts_text}, of these types:
    perception - Test what is visible in the images (road layout, lane
                 markings, traffic lights and signs, other road users, weather
                 and lighting); ask only about things that are visible, or
                 whose absence is worth stating; never presuppose objects
                 that are not there,
    prediction - Test what the recorded future trajectory shows, phrased as
                 what the ego vehicle is expected to do next,
    planning   - Test what the ego vehicle should do next and how, given the
                 ground-truth action,
    behaviour  - Test the ego vehicle's current motion state and maneuver

Rules:
1. Be specific to this scene; a question that fits any frame is a weak one.
2. Ask for a value instead of confirming it: "What is the ego vehicle's
   current speed?", not "Is the ego vehicle driving at 8.06 m/s?".
3. Each question stands alone (no "the object above"), and no two questions
   should ever ask the same thing.
4. Prefer questions whose answers need the images and the ground truth
   together (why the recorded action fits what is visible, what to watch for
   while executing it, how the trajectory relates to the road ahead) over
   questions answered by copying one ground-truth field; at most one question
   per type may be a plain lookup of a ground-truth value.
5. Vary phrasing.

Output a single JSON object, no markdown fences, no commentary:
{{"questions": [{{"type": ..., "question": ...}}, ...]}} with exactly {n_total}
items ({counts_text}), type one of "perception", "prediction", "planning",
"behaviour".
"""

ANNOTATOR_SYSTEM = """\
Annotate samples for an autonomous driving dataset collected in the
CARLA simulator by describing the scene and answering questions as instructed. 
Each sample contains 6 camera images from the ego vehicle and a ground-truth record 
from the simulator API. Write captions and answers to a fixed 
list of questions, used to train a driving vision-language model.

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

The user message lists the questions to answer, each with an id and its type:
    perception - what is visible in the scene,
    prediction - what the recorded future trajectory shows, phrased as what
                 the ego vehicle is expected to do next,
    planning   - what the ego vehicle should do next and how, consistent with
                 the ground-truth action,
    behaviour  - the ego vehicle's current motion state and maneuver
If a question presupposes something that is neither visible nor in the
ground truth, say so briefly instead of inventing it.

Style: answers are direct and factual - one or two sentences, at most
{answer_max_words} words each - with no preamble, no restating of the
question and no commentary. caption_short: one sentence, at most
{caption_short_max_words} words. caption_detailed: 2-4 sentences,
{caption_detailed_min_words}-{caption_detailed_max_words} words. Word limits
are enforced.

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

USER_GT = """\
Ground truth for this frame (simulator API, exact):
- current driving action label: {action_label} ({action_text})
- ego forward velocity: {v:.2f} m/s
- ego angular velocity: {w:.3f} rad/s (positive = left turn)
- dominant action over the past {past_window_sec:.1f} s: {past_action}
- recorded future trajectory (next {horizon_sec:.0f} s): {traj_summary}

The images above are the current key frame from the """ + \
    ", ".join(CAMERAS[:-1]) + " and " + CAMERAS[-1] + " cameras.\n"


def _prompt_hash(*texts: str) -> str:
    return hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]


QUESTION_WRITER_PROMPT_ID = _prompt_hash(QUESTION_WRITER_SYSTEM, USER_GT)
ANNOTATOR_PROMPT_ID = _prompt_hash(ANNOTATOR_SYSTEM, USER_GT)


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

MOVING_SEG_M = 0.15


def speed_profile(waypoints: np.ndarray, period_sec: float) -> str:
    pts = np.vstack([[0.0, 0.0], np.asarray(waypoints, dtype=float)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    speeds = seg / period_sec
    moving = seg > MOVING_SEG_M
    if not moving.any():
        return "stationary throughout"
    first, last = float(speeds[0]), float(speeds[-1])
    if moving[0] and not moving[-1]:
        stop_step = int(np.argmax(~moving))
        stop_m = float(pts[stop_step][0])
        return (f"decelerating from about {first:.1f} m/s to a full stop after "
                f"about {stop_m:.1f} m (within about {stop_step * period_sec:.1f} s)")
    if not moving[0] and moving[-1]:
        start_step = int(np.argmax(moving))
        return (f"pulling away from standstill after about "
                f"{start_step * period_sec:.1f} s, reaching about {last:.1f} m/s")
    if last < 0.7 * first:
        return f"slowing from about {first:.1f} m/s to about {last:.1f} m/s"
    if last > 1.4 * first:
        return f"accelerating from about {first:.1f} m/s to about {last:.1f} m/s"
    return f"at a roughly steady {speeds.mean():.1f} m/s"


def summarize_trajectory(waypoints: np.ndarray, traj_type: str,
                         horizon_sec: float, period_sec: float = 0.5) -> str:
    fwd, lat = float(waypoints[-1][0]), float(waypoints[-1][1])
    if traj_type == "STOPPING":
        return (f"the vehicle stays essentially stationary "
                f"({fwd:.1f} m forward displacement over {horizon_sec:.0f} s)")
    side = "left" if lat > 0 else "right"
    curve = {"STRAIGHT": "in a straight line",
             "LEFT_CURVE": "curving left",
             "RIGHT_CURVE": "curving right"}.get(traj_type, "")
    return (f"the vehicle moves {fwd:.1f} m forward {curve}, "
            f"{speed_profile(waypoints, period_sec)}, ending "
            f"{abs(lat):.1f} m to the {side} of its current heading")


class GroundTruth(StrictModel):
    sample_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    key_frame_id: int = Field(ge=0)
    action_label: ActionLabel
    past_action: ActionLabel
    trajectory_type: TrajectoryType
    v: FiniteFloat
    w: FiniteFloat
    past_window_sec: FiniteFloat = Field(gt=0)
    horizon_sec: FiniteFloat = Field(gt=0)
    waypoint_period_sec: FiniteFloat = Field(gt=0)
    future_waypoints_ego_frame: list[tuple[FiniteFloat, FiniteFloat]] = \
        Field(min_length=1)

    @property
    def action_text(self) -> str:
        return ACTION_TEXT[self.action_label]

    def traj_summary(self) -> str:
        return summarize_trajectory(
            np.asarray(self.future_waypoints_ego_frame, dtype=float),
            self.trajectory_type, self.horizon_sec, self.waypoint_period_sec)

    def prompt_block(self) -> str:
        return USER_GT.format(
            action_label=self.action_label, action_text=self.action_text,
            v=self.v, w=self.w,
            past_window_sec=self.past_window_sec, past_action=self.past_action,
            horizon_sec=self.horizon_sec, traj_summary=self.traj_summary())

    def action_block(self) -> dict:
        stopped = self.action_label == "STOP"
        return {"action_text": self.action_text,
                "action_label": self.action_label,
                "linear_velocity_target": 0.0 if stopped else round(self.v, 2),
                "angular_velocity_target": 0.0 if stopped else round(self.w, 3)}

    def record(self) -> dict:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------
# samples: .h5 -> SamplePayload
# --------------------------------------------------------------------------

@dataclass
class SamplePayload:
    gt: GroundTruth
    image_blocks: list
    frames: dict


def encode_jpeg(rgb: np.ndarray, width: int) -> bytes:
    img = Image.fromarray(rgb)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def past_indices(clip: np.ndarray) -> np.ndarray:
    return clip[: len(clip) // 2 + 1]


def load_sample(f: h5py.File, sample: int, image_width: int) -> SamplePayload:
    si = f["sample_index"]
    key = int(si["key_index"][sample])
    cols = [c.decode() if isinstance(c, bytes) else c
            for c in f["telemetry/data"].attrs["columns"]]
    tel = {c: f["telemetry/data"][:, i] for i, c in enumerate(cols)}

    def as_str(x):
        return x.decode() if isinstance(x, bytes) else str(x)

    clip = si["clip_frame_indices"][sample]
    labels = [action_label_from_velocity(float(tel["v"][i]), float(tel["w"][i]))
              for i in past_indices(clip)]
    clip_sec = float(f.attrs.get("clip_sec", 3.0))

    gt = GroundTruth(
        sample_id=as_str(si["sample_id"][sample]),
        sample_index=sample,
        key_frame_id=int(si["key_frame_id"][sample]),
        action_label=as_str(f["action/action_label"][sample]),
        past_action=Counter(labels).most_common(1)[0][0],
        trajectory_type=as_str(f["trajectory/trajectory_type"][sample]),
        v=float(tel["v"][key]), w=float(tel["w"][key]),
        past_window_sec=clip_sec / 2,
        horizon_sec=float(f.attrs.get("horizon_sec", 3.0)),
        waypoint_period_sec=float(f.attrs.get("waypoint_period_sec", 0.5)),
        future_waypoints_ego_frame=[
            (float(x), float(y))
            for x, y in f["trajectory/future_waypoints_ego_frame"][sample]])

    frames = {cam: encode_jpeg(f["images"][cam][key], image_width) for cam in CAMERAS}
    image_blocks = []
    for cam in CAMERAS:
        b64 = base64.b64encode(frames[cam]).decode()
        image_blocks += [
            {"type": "text", "text": f"Image from the {cam} camera:"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    return SamplePayload(gt=gt, image_blocks=image_blocks, frames=frames)


def user_content(payload: SamplePayload, tail: str) -> list:
    return payload.image_blocks + [
        {"type": "text", "text": payload.gt.prompt_block() + "\n" + tail}]


def list_run_paths(api: HfApi, repo_id: str) -> list[str]:
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith(f"{PATH_PREFIX}/") and f.endswith(".h5")]
    if not files:
        sys.exit(f"no .h5 runs found in {repo_id}/{PATH_PREFIX}")
    return sorted(files)


def resolve_run_labels(api: HfApi | None, cfg: AnnotateConfig,
                       h5: Path | None) -> list[str]:
    """HF repo paths (or the local --h5 path) in walk order. Download is lazy."""
    if h5 is not None:
        if not h5.is_file():
            sys.exit(f"no such run file: {h5}")
        return [str(h5)]
    assert api is not None
    if cfg.samples.run:
        path = f"{PATH_PREFIX}/{cfg.samples.run}.h5"
        available = list_run_paths(api, cfg.repo_id)
        if path not in available:
            sys.exit(f"{path} not in repo; available: {available}")
        return [path]
    return list_run_paths(api, cfg.repo_id)


def local_h5(repo_id: str, run_label: str, h5: Path | None) -> str:
    if h5 is not None:
        return str(h5)
    log.info("downloading %s/%s", repo_id, run_label)
    return hf_hub_download(repo_id, run_label, repo_type="dataset")


def pick_samples(n: int, cfg: SamplesConfig, run_path: str) -> list[int]:
    """Sorted sample indices to annotate (never index 0)."""
    if n <= 1:
        sys.exit(f"{run_path} has {n} samples; nothing beyond the spawn frame")
    if cfg.indices:
        bad = [i for i in cfg.indices if not 0 < i < n]
        if bad:
            sys.exit(f"indices {bad} out of range 1..{n - 1}")
        return sorted(cfg.indices)
    return list(range(1, n))


# --------------------------------------------------------------------------
# output checks
# --------------------------------------------------------------------------

def _strict(name: str, properties: dict) -> dict:
    return {"name": name, "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "required": list(properties), "properties": properties}}


def _array(n: int, item_props: dict) -> dict:
    return {"type": "array", "minItems": n, "maxItems": n,
            "items": {"type": "object", "additionalProperties": False,
                      "required": list(item_props), "properties": item_props}}


def schema_questions(n_total: int) -> dict:
    return _strict("question_set", {
        "questions": _array(n_total, {"type": {"type": "string", "enum": QA_TYPES},
                                      "question": {"type": "string"}})})


def schema_answers(ids: list[str]) -> dict:
    return _strict("annotation", {
        "caption_short": {"type": "string"},
        "caption_detailed": {"type": "string"},
        "answers": _array(len(ids), {"id": {"type": "string", "enum": ids},
                                     "answer": {"type": "string"}})})


def _nonempty_str(obj: dict, key: str, where: str, errors: list[str]) -> None:
    if not isinstance(obj.get(key), str) or not obj[key].strip():
        errors.append(f"{where}'{key}' missing or not a non-empty string")


def _check_words(text, where: str, lo: int, hi: int, errors: list[str]) -> None:
    if not isinstance(text, str):
        return
    n = len(text.split())
    if n > hi:
        errors.append(f"{where} has {n} words (max {hi})")
    elif n < lo:
        errors.append(f"{where} has {n} words (min {lo})")


def _check_captions(obj: dict, limits: LimitsConfig, errors: list[str]) -> None:
    for key in ("caption_short", "caption_detailed"):
        _nonempty_str(obj, key, "", errors)
    _check_words(obj.get("caption_short"), "caption_short", 1,
                 limits.caption_short_max_words, errors)
    _check_words(obj.get("caption_detailed"), "caption_detailed",
                 limits.caption_detailed_min_words,
                 limits.caption_detailed_max_words, errors)


def _check_typed_items(items, counts: dict[str, int], fields: tuple[str, ...],
                       name: str) -> list[str]:
    errors = []
    found = dict.fromkeys(QA_TYPES, 0)
    for i, p in enumerate(items):
        if not isinstance(p, dict):
            errors.append(f"{name}[{i}] is not an object")
            continue
        t = p.get("type")
        if t not in QA_TYPES:
            errors.append(f"{name}[{i}].type '{t}' not in {QA_TYPES}")
        else:
            found[t] += 1
        for key in fields:
            _nonempty_str(p, key, f"{name}[{i}].", errors)
    for t, c in found.items():
        if c != counts[t]:
            errors.append(f"{c} '{t}' items (need exactly {counts[t]})")
    return errors


def validate_questions(obj, counts: dict[str, int]) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    qs = obj.get("questions")
    if not isinstance(qs, list):
        return ["'questions' missing or not a list"]
    errors = _check_typed_items(qs, counts, ("question",), "questions")
    texts = [q.get("question", "").strip().lower() for q in qs if isinstance(q, dict)]
    if len(set(texts)) != len(texts):
        errors.append("duplicate questions")
    return errors


def validate_answers(obj, ids: list[str], limits: LimitsConfig) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    errors = []
    _check_captions(obj, limits, errors)
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
        _check_words(a.get("answer"), f"answers[{a.get('id', i)}].answer", 1,
                     limits.answer_max_words, errors)
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


def canonical_order(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda p: QA_TYPES.index(p["type"]))


def question_ids(n_total: int) -> list[str]:
    return [f"q{i:02d}" for i in range(1, n_total + 1)]


def question_listing(questions: list[dict]) -> str:
    return "\n".join(f"{q['id']} [{q['type']}] {q['question']}" for q in questions)


def question_set_id(questions: list[dict]) -> str:
    blob = json.dumps([[q["type"], q["question"]] for q in questions])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# local OpenAI-compatible client (vLLM)
# --------------------------------------------------------------------------

def parse_json_content(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


class VllmClient:
    """Chat-completions client: transport retries, json_schema, validator feedback."""

    HTTP_RETRIES = 5
    RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, cfg: InferenceConfig):
        self._url = f"{cfg.base_url}/chat/completions"
        self._models_url = f"{cfg.base_url}/models"
        self._timeout = cfg.timeout_sec
        self._chat_template_kwargs = cfg.chat_template_kwargs
        self.served_model = cfg.model

    def ping(self, model: str) -> None:
        req = urllib.request.Request(self._models_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            sys.exit(f"vLLM not reachable at {self._url}: {exc}\n"
                     "start it with scripts/serve_annotator.sh")
        ids = [m.get("id") for m in body.get("data", [])]
        if model in ids or not ids:
            self.served_model = model
            return
        if len(ids) == 1:
            log.warning("config model %r not served; using %r", model, ids[0])
            self.served_model = ids[0]
            return
        sys.exit(f"served models {ids} do not include {model!r}; "
                 "check inference.model vs --served-model-name")

    def _post(self, body: dict) -> dict:
        req_body = json.dumps(body).encode()
        for i in range(1, self.HTTP_RETRIES + 1):
            req = urllib.request.Request(
                self._url, data=req_body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code not in self.RETRY_STATUS or i == self.HTTP_RETRIES:
                    raise
                log.warning("HTTP %s; retry %s/%s in %ss",
                            exc.code, i, self.HTTP_RETRIES - 1, 15 * i)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                if i == self.HTTP_RETRIES:
                    raise RuntimeError(f"vLLM unreachable: {exc}") from exc
                log.warning("%s: %s; retry %s/%s in %ss",
                            type(exc).__name__, exc, i, self.HTTP_RETRIES - 1, 15 * i)
            time.sleep(15 * i)
        raise RuntimeError("vLLM unreachable")

    def call(self, model: str, system: str, content: list, json_schema: dict,
             validate, gen: GenerationConfig) -> tuple[dict, dict]:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        response_format = {"type": "json_schema", "json_schema": json_schema}
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "reasoning_tokens": 0, "cost_usd": 0.0}
        errors_seen = []

        def add_usage(resp: dict) -> None:
            u = resp.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
            usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0) or 0

        for attempt in range(1, gen.max_attempts + 1):
            body = {"model": self.served_model, "messages": messages,
                    "max_tokens": gen.max_tokens, "temperature": gen.temperature,
                    "response_format": response_format,
                    "chat_template_kwargs": self._chat_template_kwargs}
            try:
                resp = None
                for i in range(1, self.HTTP_RETRIES + 1):
                    resp = self._post(body)
                    add_usage(resp)
                    if resp.get("choices"):
                        break
                    err = str(resp.get("error", resp))[:300]
                    if i == self.HTTP_RETRIES:
                        raise RuntimeError(
                            f"provider error after {self.HTTP_RETRIES} tries: {err}")
                    log.warning("provider error: %s; retry %s/%s in %ss",
                                err[:120], i, self.HTTP_RETRIES - 1, 15 * i)
                    time.sleep(15 * i)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                if exc.code == 400 and response_format.get("type") == "json_schema":
                    log.warning("json_schema rejected (%s); falling back to json_object",
                                detail[:120])
                    response_format = {"type": "json_object"}
                    continue
                raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc

            choice = resp["choices"][0]
            raw = choice["message"].get("content") or ""
            try:
                obj = parse_json_content(raw)
                errors = validate(obj)
            except (json.JSONDecodeError, IndexError) as exc:
                errors = [f"output is not valid JSON: {exc}"]
                if choice.get("finish_reason") == "length":
                    errors.append("output truncated at max_tokens (finish_reason="
                                  "length); raise generation.max_tokens or disable thinking")
            if not errors:
                return obj, {"attempts": attempt, "usage": usage,
                             "provider": "vllm", "errors_seen": errors_seen,
                             "response_format": response_format["type"]}

            errors_seen.append(errors)
            log.warning("attempt %s invalid: %s", attempt, "; ".join(errors[:4]))
            if not raw.strip():
                continue
            messages = messages[:2] + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    "Your output violated the required schema: " + "; ".join(errors)
                    + ". Reply again with ONLY the corrected JSON object."}]

        raise RuntimeError(f"no valid output after {gen.max_attempts} attempts: "
                           f"{errors_seen[-1] if errors_seen else 'no response'}")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

class Annotator:
    def __init__(self, cfg: AnnotateConfig, client: VllmClient):
        self.cfg = cfg
        self.client = client
        self.failures: list[tuple[str, str]] = []
        self._h5_lock = threading.Lock()
        self._fail_lock = threading.Lock()

    def question_path(self, sample_id: str) -> Path:
        return self.cfg.out_dir / "questions" / f"{sample_id}.json"

    def result_path(self, sample_id: str) -> Path:
        suffix = self.cfg.inference.model.split("/")[-1]
        return self.cfg.out_dir / f"{sample_id}__{suffix}.json"

    def load_question_set(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        qs = json.loads(path.read_text())
        if (qs.get("model") == self.cfg.inference.model
                and qs.get("question_prompt_id") == QUESTION_WRITER_PROMPT_ID
                and qs.get("counts") == self.cfg.questions.counts.as_dict()):
            return qs
        return None

    def write_question_set(self, payload: SamplePayload, path: Path) -> dict:
        counts = self.cfg.questions.counts
        system = QUESTION_WRITER_SYSTEM.format(n_total=counts.total,
                                               counts_text=counts.text())
        content = user_content(payload, "Write the question set JSON now.")
        obj, meta = self.client.call(
            self.cfg.inference.model, system, content,
            schema_questions(counts.total),
            lambda o: validate_questions(o, counts.as_dict()),
            self.cfg.generation)
        questions = [{"id": qid, "type": q["type"], "question": q["question"].strip()}
                     for qid, q in zip(question_ids(counts.total),
                                       canonical_order(obj["questions"]))]
        qs = {"sample_id": payload.gt.sample_id, "model": self.cfg.inference.model,
              "question_prompt_id": QUESTION_WRITER_PROMPT_ID,
              "counts": counts.as_dict(),
              "id": question_set_id(questions), "questions": questions, "meta": meta}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(qs, indent=2))
        return qs

    def annotate(self, payload: SamplePayload, qs: dict) -> tuple[dict, dict]:
        questions = qs["questions"]
        ids = [q["id"] for q in questions]
        system = ANNOTATOR_SYSTEM.format(n_total=len(ids),
                                         **self.cfg.limits.model_dump())
        listing = question_listing(questions)
        content = user_content(payload, "Questions to answer:\n" + listing
                               + "\n\nWrite the annotation JSON now.")
        obj, meta = self.client.call(
            self.cfg.inference.model, system, content, schema_answers(ids),
            lambda o: validate_answers(o, ids, self.cfg.limits),
            self.cfg.generation)
        by_id = {a["id"]: a["answer"].strip() for a in obj["answers"]}
        return ({"caption_short": obj["caption_short"],
                 "caption_detailed": obj["caption_detailed"],
                 "qa_pairs": [{**q, "answer": by_id[q["id"]]} for q in questions]},
                meta)

    def result_is_current(self, path: Path, qs: dict) -> bool:
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        cfg = self.cfg
        return (d.get("model") == cfg.inference.model
                and d.get("prompt_id") == ANNOTATOR_PROMPT_ID
                and d.get("limits") == cfg.limits.model_dump()
                and d.get("qa_counts") == cfg.questions.counts.as_dict()
                and (d.get("question_set") or {}).get("id") == qs["id"])

    def process_sample(self, f: h5py.File, sample: int, frames_dir: Path,
                       force: bool, regenerate_questions: bool) -> None:
        with self._h5_lock:
            payload = load_sample(f, sample, self.cfg.generation.image_width)
        gt = payload.gt
        for cam, data in payload.frames.items():
            frame_path = frames_dir / f"{gt.sample_id}_{cam}.jpg"
            if not frame_path.exists():
                frame_path.write_bytes(data)

        q_path = self.question_path(gt.sample_id)
        qs = None if regenerate_questions else self.load_question_set(q_path)
        if qs is None:
            log.info("writing question set for %s (gt %s)", gt.sample_id, gt.action_label)
            try:
                qs = self.write_question_set(payload, q_path)
            except RuntimeError as exc:
                log.error("FAILED questions %s: %s", gt.sample_id, exc)
                with self._fail_lock:
                    self.failures.append((gt.sample_id, str(exc)))
                return
            log.info("  -> %s (set %s, %s attempt(s))",
                     q_path, qs["id"], qs["meta"]["attempts"])

        out_path = self.result_path(gt.sample_id)
        if not force and self.result_is_current(out_path, qs):
            log.info("%s: current, skipping", gt.sample_id)
            return
        log.info("annotating %s (gt %s)", gt.sample_id, gt.action_label)
        try:
            annotation, meta = self.annotate(payload, qs)
        except RuntimeError as exc:
            log.error("FAILED annotate %s: %s", gt.sample_id, exc)
            with self._fail_lock:
                self.failures.append((gt.sample_id, str(exc)))
            return
        annotation["action"] = gt.action_block()
        out_path.write_text(json.dumps({
            "model": self.cfg.inference.model,
            "prompt_id": ANNOTATOR_PROMPT_ID,
            "limits": self.cfg.limits.model_dump(),
            "qa_counts": self.cfg.questions.counts.as_dict(),
            "question_set": {"id": qs["id"], "model": qs["model"]},
            "examples": None,
            "ground_truth": gt.record(),
            "annotation": annotation,
            "meta": meta,
        }, indent=2))
        log.info("  -> %s (%s attempt(s), %s out tokens)",
                 out_path, meta["attempts"], meta["usage"]["completion_tokens"])

    def run_file(self, f: h5py.File, picks: list[int], force: bool,
                 regenerate_questions: bool) -> None:
        frames_dir = self.cfg.out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        workers = min(self.cfg.inference.concurrency, len(picks)) or 1
        if workers == 1:
            for sample in picks:
                self.process_sample(f, sample, frames_dir, force, regenerate_questions)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(self.process_sample, f, sample, frames_dir,
                                force, regenerate_questions)
                    for sample in picks]
            for fut in as_completed(futs):
                fut.result()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Annotate HF dataset samples with a local vLLM endpoint.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help=f"annotation config (default {DEFAULT_CONFIG})")
    ap.add_argument("--h5", type=Path,
                    help="local run .h5; skips the HF download")
    ap.add_argument("--base-url",
                    help="override inference.base_url (OpenAI-compatible root)")
    ap.add_argument("--run", help="override samples.run (one run id in the repo)")
    ap.add_argument("--indices", help="comma-separated sample indices (requires --run or --h5)")
    ap.add_argument("--limit", type=int, help="stop after N samples")
    ap.add_argument("--force", action="store_true",
                    help="recompute results that are already current on disk")
    ap.add_argument("--regenerate-questions", action="store_true",
                    help="rewrite cached question sets (invalidates results)")
    return ap


def apply_cli_overrides(cfg: AnnotateConfig, args) -> AnnotateConfig:
    if getattr(args, "base_url", None):
        inf = cfg.inference.model_copy(update={"base_url": args.base_url.rstrip("/")})
        cfg = cfg.model_copy(update={"inference": inf})
    samples = cfg.samples
    if args.run:
        samples = samples.model_copy(update={"run": args.run})
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
        samples = samples.model_copy(update={"indices": indices})
        if samples.run is None and args.h5 is None:
            sys.exit("--indices requires --run or --h5")
    if args.limit is not None:
        samples = samples.model_copy(update={"limit": args.limit})
    if samples.indices is not None and samples.run is None and args.h5 is None:
        sys.exit("samples.indices requires samples.run or --h5")
    return cfg.model_copy(update={"samples": samples})


def run(args) -> int:
    cfg = apply_cli_overrides(load_config(args.config), args)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    client = VllmClient(cfg.inference)
    client.ping(cfg.inference.model)
    annotator = Annotator(cfg, client)

    remaining = cfg.samples.limit
    api = None if args.h5 else HfApi()
    for run_path in resolve_run_labels(api, cfg, args.h5):
        if remaining is not None and remaining <= 0:
            break
        h5_path = local_h5(cfg.repo_id, run_path, args.h5)
        with h5py.File(h5_path, "r") as f:
            n = len(f["sample_index/sample_id"])
            picks = pick_samples(n, cfg.samples, run_path)
            if remaining is not None:
                picks = picks[:remaining]
                remaining -= len(picks)
            log.info("%s: %s samples in file, annotating %s", run_path, n, picks)
            annotator.run_file(f, picks, force=args.force,
                               regenerate_questions=args.regenerate_questions)

    log.info("done: %s failure(s)", len(annotator.failures))
    for sample_id, err in annotator.failures:
        log.error("FAILED %s: %s", sample_id, err[:200])
    return 1 if annotator.failures else 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
