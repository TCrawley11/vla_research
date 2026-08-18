"""Annotation benchmark: compare VLM annotators on dataset samples via OpenRouter.

Reads an annotation and question generation config, as well as an examples config 
(configs/annotation/*.yaml). Takes samples from one run .h5 (downloaded from the team 
HF dataset repo) and runs two model stages per sample, both fed the 
key-frame camera images plus a ground-truth block:

(Currently) optional examples block in the config: k finished example annotations from a
validated YAML pool are appended to the annotator prompt as format/style
reference, rotated deterministically per sample (seeded by sample_id) and
hashed into the result prompt_id.

Every (sample, model) result follows the team schema: caption_short /
caption_detailed / qa_pairs typed perception | prediction | planning |
behaviour, plus a code-generated action block.

Output shape is enforced with response_format json_schema plus code-side
validation and retries. Prompts are not versioned by hand: result files and
cached question sets store a hash of the prompt text that produced them
(prompt_id / question_prompt_id), so editing a prompt invalidates exactly the
files it affects. Re-running annotates only the missing or stale pairs. Not
the production annotate stage.

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
from collections import Counter
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

# scripts/ is sys.path[0] when run as a file; the package lives one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_data_pipeline.build_samples import action_label_from_velocity

DEFAULT_CONFIG = Path("configs/annotation/benchmark.yaml")
DEFAULT_REPO_ID = "VLA-uwo-2026/six_cam_1600x900"
PATH_PREFIX = "runs"
# We need to use all 6 cameras - as instructed by professors
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
QaType = Literal["perception", "prediction", "planning", "behaviour"]
QA_TYPES = list(get_args(QaType))

ActionLabel = Literal["STOP", "LEFT_TURN", "RIGHT_TURN",
                      "SLOW_FORWARD", "FORWARD", "UNKNOWN"]
TrajectoryType = Literal["STOPPING", "LEFT_CURVE", "RIGHT_CURVE", "STRAIGHT"]

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


class GenerationOverride(StrictModel):
    """Partial GenerationConfig: only the fields set here replace the base."""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=256)
    max_attempts: Optional[int] = Field(None, ge=1)
    reasoning: Optional[dict] = None
    image_width: Optional[int] = Field(None, ge=64)


class LimitsConfig(StrictModel):
    """Length limits, stated in the prompt and enforced code-side (a violation
    triggers a retry with feedback, like a schema violation)."""
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
    """Questions per QA class, per sample."""
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
        """'6 perception, 4 prediction, 4 planning, 4 behaviour' (skips zeros)."""
        return ", ".join(f"{n} {t}" for t, n in self.as_dict().items() if n)


class QuestionsConfig(StrictModel):
    """The question stage: who writes the per-sample question set and how big
    it is. The author is deliberately separate from the candidates so no model
    answers questions it wrote itself."""
    model: str = Field(
        ..., min_length=1,
        description="question author (OpenRouter model id); must not be a candidate")
    counts: QaCounts = Field(QaCounts(), description="questions per QA class")
    generation: Optional[GenerationOverride] = Field(
        None, description="overrides of the top-level generation block for the "
                          "question author only (it is not a candidate)")

    @field_validator("model")
    @classmethod
    def _strip(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("questions.model must not be empty")
        return v


class ExamplesConfig(StrictModel):
    """Format/style examples for the annotator prompt. k examples are picked
    per sample, seeded by the sample id: every candidate sees the identical
    prompt for a given sample (fair comparison), while different samples get
    different examples (no single phrasing to overfit)."""
    path: Path = Path("configs/annotation/examples.yaml")
    k: int = Field(2, ge=1)


class BenchmarkConfig(StrictModel):
    models: list[str] = Field(..., min_length=1,
                              description="candidate annotators (OpenRouter model ids)")
    samples: SamplesConfig = SamplesConfig()
    questions: QuestionsConfig
    generation: GenerationConfig = GenerationConfig()
    limits: LimitsConfig = LimitsConfig()
    examples: Optional[ExamplesConfig] = Field(
        None, description="format/style examples appended to the annotator "
                          "prompt; null = no examples")
    repo_id: str = DEFAULT_REPO_ID
    out_dir: Path = Path("data/annotation_test")

    @field_validator("models")
    @classmethod
    def _models_unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("models contains duplicates")
        return v

    @model_validator(mode="after")
    def _author_not_a_candidate(self):
        if self.questions.model in self.models:
            raise ValueError(f"questions.model {self.questions.model!r} is also a "
                             "candidate; the question author must not answer "
                             "its own questions")
        return self

    def question_generation(self) -> GenerationConfig:
        """Generation settings for the question author: base block + overrides."""
        if self.questions.generation is None:
            return self.generation
        return self.generation.model_copy(
            update=self.questions.generation.model_dump(exclude_unset=True))


def load_config(path: Path) -> BenchmarkConfig:
    if not path.is_file():
        sys.exit(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        sys.exit(f"config must be a YAML mapping: {path}")
    return BenchmarkConfig.model_validate(raw)


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

MOVING_SEG_M = 0.15  # a waypoint step shorter than this counts as standing still


def speed_profile(waypoints: np.ndarray, period_sec: float) -> str:
    """Describe how speed evolves along the future waypoints (T, 2), spaced
    period_sec apart, starting at the ego origin.

    The action label only reflects the current velocity, so a car labelled
    FORWARD may be braking to a stop within the horizon; without this the
    annotator has no way to know."""
    pts = np.vstack([[0.0, 0.0], np.asarray(waypoints, dtype=float)])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)          # m per step
    speeds = seg / period_sec
    moving = seg > MOVING_SEG_M
    if not moving.any():
        return "stationary throughout"
    first, last = float(speeds[0]), float(speeds[-1])
    if moving[0] and not moving[-1]:
        stop_step = int(np.argmax(~moving))                     # first still step
        stop_m = float(pts[stop_step][0])                        # forward displacement there
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
    """Turn ego-frame future waypoints (T, 2) into a short factual phrase:
    displacement, curvature and the speed profile."""
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
    """The per-sample ground truth shown to the model and stored in the result
    file. Validated so a corrupt or misread run file fails loudly instead of
    silently producing a wrong prompt."""
    sample_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    key_frame_id: int = Field(ge=0)
    action_label: ActionLabel
    past_action: ActionLabel = Field(
        description="majority action label over the past_window_sec before "
                    "the key frame (key frame included)")
    trajectory_type: TrajectoryType
    v: FiniteFloat = Field(description="ego forward velocity, m/s")
    w: FiniteFloat = Field(description="ego angular velocity, rad/s, +left")
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
        """The ground-truth block of the user message (USER_GT filled in)."""
        return USER_GT.format(
            action_label=self.action_label, action_text=self.action_text,
            v=self.v, w=self.w,
            past_window_sec=self.past_window_sec, past_action=self.past_action,
            horizon_sec=self.horizon_sec, traj_summary=self.traj_summary())

    def action_block(self) -> dict:
        """Deterministic action annotation from ground truth (never the model)."""
        stopped = self.action_label == "STOP"
        return {"action_text": self.action_text,
                "action_label": self.action_label,
                "linear_velocity_target": 0.0 if stopped else round(self.v, 2),
                "angular_velocity_target": 0.0 if stopped else round(self.w, 3)}

    def record(self) -> dict:
        """JSON-ready dict for the result file."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------
# samples: .h5 -> SamplePayload
# --------------------------------------------------------------------------

@dataclass
class SamplePayload:
    gt: GroundTruth
    image_blocks: list       # OpenAI-style content blocks with the cameras
    frames: dict             # camera -> jpeg bytes (saved for inspection)


def encode_jpeg(rgb: np.ndarray, width: int) -> bytes:
    img = Image.fromarray(rgb)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def past_indices(clip: np.ndarray) -> np.ndarray:
    """The past half of the clip, key frame included. clip_frame_indices is
    symmetric about the key frame (build_samples.clip_indices), so using the
    whole clip would leak the future into a 'past behavior' label."""
    return clip[: len(clip) // 2 + 1]


def load_sample(f: h5py.File, sample: int, image_width: int) -> SamplePayload:
    """Extract key-frame images + validated GT for one sample index."""
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


def pick_samples(f: h5py.File, cfg: SamplesConfig, rng: random.Random,
                 run_path: str) -> list[int]:
    """The sorted sample indices to benchmark on (never index 0, the spawn frame)."""
    n = len(f["sample_index/sample_id"])
    if cfg.indices:
        bad = [i for i in cfg.indices if not 0 < i < n]
        if bad:
            sys.exit(f"indices {bad} out of range 1..{n - 1}")
        return sorted(cfg.indices)
    if n <= 1:
        sys.exit(f"{run_path} has {n} samples; nothing beyond the first")
    return sorted(rng.sample(range(1, n), min(cfg.num_samples, n - 1)))


# --------------------------------------------------------------------------
# output checks: response_format schemas (strict) + code-side validators;
# the validators stay the source of truth either way
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
        return  # reported by _nonempty_str
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
    """Stable sort by QA type in the team order."""
    return sorted(items, key=lambda p: QA_TYPES.index(p["type"]))


def question_ids(n_total: int) -> list[str]:
    return [f"q{i:02d}" for i in range(1, n_total + 1)]


def question_listing(questions: list[dict]) -> str:
    """The question list as shown to the annotator (and in prompt examples)."""
    return "\n".join(f"{q['id']} [{q['type']}] {q['question']}" for q in questions)


def question_set_id(questions: list[dict]) -> str:
    blob = json.dumps([[q["type"], q["question"]] for q in questions])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# example pool (optional): finished annotations from other scenes, shown to
# the annotator as format/style reference. The pool file and k are hashed
# into the result prompt_id, so editing the pool invalidates the results it
# shaped.
# --------------------------------------------------------------------------

class ExampleQa(StrictModel):
    type: QaType
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class ExampleAnnotation(StrictModel):
    """One entry of the example pool (see configs/annotation/examples.yaml),
    stored as natural QA pairs and rendered in the exact task shape."""
    scene: str = Field(min_length=1, description="one-line scene setter shown "
                                                 "above the example")
    caption_short: str
    caption_detailed: str
    qa_pairs: list[ExampleQa] = Field(min_length=1)

    def questions(self) -> list[dict]:
        return [{"id": qid, "type": q.type, "question": q.question}
                for qid, q in zip(question_ids(len(self.qa_pairs)), self.qa_pairs)]

    def output(self) -> dict:
        """The example's annotator response, in the exact output shape."""
        return {"caption_short": self.caption_short,
                "caption_detailed": self.caption_detailed,
                "answers": [{"id": qid, "answer": q.answer}
                            for qid, q in zip(question_ids(len(self.qa_pairs)),
                                              self.qa_pairs)]}


def load_examples(cfg: ExamplesConfig, limits: LimitsConfig) -> list[ExampleAnnotation]:
    """Load the pool; every example must pass the same validators the
    annotators are held to, so an example can never contradict the limits
    stated in the prompt."""
    if not cfg.path.is_file():
        sys.exit(f"examples file not found: {cfg.path}")
    raw = yaml.safe_load(cfg.path.read_text())
    if not isinstance(raw, list) or not raw:
        sys.exit(f"examples file must be a non-empty YAML list: {cfg.path}")
    pool = [ExampleAnnotation.model_validate(e) for e in raw]
    for i, ex in enumerate(pool):
        ids = [q["id"] for q in ex.questions()]
        errors = validate_answers(ex.output(), ids, limits)
        if errors:
            sys.exit(f"examples[{i}] ({ex.scene!r}) violates the configured "
                     f"limits: {errors}")
    if cfg.k > len(pool):
        sys.exit(f"examples.k = {cfg.k} but {cfg.path} has only {len(pool)} entries")
    return pool


def select_examples(pool: list[ExampleAnnotation], k: int,
                    sample_id: str) -> list[ExampleAnnotation]:
    """Deterministic per-sample rotation. random.Random(str) is stable across
    processes (unlike hash()), so every candidate gets the identical prompt
    for a given sample while examples still rotate across samples."""
    return random.Random(sample_id).sample(pool, k)


def render_examples(examples: list[ExampleAnnotation]) -> str:
    """The examples block appended to the annotator system prompt: each
    example shown exactly in task shape (question listing -> output JSON)."""
    blocks = [f"Example {i} (scene: {ex.scene}):\n"
              "Questions to answer:\n" + question_listing(ex.questions())
              + "\n" + json.dumps(ex.output(), indent=2)
              for i, ex in enumerate(examples, 1)]
    return ("\nExamples of finished annotations from other scenes. Match their "
            "format, tone and level of detail; they describe different frames, "
            "so never copy facts or numbers from them.\n\n"
            + "\n\n".join(blocks) + "\n")


# --------------------------------------------------------------------------
# OpenRouter client
# --------------------------------------------------------------------------

def parse_json_content(text: str):
    """Parse model output, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


class OpenRouterClient:
    """Chat-completions client with retries at three levels: transport errors
    and retryable HTTP statuses, provider errors returned inside 200 bodies,
    and schema/validator violations (retried with feedback to the model)."""

    URL = "https://openrouter.ai/api/v1/chat/completions"
    HTTP_RETRIES = 5
    RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _post(self, body: dict) -> dict:
        """POST with backoff on transport errors and retryable statuses.
        Non-retryable HTTP errors propagate as urllib.error.HTTPError."""
        req_body = json.dumps(body).encode()
        for i in range(1, self.HTTP_RETRIES + 1):
            req = urllib.request.Request(
                self.URL, data=req_body,
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code not in self.RETRY_STATUS or i == self.HTTP_RETRIES:
                    raise
                print(f"  HTTP {exc.code}; retry {i}/{self.HTTP_RETRIES - 1} in {15 * i}s")
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                if i == self.HTTP_RETRIES:
                    raise RuntimeError(f"OpenRouter unreachable: {exc}") from exc
                print(f"  {type(exc).__name__}: {exc}; "
                      f"retry {i}/{self.HTTP_RETRIES - 1} in {15 * i}s")
            time.sleep(15 * i)

    def call(self, model: str, system: str, content: list, json_schema: dict,
             validate, gen: GenerationConfig) -> tuple[dict, dict]:
        """Call the model with schema enforcement; validate; retry with feedback.

        Returns (object, meta) where meta records attempts, token usage, cost
        and the serving provider."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        response_format = {"type": "json_schema", "json_schema": json_schema}
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "reasoning_tokens": 0, "cost_usd": 0.0}
        errors_seen = []
        provider = None

        def add_usage(resp: dict) -> None:
            nonlocal provider
            u = resp.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
            usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0) or 0
            usage["cost_usd"] += float(u.get("cost", 0) or 0)
            provider = resp.get("provider", provider)

        for attempt in range(1, gen.max_attempts + 1):
            body = {"model": model, "messages": messages,
                    "max_tokens": gen.max_tokens, "temperature": gen.temperature,
                    "response_format": response_format}
            if gen.reasoning is not None:
                body["reasoning"] = gen.reasoning
            try:
                resp = None
                for i in range(1, self.HTTP_RETRIES + 1):
                    resp = self._post(body)
                    add_usage(resp)
                    if resp.get("choices"):
                        break
                    # OpenRouter reports provider errors (429/5xx upstream) as
                    # 200s with an error object; back off like a transport error
                    err = str(resp.get("error", resp))[:300]
                    if i == self.HTTP_RETRIES:
                        raise RuntimeError(
                            f"provider error after {self.HTTP_RETRIES} tries: {err}")
                    print(f"  provider error: {err[:120]}; "
                          f"retry {i}/{self.HTTP_RETRIES - 1} in {15 * i}s")
                    time.sleep(15 * i)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                # provider rejected json_schema enforcement -> degrade once
                if exc.code == 400 and response_format.get("type") == "json_schema":
                    print(f"  json_schema rejected ({detail[:120]}); "
                          "falling back to json_object")
                    response_format = {"type": "json_object"}
                    continue
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc

            choice = resp["choices"][0]
            raw = choice["message"].get("content") or ""
            try:
                obj = parse_json_content(raw)
                errors = validate(obj)
            except (json.JSONDecodeError, IndexError) as exc:
                errors = [f"output is not valid JSON: {exc}"]
                if choice.get("finish_reason") == "length":
                    errors.append("output truncated at max_tokens (finish_reason="
                                  "length); raise generation.max_tokens or limit reasoning")
            if not errors:
                usage["cost_usd"] = round(usage["cost_usd"], 6)
                return obj, {"attempts": attempt, "usage": usage,
                             "provider": provider, "errors_seen": errors_seen,
                             "response_format": response_format["type"]}

            errors_seen.append(errors)
            print(f"  attempt {attempt} invalid: {'; '.join(errors[:4])}")
            if not raw.strip():
                continue  # nothing to give feedback on (e.g. truncated while thinking)
            messages = messages[:2] + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    "Your output violated the required schema: " + "; ".join(errors)
                    + ". Reply again with ONLY the corrected JSON object."}]

        raise RuntimeError(f"no valid output after {gen.max_attempts} attempts: "
                           f"{errors_seen[-1] if errors_seen else 'no response'}")


# --------------------------------------------------------------------------
# benchmark runner
# --------------------------------------------------------------------------

class Benchmark:
    """Runs the benchmark over one run file: writes/caches the question set
    per sample, has every candidate answer it, skips results that are
    current, and accumulates cost and failures."""

    def __init__(self, cfg: BenchmarkConfig, client: OpenRouterClient | None):
        self.cfg = cfg
        self.client = client
        self.examples = (load_examples(cfg.examples, cfg.limits)
                         if cfg.examples else None)
        self.examples_id = (_prompt_hash(cfg.examples.path.read_text(),
                                         str(cfg.examples.k))
                            if cfg.examples else None)
        self.failures: list[tuple[str, str, str]] = []
        self.total_cost = 0.0

    # -- paths ------------------------------------------------------------

    def question_path(self, sample_id: str) -> Path:
        return self.cfg.out_dir / "questions" / f"{sample_id}.json"

    def result_path(self, sample_id: str, model: str) -> Path:
        return self.cfg.out_dir / f"{sample_id}__{model.split('/')[-1]}.json"

    # -- prompt identity + examples ----------------------------------------

    def prompt_id(self) -> str:
        """Hash of everything that shapes the annotator prompt: the templates,
        plus the example pool and k when examples are enabled."""
        if not self.examples_id:
            return ANNOTATOR_PROMPT_ID
        return _prompt_hash(ANNOTATOR_PROMPT_ID, self.examples_id)

    def _examples_suffix(self, sample_id: str) -> str:
        if not self.examples:
            return ""
        return render_examples(
            select_examples(self.examples, self.cfg.examples.k, sample_id))

    # -- stage 1: question sets --------------------------------------------

    def load_question_set(self, path: Path) -> dict | None:
        """The cached question set if it matches the config and the current
        question prompt, else None."""
        if not path.exists():
            return None
        qs = json.loads(path.read_text())
        if (qs.get("model") == self.cfg.questions.model
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
            self.cfg.questions.model, system, content,
            schema_questions(counts.total),
            lambda o: validate_questions(o, counts.as_dict()),
            self.cfg.question_generation())
        questions = [{"id": qid, "type": q["type"], "question": q["question"].strip()}
                     for qid, q in zip(question_ids(counts.total),
                                       canonical_order(obj["questions"]))]
        qs = {"sample_id": payload.gt.sample_id, "model": self.cfg.questions.model,
              "question_prompt_id": QUESTION_WRITER_PROMPT_ID,
              "counts": counts.as_dict(),
              "id": question_set_id(questions), "questions": questions, "meta": meta}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(qs, indent=2))
        return qs

    # -- stage 2: captions + answers ---------------------------------------

    def annotate(self, model: str, payload: SamplePayload,
                 qs: dict) -> tuple[dict, dict]:
        """One (sample, model) annotation: captions plus answers to the
        sample's question set."""
        questions = qs["questions"]
        ids = [q["id"] for q in questions]
        system = ANNOTATOR_SYSTEM.format(n_total=len(ids),
                                       **self.cfg.limits.model_dump())
        system += self._examples_suffix(payload.gt.sample_id)
        listing = question_listing(questions)
        content = user_content(payload, "Questions to answer:\n" + listing
                               + "\n\nWrite the annotation JSON now.")
        obj, meta = self.client.call(
            model, system, content, schema_answers(ids),
            lambda o: validate_answers(o, ids, self.cfg.limits),
            self.cfg.generation)
        by_id = {a["id"]: a["answer"].strip() for a in obj["answers"]}
        return ({"caption_short": obj["caption_short"],
                 "caption_detailed": obj["caption_detailed"],
                 "qa_pairs": [{**q, "answer": by_id[q["id"]]} for q in questions]},
                meta)

    def result_is_current(self, path: Path, model: str, qs: dict) -> bool:
        """True when the result on disk was produced with the same model,
        prompts, limits, counts and question set, so it need not be recomputed."""
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        cfg = self.cfg
        return (d.get("model") == model
                and d.get("prompt_id") == self.prompt_id()
                and d.get("limits") == cfg.limits.model_dump()
                and d.get("qa_counts") == cfg.questions.counts.as_dict()
                and (d.get("question_set") or {}).get("id") == qs["id"])

    # -- run loop ---------------------------------------------------------

    def run(self, f: h5py.File, picks: list[int], models: list[str],
            force: bool = False, regenerate_questions: bool = False) -> None:
        frames_dir = self.cfg.out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for sample in picks:
            payload = load_sample(f, sample, self.cfg.generation.image_width)
            gt = payload.gt
            for cam, data in payload.frames.items():
                frame_path = frames_dir / f"{gt.sample_id}_{cam}.jpg"
                if not frame_path.exists():
                    frame_path.write_bytes(data)

            q_path = self.question_path(gt.sample_id)
            qs = None if regenerate_questions else self.load_question_set(q_path)
            if qs is None:
                print(f"writing question set for {gt.sample_id} "
                      f"(gt {gt.action_label}) with {self.cfg.questions.model} ...")
                try:
                    qs = self.write_question_set(payload, q_path)
                except RuntimeError as exc:
                    print(f"  FAILED: {exc}")
                    self.failures.append(
                        (gt.sample_id, self.cfg.questions.model, str(exc)))
                    continue
                self.total_cost += qs["meta"]["usage"]["cost_usd"]
                print(f"  -> {q_path} ({qs['meta']['attempts']} attempt(s), "
                      f"set {qs['id']})")

            for model in models:
                out_path = self.result_path(gt.sample_id, model)
                if not force and self.result_is_current(out_path, model, qs):
                    print(f"{gt.sample_id} {model}: current, skipping")
                    continue
                print(f"annotating {gt.sample_id} (gt {gt.action_label}) "
                      f"with {model} ...")
                try:
                    annotation, meta = self.annotate(model, payload, qs)
                except RuntimeError as exc:
                    print(f"  FAILED: {exc}")
                    self.failures.append((gt.sample_id, model, str(exc)))
                    continue
                annotation["action"] = gt.action_block()
                out_path.write_text(json.dumps({
                    "model": model,
                    "prompt_id": self.prompt_id(),
                    "limits": self.cfg.limits.model_dump(),
                    "qa_counts": self.cfg.questions.counts.as_dict(),
                    "question_set": {"id": qs["id"], "model": qs["model"]},
                    "examples": (None if not self.examples else
                                 {"k": self.cfg.examples.k,
                                  "pool_id": self.examples_id,
                                  "scenes": [e.scene for e in select_examples(
                                      self.examples, self.cfg.examples.k,
                                      gt.sample_id)]}),
                    "ground_truth": gt.record(),
                    "annotation": annotation,
                    "meta": meta,
                }, indent=2))
                self.total_cost += meta["usage"]["cost_usd"]
                print(f"  -> {out_path} ({meta['attempts']} attempt(s), "
                      f"{meta['usage']['completion_tokens']} out tokens incl. "
                      f"{meta['usage']['reasoning_tokens']} reasoning, "
                      f"${meta['usage']['cost_usd']:.4f}, {meta['provider']})")


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
                    help="rewrite cached question sets (invalidates results)")
    args = ap.parse_args()

    cfg = load_config(args.config)
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

    bench = Benchmark(cfg, OpenRouterClient(api_key))
    with h5py.File(h5_path, "r") as f:
        picks = pick_samples(f, cfg.samples, rng, run_path)
        print(f"{run_path}: {len(f['sample_index/sample_id'])} samples, "
              f"questions by {cfg.questions.model}, models {models}, "
              f"indices {picks}")
        bench.run(f, picks, models, force=args.force,
                  regenerate_questions=args.regenerate_questions)

    print(f"done: {len(picks)} samples x {len(models)} models, "
          f"${bench.total_cost:.4f} spent this run, {len(bench.failures)} failure(s)")
    for sample_id, model, err in bench.failures:
        print(f"  FAILED {sample_id} {model}: {err[:200]}")
    if bench.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
