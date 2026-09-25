"""Stage 4: annotate dataset samples with a local OpenAI-compatible VLM.

Walks the team Hugging Face dataset sequentially (every `runs/*.h5`, samples
1..n-1; index 0 is the spawn frame). Each sample is six key-frame cameras plus
a ground-truth block. The same local model writes the question set and then
answers it. Output is enforced with `response_format` json_schema, code-side
validators, and retries with feedback.

Owns the production prompts, schemas, validation and sample loading. The hosted
benchmark imports this module; production never depends on benchmark code.

Serve the model first (`scripts/serve_annotator_llama.sh`); this module is a client.

Usage:
  python -m carla_data_pipeline annotate
  python -m carla_data_pipeline annotate --h5 data/runs/run43.h5 --limit 2
  python -m carla_data_pipeline.annotate --config configs/annotation/local.yaml
  python -m carla_data_pipeline annotate --config configs/annotation/smoke.yaml
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import tempfile
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


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


# Order is significant: writers produce each type's questions in this order.
QUESTION_PURPOSES = {
    "perception": ["road_layout", "road_users", "signals_signs", "markings",
                   "visibility_surface", "another_view"],
    "prediction": ["path", "speed_change", "stop_or_move", "path_scene_relation"],
    "planning": ["maneuver", "speed_timing", "monitor", "scene_constraint"],
    "behaviour": ["motion_state", "rotation", "recent_change", "scene_relation"],
}

QUESTION_WRITER_SYSTEM = """\
Write the question set for one sample of a CARLA driving dataset.
Write questions only, never answers. Use the six camera images and the
recorded motion. Write exactly {n_total} questions, {counts_text}.

Keep the task easy for a quantized 27B model: one short question, one purpose,
one simple observation or comparison. Prefer fewer than 22 words. Use ordinary
words. Avoid multi-part questions, calculations, exact object counts, steering
angles, hidden intent, and long explanations. Never presuppose objects that
are not there. A question may establish a relevant object's absence.

Write each type in the following purpose order. Use the first N purposes if
fewer questions of that type are requested. For additional slots, ask about a
new visible detail, never paraphrase an earlier question.
perception (6):
  1. Road layout: straight road, bend, or junction.
  2. Other road users: identify or locate a visible agent, or establish absence.
  3. Traffic lights/signs: ask about one visible signal/sign, or absence.
  4. Road markings: identify a marking or describe an unmarked road.
  5. Visibility or road surface: choose one detail that matters here.
  6. Another view: name a camera other than FRONT and ask about a useful detail.
prediction (4), about the RECORDED future, not guesses:
  1. Path: which way the recorded path goes relative to the visible road.
  2. Speed change: whether the vehicle slows, speeds up, or holds speed here.
  3. Stop or move: whether it stops, stays stopped, or starts moving in the horizon.
  4. Path/scene relation: one simple relation to a visible bend, junction, or marking.
     Do not claim a precise crossing position without supporting geometry.
planning (4):
  1. Maneuver: the next movement consistent with the recorded future path.
  2. Speed/timing: one speed adjustment or waiting condition.
  3. Monitor: one visible feature or road user to watch.
  4. Scene constraint: how visible road geometry, a boundary, or an obstacle
     relates to the maneuver. Never ask what causes, explains, justifies,
     supports, permits, or allows the recorded action.
behaviour (4), about the PRESENT and recorded PAST:
  1. Motion state: stopped, creeping, or moving.
  2. Rotation: turning left/right or approximately straight, related to the scene.
  3. Recent change: compare the past dominant action with the current motion.
     A dominant label is not proof of constant speed or intent.
  4. Scene relation: describe the current movement relative to a visible road feature.

The purposes may share evidence but must ask for different information.
Do not repeat a question across types. Avoid repeated steer-and-hold instructions.
Prefer qualitative answers tied to this scene. At most one plain numeric or
label lookup per type. Never ask for unknown future angular velocity, future
signal changes, or the driver's intentions. Current action labels describe
current motion, not commands. FORWARD can coexist with braking to a stop.
Do not assume a crosswalk alone caused a stop. Name a camera for view-specific
questions. Treat all six cameras equally; none has priority.

Output only this JSON object, exactly {n_total} items:
{{"questions": [{{"type": ..., "question": ...}}, ...]}}
Types: perception, prediction, planning, behaviour.
"""

ANNOTATOR_SYSTEM = """\
Annotate one CARLA driving sample using six camera images and recorded motion.
Answer the listed questions by id, without rewriting them.

Evidence rules:
- Images establish visible objects, signals, markings, and conditions. Name the
  camera for view-specific claims. Treat all six cameras equally.
- Telemetry establishes present motion. Action labels are coarse descriptions
  of present motion, NOT commands or proof of intention.
- The recorded trajectory establishes future ego motion. Its speeds and stop
  times are approximate values derived from spaced waypoints. A current FORWARD
  label can accompany braking. A STOP label can accompany slow creeping.
- Copy supplied numbers with their units and time scope. Do not infer future
  angular velocity from current angular velocity. Do not calculate missing values.
- Planning should follow the recorded future and visible evidence. State a
  mismatch or unknown cause instead of forcing agreement. A crosswalk alone
  does not establish the reason for a stop. A green side-facing signal need
  not control the ego lane. Do not guess which lane a signal controls.
- Do not invent objects, exact counts, steering angles, acceleration, intentions,
  or future signal changes. Say briefly when an answer is not determinable.
- Distinguish observation from advice: "watch for pedestrians" does not assert
  that pedestrians are present. Absence in one camera is not absence in all.
- caption_detailed must mention visible dynamic agents (pedestrians, other
  vehicles, cyclists), or explicitly state absence in the inspected views.
  Ego itself does not satisfy this requirement. Do not infer agent motion
  or following distance changes from a single image.

Style: direct answers, one or two sentences, at most {answer_max_words} words each.
No preamble or restating the question. caption_short: one sentence, at most
{caption_short_max_words} words. caption_detailed: 2-4 sentences,
{caption_detailed_min_words}-{caption_detailed_max_words} words.

Output only this JSON object:
{{
  "caption_short": one sentence about the current situation,
  "caption_detailed": a grounded description of the visible scene,
  "answers": one item per listed question ({n_total} items),
             each {{"id": the question id, "answer": ...}}
}}
"""

USER_GT = """\
Recorded motion for this frame:
- current action label (coarse motion class, not a command): {action_label}
- current ego speed: {v:.2f} m/s
- current ego angular velocity: {w:.3f} rad/s (positive = left turn)
- dominant action over the past {past_window_sec:.1f} s: {past_action}
- recorded future trajectory (next {horizon_sec:g} s): {traj_summary}
The camera images are the current key frame, not a video. There is no measured
future angular velocity, signal-change time, or actor intent in this record.
""" + "Images: " + ", ".join(CAMERAS) + ".\n"


def _prompt_hash(*texts: str) -> str:
    return hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]


QUESTION_WRITER_PROMPT_ID = _prompt_hash(QUESTION_WRITER_SYSTEM, USER_GT)
ANNOTATOR_PROMPT_ID = _prompt_hash(ANNOTATOR_SYSTEM, USER_GT)


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

# Noise tolerance is in speed units so it is independent of waypoint spacing.
STATIONARY_MPS = 0.05
CREEPING_MPS = 0.5


def trajectory_speeds(waypoints: np.ndarray, period_sec: float) -> np.ndarray:
    pts = np.vstack([[0.0, 0.0], np.asarray(waypoints, dtype=float)])
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / period_sec


def motion_state(v: float) -> str:
    if abs(v) < STATIONARY_MPS:
        return "stationary"
    return "creeping" if abs(v) < CREEPING_MPS else "moving"


def speed_profile(waypoints: np.ndarray, period_sec: float) -> str:
    speeds = trajectory_speeds(waypoints, period_sec)
    moving = speeds >= STATIONARY_MPS
    if not moving.any():
        return "stationary throughout"
    first, last = float(speeds[0]), float(speeds[-1])
    # Only describe a sustained stop, not a temporary pause before moving again.
    if moving[0] and not moving[-1]:
        stop_step = int(np.flatnonzero(moving)[-1]) + 1
        stop_m = float(waypoints[stop_step - 1][0])
        return (f"decelerating from about {first:.1f} m/s to a full stop after "
                f"about {stop_m:.1f} m forward displacement "
                f"(within about {stop_step * period_sec:g} s)")
    if not moving[0] and moving[-1]:
        start_step = int(np.flatnonzero(moving)[0])
        return (f"pulling away from standstill after about "
                f"{start_step * period_sec:g} s, reaching about {last:.1f} m/s")
    if not moving.all():
        return (f"moving with an intervening stop, ending at about {last:.1f} m/s")
    # Use both absolute and relative change: a 28% speed drop is not steady.
    tolerance = max(0.2, 0.1 * max(first, last))
    if last < first - tolerance:
        return f"slowing from about {first:.1f} m/s to about {last:.1f} m/s"
    if last > first + tolerance:
        return f"accelerating from about {first:.1f} m/s to about {last:.1f} m/s"
    if float(np.ptp(speeds)) > max(0.3, 0.15 * float(speeds.mean())):
        return (f"varying between about {speeds.min():.1f} and {speeds.max():.1f} m/s")
    prefix = "creeping" if speeds.max() < CREEPING_MPS else "moving"
    return f"{prefix} at a roughly steady {speeds.mean():.1f} m/s"


def summarize_trajectory(waypoints: np.ndarray, traj_type: str,
                         horizon_sec: float, period_sec: float = 0.5) -> str:
    fwd, lat = map(float, waypoints[-1])
    profile = speed_profile(waypoints, period_sec)
    if profile == "stationary throughout":
        return (f"stationary throughout ({fwd:.1f} m forward displacement "
                f"over {horizon_sec:g} s)")
    # STOPPING in the dataset means small net forward displacement, not no motion.
    curve = {"STRAIGHT": "in a straight line", "LEFT_CURVE": "curving left",
             "RIGHT_CURVE": "curving right"}.get(traj_type, "")
    lateral = ("with negligible lateral displacement" if abs(lat) < 0.05 else
               f"ending {abs(lat):.1f} m to the {'left' if lat > 0 else 'right'} "
               "of its current heading")
    return (f"{profile}; {fwd:.1f} m forward displacement over {horizon_sec:g} s "
            f"{curve}, {lateral}")


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

    @model_validator(mode="after")
    def _waypoint_grid(self):
        if not np.isclose(len(self.future_waypoints_ego_frame) * self.waypoint_period_sec,
                          self.horizon_sec):
            raise ValueError("waypoint count and period must cover horizon_sec exactly")
        return self

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
        """Version 2: observed velocities, never mislabelled as control targets."""
        return {"action_label": self.action_label,
                "motion_state": motion_state(self.v),
                "linear_velocity_current": round(self.v, 2),
                "angular_velocity_current": round(self.w, 3),
                "future_motion": self.traj_summary()}

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
    tel = {c: f["telemetry/data"][:, i] for i, c in enumerate(cols)
           if c in {"v", "w"}}

    def as_str(x):
        return x.decode() if isinstance(x, bytes) else str(x)

    clip = si["clip_frame_indices"][sample]
    if "motion_telemetry" in f:
        mt = f["motion_telemetry"]
        labels = [action_label_from_velocity(
            float(mt["smoothed_speed_mps"][i]), float(tel["w"][i]),
            motion_state=as_str(mt["motion_state"][i])) for i in past_indices(clip)]
    else:
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
        past_window_sec=(key - int(past_indices(clip)[0])) / float(f.attrs["raw_fps"]),
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


PERCEPTION_SLOT_MIN = 3
LOOKUP_CAP_PER_TYPE = 1
PLANNING_PARAPHRASE_MIN = 3
_NON_FRONT_CAMERAS = [c for c in CAMERAS if c != "FRONT"]
_CAM_RE = re.compile(
    r"\b(?:FRONT(?:[_ -](?:LEFT|RIGHT))?|BACK(?:[_ -](?:LEFT|RIGHT))?)\b", re.I)
_NON_FRONT_CAM_RE = re.compile(
    r"\b(?:FRONT[_ -](?:LEFT|RIGHT)|BACK(?:[_ -](?:LEFT|RIGHT))?)\b", re.I)
_ROAD_USER_RE = re.compile(
    r"\b(?:pedestrians?|walkers?|cyclists?|vehicles?|cars?|trucks?|"
    r"buses|motorcycles?|motorcyclists?|bicycles?|people|person|road users?)\b", re.I)
_SIGNAL_RE = re.compile(
    r"\b(?:traffic lights?|(?:traffic |stop |circular )?signs?|signals?)\b", re.I)
_AGENT_RE = _ROAD_USER_RE
_ABSENCE_RE = re.compile(
    r"\bno (?:other )?(?:pedestrians?|road users|vehicles?|dynamic agents?|"
    r"cars?|people)\b", re.I)
_FOG_RE = re.compile(r"\b(?:fog|foggy|haze|hazy|mist|misty|poor visibility)\b", re.I)
_CLEAR_RE = re.compile(r"\b(?:clear|sunny|good visibility)\b", re.I)
_PED_ABSENT_RE = re.compile(r"\bno (?:pedestrians?|people|walkers?|persons?)\b", re.I)
_PED_PRESENT_RE = re.compile(r"\b(?:pedestrians?|people|walkers?|persons?|person)\b", re.I)
_STEER_RE = re.compile(r"\b(?:steer|steering|turn|turning|continue|continuing)\b", re.I)
_HOLD_RE = re.compile(r"\b(?:hold|holding|maintain|maintaining|speed)\b", re.I)
_STRIP_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*(?:m/s|rad/s|m|s))?", re.I)
_CAUSAL_QUESTION_RE = re.compile(
    r"\b(?:what|which)\b[^?]{0,100}\b(?:causes?|explains?|justifies?|"
    r"supports?|permits?|allows?)\b|\bwhy should\b", re.I)
_LOOKUP_PATTERNS: dict[str, list[re.Pattern]] = {
    "behaviour": [
        re.compile(r"\b(?:current\s+)?(?:speed|velocity|forward velocity|"
                   r"linear velocity)\b", re.I),
        re.compile(r"\bangular velocity\b", re.I),
        re.compile(r"\baction label\b", re.I),
        re.compile(r"\bpast (?:driving )?action\b", re.I),
    ],
    "prediction": [
        re.compile(r"\b(?:how far|forward (?:displacement|distance)|"
                   r"metres? forward|meters? forward)\b", re.I),
        re.compile(r"\b(?:lateral displacement|metres? to the (?:left|right)|"
                   r"meters? to the (?:left|right))\b", re.I),
        re.compile(r"\bexpected trajectory\b", re.I),
        re.compile(r"\bwaypoint\b", re.I),
    ],
    "planning": [
        re.compile(r"\baction label\b", re.I),
        re.compile(r"\brecorded future trajectory\b", re.I),
    ],
    "perception": [],
}


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
        if set(p) != {"type", *fields}:
            errors.append(f"{name}[{i}] has missing or extra fields")
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


def _is_gt_lookup(question: str, qa_type: str) -> bool:
    """Conservative audit hint, not a semantic acceptance gate.

    Scene relationships are allowed even when they mention speed/trajectory.
    Label/numeric retrieval is counted; qualitative motion questions are useful.
    """
    if re.search(r"\b(?:relat\w*|align\w*|visible|camera|road|crosswalk|signal|pedestrian|curve|bend)\b",
                 question, re.I):
        return False
    return bool(re.search(
        r"\b(?:how (?:far|fast)|(?:current|forward|angular|linear|longitudinal) velocity|"
        r"(?:current|exact|average|expected) speed|displacement|action label|"
        r"dominant action|past (?:driving )?action|waypoints?)\b", question, re.I))


def _lookup_cap_errors(questions: list) -> list[str]:
    counts: dict[str, int] = dict.fromkeys(QA_TYPES, 0)
    for q in questions:
        if not isinstance(q, dict):
            continue
        t = q.get("type")
        text = q.get("question", "")
        if t in QA_TYPES and isinstance(text, str) and _is_gt_lookup(text, t):
            counts[t] += 1
    return [f"{n} '{t}' questions are ground-truth lookups "
            f"(max {LOOKUP_CAP_PER_TYPE} per type)"
            for t, n in counts.items() if n > LOOKUP_CAP_PER_TYPE]


def _is_steer_and_hold(text: str) -> bool:
    stripped = _STRIP_NUM_RE.sub(" ", text)
    return bool(_STEER_RE.search(stripped) and _HOLD_RE.search(stripped))


def _planning_paraphrase_errors(questions: list) -> list[str]:
    planning = [q.get("question", "") for q in questions
                if isinstance(q, dict) and q.get("type") == "planning"
                and isinstance(q.get("question"), str)]
    n = sum(1 for t in planning if _is_steer_and_hold(t))
    if n >= PLANNING_PARAPHRASE_MIN:
        return ["planning questions are paraphrases of the same "
                "steer-and-hold instruction"]
    return []


def _perception_slot_errors(questions: list, counts: dict[str, int]) -> list[str]:
    if counts.get("perception", 0) < PERCEPTION_SLOT_MIN:
        return []
    perc = [q.get("question", "") for q in questions
            if isinstance(q, dict) and q.get("type") == "perception"
            and isinstance(q.get("question"), str)]
    errors = []
    if not any(_ROAD_USER_RE.search(t) for t in perc):
        errors.append("perception questions omit the road-users slot")
    if not any(_SIGNAL_RE.search(t) for t in perc):
        errors.append("perception questions omit the traffic-signals slot")
    if not any(_NON_FRONT_CAM_RE.search(t) for t in perc):
        errors.append("perception questions omit a named non-FRONT camera")
    return errors


def validate_questions(obj, counts: dict[str, int]) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    qs = obj.get("questions")
    if not isinstance(qs, list):
        return ["'questions' missing or not a list"]
    errors = _check_typed_items(qs, counts, ("question",), "questions")
    if set(obj) != {"questions"}:
        errors.append("question output must contain only questions")
    texts = [q["question"].strip().lower() for q in qs
             if isinstance(q, dict) and isinstance(q.get("question"), str)]
    if len(set(texts)) != len(texts):
        errors.append("duplicate questions")
    return errors


def _caption_blob(obj: dict) -> str:
    parts = [obj.get("caption_short") or "", obj.get("caption_detailed") or ""]
    return " ".join(p for p in parts if isinstance(p, str))


def _weather_polarity(text: str) -> str | None:
    # "clear road" and "clear of traffic" are not weather claims.
    fog = bool(_FOG_RE.search(text))
    clear = bool(re.search(r"\b(?:clear (?:weather|skies|sky)|sunny|good visibility)\b", text, re.I))
    return "fog" if fog and not clear else "clear" if clear and not fog else None


def _pedestrian_polarity(text: str) -> str | None:
    # Advice, hypotheticals, and crosswalk descriptions assert no agent presence.
    if re.search(r"\b(?:watch|monitor|check|if|could|might|may|should|before)\b", text, re.I):
        return None
    if _PED_ABSENT_RE.search(text):
        return "absent"
    text = re.sub(r"pedestrian (?:crosswalk|crossing|signal)s?", "", text, flags=re.I)
    return "present" if _PED_PRESENT_RE.search(text) else None


def _camera_scope(text: str) -> frozenset[str]:
    if re.search(r"\b(?:all|any|six) (?:camera|view)s?\b", text, re.I):
        return frozenset(CAMERAS)
    return frozenset(re.sub(r"[- ]", "_", m.group().upper())
                     for m in _CAM_RE.finditer(text))


def _causal_question_errors(questions: list[dict]) -> list[str]:
    """Flag planning questions that presuppose an unobserved reason.

    This remains an audit hint: wording alone cannot prove that a causal claim
    is false, so it must not trigger model retries.
    """
    return [
        f"review {q.get('id', '?')}: planning question presupposes why the action was chosen"
        for q in questions
        if isinstance(q, dict) and q.get("type") == "planning"
        and isinstance(q.get("question"), str)
        and _CAUSAL_QUESTION_RE.search(q["question"])
    ]


def _caption_content_errors(obj: dict) -> list[str]:
    # These are audit hints only: natural-language evidence cannot be proved
    # by a keyword match. In particular "ego vehicle" is not another agent.
    detailed = obj.get("caption_detailed")
    if not isinstance(detailed, str) or not detailed.strip():
        return []
    errors = []
    if not _CAM_RE.search(detailed):
        errors.append("caption_detailed does not name a camera")
    others = re.sub(r"(?:the )?ego vehicle|pedestrian (?:crosswalk|crossing|signal)s?",
                    "", detailed, flags=re.I)
    if not (_AGENT_RE.search(others) or _ABSENCE_RE.search(others)):
        errors.append("caption_detailed should mention a visible dynamic agent "
                      "or state absence in the inspected views")
    return errors


def _contradiction_errors(obj: dict, answers: list, questions: list | None = None) -> list[str]:
    """Scope-aware review hints. Never force the model to rewrite based on these."""
    caption = _caption_blob(obj)
    q_by_id = {q.get("id"): q for q in questions or []}
    claims = re.split(r"(?<=[.!?])\s+|;\s*", caption)
    issues = []
    for a in answers:
        if not isinstance(a, dict) or not isinstance(a.get("answer"), str):
            continue
        text = a["answer"]
        q = q_by_id.get(a.get("id"), {})
        # Planning advice is not an assertion of current pedestrian presence.
        if q.get("type") == "planning":
            continue
        scope = _camera_scope(text) or _camera_scope(q.get("question", ""))
        for claim in claims:
            cap_scope = _camera_scope(claim)
            if scope and cap_scope and not scope.intersection(cap_scope):
                continue
            # Unscoped claims are too ambiguous to assert a contradiction.
            if bool(scope) != bool(cap_scope):
                continue
            for label, polarity in [("weather/visibility", _weather_polarity),
                                     ("pedestrians", _pedestrian_polarity)]:
                left, right = polarity(claim), polarity(text)
                if left and right and left != right:
                    issues.append(f"review {a.get('id', '?')}: caption/answer disagree on {label} in the same scope")
    return list(dict.fromkeys(issues))


_NUMBER_UNIT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(m/s|rad/s)\b")


def numeric_errors(obj: dict, questions: list[dict], gt: GroundTruth) -> list[str]:
    """Check explicit current-state numeric answers against their exact field.

    Future/conditional questions are not compared against current telemetry.
    General prose is not treated as machine-checkable ground truth.
    """
    by_id = {q["id"]: q for q in questions}
    errors = []
    for answer in obj.get("answers", []):
        if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
            continue
        q = by_id.get(answer.get("id"), {})
        text = q.get("question", "").lower()
        if q.get("type") != "behaviour" or not re.search(r"\b(?:current|now|present)\b", text):
            continue
        if re.search(r"\b(?:past|future|change|compare|difference)\b", text):
            continue
        if not _is_gt_lookup(text, "behaviour"):
            continue
        field = "w" if "angular velocity" in text else "v" if re.search(r"\b(?:speed|forward velocity|linear velocity)\b", text) else None
        if field is None:
            continue
        unit, expected, decimals = ("rad/s", gt.w, 3) if field == "w" else ("m/s", gt.v, 2)
        for number, found_unit in _NUMBER_UNIT.findall(answer["answer"]):
            if found_unit == unit and abs(float(number) - round(expected, decimals)) > 0.5 * 10**(-decimals) + 1e-9:
                errors.append(f"{answer['id']}: current {field} must be {expected:.{decimals}f} {unit}, not {number} {unit}")
    return errors


def quality_issues(obj: dict, questions: list[dict]) -> list[str]:
    """Transparent review flags, not accuracy scores or automatic rejections."""
    issues = _caption_content_errors(obj)
    issues += _contradiction_errors(obj, obj.get("answers", []), questions)
    issues += _perception_slot_errors(questions, Counter(q["type"] for q in questions))
    issues += _lookup_cap_errors(questions)
    issues += _planning_paraphrase_errors(questions)
    issues += _causal_question_errors(questions)
    normalized = [re.sub(r"[^a-z0-9 ]", "", q["question"].lower()) for q in questions]
    for i, left in enumerate(normalized):
        for j in range(i):
            a, b = set(left.split()), set(normalized[j].split())
            if len(a | b) and len(a & b) / len(a | b) > 0.8:
                issues.append(f"review question similarity: {questions[j]['id']} / {questions[i]['id']}")
    return list(dict.fromkeys(issues))


def validate_answers(obj, ids: list[str], limits: LimitsConfig,
                     questions: list[dict] | None = None, gt: GroundTruth | None = None) -> list[str]:
    if not isinstance(obj, dict):
        return ["top level is not a JSON object"]
    errors = []
    if set(obj) != {"caption_short", "caption_detailed", "answers"}:
        errors.append("answer output has missing or extra fields")
    _check_captions(obj, limits, errors)
    answers = obj.get("answers")
    if not isinstance(answers, list):
        return errors + ["'answers' missing or not a list"]
    seen = []
    for i, a in enumerate(answers):
        if not isinstance(a, dict):
            errors.append(f"answers[{i}] is not an object")
            continue
        if set(a) != {"id", "answer"}:
            errors.append(f"answers[{i}] must contain only id and answer")
        seen.append(a.get("id"))
        _nonempty_str(a, "answer", f"answers[{i}].", errors)
        _check_words(a.get("answer"), f"answers[{a.get('id', i)}].answer", 1,
                     limits.answer_max_words, errors)
    missing = [i for i in ids if i not in seen]
    extra = [i for i in seen if i not in ids]
    dupes = [i for i in ids if seen.count(i) > 1]
    if missing:
        errors.append(f"unanswered question ids: {missing}")
    if extra:
        errors.append(f"unknown question ids: {extra}")
    if dupes:
        errors.append(f"question ids answered more than once: {dupes}")
    if not errors and questions is not None and gt is not None:
        errors.extend(numeric_errors(obj, questions, gt))
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


class ExamplesConfig(StrictModel):
    """Format/style examples for the annotator prompt. k examples are picked
    per sample, seeded by the sample id: every candidate sees the identical
    prompt for a given sample (fair comparison), while different samples get
    different examples (no single phrasing to overfit)."""
    path: Path = Path("configs/annotation/examples.yaml")
    k: int = Field(1, ge=1)
    questions: bool = True
    answers: bool = True


class ExampleQa(StrictModel):
    type: QaType
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class ExampleAnnotation(StrictModel):
    """One entry of the example pool (see configs/annotation/examples.yaml),
    stored as natural QA pairs and rendered in the exact task shape."""
    scene: str = Field(min_length=1, description="one-line scene setter shown "
                                                 "above the example")
    source_run: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    ground_truth: GroundTruth
    observations: dict[str, str] = Field(min_length=1)
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
        errors = validate_answers(ex.output(), ids, limits, ex.questions(), ex.ground_truth)
        if set(ex.observations) - set(CAMERAS):
            errors.append("observations contain an unknown camera")
        if not ex.ground_truth.sample_id.startswith(ex.source_run + "_"):
            errors.append("source_run does not match the source sample")
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
    run_id = sample_id.rsplit("_", 1)[0]
    eligible = [ex for ex in pool if ex.source_run != run_id]
    if len(eligible) < k:
        raise ValueError("not enough examples after excluding the target run")
    return random.Random(sample_id).sample(eligible, k)


def render_examples(examples: list[ExampleAnnotation], stage: str = "answers") -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        evidence = (f"Example {i}: {ex.scene}\n"
                    + "Reviewed image observations:\n"
                    + "\n".join(f"{cam}: {obs}" for cam, obs in ex.observations.items())
                    + "\n" + ex.ground_truth.prompt_block())
        if stage == "questions":
            output = {"questions": [{"type": q.type, "question": q.question} for q in ex.qa_pairs]}
            blocks.append(evidence + "\nQuestion-writer output:\n" + json.dumps(output))
        else:
            blocks.append(evidence + "\nQuestions to answer:\n" + question_listing(ex.questions())
                          + "\nAnswer output:\n" + json.dumps(ex.output()))
    return ("\nExamples of finished annotations from other runs. Follow their simple "
            "questions and grounded answers. Never copy their objects or numbers "
            "into the target scene. Observations below belong only to each example.\n\n"
            + "\n\n".join(blocks) + "\n")


# Cache identity includes implementation, source input, configuration and model.
# Cache identity tracks the production implementation, including validators.
CONTRACT_ID = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
RESULT_SCHEMA_VERSION = 2


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def input_id(payload: SamplePayload) -> str:
    return digest({"ground_truth": payload.gt.record(),
                   "images": {cam: hashlib.sha256(data).hexdigest()
                              for cam, data in payload.frames.items()}})


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _valid_cached_meta(meta) -> bool:
    if not isinstance(meta, dict) or not isinstance(meta.get("usage"), dict):
        return False
    attempts = meta.get("attempts")
    issues = meta.get("quality_issues", [])
    return (type(attempts) is int and attempts > 0 and isinstance(issues, list)
            and all(isinstance(issue, str) for issue in issues))


def cached_questions(path: Path, identity: str, counts: dict[str, int],
                     payload_id: str | None) -> dict | None:
    qs = read_json(path)
    if not qs or qs.get("identity") != identity or qs.get("input_id") != payload_id:
        return None
    questions = qs.get("questions")
    if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
        return None
    raw = {"questions": [{"type": q.get("type"), "question": q.get("question")} for q in questions]}
    if validate_questions(raw, counts):
        return None
    if [q.get("id") for q in questions] != question_ids(sum(counts.values())):
        return None
    if questions != purpose_questions(raw["questions"]) or not _valid_cached_meta(qs.get("meta")):
        return None
    return qs if qs.get("id") == question_set_id(questions) else None


def cached_result(path: Path, identity: str, qs: dict, limits: LimitsConfig,
                  payload: SamplePayload | None) -> bool:
    d = read_json(path)
    if not d or d.get("identity") != identity or d.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False
    if payload is None or d.get("input_id") != input_id(payload):
        return False
    question_set = d.get("question_set")
    if not isinstance(question_set, dict) or question_set.get("id") != qs.get("id"):
        return False
    if not _valid_cached_meta(d.get("meta")):
        return False
    ann = d.get("annotation")
    if not isinstance(ann, dict) or not isinstance(ann.get("qa_pairs"), list):
        return False
    pairs = ann["qa_pairs"]
    if not all(isinstance(p, dict) for p in pairs) or len(pairs) != len(qs["questions"]):
        return False
    if [{k: p.get(k) for k in ("id", "type", "question", "purpose")} for p in pairs] != [
            {k: q.get(k) for k in ("id", "type", "question", "purpose")} for q in qs["questions"]]:
        return False
    obj = {"caption_short": ann.get("caption_short"), "caption_detailed": ann.get("caption_detailed"),
           "answers": [{"id": p.get("id"), "answer": p.get("answer")} for p in pairs]}
    return (not validate_answers(obj, [q["id"] for q in qs["questions"]], limits,
                                 qs["questions"], payload.gt)
            and ann.get("action") == payload.gt.action_block()
            and d.get("ground_truth") == payload.gt.record())


def purpose_questions(raw: list[dict]) -> list[dict]:
    ordered = canonical_order(raw)
    used = Counter()
    out = []
    for qid, q in zip(question_ids(len(ordered)), ordered):
        qa_type = q["type"]
        index = used[qa_type]
        used[qa_type] += 1
        purposes = QUESTION_PURPOSES[qa_type]
        out.append({"id": qid, "type": qa_type, "question": q["question"].strip(),
                    "purpose": purposes[index] if index < len(purposes) else f"detail_{index + 1}"})
    return out


log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/annotation/local.yaml")
DEFAULT_REPO_ID = "VLA-uwo-2026/six_cam_1600x900"
PATH_PREFIX = "runs"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

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
    model: str = Field("auto", min_length=1)
    model_revision: str | None = None
    context_length: int = Field(32768, ge=4096)
    concurrency: int = Field(2, ge=1)
    timeout_sec: int = Field(300, ge=30)
    chat_template_kwargs: dict = Field(
        default_factory=lambda: {"enable_thinking": False})

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v):
        return v.rstrip("/")


class QuestionsConfig(StrictModel):
    counts: QaCounts = Field(default_factory=QaCounts)
    enable_thinking: bool = Field(
        True, description="thinking for the question-writer call only")
    max_tokens: Optional[int] = Field(
        10000, ge=256, description="token cap for question writing; "
                                  "null uses generation.max_tokens")


class AnnotateConfig(StrictModel):
    samples: SamplesConfig = Field(default_factory=SamplesConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    repo_id: str = DEFAULT_REPO_ID
    out_dir: Path = Path("data/annotations")
    examples: ExamplesConfig | None = None
    revision: str | None = None


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

def list_run_paths(api: HfApi, repo_id: str, revision: str | None = None) -> list[str]:
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
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
        available = list_run_paths(api, cfg.repo_id, cfg.revision)
        if path not in available:
            sys.exit(f"{path} not in repo; available: {available}")
        return [path]
    return list_run_paths(api, cfg.repo_id, cfg.revision)


def local_h5(repo_id: str, run_label: str, h5: Path | None,
             revision: str | None = None) -> str:
    if h5 is not None:
        return str(h5)
    log.info("downloading %s/%s", repo_id, run_label)
    return hf_hub_download(repo_id, run_label, repo_type="dataset", revision=revision)


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
        self.context_length = cfg.context_length

    def ping(self, model: str) -> None:
        req = urllib.request.Request(self._models_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            sys.exit(f"local server not reachable at {self._url}: {exc}\n"
                     "start it with scripts/serve_annotator_llama.sh")
        ids = [m.get("id") for m in body.get("data", [])]
        if model == "auto" and len(ids) == 1:
            model = ids[0]
        if model not in ids:
            sys.exit(f"served models {ids} do not include {model!r}; "
                     "set inference.model to the actual model id or auto for a single-model server")
        self.served_model = model
        info = next(m for m in body["data"] if m.get("id") == model)
        # Model-list timestamps change on every server restart. Keep model
        # properties in the cache identity, not the process creation time.
        self.model_metadata = {k: v for k, v in info.items() if k != "created"}
        self.provider = info.get("owned_by", "local")
        advertised_context = info.get("max_model_len") or (info.get("meta") or {}).get("n_ctx")
        if isinstance(advertised_context, int):
            self.context_length = advertised_context


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
                    raise RuntimeError(f"local server unreachable: {exc}") from exc
                log.warning("%s: %s; retry %s/%s in %ss",
                            type(exc).__name__, exc, i, self.HTTP_RETRIES - 1, 15 * i)
            time.sleep(15 * i)
        raise RuntimeError("local server unreachable")

    def call(self, model: str, system: str, content: list, json_schema: dict,
             validate, gen: GenerationConfig,
             chat_template_kwargs: dict | None = None) -> tuple[dict, dict]:
        if gen.max_tokens >= self.context_length:
            raise RuntimeError("output token budget leaves no room for images and prompt; increase context_length")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        response_format = {"type": "json_schema", "json_schema": json_schema}
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "reasoning_tokens": 0, "cost_usd": 0.0}
        errors_seen = []
        template_kwargs = (chat_template_kwargs if chat_template_kwargs is not None
                           else self._chat_template_kwargs)

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
                    "chat_template_kwargs": template_kwargs}
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
                if (exc.code == 400 and response_format.get("type") == "json_schema"
                        and any(x in detail.lower() for x in ("not support", "unsupported"))
                        and any(x in detail.lower() for x in ("json_schema", "response_format"))):
                    log.warning("json_schema rejected (%s); falling back to json_object",
                                detail[:120])
                    response_format = {"type": "json_object"}
                    continue
                raise RuntimeError(f"local server HTTP {exc.code}: {detail}") from exc

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
                             "provider": getattr(self, "provider", "local"), "errors_seen": errors_seen,
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

@dataclass(frozen=True)
class PromptContext:
    system: str
    id: str
    examples: dict | None


def prepare_prompt(stage: str, sample_id: str, counts: QaCounts,
                   limits: LimitsConfig, config: ExamplesConfig | None,
                   pool: list[ExampleAnnotation], pool_id: str | None) -> PromptContext:
    """Build prompt text and matching provenance from one example selection."""
    if stage == "questions":
        system = QUESTION_WRITER_SYSTEM.format(n_total=counts.total, counts_text=counts.text())
        prompt_id = QUESTION_WRITER_PROMPT_ID
    elif stage == "answers":
        system = ANNOTATOR_SYSTEM.format(n_total=counts.total, **limits.model_dump())
        prompt_id = ANNOTATOR_PROMPT_ID
    else:
        raise ValueError(f"unknown annotation stage: {stage}")
    record = None
    if config and getattr(config, stage) and pool:
        selected = select_examples(pool, config.k, sample_id)
        system += render_examples(selected, stage)
        prompt_id = _prompt_hash(prompt_id, pool_id)
        record = {"k": config.k, "pool_id": pool_id,
                  "scenes": [e.scene for e in selected]}
    return PromptContext(system, prompt_id, record)


class Annotator:
    def __init__(self, cfg: AnnotateConfig, client: VllmClient):
        self.cfg = cfg
        self.client = client
        self.failures: list[tuple[str, str]] = []
        self._fail_lock = threading.Lock()
        self.examples = load_examples(cfg.examples, cfg.limits) if cfg.examples else []
        self.examples_id = (
            digest({"config": cfg.examples.model_dump(mode="json"),
                    "pool": [e.model_dump(mode="json") for e in self.examples]})
            if cfg.examples else None)

    def _identity(self, stage: str) -> str:
        cfg = self.cfg
        return digest({"contract": CONTRACT_ID, "stage": stage,
                       "model": cfg.inference.model, "inference": cfg.inference.model_dump(),
                       "server": getattr(self.client, "model_metadata", {}),
                       "generation": cfg.generation.model_dump(),
                       "questions": cfg.questions.model_dump(), "limits": cfg.limits.model_dump(),
                       "examples_config": cfg.examples.model_dump() if cfg.examples else None,
                       "examples": [e.model_dump(mode="json") for e in self.examples],
                       "repo_id": cfg.repo_id, "revision": cfg.revision})

    def prompt(self, stage: str, sample_id: str) -> PromptContext:
        return prepare_prompt(stage, sample_id, self.cfg.questions.counts,
                              self.cfg.limits, self.cfg.examples,
                              self.examples, self.examples_id)

    def question_path(self, sample_id: str) -> Path:
        return self.cfg.out_dir / "questions" / f"{sample_id}.json"

    def result_path(self, sample_id: str) -> Path:
        suffix = self.cfg.inference.model.split("/")[-1]
        return self.cfg.out_dir / f"{sample_id}__{suffix}.json"

    def load_question_set(self, path: Path, payload: SamplePayload | None = None) -> dict | None:
        if payload is None:
            return None
        return cached_questions(path, self._identity("questions"),
                                self.cfg.questions.counts.as_dict(), input_id(payload))

    def write_question_set(self, payload: SamplePayload, path: Path) -> dict:
        counts = self.cfg.questions.counts
        prompt = self.prompt("questions", payload.gt.sample_id)
        content = user_content(payload, "Write the question set JSON now.")
        gen = self.cfg.generation
        if self.cfg.questions.max_tokens is not None:
            gen = gen.model_copy(update={"max_tokens": self.cfg.questions.max_tokens})
        obj, meta = self.client.call(
            self.cfg.inference.model, prompt.system, content,
            schema_questions(counts.total),
            lambda o: validate_questions(o, counts.as_dict()),
            gen,
            chat_template_kwargs={
                "enable_thinking": self.cfg.questions.enable_thinking})
        questions = purpose_questions(obj["questions"])
        qs = {"sample_id": payload.gt.sample_id, "model": self.cfg.inference.model,
              "question_prompt_id": prompt.id,
              "counts": counts.as_dict(),
              "id": question_set_id(questions), "questions": questions, "meta": meta,
              "examples": prompt.examples,
              "identity": self._identity("questions"), "input_id": input_id(payload)}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, qs)
        return qs

    def annotate(self, payload: SamplePayload, qs: dict) -> tuple[dict, dict]:
        questions = qs["questions"]
        ids = [q["id"] for q in questions]
        prompt = self.prompt("answers", payload.gt.sample_id)
        listing = question_listing(questions)
        content = user_content(payload, "Questions to answer:\n" + listing
                               + "\n\nWrite the annotation JSON now.")
        obj, meta = self.client.call(
            self.cfg.inference.model, prompt.system, content, schema_answers(ids),
            lambda o: validate_answers(o, ids, self.cfg.limits, questions, payload.gt),
            self.cfg.generation)
        meta["quality_issues"] = quality_issues(obj, questions)
        by_id = {a["id"]: a["answer"].strip() for a in obj["answers"]}
        return ({"caption_short": obj["caption_short"],
                 "caption_detailed": obj["caption_detailed"],
                 "qa_pairs": [{**q, "answer": by_id[q["id"]]} for q in questions]},
                meta)

    def result_is_current(self, path: Path, qs: dict, payload: SamplePayload | None = None) -> bool:
        return cached_result(path, self._identity("answers"), qs, self.cfg.limits, payload)

    def process_payload(self, payload: SamplePayload, frames_dir: Path,
                        force: bool = False, regenerate_questions: bool = False) -> None:
        gt = payload.gt
        frames_dir.mkdir(parents=True, exist_ok=True)
        for cam, data in payload.frames.items():
            frame_path = frames_dir / f"{gt.sample_id}_{cam}.jpg"
            if not frame_path.exists() or frame_path.read_bytes() != data:
                frame_path.write_bytes(data)

        stage = "questions"
        try:
            q_path = self.question_path(gt.sample_id)
            qs = None if regenerate_questions else self.load_question_set(q_path, payload)
            if qs is None:
                log.info("writing question set for %s (gt %s)", gt.sample_id, gt.action_label)
                qs = self.write_question_set(payload, q_path)
                log.info("  -> %s (set %s, %s attempt(s))",
                         q_path, qs["id"], qs["meta"]["attempts"])

            out_path = self.result_path(gt.sample_id)
            if not force and self.result_is_current(out_path, qs, payload):
                log.info("%s: current, skipping", gt.sample_id)
                return
            log.info("annotating %s (gt %s)", gt.sample_id, gt.action_label)
            stage = "answers"
            annotation, meta = self.annotate(payload, qs)
            annotation["action"] = gt.action_block()
            prompt = self.prompt("answers", gt.sample_id)
            atomic_json(out_path, {
                "schema_version": RESULT_SCHEMA_VERSION,
                "identity": self._identity("answers"), "input_id": input_id(payload),
                "model": self.cfg.inference.model,
                "prompt_id": prompt.id,
                "limits": self.cfg.limits.model_dump(),
                "qa_counts": self.cfg.questions.counts.as_dict(),
                "question_set": {"id": qs["id"], "model": qs["model"]},
                "examples": prompt.examples,
                "ground_truth": gt.record(),
                "annotation": annotation,
                "meta": meta,
            })
            log.info("  -> %s (%s attempt(s), %s out tokens)",
                     out_path, meta["attempts"], meta["usage"]["completion_tokens"])
        except RuntimeError as exc:
            log.error("FAILED %s %s: %s", stage, gt.sample_id, exc)
            with self._fail_lock:
                self.failures.append((gt.sample_id, str(exc)))

    def run_file(self, f: h5py.File, picks: list[int], force: bool,
                 regenerate_questions: bool) -> None:
        frames_dir = self.cfg.out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        h5_lock = threading.Lock()

        def process(sample):
            with h5_lock:
                payload = load_sample(f, sample, self.cfg.generation.image_width)
            self.process_payload(payload, frames_dir, force, regenerate_questions)

        workers = min(self.cfg.inference.concurrency, len(picks)) or 1
        if workers == 1:
            for sample in picks:
                process(sample)
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(process, sample)
                    for sample in picks]
            for fut in as_completed(futs):
                fut.result()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Annotate HF dataset samples with a local OpenAI-compatible endpoint.")
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
    return AnnotateConfig.model_validate({**cfg.model_dump(), "samples": samples.model_dump()})


def run(args) -> int:
    cfg = apply_cli_overrides(load_config(args.config), args)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    client = VllmClient(cfg.inference)
    client.ping(cfg.inference.model)
    cfg.inference.model = client.served_model
    annotator = Annotator(cfg, client)

    remaining = cfg.samples.limit
    api = None if args.h5 else HfApi()
    for run_path in resolve_run_labels(api, cfg, args.h5):
        if remaining is not None and remaining <= 0:
            break
        h5_path = local_h5(cfg.repo_id, run_path, args.h5, cfg.revision)
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
