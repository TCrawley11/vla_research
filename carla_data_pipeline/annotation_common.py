"""Shared annotation contract for local inference and the hosted benchmark.

Prompts, motion descriptions, examples and validation have a single source of
truth. Changes to this file invalidate cached annotations through CONTRACT_ID.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import tempfile
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args

import h5py
import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

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
