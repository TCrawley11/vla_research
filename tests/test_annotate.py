"""Config, prompt, validator and sample-walk checks for
carla_data_pipeline.annotate (no network, no GPU)."""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from carla_data_pipeline import annotate as an

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "annotation" / "local.yaml"


def _cfg(**overrides):
    raw = yaml.safe_load(CONFIG.read_text())
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return an.AnnotateConfig.model_validate(raw)


def _args(**over):
    base = dict(config=CONFIG, h5=None, run=None, indices=None, limit=None,
                force=False, regenerate_questions=False)
    base.update(over)
    return Namespace(**base)


def test_shipped_config_loads():
    cfg = an.load_config(CONFIG)
    assert cfg.inference.model == "qwen3.5-9b-q6_k"
    assert cfg.samples.run is None and cfg.samples.indices is None
    assert cfg.questions.counts.as_dict() == {
        "perception": 6, "prediction": 3, "planning": 4, "behaviour": 1}
    assert cfg.questions.counts.total == 14
    assert cfg.inference.chat_template_kwargs["enable_thinking"] is False
    assert cfg.questions.enable_thinking is True
    assert cfg.questions.max_tokens == 8000


def test_smoke_config_spread_indices():
    smoke = an.load_config(REPO / "configs" / "annotation" / "smoke.yaml")
    assert smoke.samples.run == "run43"
    assert smoke.samples.indices == [4, 8, 12]
    assert min(smoke.samples.indices) >= 4
    assert smoke.inference.model == an.load_config(CONFIG).inference.model
    assert smoke.questions.enable_thinking is True
    assert smoke.inference.chat_template_kwargs["enable_thinking"] is False
    assert smoke.questions.max_tokens == 8000


@pytest.mark.parametrize("overrides, msg", [
    ({"bogus": 1}, "extra"),
    ({"samples.indices": [0, 3]}, ">= 1"),
    ({"samples.indices": []}, "must not be empty"),
    ({"questions.counts": {"perception": 0, "prediction": 0,
                           "planning": 0, "behaviour": 0}},
     "at least one question"),
    ({"questions.counts": {"rest": 4}}, "extra"),
    ({"inference.concurrency": 0}, "greater than or equal"),
])
def test_config_rejects(overrides, msg):
    with pytest.raises(Exception, match=msg):
        _cfg(**overrides)


def test_indices_require_run_or_h5():
    cfg = _cfg(**{"samples.indices": [4, 5]})
    with pytest.raises(SystemExit, match="indices requires"):
        an.apply_cli_overrides(cfg, _args())
    out = an.apply_cli_overrides(cfg, _args(h5=Path("data/runs/run43.h5")))
    assert out.samples.indices == [4, 5]
    out = an.apply_cli_overrides(_cfg(), _args(run="run43", indices="4,8"))
    assert out.samples.run == "run43" and out.samples.indices == [4, 8]


def test_cli_limit_and_run_overrides():
    cfg = an.apply_cli_overrides(_cfg(), _args(run="run07", limit=3))
    assert cfg.samples.run == "run07" and cfg.samples.limit == 3


def test_pick_samples_sequential_skips_spawn():
    cfg = an.SamplesConfig()
    assert an.pick_samples(5, cfg, "runs/run43.h5") == [1, 2, 3, 4]


def test_pick_samples_explicit_indices():
    cfg = an.SamplesConfig(run="run43", indices=[4, 2, 8])
    assert an.pick_samples(10, cfg, "runs/run43.h5") == [2, 4, 8]
    with pytest.raises(SystemExit, match="out of range"):
        an.pick_samples(5, cfg, "runs/run43.h5")
    with pytest.raises(SystemExit, match="spawn"):
        an.pick_samples(1, an.SamplesConfig(), "runs/run43.h5")


def test_qa_counts_helpers():
    c = an.QaCounts(perception=6, prediction=4, planning=4, behaviour=0)
    assert c.total == 14
    assert c.text() == "6 perception, 4 prediction, 4 planning"
    assert list(c.as_dict()) == an.QA_TYPES


def _gt(**over):
    base = dict(sample_id="run43_k0100", sample_index=4, key_frame_id=100,
                action_label="STOP", past_action="FORWARD",
                trajectory_type="STOPPING", v=0.02, w=0.001,
                past_window_sec=1.5, horizon_sec=3.0, waypoint_period_sec=0.5,
                future_waypoints_ego_frame=[[0.05, 0.0]] * 6)
    base.update(over)
    return an.GroundTruth(**base)


def test_ground_truth_prompt_and_action():
    block = _gt().prompt_block()
    assert "current action label (coarse motion class, not a command): STOP" in block
    for cam in an.CAMERAS:
        assert cam in block
    stop = _gt().action_block()
    assert stop["linear_velocity_current"] == 0.02
    fwd = _gt(action_label="FORWARD", v=8.062, w=-0.0014).action_block()
    assert fwd["linear_velocity_current"] == 8.06
    assert fwd["angular_velocity_current"] == -0.001


def test_past_indices_take_past_half_and_key():
    clip = np.array([94, 96, 98, 100, 102, 104, 106])
    assert an.past_indices(clip).tolist() == [94, 96, 98, 100]


LIMITS = an.LimitsConfig()
FULL_COUNTS = dict(perception=6, prediction=4, planning=4, behaviour=4)


def _ok_detailed(min_words=30):
    head = ("FRONT shows a residential street with the road curving right. "
            "BACK looks over a junction just passed. No pedestrians are "
            "visible in any camera.")
    words = head.split()
    if len(words) < min_words:
        words += ["stone"] * (min_words - len(words))
    return " ".join(words)


def _ok_answers(ids):
    return {"caption_short": "Ego continues along a residential curve.",
            "caption_detailed": _ok_detailed(),
            "answers": [{"id": i, "answer": f"a {i}"} for i in ids]}


def _valid_question_set(**over):
    qs = {
        "perception": [
            "Are any pedestrians or other road users visible?",
            "Are any traffic lights or traffic signs visible?",
            "What does the BACK camera show behind the ego vehicle?",
            "What is the weather condition in the FRONT view?",
            "Are lane markings visible on the road ahead?",
            "What kind of buildings stand on the left?",
        ],
        "prediction": [
            "Does the recorded action fit the curve visible ahead?",
            "What should the ego watch for while following the trajectory?",
            "How does the future path relate to the road layout?",
            "Is the ego expected to stop within the horizon?",
        ],
        "planning": [
            "What maneuver should the ego vehicle take next?",
            "What speed or timing should it use for that maneuver?",
            "What should it watch for while executing the action?",
            "Why does the recorded action fit what is visible ahead?",
        ],
        "behaviour": [
            "What is the ego vehicle's current speed?",
            "Is the ego currently turning or going straight?",
            "Does the current motion match the visible road curve?",
            "How does the current motion relate to the road ahead?",
        ],
    }
    qs.update(over)
    return {"questions": [{"type": t, "question": q}
                          for t, items in qs.items() for q in items]}


def test_schema_and_validators():
    ids = an.question_ids(4)
    schema = an.schema_answers(ids)["schema"]
    assert schema["properties"]["answers"]["items"]["properties"]["id"]["enum"] == ids
    qs = {"questions": [{"type": t, "question": f"{t} {k}?"}
                        for t in an.QA_TYPES for k in range(2)]}
    two_each = dict.fromkeys(an.QA_TYPES, 2)
    assert an.validate_questions(qs, two_each) == []
    answers = _ok_answers(ids)
    assert an.validate_answers(answers, ids, LIMITS) == []
    answers["answers"][0]["answer"] = " ".join(["w"] * 31)
    assert any("31 words" in e for e in an.validate_answers(answers, ids, LIMITS))


def test_validate_questions_accepts_slotted_set():
    assert an.validate_questions(_valid_question_set(), FULL_COUNTS) == []


def test_validate_questions_requires_perception_slots():
    qs = _valid_question_set(perception=[
        "What is the weather in the FRONT view?",
        "Are lane markings visible on the road ahead?",
        "What kind of buildings stand on the left?",
        "What is the lighting like?",
        "Is the pavement wet or dry?",
        "How many lanes are visible ahead?",
    ])
    errors = an._perception_slot_errors(qs["questions"], FULL_COUNTS)
    assert an.validate_questions(qs, FULL_COUNTS) == []
    assert "perception questions omit the road-users slot" in errors
    assert "perception questions omit the traffic-signals slot" in errors
    assert "perception questions omit a named non-FRONT camera" in errors


def test_validate_questions_lookup_cap():
    qs = _valid_question_set(behaviour=[
        "What is the ego vehicle's current speed?",
        "What is the ego angular velocity?",
        "What is the current driving action label?",
        "What was the past action over the last second?",
    ])
    errors = an._lookup_cap_errors(qs["questions"])
    assert an.validate_questions(qs, FULL_COUNTS) == []
    assert any("behaviour" in e and "ground-truth lookups" in e for e in errors)


def test_validate_questions_planning_paraphrases():
    qs = _valid_question_set(planning=[
        "Steer right and hold 8 m/s.",
        "Turn right while holding the same speed.",
        "Continue right and maintain speed.",
        "Keep steering right at the current speed.",
    ])
    errors = an._planning_paraphrase_errors(qs["questions"])
    assert an.validate_questions(qs, FULL_COUNTS) == []
    assert any("steer-and-hold" in e for e in errors)


def test_validate_answers_requires_camera_and_agent_claim():
    ids = an.question_ids(4)
    answers = _ok_answers(ids)
    answers["caption_detailed"] = " ".join(["word"] * 40)
    errors = an._caption_content_errors(answers)
    assert an.validate_answers(answers, ids, LIMITS) == []
    assert "caption_detailed does not name a camera" in errors
    assert any("dynamic agent" in e for e in errors)


def test_semantic_hints_do_not_force_rewrites():
    ids = an.question_ids(4)
    answers = _ok_answers(ids)
    answers["caption_short"] = "The road ahead looks clear."
    answers["answers"][0]["answer"] = "The scene is foggy with poor visibility."
    assert an.validate_answers(answers, ids, LIMITS) == []
    assert an._contradiction_errors(answers, answers["answers"]) == []


def test_parse_json_content():
    assert an.parse_json_content('{"a": 1}') == {"a": 1}
    assert an.parse_json_content("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_prompt_ids_stable_and_distinct():
    assert an.QUESTION_WRITER_PROMPT_ID != an.ANNOTATOR_PROMPT_ID
    for pid in (an.QUESTION_WRITER_PROMPT_ID, an.ANNOTATOR_PROMPT_ID):
        assert len(pid) == 12 and int(pid, 16) >= 0


def test_prompts_format():
    counts = an.QaCounts()
    qs = an.QUESTION_WRITER_SYSTEM.format(n_total=counts.total, counts_text=counts.text())
    answers = an.ANNOTATOR_SYSTEM.format(n_total=counts.total, **LIMITS.model_dump())
    assert "Write exactly 14 questions" in qs
    assert "{{" not in qs and "}}" not in qs
    assert "(14 items)" in answers
    assert "camera other than FRONT" in qs
    assert "visible dynamic agents" in answers
