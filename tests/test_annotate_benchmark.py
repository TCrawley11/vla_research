"""Config, prompt, ground-truth and validator checks for
scripts/annotate_benchmark.py (no network)."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from scripts import annotate_benchmark as ab

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "annotation" / "benchmark.yaml"


def _cfg(**overrides):
    raw = yaml.safe_load(CONFIG.read_text())
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return ab.BenchmarkConfig.model_validate(raw)


def _bench(cfg=None):
    return ab.Benchmark(cfg or _cfg(), client=None)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_shipped_config_loads():
    cfg = ab.load_config(CONFIG)
    assert cfg.questions.model == "qwen/qwen3.5-397b-a17b"
    assert cfg.questions.counts.as_dict() == {"perception": 6, "prediction": 4,
                                              "planning": 4, "behaviour": 4}
    assert cfg.questions.counts.total == 18
    assert cfg.questions.model not in cfg.models, "question author must not be a candidate"
    assert len(cfg.models) >= 2


@pytest.mark.parametrize("overrides, msg", [
    ({"bogus": 1}, "extra"),
    ({"questions.model": None}, "questions.model"),
    ({"questions.model": "  "}, "must not be empty"),
    ({"questions.model": "qwen/qwen3.5-27b"}, "also a candidate"),
    ({"questions.mode": "shared"}, "extra"),           # own/shared modes removed
    ({"prompt_version": "team-schema-v2"}, "extra"),   # versioning removed
    ({"models": ["a/b", "a/b"]}, "duplicates"),
    ({"samples.indices": [0, 3]}, ">= 1"),
    ({"samples.indices": []}, "must not be empty"),
    ({"questions.counts": {"perception": 0, "prediction": 0, "planning": 0, "behaviour": 0}},
     "at least one question"),
    ({"questions.counts": {"rest": 4}}, "extra"),
    ({"questions.per_type": 3}, "extra"),
])
def test_config_rejects(overrides, msg):
    with pytest.raises(Exception, match=msg):
        _cfg(**overrides)


def test_question_generation_override():
    cfg = _cfg()
    qg = cfg.question_generation()
    assert qg.max_tokens == 16000 and qg.temperature == cfg.generation.temperature
    assert cfg.generation.max_tokens != 16000
    plain = _cfg(**{"questions.generation": None})
    assert plain.question_generation() == plain.generation
    with pytest.raises(Exception, match="extra"):
        _cfg(**{"questions.generation": {"nope": 1}})


def test_questions_block_is_required():
    raw = yaml.safe_load(CONFIG.read_text())
    del raw["questions"]
    with pytest.raises(ValidationError, match="questions"):
        ab.BenchmarkConfig.model_validate(raw)


def test_qa_counts_helpers():
    c = ab.QaCounts(perception=6, prediction=4, planning=4, behaviour=0)
    assert c.total == 14
    assert c.text() == "6 perception, 4 prediction, 4 planning"
    assert list(c.as_dict()) == ab.QA_TYPES


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def _gt(**over):
    base = dict(sample_id="run43_k0100", sample_index=4, key_frame_id=100,
                action_label="STOP", past_action="FORWARD",
                trajectory_type="STOPPING", v=0.02, w=0.001,
                past_window_sec=1.5, horizon_sec=3.0, waypoint_period_sec=0.5,
                future_waypoints_ego_frame=[[0.05, 0.0]] * 6)
    base.update(over)
    return ab.GroundTruth(**base)


@pytest.mark.parametrize("over", [
    {"action_label": "LANE_KEEP"},              # never produced by the pipeline
    {"past_action": "vibes"},
    {"trajectory_type": "DONUT"},
    {"v": float("nan")},
    {"w": float("inf")},
    {"past_window_sec": 0.0},
    {"future_waypoints_ego_frame": []},
    {"future_waypoints_ego_frame": [[1.0, 2.0, 3.0]]},
    {"future_waypoints_ego_frame": [[1.0, float("nan")]]},
    {"sample_id": ""},
    {"sample_index": -1},
])
def test_ground_truth_rejects(over):
    with pytest.raises(ValidationError):
        _gt(**over)


def test_ground_truth_prompt_block():
    block = _gt().prompt_block()
    assert "action label: STOP (Stop and wait before continuing.)" in block
    assert "velocity: 0.02 m/s" in block
    assert "dominant action over the past 1.5 s: FORWARD" in block
    assert "stays essentially stationary" in block
    assert "map" not in block.lower()
    for cam in ab.CAMERAS:                       # all six cameras are attached
        assert cam in block


def test_ground_truth_action_block():
    stop = _gt().action_block()
    assert stop["linear_velocity_target"] == 0.0
    assert stop["angular_velocity_target"] == 0.0
    fwd = _gt(action_label="FORWARD", v=8.062, w=-0.0014).action_block()
    assert fwd == {"action_text": "Continue driving forward.",
                   "action_label": "FORWARD",
                   "linear_velocity_target": 8.06,
                   "angular_velocity_target": -0.001}


def test_ground_truth_record_is_json_ready():
    rec = _gt().record()
    assert rec["past_action"] == "FORWARD"
    assert rec["future_waypoints_ego_frame"][0] == [0.05, 0.0]
    assert "map_name" not in rec and "his_action" not in rec
    json.dumps(rec)


def test_past_indices_take_past_half_and_key():
    clip = np.array([94, 96, 98, 100, 102, 104, 106])   # key = 100, 3 each side
    assert ab.past_indices(clip).tolist() == [94, 96, 98, 100]


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

LIMITS = ab.LimitsConfig(answer_max_words=30, caption_short_max_words=25,
                         caption_detailed_min_words=30, caption_detailed_max_words=70)


def test_prompts_format_for_both_stages():
    counts = ab.QaCounts()
    qs = ab.QUESTION_WRITER_SYSTEM.format(n_total=counts.total, counts_text=counts.text())
    answers = ab.ANNOTATOR_SYSTEM.format(n_total=counts.total, **LIMITS.model_dump())
    assert "Write exactly 18 questions, 6 perception, 4 prediction, 4 planning, 4 behaviour" in qs
    assert "Write the question set for one sample" in qs
    assert "(18 items)" in answers
    assert "30 words each" in answers and "30-70 words" in answers
    assert '{"questions": [{"type": ..., "question": ...}, ...]}' in qs
    assert '"answers": one item per listed question' in answers
    for text in (qs, answers):
        assert "{{" not in text and "}}" not in text


def test_prompt_ids():
    assert ab.QUESTION_WRITER_PROMPT_ID != ab.ANNOTATOR_PROMPT_ID
    for pid in (ab.QUESTION_WRITER_PROMPT_ID, ab.ANNOTATOR_PROMPT_ID):
        assert len(pid) == 12 and int(pid, 16) >= 0


# --------------------------------------------------------------------------
# schemas + validators
# --------------------------------------------------------------------------

def test_schema_answers_enumerates_ids():
    ids = ab.question_ids(12)
    schema = ab.schema_answers(ids)["schema"]
    answers = schema["properties"]["answers"]
    assert answers["minItems"] == answers["maxItems"] == 12
    assert answers["items"]["properties"]["id"]["enum"] == ids
    assert ids[0] == "q01" and ids[-1] == "q12"
    # candidates answer by id only; they never author question text
    assert set(answers["items"]["properties"]) == {"id", "answer"}
    assert set(schema["properties"]) == {"caption_short", "caption_detailed", "answers"}


def test_no_own_mode_surface():
    """The benchmark is strictly two-stage (question writer -> annotators).
    Removed 2026-08-18; this guards against the single-pass "model writes
    its own questions and answers" path coming back under any name."""
    leaked = [n for n in dir(ab) if "own" in n.lower() and "download" not in n.lower()]
    assert leaked == [], f"own-mode symbols back in annotate_benchmark: {leaked}"
    for cls in (ab.QuestionsConfig, ab.BenchmarkConfig):
        assert "mode" not in cls.model_fields, f"{cls.__name__} grew a mode switch"
    with pytest.raises(ValidationError, match="extra"):
        _cfg(**{"questions.mode": "own"})
    # only two output schemas exist: questions (writer) and answers (annotator)
    schema_fns = sorted(n for n in dir(ab) if n.startswith("schema_"))
    assert schema_fns == ["schema_answers", "schema_questions"], schema_fns
    validators = sorted(n for n in dir(ab) if n.startswith("validate_"))
    assert validators == ["validate_answers", "validate_questions"], validators


def _answers(ids):
    return {"caption_short": "s", "caption_detailed": " ".join(["w"] * 40),
            "answers": [{"id": i, "answer": f"a {i}"} for i in ids]}


def test_validate_answers():
    ids = ab.question_ids(4)
    assert ab.validate_answers(_answers(ids), ids, LIMITS) == []
    bad = _answers(ids[:-1] + ["q09"])
    errors = ab.validate_answers(bad, ids, LIMITS)
    assert any("unanswered" in e and "q04" in e for e in errors)
    assert any("unknown" in e and "q09" in e for e in errors)
    dup = _answers(ids[:-1] + ["q01"])
    assert any("more than once" in e for e in ab.validate_answers(dup, ids, LIMITS))
    empty = _answers(ids)
    empty["answers"][0]["answer"] = "  "
    assert any("answers[0].'answer'" in e for e in ab.validate_answers(empty, ids, LIMITS))
    assert ab.validate_answers([], ids, LIMITS) == ["top level is not a JSON object"]


def test_word_limits():
    ids = ab.question_ids(2)
    ok = _answers(ids)
    ok["answers"][1]["answer"] = " ".join(["w"] * 31)
    errors = ab.validate_answers(ok, ids, LIMITS)
    assert errors == ["answers[q02].answer has 31 words (max 30)"]
    long_cap = _answers(ids)
    long_cap["caption_short"] = " ".join(["w"] * 26)
    long_cap["caption_detailed"] = " ".join(["w"] * 10)
    errors = ab.validate_answers(long_cap, ids, LIMITS)
    assert "caption_short has 26 words (max 25)" in errors
    assert "caption_detailed has 10 words (min 30)" in errors
    with pytest.raises(Exception, match="exceeds"):
        ab.LimitsConfig(caption_detailed_min_words=80, caption_detailed_max_words=70)


def test_trajectory_summary_reports_speed_profile():
    braking = np.array([[3.08, -0.17], [5.94, -0.43], [7.43, -0.64],
                        [7.43, -0.64], [7.43, -0.64], [7.43, -0.64]])
    s = ab.summarize_trajectory(braking, "STRAIGHT", 3.0, 0.5)
    assert "full stop after about 7.4 m" in s and "within about 1.5 s" in s
    steady = np.array([[4, -0.2], [8, -0.7], [12, -1.4], [15.9, -2.5], [19.7, -3.8], [23.4, -5.3]])
    assert "roughly steady 8." in ab.summarize_trajectory(steady, "RIGHT_CURVE", 3.0, 0.5)
    pull_away = np.array([[0, 0], [0, 0], [0.5, 0], [2, 0], [4, 0], [7, 0]])
    assert "pulling away from standstill after about 1.0 s" in ab.summarize_trajectory(pull_away, "STRAIGHT", 3.0, 0.5)
    slowing = np.array([[4, 0], [7, 0], [9, 0], [10.5, 0], [11.5, 0], [12, 0]])
    assert "slowing from about 8.0 m/s to about 1.0 m/s" in ab.summarize_trajectory(slowing, "STRAIGHT", 3.0, 0.5)
    still = np.zeros((6, 2))
    assert "stationary" in ab.summarize_trajectory(still, "STOPPING", 3.0, 0.5)


def test_validate_questions():
    two_each = dict.fromkeys(ab.QA_TYPES, 2)
    qs = {"questions": [{"type": t, "question": f"{t} {k}?"}
                        for t in ab.QA_TYPES for k in range(2)]}
    assert ab.validate_questions(qs, two_each) == []
    uneven = {"perception": 3, "prediction": 2, "planning": 2, "behaviour": 1}
    errors = ab.validate_questions(qs, uneven)
    assert "2 'perception' items (need exactly 3)" in errors
    assert "2 'behaviour' items (need exactly 1)" in errors
    assert not any("prediction" in e or "planning" in e for e in errors)
    qs["questions"][1]["question"] = qs["questions"][0]["question"]
    assert "duplicate questions" in ab.validate_questions(qs, two_each)
    qs["questions"][0]["type"] = "vibes"
    assert any("not in" in e for e in ab.validate_questions(qs, two_each))


# --------------------------------------------------------------------------
# question set cache + result staleness (Benchmark, no client needed)
# --------------------------------------------------------------------------

def test_question_set_cache_keys_on_question_prompt(tmp_path, monkeypatch):
    bench = _bench()
    path = tmp_path / "s.json"
    assert bench.load_question_set(path) is None
    qs = {"model": bench.cfg.questions.model,
          "question_prompt_id": ab.QUESTION_WRITER_PROMPT_ID,
          "counts": bench.cfg.questions.counts.as_dict(),
          "questions": [{"id": f"q{i:02d}", "type": "perception", "question": "x?"}
                        for i in range(1, 19)]}
    path.write_text(json.dumps(qs))
    assert bench.load_question_set(path) is not None
    # question-side prompt edit -> stale
    monkeypatch.setattr(ab, "QUESTION_WRITER_PROMPT_ID", "000000000000")
    assert bench.load_question_set(path) is None
    monkeypatch.undo()
    # answer-side change only -> still valid
    tighter = _bench(_cfg(**{"limits.answer_max_words": 20}))
    assert tighter.load_question_set(path) is not None
    fewer = _bench(_cfg(**{"questions.counts": {"perception": 3}}))
    assert fewer.cfg.questions.counts.total == 15
    assert fewer.load_question_set(path) is None


def test_question_set_id_depends_on_text_only():
    a = [{"id": "q01", "type": "perception", "question": "x?"}]
    b = [{"id": "q77", "type": "perception", "question": "x?"}]
    c = [{"id": "q01", "type": "perception", "question": "y?"}]
    assert ab.question_set_id(a) == ab.question_set_id(b) != ab.question_set_id(c)


def test_result_is_current(tmp_path):
    bench = _bench()
    cfg = bench.cfg
    qs = {"id": "abc123", "model": cfg.questions.model}
    model = cfg.models[0]
    path = tmp_path / "s__m.json"
    assert not bench.result_is_current(path, model, qs)
    base = {"model": model, "prompt_id": ab.ANNOTATOR_PROMPT_ID,
            "limits": cfg.limits.model_dump(),
            "qa_counts": cfg.questions.counts.as_dict(), "question_set": qs}
    path.write_text(json.dumps(base))
    assert bench.result_is_current(path, model, qs)
    assert not bench.result_is_current(path, model, {"id": "other"})
    assert not bench.result_is_current(path, cfg.models[1], qs)
    # pre-refactor own-mode file (no question set) -> stale
    no_set = dict(base)
    del no_set["question_set"]
    path.write_text(json.dumps(no_set))
    assert not bench.result_is_current(path, model, qs)
    # pre-refactor file: prompt_version instead of prompt_id -> stale
    old = dict(base)
    del old["prompt_id"]
    old["prompt_version"] = "team-schema-v2"
    path.write_text(json.dumps(old))
    assert not bench.result_is_current(path, model, qs)
    # tighter limits in the config -> the prompt changed -> stale
    path.write_text(json.dumps(base))
    tighter = _bench(_cfg(**{"limits.answer_max_words": 20}))
    assert not tighter.result_is_current(path, model, qs)
    more = _bench(_cfg(**{"questions.counts": {"perception": 8}}))
    assert not more.result_is_current(path, model, qs)


# --------------------------------------------------------------------------
# example pool
# --------------------------------------------------------------------------

EXAMPLE = {
    "scene": "toy scene",
    "caption_short": "Short caption.",
    "caption_detailed": " ".join(["word"] * 35),
    "qa_pairs": [{"type": "perception", "question": "What is visible?",
                  "answer": "A road."}],
}


def _pool_cfg(tmp_path, pool, k=1):
    p = tmp_path / "examples.yaml"
    p.write_text(yaml.safe_dump(pool))
    return ab.ExamplesConfig(path=p, k=k)


def test_examples_pool_validates(tmp_path):
    pool = ab.load_examples(_pool_cfg(tmp_path, [EXAMPLE]), LIMITS)
    assert pool[0].scene == "toy scene"
    assert pool[0].questions()[0]["id"] == "q01"
    too_long = dict(EXAMPLE, caption_short=" ".join(["w"] * 26))
    with pytest.raises(SystemExit, match="limits"):
        ab.load_examples(_pool_cfg(tmp_path, [too_long]), LIMITS)
    with pytest.raises(SystemExit, match="only 1"):
        ab.load_examples(_pool_cfg(tmp_path, [EXAMPLE], k=3), LIMITS)
    bad_type = dict(EXAMPLE, qa_pairs=[{"type": "vibes", "question": "q?",
                                        "answer": "a"}])
    with pytest.raises(ValidationError):
        ab.load_examples(_pool_cfg(tmp_path, [bad_type]), LIMITS)


def test_example_rotation_deterministic_per_sample(tmp_path):
    raw = [dict(EXAMPLE, scene=f"scene {i}") for i in range(5)]
    pool = ab.load_examples(_pool_cfg(tmp_path, raw, k=2), LIMITS)

    def pick(sid):
        return [e.scene for e in ab.select_examples(pool, 2, sid)]

    assert pick("run43_000165") == pick("run43_000165")
    assert len({tuple(pick(f"run43_{i:06d}")) for i in range(20)}) > 1


def test_render_examples_matches_task_shape(tmp_path):
    raw = [dict(EXAMPLE, scene=f"scene {i}") for i in range(2)]
    pool = ab.load_examples(_pool_cfg(tmp_path, raw, k=2), LIMITS)
    text = ab.render_examples(pool)
    assert "Example 1 (scene: scene 0):" in text
    assert "q01 [perception] What is visible?" in text
    assert '"answers"' in text and '"caption_detailed"' in text
    assert "never copy facts" in text


def test_examples_change_prompt_id_and_stale_results(tmp_path):
    raw = [dict(EXAMPLE, scene=f"scene {i}") for i in range(3)]
    pool_path = tmp_path / "examples.yaml"
    pool_path.write_text(yaml.safe_dump(raw))
    plain = _bench()
    assert plain.prompt_id() == ab.ANNOTATOR_PROMPT_ID
    assert plain._examples_suffix("run43_000165") == ""
    with_ex = _bench(_cfg(examples={"path": str(pool_path), "k": 2}))
    assert with_ex.prompt_id() != plain.prompt_id()
    assert "Examples of finished annotations" in with_ex._examples_suffix("run43_000165")
    # a result produced without examples goes stale once examples are enabled
    qs = {"id": "abc123", "model": plain.cfg.questions.model}
    path = tmp_path / "s__m.json"
    path.write_text(json.dumps({
        "model": plain.cfg.models[0], "prompt_id": plain.prompt_id(),
        "limits": plain.cfg.limits.model_dump(),
        "qa_counts": plain.cfg.questions.counts.as_dict(),
        "question_set": qs}))
    assert plain.result_is_current(path, plain.cfg.models[0], qs)
    assert not with_ex.result_is_current(path, plain.cfg.models[0], qs)


def test_shipped_examples_pool_loads():
    cfg = ab.ExamplesConfig(path=REPO / "configs" / "annotation" / "examples.yaml", k=2)
    pool = ab.load_examples(cfg, ab.LimitsConfig())
    assert len(pool) >= 3
    scenes = [e.scene for e in pool]
    assert len(set(scenes)) == len(scenes)
