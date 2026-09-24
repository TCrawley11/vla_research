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
import re
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
from carla_data_pipeline.annotation_common import (
    StrictModel,
    LimitsConfig,
    QaCounts,
    _prompt_hash,
    speed_profile,
    summarize_trajectory,
    GroundTruth,
    SamplePayload,
    encode_jpeg,
    past_indices,
    load_sample,
    user_content,
    _strict,
    _array,
    schema_questions,
    schema_answers,
    _nonempty_str,
    _check_words,
    _check_captions,
    _check_typed_items,
    _is_gt_lookup,
    _lookup_cap_errors,
    _is_steer_and_hold,
    _planning_paraphrase_errors,
    _perception_slot_errors,
    validate_questions,
    _caption_blob,
    _weather_polarity,
    _pedestrian_polarity,
    _caption_content_errors,
    _contradiction_errors,
    validate_answers,
    canonical_order,
    question_ids,
    question_listing,
    question_set_id,
    ExamplesConfig,
    ExampleQa,
    ExampleAnnotation,
    load_examples,
    select_examples,
    render_examples,
    CAMERAS,
    QA_TYPES,
    QaType,
    ActionLabel,
    TrajectoryType,
    ACTION_TEXT,
    QUESTION_WRITER_SYSTEM,
    ANNOTATOR_SYSTEM,
    USER_GT,
    QUESTION_WRITER_PROMPT_ID,
    ANNOTATOR_PROMPT_ID
)

from carla_data_pipeline.annotation_common import (
    CONTRACT_ID, RESULT_SCHEMA_VERSION, atomic_json, read_json, digest, input_id,
    cached_questions, cached_result, purpose_questions, quality_issues,
)

DEFAULT_CONFIG = Path("configs/annotation/benchmark.yaml")
DEFAULT_REPO_ID = "VLA-uwo-2026/six_cam_1600x900"
PATH_PREFIX = "runs"
# We need to use all 6 cameras - as instructed by professors


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

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
    revision: str | None = None

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

def pick_run(api: HfApi, repo_id: str, rng: random.Random, run_id: str | None,
             revision: str | None = None) -> str:
    """Return the repo path of the run .h5 to benchmark on."""
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
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
                if (exc.code == 400 and response_format.get("type") == "json_schema"
                        and any(x in detail.lower() for x in ("not support", "unsupported"))
                        and any(x in detail.lower() for x in ("json_schema", "response_format"))):
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

    def _identity(self, stage: str, model: str | None = None) -> str:
        cfg = self.cfg
        return digest({"contract": CONTRACT_ID, "stage": stage,
                       "model": model or cfg.questions.model,
                       "generation": cfg.generation.model_dump(),
                       "questions": cfg.questions.model_dump(), "limits": cfg.limits.model_dump(),
                       "examples_config": cfg.examples.model_dump() if cfg.examples else None,
                       "examples": [e.model_dump(mode="json") for e in self.examples or []],
                       "repo_id": cfg.repo_id, "revision": cfg.revision})


    # -- paths ------------------------------------------------------------

    def question_path(self, sample_id: str) -> Path:
        return self.cfg.out_dir / "questions" / f"{sample_id}.json"

    def result_path(self, sample_id: str, model: str) -> Path:
        return self.cfg.out_dir / f"{sample_id}__{model.split('/')[-1]}.json"

    # -- prompt identity + examples ----------------------------------------

    def prompt_id(self, stage: str = "answers") -> str:
        """Hash the stage template plus enabled example-pool settings."""
        base = QUESTION_WRITER_PROMPT_ID if stage == "questions" else ANNOTATOR_PROMPT_ID
        if not self.examples_id or not getattr(self.cfg.examples, stage):
            return base
        return _prompt_hash(base, self.examples_id)

    def _examples_suffix(self, sample_id: str, stage: str = "answers") -> str:
        if not self.examples or not getattr(self.cfg.examples, stage):
            return ""
        return render_examples(
            select_examples(self.examples, self.cfg.examples.k, sample_id), stage)

    def example_record(self, sample_id: str, stage: str) -> dict | None:
        if not self.examples or not getattr(self.cfg.examples, stage):
            return None
        selected = select_examples(self.examples, self.cfg.examples.k, sample_id)
        return {"k": self.cfg.examples.k, "pool_id": self.examples_id,
                "scenes": [e.scene for e in selected]}

    # -- stage 1: question sets --------------------------------------------

    def load_question_set(self, path: Path, payload: SamplePayload | None = None) -> dict | None:
        if payload is None:
            return None
        return cached_questions(path, self._identity("questions"),
                                self.cfg.questions.counts.as_dict(), input_id(payload))

    def write_question_set(self, payload: SamplePayload, path: Path) -> dict:
        counts = self.cfg.questions.counts
        system = QUESTION_WRITER_SYSTEM.format(n_total=counts.total,
                                         counts_text=counts.text())
        system += self._examples_suffix(payload.gt.sample_id, "questions")
        content = user_content(payload, "Write the question set JSON now.")
        obj, meta = self.client.call(
            self.cfg.questions.model, system, content,
            schema_questions(counts.total),
            lambda o: validate_questions(o, counts.as_dict()),
            self.cfg.question_generation())
        questions = purpose_questions(obj["questions"])
        qs = {"sample_id": payload.gt.sample_id, "model": self.cfg.questions.model,
              "question_prompt_id": self.prompt_id("questions"),
              "counts": counts.as_dict(),
              "id": question_set_id(questions), "questions": questions, "meta": meta,
              "examples": self.example_record(payload.gt.sample_id, "questions"),
              "identity": self._identity("questions"), "input_id": input_id(payload)}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, qs)
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
            lambda o: validate_answers(o, ids, self.cfg.limits, questions, payload.gt),
            self.cfg.generation)
        meta["quality_issues"] = quality_issues(obj, questions)
        by_id = {a["id"]: a["answer"].strip() for a in obj["answers"]}
        return ({"caption_short": obj["caption_short"],
                 "caption_detailed": obj["caption_detailed"],
                 "qa_pairs": [{**q, "answer": by_id[q["id"]]} for q in questions]},
                meta)

    def result_is_current(self, path: Path, model: str, qs: dict,
                          payload: SamplePayload | None = None) -> bool:
        return cached_result(path, self._identity("answers", model), qs, self.cfg.limits, payload)

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
                if not frame_path.exists() or frame_path.read_bytes() != data:
                    frame_path.write_bytes(data)

            q_path = self.question_path(gt.sample_id)
            qs = None if regenerate_questions else self.load_question_set(q_path, payload)
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
                if not force and self.result_is_current(out_path, model, qs, payload):
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
                atomic_json(out_path, {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "identity": self._identity("answers", model), "input_id": input_id(payload),
                    "model": model,
                    "prompt_id": self.prompt_id(),
                    "limits": self.cfg.limits.model_dump(),
                    "qa_counts": self.cfg.questions.counts.as_dict(),
                    "question_set": {"id": qs["id"], "model": qs["model"]},
                    "examples": self.example_record(gt.sample_id, "answers"),
                    "ground_truth": gt.record(),
                    "annotation": annotation,
                    "meta": meta,
                })
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
        run_path = pick_run(HfApi(), cfg.repo_id, rng, cfg.samples.run, cfg.revision)
        print(f"downloading {cfg.repo_id}/{run_path} ...")
        h5_path = hf_hub_download(cfg.repo_id, run_path, repo_type="dataset", revision=cfg.revision)

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
