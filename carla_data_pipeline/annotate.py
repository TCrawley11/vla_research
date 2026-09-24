"""Stage 4: annotate dataset samples with a local OpenAI-compatible VLM.

Walks the team Hugging Face dataset sequentially (every `runs/*.h5`, samples
1..n-1; index 0 is the spawn frame). Each sample is six key-frame cameras plus
a ground-truth block. The same local model writes the question set and then
answers it. Output is enforced with `response_format` json_schema, code-side
validators, and retries with feedback.

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
import re
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

from carla_data_pipeline.annotation_common import (
    StrictModel,
    GenerationConfig,
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
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
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
# output checks
# --------------------------------------------------------------------------



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

class Annotator:
    def __init__(self, cfg: AnnotateConfig, client: VllmClient):
        self.cfg = cfg
        self.client = client
        self.failures: list[tuple[str, str]] = []
        self._h5_lock = threading.Lock()
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

    def _examples_suffix(self, sample_id: str, stage: str) -> str:
        if not self.examples or not getattr(self.cfg.examples, stage):
            return ""
        return render_examples(select_examples(self.examples, self.cfg.examples.k, sample_id), stage)

    def prompt_id(self, stage: str) -> str:
        base = QUESTION_WRITER_PROMPT_ID if stage == "questions" else ANNOTATOR_PROMPT_ID
        if not self.examples_id or not getattr(self.cfg.examples, stage):
            return base
        return _prompt_hash(base, self.examples_id)

    def example_record(self, sample_id: str, stage: str) -> dict | None:
        if not self.examples or not getattr(self.cfg.examples, stage):
            return None
        selected = select_examples(self.examples, self.cfg.examples.k, sample_id)
        return {"k": self.cfg.examples.k, "pool_id": self.examples_id,
                "scenes": [e.scene for e in selected]}


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
        system = QUESTION_WRITER_SYSTEM.format(n_total=counts.total,
                                               counts_text=counts.text())
        system += self._examples_suffix(payload.gt.sample_id, "questions")
        content = user_content(payload, "Write the question set JSON now.")
        gen = self.cfg.generation
        if self.cfg.questions.max_tokens is not None:
            gen = gen.model_copy(update={"max_tokens": self.cfg.questions.max_tokens})
        obj, meta = self.client.call(
            self.cfg.inference.model, system, content,
            schema_questions(counts.total),
            lambda o: validate_questions(o, counts.as_dict()),
            gen,
            chat_template_kwargs={
                "enable_thinking": self.cfg.questions.enable_thinking})
        questions = purpose_questions(obj["questions"])
        qs = {"sample_id": payload.gt.sample_id, "model": self.cfg.inference.model,
              "question_prompt_id": self.prompt_id("questions"),
              "counts": counts.as_dict(),
              "id": question_set_id(questions), "questions": questions, "meta": meta,
              "examples": self.example_record(payload.gt.sample_id, "questions"),
              "identity": self._identity("questions"), "input_id": input_id(payload)}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, qs)
        return qs

    def annotate(self, payload: SamplePayload, qs: dict) -> tuple[dict, dict]:
        questions = qs["questions"]
        ids = [q["id"] for q in questions]
        system = ANNOTATOR_SYSTEM.format(n_total=len(ids),
                                         **self.cfg.limits.model_dump())
        system += self._examples_suffix(payload.gt.sample_id, "answers")
        listing = question_listing(questions)
        content = user_content(payload, "Questions to answer:\n" + listing
                               + "\n\nWrite the annotation JSON now.")
        obj, meta = self.client.call(
            self.cfg.inference.model, system, content, schema_answers(ids),
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

    def process_sample(self, f: h5py.File, sample: int, frames_dir: Path,
                       force: bool, regenerate_questions: bool) -> None:
        with self._h5_lock:
            payload = load_sample(f, sample, self.cfg.generation.image_width)
        self.process_payload(payload, frames_dir, force, regenerate_questions)

    def process_payload(self, payload: SamplePayload, frames_dir: Path,
                        force: bool = False, regenerate_questions: bool = False) -> None:
        gt = payload.gt
        for cam, data in payload.frames.items():
            frame_path = frames_dir / f"{gt.sample_id}_{cam}.jpg"
            if not frame_path.exists() or frame_path.read_bytes() != data:
                frame_path.write_bytes(data)

        q_path = self.question_path(gt.sample_id)
        qs = None if regenerate_questions else self.load_question_set(q_path, payload)
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
        if not force and self.result_is_current(out_path, qs, payload):
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
        atomic_json(out_path, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": self._identity("answers"), "input_id": input_id(payload),
            "model": self.cfg.inference.model,
            "prompt_id": self.prompt_id("answers"),
            "limits": self.cfg.limits.model_dump(),
            "qa_counts": self.cfg.questions.counts.as_dict(),
            "question_set": {"id": qs["id"], "model": qs["model"]},
            "examples": self.example_record(gt.sample_id, "answers"),
            "ground_truth": gt.record(),
            "annotation": annotation,
            "meta": meta,
        })
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
