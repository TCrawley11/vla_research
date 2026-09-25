# vla_research

Headless, config-driven data collection from CARLA for VLA training.

## Structure

```
carla_data_pipeline/   the pipeline package (CLI: python -m carla_data_pipeline)
configs/               base.yaml, camera_spec/ rigs, scenarios/, annotation/
models/                local GGUF + mmproj (gitignored; serve_annotator.sh fills this)
data/runs/             output: <run_id>.h5 + <run_id>.json per run (schema: data/README.md)
tests/                 pytest suite (no CARLA needed)
```

## Usage

```sh
uv sync                # needs the CARLA 0.9.16 cp311 wheel path in pyproject.toml

# start the server headless
# (no -quality-level=Low: it makes load_world segfault on some towns,
#  e.g. Town03/Town05 - carla-simulator/carla#4940)
~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen -nosound &

# stage 1: capture a run from a scenario config
python -m carla_data_pipeline collect configs/scenarios/town10_light_traffic.yaml

# stage 2: build the sample groups into the run file (offline); when
# upload.enabled in configs/base.yaml this auto-runs stage 3 afterwards
python -m carla_data_pipeline build-samples run01

# stage 3 (usually automatic): upload the finished run to the private HF
# dataset repo (upload.repo_id), verify, then delete the local .h5.
# One-time setup: `hf auth login`. Backfill/retry: upload --all
python -m carla_data_pipeline upload run01

# viewing tool: rebuild per-camera mp4s from a run (needs ffmpeg)
python -m carla_data_pipeline export-video run01 --camera FRONT
```

## Annotation benchmark

Compares candidate VLM annotators (OpenRouter) on a fixed set of dataset
samples, driven by `configs/annotation/benchmark.yaml` (question author,
candidate models, samples, generation settings). Per sample the question
author (`questions.model`) writes one question set once; every candidate
answers exactly that set, so answers compare one-to-one across models.

```sh
OPENROUTER_API_KEY=... uv run python scripts/annotate_benchmark.py   # --h5 <local run> to skip the download
uv run python scripts/build_inspection.py                             # -> data/annotation_test/inspection.html
```

`collect --dry-run` validates and prints the resolved config without CARLA;
`collect --verify-only` connects and checks map/spawn/blueprints without
spawning. Full manual and config reference:

```sh
python -m carla_data_pipeline man
python -m carla_data_pipeline man config
```

## Local annotation (quantized 27B)

Six-camera samples from the team HF dataset, annotated by a local
OpenAI-compatible server. Question writing and answering use the same model.
The current local launcher uses the cached Qwen3.6-27B Q3_K_M and its BF16
vision projector through llama.cpp. It pins the cached model revision and
does not download weights. Set `LLAMA_SERVER`, `MODEL`, and `MMPROJ` to use
other executable or weight paths. A compatible vLLM endpoint also works;
`serve_annotator.sh` requires explicit model, tokenizer, and serving settings.

```sh
scripts/serve_annotator_llama.sh    # leave running in another terminal
python -m carla_data_pipeline annotate
python -m carla_data_pipeline annotate --h5 data/runs/run43.h5 --limit 1
uv run python scripts/build_inspection.py --dir data/annotations
```

Default walk is every `runs/*.h5`, samples 1..n-1 in order. `--run`,
`--indices`, and `--limit` slice that walk. Config:
`configs/annotation/local.yaml`.

The writer produces 18 short questions: 6 perception, 4 prediction,
4 planning, and 4 behaviour, each with an assigned purpose. Both stages
receive one of 12 reviewed examples from a different source run. No camera
has priority. Images describe the present scene; telemetry describes current
motion; waypoints describe recorded future motion. The schema-v2 action
fields use `linear_velocity_current` and `angular_velocity_current`, replacing
the misleading `*_target` names. Planning remains grounded in the future path.

JSON shape, type counts, word limits, duplicate IDs, and explicit current
numeric lookups are enforced with retries. Camera/agent coverage, repeated or
causal questions, and potential contradictions are review flags shown in the
inspection page. They are deliberately not automatic visual-accuracy gates.
Cache identities include input pixels, ground truth, shared logic,
configuration, model metadata, and examples; incomplete or malformed caches
regenerate automatically. The shipped config pins `inference.model_revision`;
change it whenever weights change under the same served alias.

### Frozen local evaluation

The preparation script exports compact samples from a pinned dataset revision
without downloading entire HDF5 runs. The repo keeps the 12 example sources;
the 20 evaluation samples are generated locally into `data/annotation_eval/inputs/eval`
and are not committed. Example and eval runs stay disjoint. Both variants
generate their own questions, so this compares the complete pipeline, not
answer quality on a shared question set. Keep the hosted benchmark for that
separate comparison.

```sh
# Required before a local eval run (writes holdout samples locally, not to git):
.venv/bin/python scripts/prepare_annotation_eval.py --revision 3e3d9aea3083e530505a47e596d493afe005719a

# Six varied scenes: slowing, right turn, stopped, highway acceleration,
# braking to a stop, and left turn. Run with the local server above.
.venv/bin/python scripts/run_local_annotation_eval.py \
  --sample-ids run06_000345,run06_000615,run07_000795,run08_000555,run09_000405,run10_000285 \
  --model-revision 5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q3_K_M.gguf \
  --out-dir data/annotation_eval/baseline_2026-09-18

# Omit --sample-ids for all 20 samples. --variant baseline or examples runs
# one variant; the default is both. Repeat the same command to resume.
```

`evaluation.json` records the selected inputs, resolved configuration, model,
completion counts, retries, and review flags after each sample. Each completed
variant has its own `inspection.html` with all six camera frames. Review front
objects, unsupported claims, repetitive questions, and ground-truth copying
against the images before scaling up. A successful schema check alone does
not establish annotation quality.
