# vla_research

Headless, config-driven data collection from CARLA for VLA training.

## Structure

```
carla_data_pipeline/   the pipeline package (CLI: python -m carla_data_pipeline)
configs/               base.yaml, camera_spec/ rigs, scenarios/ to collect with
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
