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

# stage 2: build the sample groups into the run file (offline)
python -m carla_data_pipeline build-samples run01
```

`collect --dry-run` validates and prints the resolved config without CARLA;
`collect --verify-only` connects and checks map/spawn/blueprints without
spawning. Full manual and config reference:

```sh
python -m carla_data_pipeline man
python -m carla_data_pipeline man config
```
