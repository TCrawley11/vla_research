"""Schema + loader tests: the shipped example configs parse cleanly, and each
invalid fixture raises a ValidationError naming the offending field."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from carla_data_pipeline.config.load import ConfigError, load_collect_config
from carla_data_pipeline.config.schema import CollectConfig

REPO = Path(__file__).resolve().parents[1]

RIG = {"cameras": [{"name": "FRONT", "fov": 70}]}


def minimal(**overrides):
    cfg = {
        "scenario": {"map_name": "Town10HD_Opt"},
        "capture": {"stop": {"duration_sec": 60}},
        "camera_spec": RIG,
    }
    cfg.update(overrides)
    return cfg


def test_example_scenario_loads():
    cfg = load_collect_config(REPO / "configs/scenarios/town10_light_traffic.yaml")
    assert cfg.scenario.map_name == "Town10HD_Opt"
    assert cfg.capture.raw_fps == 30                # from base.yaml
    assert cfg.capture.stop.duration_sec == 30      # scenario wins the merge
    assert cfg.seed is None                         # drawn per run, recorded in sidecar
    assert cfg.traffic.enabled and cfg.traffic.num_vehicles == 20
    assert [c.name for c in cfg.camera_spec.cameras] == [
        "FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
    assert cfg.upload.enabled and cfg.upload.auto and cfg.upload.delete_local_h5
    assert cfg.upload.repo_id == "VLA-uwo-2026/six_cam_1600x900"


def test_unknown_key_rejected():
    with pytest.raises(ValidationError, match="num_vehicels"):
        CollectConfig.model_validate(
            minimal(traffic={"enabled": True, "num_vehicels": 5}))


@pytest.mark.parametrize("stop", [{}, {"duration_sec": 60, "num_frames": 100}])
def test_stop_condition_exactly_one(stop):
    with pytest.raises(ValidationError, match="exactly one"):
        CollectConfig.model_validate(minimal(capture={"stop": stop}))


def test_fps_must_divide():
    with pytest.raises(ValidationError, match="divisible"):
        CollectConfig.model_validate(
            minimal(capture={"stop": {"duration_sec": 60}, "sample_fps": 7}))


def test_run_too_short_for_one_sample():
    with pytest.raises(ValidationError, match="too short"):
        CollectConfig.model_validate(minimal(capture={"stop": {"duration_sec": 2}}))


def test_traffic_enabled_needs_actors():
    with pytest.raises(ValidationError, match="num_vehicles"):
        CollectConfig.model_validate(minimal(traffic={"enabled": True}))


def test_traffic_disabled_forbids_counts():
    with pytest.raises(ValidationError, match="must be 0"):
        CollectConfig.model_validate(minimal(traffic={"num_vehicles": 3}))


def test_rig_needs_front_camera():
    with pytest.raises(ValidationError, match="FRONT"):
        CollectConfig.model_validate(
            minimal(camera_spec={"cameras": [{"name": "BACK", "fov": 70}]}))


def test_rig_rejects_duplicate_names():
    with pytest.raises(ValidationError, match="duplicate"):
        CollectConfig.model_validate(minimal(camera_spec={
            "cameras": [{"name": "FRONT", "fov": 70}, {"name": "FRONT", "fov": 90}]}))


def test_camera_resolution_must_be_even():
    with pytest.raises(ValidationError, match="even"):
        CollectConfig.model_validate(minimal(camera_spec={
            "cameras": [{"name": "FRONT", "fov": 70, "width": 1601}]}))


def test_camera_fov_bounds():
    with pytest.raises(ValidationError, match="fov"):
        CollectConfig.model_validate(minimal(camera_spec={
            "cameras": [{"name": "FRONT", "fov": 180}]}))


def test_tm_port_must_differ():
    with pytest.raises(ValidationError, match="tm_port"):
        CollectConfig.model_validate(minimal(carla={"port": 2000, "tm_port": 2000}))


def test_upload_enabled_requires_repo_id():
    with pytest.raises(ValidationError, match="repo_id"):
        CollectConfig.model_validate(minimal(upload={"enabled": True}))


def test_upload_defaults_off():
    cfg = CollectConfig.model_validate(minimal())
    assert cfg.upload.enabled is False and cfg.upload.delete_local_h5 is False


def test_seed_must_be_non_negative():
    with pytest.raises(ValidationError, match="seed"):
        CollectConfig.model_validate(minimal(seed=-1))


def test_missing_camera_spec_file(tmp_path):
    scenario = tmp_path / "s.yaml"
    scenario.write_text(
        "scenario: {map_name: Town10HD_Opt}\n"
        "capture: {stop: {duration_sec: 60}}\n"
        "camera_spec: does_not_exist.yaml\n")
    with pytest.raises(ConfigError, match="not found"):
        load_collect_config(scenario)


def test_extends_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_collect_config(tmp_path / "a.yaml")


def test_camera_spec_path_is_relative_to_declaring_file(tmp_path):
    (tmp_path / "rigs").mkdir()
    (tmp_path / "rigs/rig.yaml").write_text(
        "cameras:\n  - {name: FRONT, fov: 70}\n")
    (tmp_path / "base.yaml").write_text("camera_spec: rigs/rig.yaml\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/s.yaml").write_text(
        "extends: ../base.yaml\n"
        "scenario: {map_name: Town10HD_Opt}\n"
        "capture: {stop: {duration_sec: 60}}\n")
    cfg = load_collect_config(tmp_path / "sub/s.yaml")
    assert cfg.camera_spec.cameras[0].name == "FRONT"
