"""Stage 2 round-trip on a tiny synthetic run: shapes/dtypes/attrs per
data/README.md, sidecar updated, rerun idempotent. No CARLA involved."""
import json

import h5py
import numpy as np

from carla_data_pipeline.build_samples import ACTION_LABELS, build_samples

RAW_FPS, SAMPLE_FPS = 30, 5


def make_run(tmp_path, n_frames=400, run_id="run99"):
    """Straight drive along +x at 5 m/s."""
    path = tmp_path / f"{run_id}.h5"
    t = np.arange(n_frames) / RAW_FPS
    data = np.stack([t, 5.0 * t, np.zeros(n_frames), np.zeros(n_frames),
                     np.full(n_frames, 5.0), np.zeros(n_frames)], axis=1)
    with h5py.File(path, "w") as f:
        f.attrs.update({"run_id": run_id, "raw_fps": RAW_FPS, "sample_fps": SAMPLE_FPS,
                        "clip_sec": 3.0, "sample_period_sec": 1.0, "horizon_sec": 3.0})
        tel = f.create_group("telemetry")
        tel.create_dataset("frame_id", data=np.arange(n_frames, dtype=np.int32))
        d = tel.create_dataset("data", data=data)
        d.attrs["columns"] = ["sim_time", "x", "y", "yaw", "v", "w"]
    path.with_suffix(".json").write_text(
        json.dumps({"run_id": run_id, "status": "collected"}))
    return path


def test_round_trip(tmp_path):
    path = make_run(tmp_path)
    assert build_samples(path) == 9

    with h5py.File(path) as f:
        keys = f["sample_index/key_index"][:]
        assert keys.dtype == np.int32
        assert list(keys) == list(range(45, 310, 30))
        assert f["sample_index/clip_frame_indices"].shape == (9, 15)
        assert f["sample_index/sample_id"][0].decode() == "run99_000045"
        assert f["sample_index/key_timestamp"][0] == 45 / RAW_FPS
        # window bounds are the first/last of the respective index arrays, inclusive
        assert f["sample_index/clip_start_index"][0] == 45 - 42
        assert f["sample_index/clip_end_index"][0] == 45 + 42
        assert f["sample_index/future_start_index"][0] == 45 + 6
        assert f["sample_index/future_end_index"][0] == 45 + 90

        assert f["trajectory/future_waypoints_map_frame"].shape == (9, 15, 2)
        ego = f["trajectory/future_waypoints_ego_frame"][:]
        assert ego.shape == (9, 15, 2)
        assert np.allclose(ego[:, :, 1], 0, atol=1e-9)   # straight -> ego_y ~ 0
        assert np.all(np.diff(ego[0, :, 0]) > 0)          # moving forward
        assert {t.decode() for t in f["trajectory/trajectory_type"][:]} == {"straight"}

        labels = [b.decode() for b in f["action/action_label"][:]]
        assert set(labels) == {"FORWARD"}
        assert list(f["action/action_id"][:]) == [ACTION_LABELS.index("FORWARD")] * 9
        assert f["action/action_id"].dtype == np.int32

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["status"] == "samples_built"
    assert sidecar["num_samples"] == 9


def test_rerun_is_idempotent(tmp_path):
    path = make_run(tmp_path)
    assert build_samples(path) == 9
    assert build_samples(path) == 9
    with h5py.File(path) as f:
        assert f["sample_index/key_index"].shape == (9,)


def test_too_short_run_yields_zero_samples(tmp_path):
    # 100 frames: first key (45) + horizon (90) overruns the log
    path = make_run(tmp_path, n_frames=100)
    assert build_samples(path) == 0
    with h5py.File(path) as f:
        assert f["sample_index/key_index"].shape == (0,)
        assert f["trajectory/future_waypoints_ego_frame"].shape == (0, 15, 2)
