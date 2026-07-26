"""Stage 2: build the S-axis sample groups into a run .h5 (offline, no CARLA).

Opens `data/runs/<run_id>.h5` r+, reads the sampling parameters from the root
attrs written by Stage 1, and appends `/sample_index`, `/trajectory` and
`/action` per `data/README.md`. Re-running replaces the sample groups, so the
step is idempotent. The run's `.json` sidecar is updated with the sample count.

Absorbs the old root-level dataset_builder.py: `action_label_from_velocity`
and the sampling/slicing responsibilities live here now, operating on arrays
in the h5 instead of dicts destined for JSON.

Coordinate convention: ROS REP-103 (x forward, y left, yaw rad, +w = left),
already applied by Stage 1 at log time.

Index spaces: N = raw-fps frames, S = built samples. A key frame qualifies
only if the *full future horizon* fits inside the log - the horizon extends
further past the key frame than the clip half-window does, so reusing the clip
bound would silently truncate trajectories at the end of every run.
"""
import datetime
import json
import logging
import math
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)

# Action-label thresholds, tuned for CARLA's road-vehicle speed regime (m/s,
# rad/s) rather than the document's lab-robot values. Reasonable starting points
# for urban driving; adjust to taste (named so they're easy to tune).
STOP_V = 0.5    # m/s (~1.8 km/h); below this magnitude -> STOP
SLOW_V = 3.0    # m/s (~11 km/h); below this (and roughly straight) -> SLOW_FORWARD
TURN_W = 0.15   # rad/s (~8.6 deg/s); above this magnitude -> turning

# net yaw change over the future horizon separating straight from a curve
CURVE_YAW = 0.30  # rad (~17 deg)

# final forward displacement over the horizon below which the trajectory is
# STOPPING (car-scale analogue of the instruction document's 0.05 m lab rule)
STOP_TRAJ_DIST = 1.0  # m

# fixed id mapping per data/README.md: 0..5 in this order
ACTION_LABELS = ["STOP", "LEFT_TURN", "RIGHT_TURN", "SLOW_FORWARD", "FORWARD", "UNKNOWN"]


def action_label_from_velocity(v, w, stop_v=STOP_V, slow_v=SLOW_V, turn_w=TURN_W):
    """Discrete action class from linear speed v (m/s) and yaw rate w (rad/s, +left).

    Classes: STOP, LEFT_TURN, RIGHT_TURN, SLOW_FORWARD, FORWARD, UNKNOWN.
    LANE_KEEP and LANE_RECOVERY_* are intentionally never produced.
    """
    if v is None or w is None or math.isnan(v) or math.isnan(w):
        return "UNKNOWN"
    if abs(v) < stop_v:
        return "STOP"
    if abs(w) >= turn_w:
        return "LEFT_TURN" if w > 0 else "RIGHT_TURN"
    if v < slow_v:
        return "SLOW_FORWARD"
    return "FORWARD"


def select_key_indices(n_frames, raw_fps, clip_sec,
                       sample_period_sec, horizon_sec):
    """Indices into the N axis that qualify as key frames.

    A key frame needs a full half-clip of past frames and a full future
    horizon ahead of it; keys are spaced sample_period_sec apart.
    """
    clip_half = round(clip_sec / 2 * raw_fps)
    horizon_frames = round(horizon_sec * raw_fps)
    period = round(sample_period_sec * raw_fps)
    lookahead = max(clip_half, horizon_frames)
    keys = np.arange(clip_half, n_frames, period)
    return keys[keys + lookahead <= n_frames - 1]


def clip_indices(key_index, raw_fps, sample_fps, clip_sec):
    """The K clip frame indices at sample_fps, symmetric about the key frame."""
    step = raw_fps // sample_fps
    n_side = int(clip_sec / 2 * sample_fps)  # truncate: K = 15 for a 3 s clip at 5 fps
    return key_index + np.arange(-n_side, n_side + 1) * step


def future_indices(key_index, raw_fps, horizon_sec, waypoint_period_sec):
    """The T future frame indices on the waypoint grid, strictly after the key.

    Waypoints are spaced waypoint_period_sec apart (0.5 s per the instruction
    document -> 6 per 3 s horizon), indexed into the raw-fps log where every
    grid point lands exactly on a frame (validated by the config schema).
    """
    step = round(waypoint_period_sec * raw_fps)
    t = round(horizon_sec / waypoint_period_sec)
    return key_index + np.arange(1, t + 1) * step


def waypoints_to_ego_frame(points_xy, key_xy, key_yaw):
    """Rotate map-frame waypoints (T, 2) into the key frame's ego frame.

    Straight driving -> ego_y ~ 0; a left turn -> ego_y > 0 (+y-left, REP-103).
    """
    d = np.asarray(points_xy, dtype=np.float64) - np.asarray(key_xy, dtype=np.float64)
    c, s = math.cos(key_yaw), math.sin(key_yaw)
    return np.stack([d[:, 0] * c + d[:, 1] * s,
                     -d[:, 0] * s + d[:, 1] * c], axis=1)


def trajectory_type(yaw_key, yaw_end, forward_dist,
                    curve_yaw=CURVE_YAW, stop_dist=STOP_TRAJ_DIST):
    """STOPPING / LEFT_CURVE / RIGHT_CURVE / STRAIGHT for one sample.

    STOPPING when the final ego-frame forward displacement over the horizon is
    negligible; otherwise classified by net yaw change over the horizon.
    """
    if abs(forward_dist) < stop_dist:
        return "STOPPING"
    net = (yaw_end - yaw_key + math.pi) % (2 * math.pi) - math.pi
    if net > curve_yaw:
        return "LEFT_CURVE"
    if net < -curve_yaw:
        return "RIGHT_CURVE"
    return "STRAIGHT"


def build_samples(h5_path):
    """Build and write the S-axis groups for one run file. Returns the sample count."""
    h5_path = Path(h5_path)
    str_dt = h5py.string_dtype()
    with h5py.File(h5_path, "r+") as f:
        run_id = f.attrs["run_id"]
        raw_fps = int(f.attrs["raw_fps"])
        sample_fps = int(f.attrs["sample_fps"])
        clip_sec = float(f.attrs["clip_sec"])
        sample_period_sec = float(f.attrs["sample_period_sec"])
        horizon_sec = float(f.attrs["horizon_sec"])
        # runs captured before the attr existed default to the document's 0.5 s
        wp_period = float(f.attrs.get("waypoint_period_sec", 0.5))

        frame_id = f["telemetry/frame_id"][:]
        data = f["telemetry/data"][:]
        columns = [c if isinstance(c, str) else c.decode()
                   for c in f["telemetry/data"].attrs["columns"]]
        col = {name: data[:, i] for i, name in enumerate(columns)}
        n_frames = len(frame_id)

        keys = select_key_indices(n_frames, raw_fps, clip_sec,
                                  sample_period_sec, horizon_sec)
        s = len(keys)
        log.info("%s: %d frames -> %d samples", h5_path.name, n_frames, s)

        k_clip = 2 * int(clip_sec / 2 * sample_fps) + 1
        t_horizon = round(horizon_sec / wp_period)
        clips = np.stack([clip_indices(k, raw_fps, sample_fps, clip_sec) for k in keys]) \
            if s else np.zeros((0, k_clip), dtype=np.int64)
        futures = np.stack([future_indices(k, raw_fps, horizon_sec, wp_period) for k in keys]) \
            if s else np.zeros((0, t_horizon), dtype=np.int64)

        for name in ("sample_index", "trajectory", "action"):
            if name in f:
                del f[name]

        si = f.create_group("sample_index")
        si.create_dataset("sample_id", data=[f"{run_id}_{frame_id[k]:06d}" for k in keys],
                          dtype=str_dt)
        si.create_dataset("key_index", data=keys.astype(np.int32))
        si.create_dataset("key_frame_id", data=frame_id[keys].astype(np.int32))
        si.create_dataset("key_timestamp", data=col["sim_time"][keys])
        si.create_dataset("timestamp_start", data=col["sim_time"][clips[:, 0]]
                          if s else np.zeros(0))
        si.create_dataset("timestamp_end", data=col["sim_time"][clips[:, -1]]
                          if s else np.zeros(0))
        si.create_dataset("clip_start_index", data=clips[:, 0].astype(np.int32)
                          if s else np.zeros(0, np.int32))
        si.create_dataset("clip_end_index", data=clips[:, -1].astype(np.int32)
                          if s else np.zeros(0, np.int32))
        si.create_dataset("future_start_index", data=futures[:, 0].astype(np.int32)
                          if s else np.zeros(0, np.int32))
        si.create_dataset("future_end_index", data=futures[:, -1].astype(np.int32)
                          if s else np.zeros(0, np.int32))
        si.create_dataset("clip_frame_indices", data=clips.astype(np.int32))

        xy = np.stack([col["x"], col["y"]], axis=1)
        map_frame = xy[futures] if s else np.zeros((0, t_horizon, 2))
        ego_frame = np.stack([
            waypoints_to_ego_frame(map_frame[i], xy[k], col["yaw"][k])
            for i, k in enumerate(keys)
        ]) if s else np.zeros((0, t_horizon, 2))
        types = [trajectory_type(col["yaw"][k], col["yaw"][futures[i, -1]],
                                 ego_frame[i][-1, 0])
                 for i, k in enumerate(keys)]

        tr = f.create_group("trajectory")
        tr.create_dataset("future_waypoints_map_frame", data=map_frame)
        tr.create_dataset("future_waypoints_ego_frame", data=ego_frame)
        tr.create_dataset("trajectory_type", data=types, dtype=str_dt)

        labels = [action_label_from_velocity(col["v"][k], col["w"][k]) for k in keys]
        ac = f.create_group("action")
        ac.create_dataset("action_label", data=labels, dtype=str_dt)
        ac.create_dataset("action_id",
                          data=np.array([ACTION_LABELS.index(l) for l in labels],
                                        dtype=np.int32))

    _update_sidecar(h5_path.with_suffix(".json"), s)
    return s


def _update_sidecar(sidecar_path, num_samples):
    if not sidecar_path.is_file():
        log.warning("no sidecar at %s; skipping status update", sidecar_path)
        return
    with open(sidecar_path) as fh:
        sidecar = json.load(fh)
    sidecar["status"] = "samples_built"
    sidecar["num_samples"] = num_samples
    sidecar["samples_built_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
