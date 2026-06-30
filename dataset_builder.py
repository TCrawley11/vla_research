"""Builds 3-second VLA dataset samples from a dense per-frame log.

Scope (per the instruction document, with the requested reductions):
  - Each sample covers a 3 s clip; the key frame is the clip centre.
  - Raw video is 30 fps; clip frames are subsampled to 5 fps.
  - Included fields: sample_id, video_clip, key_frame, clip_frames,
    timestamp_start/end, robot_state, map_context, action_label,
    quality_control.
  - Excluded (per instructions): caption, qa_pairs, the `action` text block,
    trajectory (and trajectory_text), and every other language-generation output.

Coordinate convention (this NEW dataset only): ROS REP-103 / the document's
convention - x forward, y left, z up; yaw in radians; angular velocity + = left.
The notebook converts CARLA -> this convention at log time, so the inputs to
build_samples() are assumed to already be ROS-convention. Note this differs from
output.csv and calibration.json, which keep CARLA's native convention
(x forward, y right, degrees) as the shared standard format.
"""
import bisect
import math

# Action-label thresholds, tuned for CARLA's road-vehicle speed regime (m/s,
# rad/s) rather than the document's lab-robot values. Reasonable starting points
# for urban driving; adjust to taste (named so they're easy to tune).
STOP_V = 0.5    # m/s (~1.8 km/h); below this magnitude -> STOP
SLOW_V = 3.0    # m/s (~11 km/h); below this (and roughly straight) -> SLOW_FORWARD
TURN_W = 0.15   # rad/s (~8.6 deg/s); above this magnitude -> turning


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


def _nearest_index(times, t):
    """Index of the frame whose sim_time is closest to t (times must be sorted)."""
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if (times[i] - t) < (t - times[i - 1]) else i - 1


def build_samples(frames, run_id="run01", primary_cam="FRONT",
                  clip_sec=3.0, sample_fps=5, sample_period_sec=1.0):
    """Slice a dense per-frame log into 3 s samples centred on a key frame.

    `frames`: list of dicts (any order), each with keys:
        frame_id, sim_time, x, y, yaw (rad), v (m/s), w (rad/s),
        action_label, map_context (dict).

    A sample is emitted every `sample_period_sec` for each key time that has a
    full half-clip of past and future frames available. Returns a list of sample
    dicts ready to dump as JSON.
    """
    if not frames:
        return []
    frames = sorted(frames, key=lambda f: f["sim_time"])
    times = [f["sim_time"] for f in frames]
    t0, t1 = times[0], times[-1]
    half = clip_sec / 2.0
    n_side = int(half * sample_fps)                    # clip frames each side of key

    samples = []
    t_key = t0 + half
    while t_key <= t1 - half + 1e-6:
        key = frames[_nearest_index(times, t_key)]
        key_id = key["frame_id"]

        # clip frames subsampled at `sample_fps`, symmetric about the key frame
        clip_ids = [frames[_nearest_index(times, t_key + j / sample_fps)]["frame_id"]
                    for j in range(-n_side, n_side + 1)]

        x0, y0, yaw0 = key["x"], key["y"], key["yaw"]

        samples.append({
            "sample_id": f"{run_id}_{key_id:06d}",
            "video_clip": f"{run_id}_clip_{key_id:06d}.mp4",
            "key_frame": f"{primary_cam}/{key_id:06d}.png",
            "clip_frames": [f"{primary_cam}/{fid:06d}.png" for fid in clip_ids],
            "timestamp_start": round(t_key - half, 3),
            "timestamp_end": round(t_key + half, 3),
            "robot_state": {
                "x": round(x0, 4),
                "y": round(y0, 4),
                "yaw": round(yaw0, 4),
                "linear_velocity": round(key["v"], 4),
                "angular_velocity": round(key["w"], 4),
                "battery_voltage": None,                # not available in CARLA
            },
            "map_context": key["map_context"],
            "action_label": key["action_label"],
            "quality_control": {"auto_generated": True,
                                "human_checked": False, "issues": []},
        })
        t_key += sample_period_sec
    return samples
