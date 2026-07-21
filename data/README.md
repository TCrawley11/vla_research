# Data directory

Canonical home for all captured data going forward. The legacy locations
(`dataset/run01/` PNG-per-frame layout and the root `dataset.h5` real-robot
capture) are superseded by this structure; new CARLA runs are captured directly
to HDF5 here.

```
data/
  runs/     one self-contained <run_id>.h5 per capture run,
            plus a <run_id>.json sidecar (see below)
```

Everything under `runs/` is gitignored (large binaries); only this README and
the directory structure are tracked. A run file is fully self-contained
(frames, telemetry, camera config, sample index, trajectories, actions) so a
single `.h5` can be shared or moved to a training machine on its own.

## Run sidecar (`data/runs/<run_id>.json`)

A small human/LLM-readable companion written by Stage 1 (`collect`) and
updated by Stage 2 (`build-samples`). It duplicates the h5 root attrs for
quick inspection without opening the h5, and adds:

- `config`: the fully resolved scenario config the run was captured with
- `seed`: the resolved seed (also recorded when the config left it unset)
- `status`: `collecting` -> `collected` (or `aborted`, with an `error` field)
  -> `samples_built`
- `num_frames`, `sim_duration_sec`, `wall_duration_sec`; after Stage 2,
  `num_samples`

## Per-run HDF5 schema (`data/runs/<run_id>.h5`)

One file per recording session. Two index spaces:

- **N** = captured frames at the raw 30 fps tick rate. All per-frame datasets
  (`/images/*`, `/telemetry/*`, `/map_context/*`) share this index, aligned
  with `/telemetry/frame_id`.
- **S** = built samples (one per key frame, every 1 s per the instruction
  document). All per-sample datasets (`/sample_index/*`, `/trajectory/*`,
  `/action/*`) share this index.

Sample datasets reference frames by **index into the N axis** (`*_index`
fields), not by frame_id, so slicing a clip is a direct array slice.

### Root attributes

| attr | type | meaning |
|---|---|---|
| `run_id` | str | e.g. `run02`; matches the filename |
| `map` | str | CARLA town name |
| `carla_version` | str | e.g. `0.9.16` |
| `vehicle` | str | ego blueprint id, e.g. `vehicle.tesla.model3` |
| `raw_fps` | int | capture tick rate (30) |
| `sample_fps` | int | clip subsampling rate (5) |
| `clip_sec` | float | sample clip length in seconds (3.0) |
| `sample_period_sec` | float | one sample per this much sim time (1.0) |
| `horizon_sec` | float | future trajectory horizon in seconds (3.0) |
| `seed` | int | resolved run seed (python/numpy/TM) |
| `coordinate_convention` | str | `"ROS REP-103: x forward, y left, z up; yaw rad; +w = left"` |
| `created_utc` | str | ISO-8601 capture start time |
| `schema_version` | int | starts at 1; bump on breaking layout changes |

### Groups and datasets

```
/images/                                 raw 30 fps frames, per-frame chunked, lzf
    ...            attrs["channel_order"]  "RGB"
    FRONT          (N, 900, 1600, 3) uint8   chunks=(1, 900, 1600, 3)
    FRONT_LEFT     (N, 900, 1600, 3) uint8
    FRONT_RIGHT    (N, 900, 1600, 3) uint8
    BACK           (N, 900, 1600, 3) uint8
    BACK_LEFT      (N, 900, 1600, 3) uint8
    BACK_RIGHT     (N, 900, 1600, 3) uint8

/telemetry/
    frame_id       (N,) int32           contiguous from 0 (what 000123.png used to encode)
    data           (N, C) float64       one row per frame, one column per signal
    ...            attrs["columns"]     list of C column names, the source of truth
                                        for what data[:, j] means

/map_context/                           CARLA waypoint fields at the ego location, per frame
    road_id        (N,) int32
    lane_id        (N,) int32
    section_id     (N,) int32
    is_junction    (N,) uint8           0/1
    lane_width     (N,) float64         m
    lane_type      (N,) vlen str        e.g. 'Driving'
    lane_change    (N,) vlen str        e.g. 'Both', 'Right', 'NONE'

/camera_config/                         one entry per camera, same order everywhere
    camera_names                (6,) vlen str      'FRONT', 'FRONT_LEFT', ...
    fov                         (6,) float64       horizontal fov, degrees
    location_xyz                (6, 3) float64     mounting, vehicle frame, m
    rotation_pitch_yaw_roll     (6, 3) float64     mounting, degrees
    intrinsic                   (6, 3, 3) float64  K matrix
    extrinsic_cam_to_vehicle    (6, 4, 4) float64  homogeneous cam -> vehicle

/sample_index/                          one row per built sample
    sample_id           (S,) vlen str   '<run_id>_<key_frame_id:06d>'
    key_index           (S,) int32      index into the N axis
    key_frame_id        (S,) int32      /telemetry/frame_id at key_index
    key_timestamp       (S,) float64    sim seconds at the key frame
    clip_start_index    (S,) int32      first entry of clip_frame_indices (inclusive)
    clip_end_index      (S,) int32      last entry of clip_frame_indices (inclusive)
    future_start_index  (S,) int32      first future frame: key_index + step (inclusive)
    future_end_index    (S,) int32      last future frame: key_index + T*step (inclusive)
    clip_frame_indices  (S, K) int32    the K clip frames subsampled at 5 fps
                                        (K = 15 for a 3 s clip)

/trajectory/                            future horizon after the key frame, at sample_fps
    future_waypoints_map_frame  (S, T, 2) float64   (x, y) map frame
    future_waypoints_ego_frame  (S, T, 2) float64   (x, y) in the key frame's ego frame
    trajectory_type             (S,) vlen str       e.g. 'straight', 'left_curve'
                                                    (T = 15 for a 3 s horizon at 5 fps)

/action/
    action_label   (S,) vlen str        STOP / LEFT_TURN / RIGHT_TURN /
                                        SLOW_FORWARD / FORWARD / UNKNOWN
    action_id      (S,) int32           fixed mapping of the labels above, in that
                                        order: 0..5 (UNKNOWN = 5)
```

### Telemetry columns

`/telemetry/data` columns (recorded in `attrs["columns"]`, which is always
authoritative over this list):

```
sim_time, x, y, yaw, v, w
```

State values follow the ROS convention (x forward, y left, yaw in rad,
+w = left); CARLA native values (x forward, y right, degrees) are converted at
log time by `carla_data_pipeline/collect.py`. New signals are added by appending
columns and extending `attrs["columns"]`; readers must look up columns by name,
never by position.

### Conventions carried over from the PNG layout

- One run per file; never mix sessions. Bump the run id to keep an old run
  instead of overwriting it.
- `frame_id` is the contiguous 30 fps index.
- Camera config is constant per run and stored once, inside the file.
- Per-frame datasets are created resizable (`maxshape=(None, ...)`) and
  appended each tick, so a crashed run keeps everything written so far. The
  per-sample groups (`/sample_index/`, `/trajectory/`, `/action/`) are written
  by the build step after capture ends.
