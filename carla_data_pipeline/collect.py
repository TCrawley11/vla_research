"""Stage 1: headless CARLA collection driven by a validated CollectConfig.

Writes the per-frame groups of `data/runs/<run_id>.h5` (/images, /telemetry,
/map_context, /camera_config, root attrs) per `data/README.md`, plus a
`<run_id>.json` sidecar. Requires a running CARLA server, e.g.:

    CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low

This module imports `carla`; the CLI only loads it once a real connection is
needed, so `--dry-run` never touches it.

Telemetry is converted CARLA -> ROS/doc convention at log time (x forward,
y left, yaw rad, +w = left turn); images are stored RGB.
"""
import datetime
import json
import logging
import math
import queue
import random
import re
import time
from pathlib import Path

import carla
import h5py
import numpy as np

from .config.schema import CameraSpec, CollectConfig, TrafficConfig

log = logging.getLogger(__name__)

TELEMETRY_COLUMNS = ["sim_time", "x", "y", "yaw", "v", "w"]
COORDINATE_CONVENTION = "ROS REP-103: x forward, y left, z up; yaw rad; +w = left"
SCHEMA_VERSION = 1


def _utcnow() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def next_run_id(data_dir: Path) -> str:
    used = [int(m.group(1)) for p in data_dir.glob("run*")
            if (m := re.fullmatch(r"run(\d+)", p.stem))]
    return f"run{max(used, default=0) + 1:02d}"


def intrinsic_matrix(width: int, height: int, fov_deg: float):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))  # square pixels, horizontal fov
    return [[f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0]]


def _spec_transform(spec: CameraSpec) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=spec.location.x, y=spec.location.y, z=spec.location.z),
        carla.Rotation(pitch=spec.rotation.pitch, yaw=spec.rotation.yaw,
                       roll=spec.rotation.roll))


class RunWriter:
    """Appends one synchronized frame per tick into resizable h5 datasets, so a
    crashed run keeps everything written so far."""

    def __init__(self, path: Path, cfg: CollectConfig, run_id: str,
                 carla_version: str, seed: int):
        self.file = h5py.File(path, "w")
        self.n = 0
        f = self.file
        f.attrs.update({
            "run_id": run_id,
            "map": cfg.scenario.map_name,
            "carla_version": carla_version,
            "vehicle": cfg.ego.blueprint,
            "raw_fps": cfg.capture.raw_fps,
            "sample_fps": cfg.capture.sample_fps,
            "clip_sec": cfg.capture.clip_sec,
            "sample_period_sec": cfg.capture.sample_period_sec,
            "horizon_sec": cfg.capture.horizon_sec,
            "waypoint_period_sec": cfg.capture.waypoint_period_sec,
            "seed": seed,
            "coordinate_convention": COORDINATE_CONVENTION,
            "created_utc": _utcnow(),
            "schema_version": SCHEMA_VERSION,
        })
        str_dt = h5py.string_dtype()
        images = f.create_group("images")
        images.attrs["channel_order"] = "RGB"
        for spec in cfg.camera_spec.cameras:
            frame_shape = (spec.height, spec.width, 3)
            images.create_dataset(spec.name, shape=(0, *frame_shape),
                                  maxshape=(None, *frame_shape),
                                  chunks=(1, *frame_shape),
                                  dtype=np.uint8, compression="lzf")
        tel = f.create_group("telemetry")
        tel.create_dataset("frame_id", shape=(0,), maxshape=(None,), dtype=np.int32)
        data = tel.create_dataset("data", shape=(0, len(TELEMETRY_COLUMNS)),
                                  maxshape=(None, len(TELEMETRY_COLUMNS)),
                                  dtype=np.float64)
        data.attrs["columns"] = TELEMETRY_COLUMNS
        mc = f.create_group("map_context")
        for name, dt in [("road_id", np.int32), ("lane_id", np.int32),
                         ("section_id", np.int32), ("is_junction", np.uint8),
                         ("lane_width", np.float64), ("lane_type", str_dt),
                         ("lane_change", str_dt), ("nearest_landmark", str_dt),
                         ("distance_to_next_turn_m", np.float64)]:
            mc.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt)

        cams = cfg.camera_spec.cameras
        cc = f.create_group("camera_config")
        cc.create_dataset("camera_names", data=[c.name for c in cams], dtype=str_dt)
        cc.create_dataset("fov", data=np.array([c.fov for c in cams], dtype=np.float64))
        cc.create_dataset("location_xyz", data=np.array(
            [[c.location.x, c.location.y, c.location.z] for c in cams]))
        cc.create_dataset("rotation_pitch_yaw_roll", data=np.array(
            [[c.rotation.pitch, c.rotation.yaw, c.rotation.roll] for c in cams]))
        cc.create_dataset("intrinsic", data=np.array(
            [intrinsic_matrix(c.width, c.height, c.fov) for c in cams]))
        cc.create_dataset("extrinsic_cam_to_vehicle", data=np.array(
            [_spec_transform(c).get_matrix() for c in cams]))

    def append(self, frame_id: int, images: dict, telemetry_row, map_ctx: dict):
        i = self.n
        f = self.file
        for name, arr in images.items():
            ds = f["images"][name]
            ds.resize(i + 1, axis=0)
            ds[i] = arr
        f["telemetry/frame_id"].resize(i + 1, axis=0)
        f["telemetry/frame_id"][i] = frame_id
        f["telemetry/data"].resize(i + 1, axis=0)
        f["telemetry/data"][i] = telemetry_row
        for name, value in map_ctx.items():
            ds = f["map_context"][name]
            ds.resize(i + 1, axis=0)
            ds[i] = value
        self.n += 1

    def close(self):
        self.file.close()


def _write_sidecar(path: Path, sidecar: dict):
    with open(path, "w") as fh:
        json.dump(sidecar, fh, indent=2)


def _telemetry_row(world, vehicle):
    snap = world.get_snapshot()
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    ang = vehicle.get_angular_velocity()   # CARLA: deg/s, +z is a right (clockwise) turn
    speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    # CARLA (y right, degrees, clockwise+) -> ROS convention (y left, rad, counter-clockwise+)
    row = [snap.timestamp.elapsed_seconds, tf.location.x, -tf.location.y,
           -math.radians(tf.rotation.yaw), speed, -math.radians(ang.z)]
    return row, tf.location


TURN_SEARCH_STEP_M = 2.0
TURN_SEARCH_MAX_M = 200.0  # distance_to_next_turn_m is clamped here (no sentinel)


def _landmark_index(carla_map):
    """All map landmarks as (names, positions (N, 2)); map queries are local."""
    landmarks = carla_map.get_all_landmarks()
    names = [lm.name or f"landmark_{lm.id}" for lm in landmarks]
    positions = np.array([[lm.transform.location.x, lm.transform.location.y]
                          for lm in landmarks]) if landmarks else np.zeros((0, 2))
    return names, positions


def _distance_to_next_junction(wp) -> float:
    """Distance along the lane to the first junction waypoint, clamped."""
    if wp.is_junction:
        return 0.0
    dist, cur = 0.0, wp
    while dist < TURN_SEARCH_MAX_M:
        ahead = cur.next(TURN_SEARCH_STEP_M)
        if not ahead:
            return TURN_SEARCH_MAX_M
        cur = ahead[0]
        dist += TURN_SEARCH_STEP_M
        if cur.is_junction:
            return dist
    return TURN_SEARCH_MAX_M


def _map_context_row(carla_map, location, landmark_names, landmark_pos) -> dict:
    if len(landmark_names):
        d2 = ((landmark_pos[:, 0] - location.x) ** 2
              + (landmark_pos[:, 1] - location.y) ** 2)
        nearest = landmark_names[int(np.argmin(d2))]
    else:
        nearest = ""
    wp = carla_map.get_waypoint(location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    if wp is None:
        return {"road_id": -1, "lane_id": 0, "section_id": -1, "is_junction": 0,
                "lane_width": 0.0, "lane_type": "NONE", "lane_change": "NONE",
                "nearest_landmark": nearest,
                "distance_to_next_turn_m": TURN_SEARCH_MAX_M}
    return {"road_id": wp.road_id, "lane_id": wp.lane_id,
            "section_id": wp.section_id, "is_junction": int(wp.is_junction),
            "lane_width": wp.lane_width, "lane_type": str(wp.lane_type),
            "lane_change": str(wp.lane_change), "nearest_landmark": nearest,
            "distance_to_next_turn_m": _distance_to_next_junction(wp)}


def _spawn_cameras(world, ego, specs: list[CameraSpec]):
    bp_lib = world.get_blueprint_library()
    cameras, queues = [], {}
    for spec in specs:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(spec.width))
        bp.set_attribute("image_size_y", str(spec.height))
        bp.set_attribute("fov", str(spec.fov))
        cam = world.spawn_actor(bp, _spec_transform(spec), attach_to=ego)
        # one queue per camera: in sync mode every tick pushes exactly one image
        # per camera, and matching on the tick's frame id bundles a synchronized
        # snapshot across all cameras
        q = queue.Queue()
        cam.listen(q.put)
        cameras.append(cam)
        queues[spec.name] = q
    return cameras, queues


def _spawn_traffic(client, world, tm, cfg: TrafficConfig, tm_port: int,
                   rng: random.Random, seed: int, ego_spawn_index: int):
    SpawnActor, SetAutopilot = carla.command.SpawnActor, carla.command.SetAutopilot
    bp_lib = world.get_blueprint_library()
    vehicle_ids, walker_ids, controller_ids = [], [], []

    vehicle_bps = list(bp_lib.filter(cfg.vehicle_filter))
    candidates = [sp for i, sp in enumerate(world.get_map().get_spawn_points())
                  if i != ego_spawn_index]
    rng.shuffle(candidates)
    batch = []
    for sp in candidates[:cfg.num_vehicles]:
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "traffic")
        batch.append(SpawnActor(bp, sp).then(
            SetAutopilot(carla.command.FutureActor, True, tm_port)))
    for resp in client.apply_batch_sync(batch, True):
        if not resp.error:
            vehicle_ids.append(resp.actor_id)

    tm.global_percentage_speed_difference(cfg.behavior.speed_delta_pct)
    tm.set_global_distance_to_leading_vehicle(cfg.behavior.follow_distance_m)
    if cfg.behavior.ignore_lights_pct or cfg.behavior.ignore_signs_pct:
        for vid in vehicle_ids:  # lights/signs knobs are per-actor only
            actor = world.get_actor(vid)
            tm.ignore_lights_percentage(actor, cfg.behavior.ignore_lights_pct)
            tm.ignore_signs_percentage(actor, cfg.behavior.ignore_signs_pct)

    if cfg.num_walkers:
        world.set_pedestrians_seed(seed)
        walker_bps = list(bp_lib.filter(cfg.walker_filter))
        batch = []
        for _ in range(cfg.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")
            batch.append(SpawnActor(bp, carla.Transform(loc)))
        for resp in client.apply_batch_sync(batch, True):
            if not resp.error:
                walker_ids.append(resp.actor_id)
        controller_bp = bp_lib.find("controller.ai.walker")
        batch = [SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
        for resp in client.apply_batch_sync(batch, True):
            if not resp.error:
                controller_ids.append(resp.actor_id)
        world.tick()  # controllers must exist in the sim before start()
        for cid in controller_ids:
            controller = world.get_actor(cid)
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)

    log.info("spawned %d/%d traffic vehicles, %d/%d walkers",
             len(vehicle_ids), cfg.num_vehicles, len(walker_ids), cfg.num_walkers)
    return vehicle_ids, walker_ids, controller_ids


def _cleanup(client, world, tm, cameras, ego, vehicle_ids, walker_ids, controller_ids):
    """Destroy everything we spawned and restore async mode, on any exit path.

    Order follows CARLA's own generate_traffic.py: async mode is restored
    *before* TM-registered vehicles are destroyed, otherwise the TM worker
    thread can hit a destroyed actor and abort the whole client process.
    """
    for cam in cameras:
        for op in (cam.stop, cam.destroy):
            try:
                op()
            except RuntimeError:
                pass
    try:
        # restore async mode so the server keeps running on its own
        tm.set_synchronous_mode(False)
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except RuntimeError:
        log.warning("could not restore async mode; is the server still up?")
    for cid in controller_ids:
        try:
            actor = world.get_actor(cid)
            if actor is not None:
                actor.stop()
        except RuntimeError:
            pass
    doomed = list(controller_ids) + list(walker_ids) + list(vehicle_ids)
    if ego is not None:
        doomed.append(ego.id)
    if doomed:
        client.apply_batch([carla.command.DestroyActor(x) for x in doomed])
    time.sleep(0.5)  # let the server process the destruction before we disconnect


def _capture_loop(world, ego, queues, writer: RunWriter, cfg: CollectConfig):
    carla_map = world.get_map()
    landmark_names, landmark_pos = _landmark_index(carla_map)
    total = cfg.capture.stop.resolve_num_frames(cfg.capture.raw_fps)
    log.info("capturing %d frames (%.1f s sim time)", total, total / cfg.capture.raw_fps)
    report_every = 5 * cfg.capture.raw_fps
    for i in range(total):
        world_frame = world.tick()
        images = {}
        for name, q in queues.items():
            # a queue may briefly hold an older frame: discard until this tick's
            # image; the timeout surfaces a stalled sensor instead of hanging
            while True:
                image = q.get(timeout=10.0)
                if image.frame == world_frame:
                    break
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            images[name] = arr.reshape(image.height, image.width, 4)[:, :, 2::-1]  # BGRA -> RGB
        row, location = _telemetry_row(world, ego)
        writer.append(i, images, row,
                      _map_context_row(carla_map, location,
                                       landmark_names, landmark_pos))
        if (i + 1) % report_every == 0:
            log.info("  %d/%d frames", i + 1, total)


def collect(cfg: CollectConfig, run_id: str | None = None,
            data_dir: Path = Path("data/runs"), verify_only: bool = False):
    """Run one capture. Returns the h5 path (None for --verify-only)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or next_run_id(data_dir)
    h5_path = data_dir / f"{run_id}.h5"
    sidecar_path = data_dir / f"{run_id}.json"
    if not verify_only and h5_path.exists():
        raise FileExistsError(
            f"{h5_path} already exists; bump the run id instead of overwriting")

    seed = cfg.seed if cfg.seed is not None else random.SystemRandom().randrange(2 ** 31)
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    log.info("run %s, seed %d", run_id, seed)

    client = carla.Client(cfg.carla.host, cfg.carla.port)
    client.set_timeout(cfg.carla.timeout_sec)
    carla_version = client.get_server_version()
    available = sorted({m.rsplit("/", 1)[-1] for m in client.get_available_maps()})
    if cfg.scenario.map_name not in available:
        raise ValueError(f"map '{cfg.scenario.map_name}' not on this server; "
                         f"available: {', '.join(available)}")

    log.info("loading map %s (CARLA %s)", cfg.scenario.map_name, carla_version)
    # heavy maps (Town03/Town05) blow well past the RPC timeout during load and
    # stay slow while assets stream in afterwards - the first sync ticks
    # included - and a tick timeout aborts the whole process (thrown on a
    # client worker thread). Run the entire setup phase on a generous budget;
    # the per-tick budget is restored right before the capture loop.
    client.set_timeout(max(cfg.carla.timeout_sec, 300.0))
    world = client.load_world(cfg.scenario.map_name)
    # load_world resets episode settings, so sync mode goes strictly after it
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / cfg.capture.raw_fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager(cfg.carla.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)

    cameras, ego = [], None
    vehicle_ids, walker_ids, controller_ids = [], [], []
    try:
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not bp_lib.filter(cfg.ego.blueprint):
            raise ValueError(f"unknown ego blueprint '{cfg.ego.blueprint}'")
        if cfg.ego.spawn_index is not None:
            if cfg.ego.spawn_index >= len(spawn_points):
                raise ValueError(f"ego.spawn_index {cfg.ego.spawn_index} out of range; "
                                 f"map has {len(spawn_points)} spawn points")
            ego_index = cfg.ego.spawn_index
        else:
            ego_index = rng.randrange(len(spawn_points))

        if verify_only:
            log.info("verify OK: map %s, %d spawn points (ego would use %d), "
                     "blueprint %s", cfg.scenario.map_name, len(spawn_points),
                     ego_index, cfg.ego.blueprint)
            return None

        sidecar = {
            "run_id": run_id,
            "status": "collecting",
            "created_utc": _utcnow(),
            "carla_version": carla_version,
            "map": cfg.scenario.map_name,
            "ego_blueprint": cfg.ego.blueprint,
            "seed": seed,
            "config": cfg.model_dump(mode="json"),
        }
        _write_sidecar(sidecar_path, sidecar)

        ego = world.spawn_actor(bp_lib.find(cfg.ego.blueprint), spawn_points[ego_index])
        world.tick()  # sync mode: advance one step so the ego actually appears
        ego.set_autopilot(True, cfg.carla.tm_port)
        driver = cfg.ego.driver
        tm.vehicle_percentage_speed_difference(ego, driver.speed_delta_pct)
        tm.distance_to_leading_vehicle(ego, driver.follow_distance_m)
        tm.ignore_lights_percentage(ego, driver.ignore_lights_pct)
        tm.ignore_signs_percentage(ego, driver.ignore_signs_pct)
        tm.auto_lane_change(ego, driver.auto_lane_change)

        cameras, queues = _spawn_cameras(world, ego, cfg.camera_spec.cameras)
        if cfg.traffic.enabled:
            vehicle_ids, walker_ids, controller_ids = _spawn_traffic(
                client, world, tm, cfg.traffic, cfg.carla.tm_port, rng, seed, ego_index)
        world.tick()  # let sensors and traffic register before frame 0
        client.set_timeout(cfg.carla.timeout_sec)  # setup done: per-tick budget

        writer = RunWriter(h5_path, cfg, run_id, carla_version, seed)
        wall_start = time.monotonic()
        try:
            _capture_loop(world, ego, queues, writer, cfg)
            sidecar["status"] = "collected"
        except BaseException as exc:
            sidecar["status"] = "aborted"
            sidecar["error"] = repr(exc)
            raise
        finally:
            sidecar["num_frames"] = writer.n
            sidecar["sim_duration_sec"] = round(writer.n / cfg.capture.raw_fps, 3)
            sidecar["wall_duration_sec"] = round(time.monotonic() - wall_start, 1)
            writer.close()
            _write_sidecar(sidecar_path, sidecar)
            log.info("%s: %d frames (%s)", h5_path, writer.n, sidecar["status"])
    finally:
        _cleanup(client, world, tm, cameras, ego,
                 vehicle_ids, walker_ids, controller_ids)
    return h5_path
