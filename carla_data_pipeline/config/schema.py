"""Pydantic models for the collection config.

A scenario YAML (configs/scenarios/*.yaml) deep-merged over configs/base.yaml
validates into a single `CollectConfig`. Every model forbids unknown keys, so a
misspelled YAML field is a hard error naming the key instead of a silently
ignored setting. Field descriptions double as the source for `man config`.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CarlaConnection(StrictModel):
    host: str = Field("localhost", description="CARLA server hostname or IP.")
    port: int = Field(2000, ge=1024, le=65535, description="CARLA RPC port.")
    tm_port: int = Field(8000, ge=1024, le=65535,
                         description="Traffic Manager port; must differ from the RPC port.")
    timeout_sec: float = Field(10.0, gt=0, description="Client RPC timeout in seconds.")

    @model_validator(mode="after")
    def _distinct_ports(self):
        if self.tm_port == self.port:
            raise ValueError("tm_port must differ from port")
        return self


class ScenarioConfig(StrictModel):
    map_name: str = Field(..., min_length=1,
                          description="CARLA map to load, e.g. 'Town10HD_Opt' "
                                      "(as accepted by client.load_world).")
    # Weather (incl. dynamic weather) joins this section later.


class EgoDriverConfig(StrictModel):
    """How the ego drives itself. Only Traffic Manager autopilot for now;
    'behavior_agent' (the agents package) is a planned later mode."""
    mode: Literal["traffic_manager"] = Field(
        "traffic_manager", description="Driving mode for the ego vehicle.")
    speed_delta_pct: float = Field(
        0.0, ge=-100,
        description="Percentage difference from the speed limit; positive drives slower "
                    "(TM vehicle_percentage_speed_difference).")
    follow_distance_m: float = Field(
        2.5, ge=0, description="Minimum distance to the leading vehicle, meters.")
    ignore_lights_pct: float = Field(
        0.0, ge=0, le=100, description="Percentage of traffic lights the ego runs.")
    ignore_signs_pct: float = Field(
        0.0, ge=0, le=100, description="Percentage of stop signs the ego ignores.")
    auto_lane_change: bool = Field(True, description="Allow TM to change lanes.")


class EgoConfig(StrictModel):
    blueprint: str = Field("vehicle.tesla.model3",
                           description="Ego vehicle blueprint id.")
    spawn_index: Optional[int] = Field(
        None, ge=0,
        description="Index into the map's spawn points; omit for a seeded random choice. "
                    "Range is checked against the loaded map by --verify-only.")
    driver: EgoDriverConfig = Field(default_factory=EgoDriverConfig,
                                    description="Ego driving behavior.")


class TrafficManagerBehavior(StrictModel):
    """TM knobs applied to background traffic vehicles."""
    speed_delta_pct: float = Field(
        0.0, ge=-100,
        description="Global percentage difference from speed limits; positive is slower.")
    follow_distance_m: float = Field(
        2.5, ge=0, description="Global minimum distance to leading vehicles, meters.")
    ignore_lights_pct: float = Field(
        0.0, ge=0, le=100, description="Percentage of lights background vehicles run.")
    ignore_signs_pct: float = Field(
        0.0, ge=0, le=100, description="Percentage of signs background vehicles ignore.")


class TrafficConfig(StrictModel):
    enabled: bool = Field(False, description="Spawn background traffic.")
    num_vehicles: int = Field(0, ge=0, description="Background vehicles to spawn.")
    num_walkers: int = Field(0, ge=0, description="Pedestrians to spawn.")
    vehicle_filter: str = Field("vehicle.*",
                                description="Blueprint filter for background vehicles.")
    walker_filter: str = Field("walker.pedestrian.*",
                               description="Blueprint filter for pedestrians.")
    behavior: TrafficManagerBehavior = Field(
        default_factory=TrafficManagerBehavior,
        description="TM behavior for background vehicles.")

    @model_validator(mode="after")
    def _counts_match_enabled(self):
        if self.enabled and self.num_vehicles + self.num_walkers < 1:
            raise ValueError("traffic.enabled requires num_vehicles + num_walkers >= 1")
        if not self.enabled and (self.num_vehicles or self.num_walkers):
            raise ValueError("traffic counts must be 0 (or omitted) when traffic.enabled is false")
        return self


class StopCondition(StrictModel):
    """Exactly one of duration_sec / num_frames."""
    duration_sec: Optional[float] = Field(
        None, gt=0, description="Stop after this much simulated time.")
    num_frames: Optional[int] = Field(
        None, gt=0, description="Stop after exactly this many captured frames.")

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.duration_sec is None) == (self.num_frames is None):
            raise ValueError("set exactly one of stop.duration_sec / stop.num_frames")
        return self

    def resolve_num_frames(self, raw_fps: int) -> int:
        if self.num_frames is not None:
            return self.num_frames
        assert self.duration_sec is not None  # _exactly_one guarantees this
        return int(round(self.duration_sec * raw_fps))


class CaptureConfig(StrictModel):
    raw_fps: int = Field(30, gt=0, description="Capture tick rate; fixed_delta_seconds = 1/raw_fps.")
    stop: StopCondition = Field(..., description="When the capture ends.")
    sample_fps: int = Field(5, gt=0, description="Clip subsampling rate; must divide raw_fps.")
    clip_sec: float = Field(3.0, gt=0, description="Sample clip length, centred on the key frame.")
    sample_period_sec: float = Field(1.0, gt=0, description="One sample per this much sim time.")
    horizon_sec: float = Field(3.0, gt=0, description="Future trajectory horizon after the key frame.")
    waypoint_period_sec: float = Field(
        0.5, gt=0, description="Future waypoint spacing in seconds; "
                               "horizon_sec / this = waypoints per sample.")

    @model_validator(mode="after")
    def _consistent(self):
        if self.raw_fps % self.sample_fps != 0:
            raise ValueError(f"raw_fps ({self.raw_fps}) must be divisible by "
                             f"sample_fps ({self.sample_fps})")
        wp_frames = self.waypoint_period_sec * self.raw_fps
        if abs(wp_frames - round(wp_frames)) > 1e-9 or round(wp_frames) < 1:
            raise ValueError(
                f"waypoint_period_sec ({self.waypoint_period_sec}) must be a whole "
                f"number of frames at raw_fps {self.raw_fps}")
        n_wp = self.horizon_sec / self.waypoint_period_sec
        if abs(n_wp - round(n_wp)) > 1e-9:
            raise ValueError(
                f"horizon_sec ({self.horizon_sec}) must be an integer multiple of "
                f"waypoint_period_sec ({self.waypoint_period_sec})")
        min_sec = self.clip_sec / 2 + self.horizon_sec
        run_sec = (self.stop.duration_sec if self.stop.duration_sec is not None
                   else self.stop.num_frames / self.raw_fps)
        if run_sec < min_sec:
            raise ValueError(
                f"run too short to yield a single sample: needs at least {min_sec:.1f}s "
                f"(clip_sec/2 + horizon_sec), stop condition gives {run_sec:.1f}s")
        return self


class Location(StrictModel):
    x: float = Field(0.0, description="Forward offset from vehicle origin, m.")
    y: float = Field(0.0, description="Right offset (CARLA convention), m.")
    z: float = Field(0.0, description="Up offset, m.")


class Rotation(StrictModel):
    pitch: float = Field(0.0, description="Degrees.")
    yaw: float = Field(0.0, description="Degrees; positive is clockwise from above (CARLA).")
    roll: float = Field(0.0, description="Degrees.")


class CameraSpec(StrictModel):
    name: str = Field(..., pattern=r"^[A-Z][A-Z0-9_]*$",
                      description="Uppercase camera name; becomes the h5 dataset name.")
    width: int = Field(1600, gt=0, description="Image width, pixels; must be even.")
    height: int = Field(900, gt=0, description="Image height, pixels; must be even.")
    fov: float = Field(..., gt=0, lt=170, description="Horizontal field of view, degrees.")
    location: Location = Field(default_factory=Location,
                               description="Mounting position in the vehicle frame.")
    rotation: Rotation = Field(default_factory=Rotation,
                               description="Mounting rotation in the vehicle frame.")

    @model_validator(mode="after")
    def _even_resolution(self):
        if self.width % 2 or self.height % 2:
            raise ValueError(f"camera '{self.name}': width and height must be even "
                             "(h264-friendly for the later compress stage)")
        return self


class CameraSpecConfig(StrictModel):
    cameras: list[CameraSpec] = Field(..., min_length=1,
                                      description="The camera rig, one entry per camera.")

    @model_validator(mode="after")
    def _rig_rules(self):
        names = [c.name for c in self.cameras]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate camera names: {sorted(dupes)}")
        if "FRONT" not in names:
            raise ValueError("camera rig must include a FRONT camera (primary sample camera)")
        return self


class CollectConfig(StrictModel):
    """Root of a fully resolved scenario config."""
    carla: CarlaConnection = Field(default_factory=CarlaConnection,
                                   description="Server connection.")
    scenario: ScenarioConfig = Field(..., description="What world to collect in.")
    seed: Optional[int] = Field(
        None, ge=0,
        description="Seeds python/numpy/TM for a reproducible run. Omit to draw one at "
                    "runtime; the resolved seed is always recorded in the run sidecar.")
    ego: EgoConfig = Field(default_factory=EgoConfig, description="Ego vehicle.")
    traffic: TrafficConfig = Field(default_factory=TrafficConfig,
                                   description="Background traffic.")
    capture: CaptureConfig = Field(..., description="Tick rate, stop condition, sampling.")
    camera_spec: CameraSpecConfig = Field(
        ..., description="Camera rig; in YAML this is a path to a camera_spec file, "
                         "resolved by the loader.")
