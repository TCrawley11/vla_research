"""Hindsight-trajectory math against synthetic telemetry (no h5, no CARLA)."""
import math

import numpy as np

from carla_data_pipeline.build_samples import (
    action_label_from_velocity, clip_indices, future_indices,
    select_key_indices, trajectory_type, waypoints_to_ego_frame)

RAW_FPS, SAMPLE_FPS = 30, 5
CLIP_SEC, PERIOD_SEC, HORIZON_SEC = 3.0, 1.0, 3.0


def test_straight_line_maps_to_zero_ego_y():
    yaw = math.radians(45)
    dists = np.arange(1, 16, dtype=float)
    points = np.stack([10 + dists * math.cos(yaw), -3 + dists * math.sin(yaw)], axis=1)
    ego = waypoints_to_ego_frame(points, key_xy=(10, -3), key_yaw=yaw)
    assert np.allclose(ego[:, 1], 0, atol=1e-9)
    assert np.allclose(ego[:, 0], dists)


def test_left_turn_gives_positive_ego_y():
    # quarter circle, radius 10, starting at origin heading +x, turning left
    # (+y is left in REP-103)
    theta = np.linspace(0.1, math.pi / 2, 15)
    points = np.stack([10 * np.sin(theta), 10 * (1 - np.cos(theta))], axis=1)
    ego = waypoints_to_ego_frame(points, key_xy=(0, 0), key_yaw=0.0)
    assert np.all(ego[:, 1] > 0)
    mirrored = waypoints_to_ego_frame(points * [1, -1], key_xy=(0, 0), key_yaw=0.0)
    assert np.all(mirrored[:, 1] < 0)


def test_trajectory_type_thresholds():
    assert trajectory_type(0.0, math.pi / 2) == "left_curve"
    assert trajectory_type(0.0, -math.pi / 2) == "right_curve"
    assert trajectory_type(0.0, 0.1) == "straight"


def test_trajectory_type_wraps_across_pi():
    # net change is +0.2 rad, not -2*pi + 0.2
    assert trajectory_type(math.pi - 0.1, -math.pi + 0.1) == "straight"


def test_key_selection_excludes_frames_without_full_horizon():
    # step=6, clip_half=45, horizon=15*6=90 raw frames
    keys = select_key_indices(400, RAW_FPS, SAMPLE_FPS, CLIP_SEC, PERIOD_SEC, HORIZON_SEC)
    assert keys[0] == 45
    assert list(keys) == list(range(45, 310, 30))  # last valid key: 285 (285+90 <= 399)
    assert len(select_key_indices(136, RAW_FPS, SAMPLE_FPS, CLIP_SEC, PERIOD_SEC, HORIZON_SEC)) == 1
    assert len(select_key_indices(135, RAW_FPS, SAMPLE_FPS, CLIP_SEC, PERIOD_SEC, HORIZON_SEC)) == 0


def test_clip_indices_symmetric_about_key():
    idx = clip_indices(45, RAW_FPS, SAMPLE_FPS, CLIP_SEC)
    assert len(idx) == 15
    assert idx[7] == 45                      # key frame at the clip centre
    assert list(idx[:2]) == [3, 9]
    assert idx[-1] == 87


def test_future_indices_strictly_after_key():
    idx = future_indices(45, RAW_FPS, SAMPLE_FPS, HORIZON_SEC)
    assert len(idx) == 15
    assert idx[0] == 51 and idx[-1] == 135   # key + step .. key + T*step


def test_action_labels():
    assert action_label_from_velocity(0.0, 0.0) == "STOP"
    assert action_label_from_velocity(5.0, 0.0) == "FORWARD"
    assert action_label_from_velocity(1.0, 0.0) == "SLOW_FORWARD"
    assert action_label_from_velocity(5.0, 0.2) == "LEFT_TURN"
    assert action_label_from_velocity(5.0, -0.2) == "RIGHT_TURN"
    assert action_label_from_velocity(float("nan"), 0.0) == "UNKNOWN"
