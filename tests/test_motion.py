"""Temporal motion semantics, including causal boundaries and missing readings."""
import numpy as np
import pytest
from pydantic import ValidationError

from carla_data_pipeline.config_utils.schema import MotionLabelConfig
from carla_data_pipeline.motion import future_motion_events, motion_states
from carla_data_pipeline.build_samples import action_label_from_velocity


def run(speed, **settings):
    return motion_states(speed, np.arange(len(speed)) / 30,
                         MotionLabelConfig(**settings))


def test_stop_requires_elapsed_dwell_and_is_causal():
    speed = np.zeros(60)
    filtered, states = run(speed)
    assert np.all(states[:15] == "UNKNOWN")
    assert np.all(states[15:] == "STATIONARY")
    for end in (10, 16, 40):
        a, b = run(speed[:end])
        np.testing.assert_array_equal(a, filtered[:end])
        np.testing.assert_array_equal(b, states[:end])


def test_median_rejects_isolated_spike_and_hysteresis_retains_stop():
    speed = np.r_[np.zeros(30), 10., np.zeros(10), np.full(30, .25)]
    filtered, states = run(speed)
    assert filtered[30] == 0
    assert np.all(states[15:] == "STATIONARY")


def test_creeping_does_not_mean_stopped_and_boundaries_do_not_chatter():
    _, states = run(np.r_[np.full(30, .6), np.tile([1.1, 1.4], 30)])
    assert np.all(states[9:] == "CREEPING")
    assert action_label_from_velocity(.2, 0, motion_state="CREEPING") == "SLOW_FORWARD"
    assert action_label_from_velocity(.3, .4, motion_state="STATIONARY") == "STOP"
    _, states = run(np.r_[np.full(30, 2.), np.full(30, 1.2)])
    assert np.all(states[9:] == "MOVING")


def test_stop_and_restart_in_one_horizon():
    _, states = run(np.r_[np.full(30, 2.), np.zeros(30), np.full(30, 2.)])
    assert future_motion_events(states[20:]) == (False, True, True, True)
    assert future_motion_events(states[50:60]) == (True, False, False, True)


def test_stop_must_be_confirmed_before_horizon_ends():
    _, states = run(np.r_[np.full(30, .6), np.zeros(30)])
    assert not future_motion_events(states[20:48])[1]
    assert future_motion_events(states[20:49])[1]


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.])
def test_invalid_speed_resets_confirmation(bad):
    _, states = run(np.r_[np.zeros(30), bad, np.zeros(30)])
    assert np.all(states[30:46] == "UNKNOWN")
    assert states[46] == "STATIONARY"
    assert future_motion_events(states[20:]) == (False, False, False, False)


def test_gap_resets_dwell_and_bad_timestamps_rejected():
    t = np.arange(60) / 30
    t[30:] += 1
    _, states = motion_states(np.zeros(60), t, MotionLabelConfig())
    assert states[29] == "STATIONARY" and states[30] == "UNKNOWN"
    with pytest.raises(ValueError, match="strictly increasing"):
        motion_states([0, 0], [1, 1], MotionLabelConfig())


@pytest.mark.parametrize('settings', [dict(median_window_frames=2),
    dict(stationary_exit_mps=.1), dict(moving_enter_sec=0),
    dict(stationary_enter_sec=float('inf'))])
def test_invalid_config_rejected(settings):
    with pytest.raises(ValidationError):
        MotionLabelConfig(**settings)


@pytest.mark.parametrize('speed', [.15, .35, 1., 1.2, 1.5])
def test_initial_speed_in_hysteresis_band_can_be_classified(speed):
    _, states = run(np.full(30, speed))
    assert np.all(states[9:] == "CREEPING")
