"""Causal motion labeling from raw telemetry. No future smoothing or backdating."""
from collections import deque

import numpy as np

from .config_utils.schema import MotionLabelConfig

LABELER_VERSION = 2


def motion_states(speed, timestamps, config: MotionLabelConfig):
    """Return trailing median speed and confirmed state at each raw frame.

    Start UNKNOWN until a state is confirmed. Dwell times measure elapsed time
    from the first qualifying filtered reading. Invalid speeds reset the filter
    and state; a timestamp gap breaks continuity too. Warm-up uses available
    readings only. Threshold comparisons are strict; equality retains state.
    """
    speed = np.asarray(speed, dtype=float)
    times = np.asarray(timestamps, dtype=float)
    if speed.ndim != 1 or times.shape != speed.shape:
        raise ValueError("speed and timestamps must be matching 1-D arrays")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    filtered = np.full(len(speed), np.nan)
    states = np.full(len(speed), "UNKNOWN", dtype="U10")
    history = deque(maxlen=config.median_window_frames)
    state = "UNKNOWN"
    starts = {}
    nominal_dt = np.median(np.diff(times)) if len(times) > 1 else np.inf
    for i, (v, t) in enumerate(zip(speed, times)):
        if i and t - times[i - 1] > 1.5 * nominal_dt:
            history.clear()
            starts.clear()
            state = "UNKNOWN"
        if not np.isfinite(v) or v < 0:
            history.clear()
            starts.clear()
            state = "UNKNOWN"
            continue
        history.append(v)
        v = filtered[i] = np.median(history)
        conditions = {
            "stop": v < config.stationary_enter_mps,
            "leave_stop": v > config.stationary_exit_mps,
            "creep": v < config.creeping_enter_mps,
            "move": v > config.moving_enter_mps,
            "initial_creep": config.stationary_enter_mps <= v <= config.moving_enter_mps,
        }
        for name, active in conditions.items():
            if active:
                starts.setdefault(name, t)
            else:
                starts.pop(name, None)

        def held(name, duration):
            return name in starts and t - starts[name] + 1e-9 >= duration

        if state == "STATIONARY":
            if held("leave_stop", config.stationary_exit_sec):
                state = "CREEPING"
        elif held("stop", config.stationary_enter_sec):
            state = "STATIONARY"
        elif state == "MOVING":
            if held("creep", config.creeping_enter_sec):
                state = "CREEPING"
        elif held("move", config.moving_enter_sec):
            state = "MOVING"
        elif state == "UNKNOWN":
            if held("initial_creep", config.creeping_enter_sec):
                state = "CREEPING"
        states[i] = state
    return filtered, states


def future_motion_events(states):
    """Flags over an inclusive key-to-horizon sequence of confirmed states.

    Stop and start transitions may both occur. UNKNOWN anywhere makes the event
    flags unavailable (all false, events_valid false), not evidence of no event.
    """
    states = np.asarray(states)
    valid = len(states) > 0 and np.all(states != "UNKNOWN")
    if not valid:
        return False, False, False, False
    stopped = states == "STATIONARY"
    return (bool(stopped.all()), bool(np.any(~stopped[:-1] & stopped[1:])),
            bool(np.any(stopped[:-1] & ~stopped[1:])), True)
