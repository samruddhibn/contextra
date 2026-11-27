"""
Shared-state helpers for CONTEXTRA.

The webcam monitor, the dashboard and the (optional) CARLA bridge run as
separate processes, so they talk through small JSON files that are written
atomically:

    driver_state.json    monitor      -> dashboard (fatigue features, heartbeat)
    context_state.json   carla_bridge -> dashboard (driving context)
    advisory_state.json  dashboard    -> monitor   (context-aware voice advisory)
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "driver_state.json"
CONTEXT_FILE = BASE_DIR / "context_state.json"
ADVISORY_FILE = BASE_DIR / "advisory_state.json"

DEFAULT_STATE = {
    "timestamp_utc": None,
    "face_detected": False,
    "eye_state": "unknown",
    "eye_closed_duration_seconds": 0.0,
    "prolonged_eye_closure": False,
    "signal_confidence": None,
    "monitor_running": False,
    "monitor_pid": None,
    "calibrated": False,
    "features": {},
}


def atomic_write_json(path, obj):
    """Atomically write JSON. Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            delete=False, suffix=".tmp",
        ) as temp_file:
            json.dump(obj, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name

        # On Windows, replace can fail briefly if a reader has the file open.
        for _ in range(5):
            try:
                os.replace(temp_path, path)
                temp_path = None
                return True
            except PermissionError:
                time.sleep(0.02)
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def read_json(path, default=None):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def age_seconds(timestamp_iso, now=None):
    """Seconds since an ISO timestamp, or None if it cannot be parsed."""
    if not timestamp_iso:
        return None
    try:
        t = datetime.fromisoformat(timestamp_iso)
    except (ValueError, TypeError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - t.astimezone(timezone.utc)).total_seconds()


# ---------------------------------------------------------------- driver state
def update_driver_state(
    face_detected,
    eye_state="unknown",
    eye_closed_duration_seconds=0.0,
    prolonged_eye_closure=False,
    signal_confidence=None,
    monitor_running=True,
    calibrated=False,
    features=None,
    path=None,
):
    state = {
        "timestamp_utc": utc_now_iso(),
        "face_detected": bool(face_detected),
        "eye_state": str(eye_state),
        "eye_closed_duration_seconds": max(0.0, float(eye_closed_duration_seconds)),
        "prolonged_eye_closure": bool(prolonged_eye_closure),
        "signal_confidence": signal_confidence,
        "monitor_running": bool(monitor_running),
        "monitor_pid": os.getpid(),
        "calibrated": bool(calibrated),
        "features": dict(features or {}),
    }
    atomic_write_json(path or STATE_FILE, state)
    return state


def get_driver_state(path=None):
    state = read_json(path or STATE_FILE, None)
    if not isinstance(state, dict):
        return {**DEFAULT_STATE, "features": {}}
    return {**DEFAULT_STATE, **state}


def is_state_fresh(state, max_age_seconds=3.0, now=None):
    """True only if the monitor says it is running AND wrote recently."""
    if not state.get("monitor_running", False):
        return False
    age = age_seconds(state.get("timestamp_utc"), now)
    if age is None:
        return False
    return -0.5 <= age <= max_age_seconds


# -------------------------------------------------------------------- advisory
def write_advisory(advisory_id, level, message, speak=True, path=None):
    return atomic_write_json(path or ADVISORY_FILE, {
        "timestamp_utc": utc_now_iso(),
        "id": str(advisory_id),
        "level": str(level),
        "message": str(message),
        "speak": bool(speak),
    })


def read_advisory(path=None, max_age_seconds=10.0):
    adv = read_json(path or ADVISORY_FILE, None)
    if not isinstance(adv, dict):
        return None
    age = age_seconds(adv.get("timestamp_utc"))
    if age is None or age > max_age_seconds:
        return None
    return adv
