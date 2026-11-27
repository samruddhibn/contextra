"""
Driving-context sources for CONTEXTRA.

The reasoning layer only depends on the DrivingContext dataclass, so any source
can feed it:

    ScenarioProvider    built-in lightweight simulator (scripted drives)
    manual sliders      handled in the dashboard
    CARLA bridge        carla_bridge.py writes context_state.json, read here

All values are SIMULATED unless the source is "carla" running a real server.
"""

from dataclasses import dataclass, asdict, fields

from driver_state import CONTEXT_FILE, atomic_write_json, read_json, age_seconds, utc_now_iso


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


@dataclass
class DrivingContext:
    speed_kmh: float = 0.0
    road_curvature: float = 0.0   # 0 straight ... 1 very sharp bend
    road_roughness: float = 0.0   # 0 smooth ... 1 very rough
    visibility: float = 1.0       # 1 clear ... 0 very poor (fog / heavy rain)
    traffic_density: float = 0.0  # 0 empty ... 1 congested
    hour_of_day: float = 12.0     # 0-24
    drive_minutes: float = 0.0    # time already spent driving
    source: str = "manual"

    @property
    def is_night(self):
        return 1.0 if (self.hour_of_day >= 19.5 or self.hour_of_day < 6.0) else 0.0

    def road_category(self):
        if self.road_curvature >= 0.6:
            return "sharp_turn"
        if self.road_roughness >= 0.6:
            return "rough_road"
        return "normal"

    def visibility_category(self):
        return "poor" if self.visibility < 0.5 else "good"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        names = {f.name for f in fields(DrivingContext)}
        return DrivingContext(**{k: v for k, v in (d or {}).items() if k in names})


# ------------------------------------------------------------ built-in simulator
# Each scenario is a looping list of (second, values) keyframes; values are
# linearly interpolated between keyframes.
SCENARIOS = {
    "Highway, daytime, clear": {
        "duration": 90,
        "frames": [
            (0,  dict(speed_kmh=95,  road_curvature=0.05, road_roughness=0.10, visibility=1.0, traffic_density=0.25, hour_of_day=14.0, drive_minutes=45)),
            (45, dict(speed_kmh=110, road_curvature=0.10, road_roughness=0.10, visibility=1.0, traffic_density=0.35, hour_of_day=14.3, drive_minutes=50)),
            (90, dict(speed_kmh=100, road_curvature=0.05, road_roughness=0.10, visibility=1.0, traffic_density=0.25, hour_of_day=14.6, drive_minutes=55)),
        ],
    },
    "Night drive, fog, winding road": {
        "duration": 90,
        "frames": [
            (0,  dict(speed_kmh=70, road_curvature=0.30, road_roughness=0.30, visibility=0.85, traffic_density=0.10, hour_of_day=23.0, drive_minutes=140)),
            (30, dict(speed_kmh=65, road_curvature=0.85, road_roughness=0.35, visibility=0.30, traffic_density=0.10, hour_of_day=23.3, drive_minutes=150)),
            (60, dict(speed_kmh=55, road_curvature=0.90, road_roughness=0.40, visibility=0.20, traffic_density=0.05, hour_of_day=23.6, drive_minutes=160)),
            (90, dict(speed_kmh=60, road_curvature=0.40, road_roughness=0.30, visibility=0.45, traffic_density=0.10, hour_of_day=23.9, drive_minutes=170)),
        ],
    },
    "City rush hour": {
        "duration": 90,
        "frames": [
            (0,  dict(speed_kmh=25, road_curvature=0.20, road_roughness=0.20, visibility=1.0, traffic_density=0.90, hour_of_day=18.0, drive_minutes=25)),
            (45, dict(speed_kmh=40, road_curvature=0.30, road_roughness=0.25, visibility=0.9, traffic_density=0.80, hour_of_day=18.3, drive_minutes=35)),
            (90, dict(speed_kmh=20, road_curvature=0.20, road_roughness=0.20, visibility=1.0, traffic_density=0.95, hour_of_day=18.6, drive_minutes=45)),
        ],
    },
    "Long overnight drive, rough road": {
        "duration": 90,
        "frames": [
            (0,  dict(speed_kmh=80, road_curvature=0.20, road_roughness=0.75, visibility=0.8, traffic_density=0.05, hour_of_day=2.5, drive_minutes=300)),
            (45, dict(speed_kmh=85, road_curvature=0.35, road_roughness=0.85, visibility=0.7, traffic_density=0.05, hour_of_day=2.8, drive_minutes=310)),
            (90, dict(speed_kmh=80, road_curvature=0.20, road_roughness=0.75, visibility=0.8, traffic_density=0.05, hour_of_day=3.1, drive_minutes=320)),
        ],
    },
}


def _interp(frames, t):
    if t <= frames[0][0]:
        return dict(frames[0][1])
    for (t0, a), (t1, b) in zip(frames, frames[1:]):
        if t0 <= t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return {k: a[k] + (b[k] - a[k]) * w for k in a}
    return dict(frames[-1][1])


class ScenarioProvider:
    """Built-in lightweight simulator: replays a scripted drive in a loop."""

    def __init__(self, name):
        if name not in SCENARIOS:
            raise KeyError(f"Unknown scenario: {name}")
        self.name = name
        self.spec = SCENARIOS[name]

    def get(self, elapsed_s):
        duration = self.spec["duration"]
        t = elapsed_s % duration
        values = _interp(self.spec["frames"], t)
        return DrivingContext(source=f"simulator: {self.name}", **values)


# --------------------------------------------------------------- file handoff
def write_context_file(ctx, path=None):
    payload = ctx.to_dict()
    payload["timestamp_utc"] = utc_now_iso()
    return atomic_write_json(path or CONTEXT_FILE, payload)


def read_context_file(path=None, max_age_seconds=3.0):
    """Returns a DrivingContext, or None if missing / stale."""
    data = read_json(path or CONTEXT_FILE, None)
    if not isinstance(data, dict):
        return None
    age = age_seconds(data.get("timestamp_utc"))
    if age is None or age > max_age_seconds or age < -0.5:
        return None
    return DrivingContext.from_dict(data)


# ------------------------------------------- CARLA value mapping (pure functions)
# Kept here (not in carla_bridge.py) so they can be unit-tested without CARLA.
def visibility_from_weather(fog_density, precipitation):
    """CARLA weather values are 0-100. Returns visibility 0-1 (1 = clear)."""
    loss = 0.8 * (fog_density / 100.0) + 0.5 * (precipitation / 100.0)
    return clamp(1.0 - loss, 0.05, 1.0)


def curvature_from_yaw_delta(delta_deg, lookahead_note="~20 m ahead"):
    """Heading change between the vehicle's waypoint and one ~20 m ahead.
    45 degrees or more maps to curvature 1.0."""
    d = abs((delta_deg + 180.0) % 360.0 - 180.0)
    return clamp(d / 45.0)


def roughness_from_vertical_accel(accel_z_ms2):
    """Crude proxy: CARLA has no road-roughness value, so vertical acceleration
    is used (gravity compensated by CARLA). 5 m/s^2 or more maps to 1.0."""
    return clamp(abs(accel_z_ms2) / 5.0)


def hour_from_sun_altitude(sun_altitude_deg):
    """Coarse day/night mapping used only to set the night flag."""
    return 14.0 if sun_altitude_deg > 0 else 23.0


def traffic_from_nearby_vehicles(count_within_50m):
    return clamp(count_within_50m / 8.0)
