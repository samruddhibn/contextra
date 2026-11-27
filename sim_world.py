"""
Built-in lightweight driving simulator for the CONTEXTRA live demo.

A simple pseudo-3D road view rendered with OpenCV/numpy. It runs on any laptop
(no GPU, no CARLA) and gives the demo a closed loop: the context comes from the
simulated drive, and the ego car reacts to the risk (slow down / pull over).

It is a SIMULATION, not CARLA, and is labelled that way on screen.
"""

import math
import random

import cv2
import numpy as np

from context_provider import DrivingContext, ScenarioProvider

ROAD_HALF = 7.0         # metres from centre line to road edge (2 lanes)
SHOULDER = 2.5          # metres of shoulder beyond the edge
LANE_X = 1.75           # ego lane centre (right lane)
SHOULDER_X = ROAD_HALF + 1.2
VIEW_DEPTH = 130.0
Z0 = 7.0                # camera distance parameter for the projection


class BuiltinSim:
    def __init__(self, scenario_name, seed=7):
        self.provider = ScenarioProvider(scenario_name)
        self.rng = random.Random(seed)
        self.t = 0.0
        self.distance = 0.0
        self.speed_kmh = 0.0
        self.ego_x = LANE_X
        self.vehicles = [self._new_vehicle(initial=True) for _ in range(5)]
        self.label = f"built-in simulator: {scenario_name}"

    # ------------------------------------------------------------- vehicles
    def _new_vehicle(self, initial=False):
        lane = self.rng.choice([-LANE_X, LANE_X])
        z = self.rng.uniform(25, VIEW_DEPTH) if initial else self.rng.uniform(VIEW_DEPTH, VIEW_DEPTH + 60)
        return {"z": z, "x": lane, "speed": self.rng.uniform(40, 90),
                "color": self.rng.choice([(200, 120, 40), (60, 160, 220), (190, 190, 190), (40, 190, 230)])}

    # ------------------------------------------------------------ simulation
    def scenario_context(self):
        return self.provider.get(self.t)

    def context(self):
        """Context seen by the reasoning layer: scenario values with the REAL ego speed."""
        c = self.scenario_context()
        c.speed_kmh = self.speed_kmh
        return c

    def step(self, dt, responder):
        self.t += dt
        target = self.scenario_context().speed_kmh * responder.speed_factor
        accel = 14.0 if responder.mode == "PULLING OVER" else 10.0   # km/h per second
        if self.speed_kmh < target:
            self.speed_kmh = min(target, self.speed_kmh + 12.0 * dt)
        else:
            self.speed_kmh = max(target, self.speed_kmh - accel * dt)

        shoulder_goal = SHOULDER_X if responder.wants_shoulder else LANE_X
        if abs(self.ego_x - shoulder_goal) > 0.01:
            step = 1.6 * dt
            self.ego_x += max(-step, min(step, shoulder_goal - self.ego_x))

        v_ms = self.speed_kmh / 3.6
        self.distance += v_ms * dt
        for v in self.vehicles:
            v["z"] += (v["speed"] / 3.6 - v_ms) * dt
        for i, v in enumerate(self.vehicles):
            if v["z"] < -12 or v["z"] > VIEW_DEPTH + 120:
                self.vehicles[i] = self._new_vehicle()

    # -------------------------------------------------------------- rendering
    def _curve_term(self, z, curvature):
        # sign flips slowly so the road bends left and right
        sign = math.sin(self.distance / 180.0)
        return sign * curvature * 380.0 * (z / VIEW_DEPTH) ** 2

    def render(self, w=640, h=480):
        ctx = self.scenario_context()
        night = ctx.is_night
        img = np.zeros((h, w, 3), dtype=np.uint8)
        hor = int(h * 0.38)
        sky = (60, 40, 25) if night else (200, 170, 120)
        ground = (35, 55, 35) if night else (70, 120, 80)
        img[:hor] = sky
        img[hor:] = ground
        S = w / 12.0

        def proj(z, xw):
            k = Z0 / (Z0 + z)
            x = w / 2 + ((xw - self.ego_x) * S) * k + self._curve_term(z, ctx.road_curvature) * k
            y = hor + (h - hor) * k
            return int(x), int(y), k

        # road + shoulder as depth strips
        zs = np.concatenate([np.linspace(0, 12, 14), np.linspace(12, VIEW_DEPTH, 40)[1:]])
        for a, b in zip(zs[:-1], zs[1:]):
            for xl, xr, col in ((-ROAD_HALF - SHOULDER, ROAD_HALF + SHOULDER, (105, 105, 100)),
                                (-ROAD_HALF, ROAD_HALF, (78, 78, 78))):
                p = [proj(a, xl)[:2], proj(a, xr)[:2], proj(b, xr)[:2], proj(b, xl)[:2]]
                cv2.fillConvexPoly(img, np.array(p, np.int32), col)
        # edge lines
        for xe in (-ROAD_HALF, ROAD_HALF):
            pts = np.array([proj(z, xe)[:2] for z in zs], np.int32)
            cv2.polylines(img, [pts], False, (230, 230, 230), 2)
        # dashed centre line
        span, dash = 14.0, 6.0
        offset = self.distance % span
        z = -offset
        while z < VIEW_DEPTH:
            z1, z2 = max(z, 0.2), z + dash
            if z2 > 0.2:
                p1, p2 = proj(z1, 0), proj(z2, 0)
                cv2.line(img, p1[:2], p2[:2], (220, 220, 220), max(1, int(4 * p1[2])))
            z += span

        # other vehicles (far to near)
        for v in sorted(self.vehicles, key=lambda q: -q["z"]):
            if 2.0 < v["z"] < VIEW_DEPTH:
                x, y, k = proj(v["z"], v["x"])
                vw, vh = int(1.9 * S * k), int(1.7 * S * k)
                cv2.rectangle(img, (x - vw // 2, y - vh), (x + vw // 2, y), v["color"], -1)
                cv2.rectangle(img, (x - vw // 2, y - vh), (x + vw // 2, y - vh + max(2, vh // 3)), (40, 40, 40), -1)

        # ego car (camera follows it, so it stays centred)
        ex, ey = w // 2, h - 28
        cv2.rectangle(img, (ex - 50, ey - 62), (ex + 50, ey), (40, 40, 200), -1)
        cv2.rectangle(img, (ex - 38, ey - 62), (ex + 38, ey - 40), (60, 60, 60), -1)
        cv2.rectangle(img, (ex - 50, ey - 8), (ex - 30, ey), (20, 20, 20), -1)
        cv2.rectangle(img, (ex + 30, ey - 8), (ex + 50, ey), (20, 20, 20), -1)

        # visibility (fog / rain) and night
        fog = float(np.clip(1.0 - ctx.visibility, 0.0, 0.95))
        if fog > 0.02:
            cv2.addWeighted(np.full_like(img, (190, 190, 190)), fog * 0.85, img, 1 - fog * 0.85, 0, img)
        if night:
            img = (img * 0.55).astype(np.uint8)
        return img
