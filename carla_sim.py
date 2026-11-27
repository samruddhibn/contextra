"""
CARLA backend for live_demo.py  (python live_demo.py --sim carla)

Same interface as sim_world.BuiltinSim, so the demo does not care which one it
drives: context(), step(dt, responder), render(w, h), label, close().

Spawns an autopilot ego vehicle with a chase camera, reads the driving context
from the simulation, and lets the Responder slow the car / change to the right
lane / stop through CARLA's Traffic Manager.

STATUS: written against the CARLA 0.9.x Python API, NOT tested on a real server
(the development laptop cannot run CARLA). Test on a GPU machine before relying on it.
"""

import random
import time

import cv2
import numpy as np

from carla_bridge import read_context, spawn_vehicle


class CarlaSim:
    def __init__(self, host="localhost", port=2000, weather=None, minutes_offset=0.0):
        import carla
        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        if weather:
            preset = getattr(carla.WeatherParameters, weather, None)
            if preset is None:
                raise SystemExit(f"Unknown weather preset: {weather}")
            self.world.set_weather(preset)

        self.vehicle = spawn_vehicle(self.world)
        self.tm = self.client.get_trafficmanager()
        self.vehicle.set_autopilot(True, self.tm.get_port())

        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "640")
        bp.set_attribute("image_size_y", "480")
        bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12.0))
        self.camera = self.world.spawn_actor(bp, cam_tf, attach_to=self.vehicle)
        self._frame = None
        self.camera.listen(self._on_image)

        self.start = time.time()
        self.minutes_offset = minutes_offset
        self._mode = "NORMAL"
        self.speed_kmh = 0.0
        self.label = "CARLA"

    def _on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self._frame = arr[:, :, :3].copy()      # BGRA -> BGR

    def context(self):
        ctx = read_context(self.world, self.vehicle, self.start, self.minutes_offset)
        self.speed_kmh = ctx.speed_kmh
        return ctx

    def step(self, dt, responder):
        if responder.mode != self._mode:
            self._mode = responder.mode
            if self._mode == "SLOWING":
                self.tm.vehicle_percentage_speed_difference(self.vehicle, 30.0)
            elif self._mode == "PULLING OVER":
                self.tm.vehicle_percentage_speed_difference(self.vehicle, 100.0)   # 100 = come to a stop
                self.tm.force_lane_change(self.vehicle, True)                      # True = right lane
            else:                                                                  # NORMAL / STOPPED
                self.tm.vehicle_percentage_speed_difference(self.vehicle, 0.0 if self._mode == "NORMAL" else 100.0)

    def render(self, w=640, h=480):
        if self._frame is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return cv2.resize(self._frame, (w, h))

    def close(self):
        try:
            self.camera.stop()
            self.camera.destroy()
            self.vehicle.destroy()
        except Exception:
            pass
