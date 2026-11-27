import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import responder as rp
import sim_world as sw


def feed(r, level, seconds, alert_ok=False, speed=60.0, dt=0.1):
    for _ in range(int(seconds / dt)):
        r.update(dt, level, alert_ok, speed)


class ResponderTests(unittest.TestCase):
    def test_low_risk_stays_normal(self):
        r = rp.Responder()
        feed(r, "LOW", 20)
        self.assertEqual(r.mode, rp.NORMAL)
        self.assertEqual(r.speed_factor, 1.0)

    def test_elevated_slows(self):
        r = rp.Responder()
        feed(r, "ELEVATED", 2)
        self.assertEqual(r.mode, rp.SLOWING)
        self.assertLess(r.speed_factor, 1.0)

    def test_brief_high_does_not_pull_over(self):
        r = rp.Responder(high_to_pullover_s=3.0)
        feed(r, "HIGH", 1.5)
        self.assertEqual(r.mode, rp.SLOWING)

    def test_sustained_high_pulls_over_then_stops(self):
        r = rp.Responder(high_to_pullover_s=3.0)
        feed(r, "HIGH", 3.5)
        self.assertEqual(r.mode, rp.PULLING_OVER)
        self.assertTrue(r.wants_shoulder)
        feed(r, "HIGH", 1, speed=0.0)
        self.assertEqual(r.mode, rp.STOPPED)

    def test_does_not_resume_until_driver_alert_long_enough(self):
        r = rp.Responder(high_to_pullover_s=1.0, alert_resume_s=5.0)
        feed(r, "HIGH", 2)
        feed(r, "HIGH", 1, speed=0.0)
        self.assertEqual(r.mode, rp.STOPPED)
        feed(r, "LOW", 4, alert_ok=True, speed=0.0)        # alert but not long enough
        self.assertEqual(r.mode, rp.STOPPED)
        feed(r, "LOW", 2, alert_ok=False, speed=0.0)       # drowsy again -> timer resets
        feed(r, "LOW", 4, alert_ok=True, speed=0.0)
        self.assertEqual(r.mode, rp.STOPPED)
        feed(r, "LOW", 2, alert_ok=True, speed=0.0)
        self.assertEqual(r.mode, rp.NORMAL)

    def test_no_driver_data_never_triggers_pull_over(self):
        r = rp.Responder()
        feed(r, None, 30)
        self.assertEqual(r.mode, rp.NORMAL)


class SimTests(unittest.TestCase):
    def test_render_shape_and_all_scenarios(self):
        from context_provider import SCENARIOS
        for name in SCENARIOS:
            sim = sw.BuiltinSim(name)
            r = rp.Responder()
            for _ in range(40):
                sim.step(0.05, r)
            img = sim.render(640, 480)
            self.assertEqual(img.shape, (480, 640, 3))
            self.assertEqual(img.dtype, np.uint8)

    def test_car_slows_stops_and_moves_to_shoulder(self):
        sim = sw.BuiltinSim("Highway, daytime, clear")
        r = rp.Responder()
        for _ in range(100):
            sim.step(0.1, r)
        cruising = sim.speed_kmh
        self.assertGreater(cruising, 50)
        r.mode = rp.PULLING_OVER
        for _ in range(400):
            sim.step(0.1, r)
        self.assertLess(sim.speed_kmh, 1.0)
        self.assertGreater(sim.ego_x, sw.ROAD_HALF)         # on the shoulder, out of the lane

    def test_context_uses_real_ego_speed(self):
        sim = sw.BuiltinSim("Highway, daytime, clear")
        sim.speed_kmh = 12.0
        self.assertEqual(sim.context().speed_kmh, 12.0)


if __name__ == "__main__":
    unittest.main()
