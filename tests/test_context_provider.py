import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import context_provider as cp


class ContextTests(unittest.TestCase):
    def test_every_scenario_produces_valid_values_and_loops(self):
        for name in cp.SCENARIOS:
            p = cp.ScenarioProvider(name)
            for t in range(0, 400, 7):
                c = p.get(t)
                self.assertTrue(0 <= c.road_curvature <= 1)
                self.assertTrue(0 <= c.visibility <= 1)
                self.assertGreaterEqual(c.speed_kmh, 0)
            a, b = p.get(10), p.get(10 + p.spec["duration"])
            self.assertAlmostEqual(a.speed_kmh, b.speed_kmh)

    def test_unknown_scenario(self):
        with self.assertRaises(KeyError):
            cp.ScenarioProvider("nope")

    def test_categories_and_night(self):
        c = cp.DrivingContext(road_curvature=0.8, visibility=0.3, hour_of_day=23)
        self.assertEqual(c.road_category(), "sharp_turn")
        self.assertEqual(c.visibility_category(), "poor")
        self.assertEqual(c.is_night, 1.0)
        self.assertEqual(cp.DrivingContext(hour_of_day=12).is_night, 0.0)

    def test_context_file_roundtrip_and_staleness(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ctx.json")
            self.assertIsNone(cp.read_context_file(path))
            cp.write_context_file(cp.DrivingContext(speed_kmh=77, source="carla"), path)
            got = cp.read_context_file(path)
            self.assertEqual(got.speed_kmh, 77)
            self.assertEqual(got.source, "carla")
            self.assertIsNone(cp.read_context_file(path, max_age_seconds=-1))

    def test_carla_mappings(self):
        self.assertEqual(cp.visibility_from_weather(0, 0), 1.0)
        self.assertLess(cp.visibility_from_weather(100, 100), 0.2)
        self.assertEqual(cp.curvature_from_yaw_delta(0), 0.0)
        self.assertEqual(cp.curvature_from_yaw_delta(90), 1.0)
        self.assertEqual(cp.curvature_from_yaw_delta(350), cp.curvature_from_yaw_delta(-10))
        self.assertEqual(cp.roughness_from_vertical_accel(-10), 1.0)
        self.assertEqual(cp.hour_from_sun_altitude(-5), 23.0)
        self.assertEqual(cp.traffic_from_nearby_vehicles(100), 1.0)


if __name__ == "__main__":
    unittest.main()
