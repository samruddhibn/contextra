import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import driver_state as ds
from contextra_engine import reason


class EngineScenarioTests(unittest.TestCase):
    def level(self, **kw):
        return reason(**kw)["risk_level"], reason(**kw)["risk_score"]

    def test_1_all_normal(self):
        self.assertEqual(self.level(driver_eye_closed=False, road_condition="normal", visibility="good"), ("LOW", 0))

    def test_2_closure_only(self):
        self.assertEqual(self.level(driver_eye_closed=True, road_condition="normal", visibility="good"), ("ELEVATED", 3))

    def test_3_closure_sharp_turn(self):
        self.assertEqual(self.level(driver_eye_closed=True, road_condition="sharp_turn", visibility="good"), ("HIGH", 5))

    def test_4_closure_poor_visibility(self):
        self.assertEqual(self.level(driver_eye_closed=True, road_condition="normal", visibility="poor"), ("HIGH", 5))

    def test_5_everything(self):
        self.assertEqual(self.level(driver_eye_closed=True, road_condition="sharp_turn", visibility="poor"), ("HIGH", 7))

    def test_deterministic(self):
        a = reason(driver_eye_closed=True, road_condition="rough_road", visibility="poor")
        b = reason(driver_eye_closed=True, road_condition="rough_road", visibility="poor")
        a.pop("timestamp_utc"); b.pop("timestamp_utc")
        self.assertEqual(a, b)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            reason(road_condition="lava")
        with self.assertRaises(ValueError):
            reason(visibility="blurry")
        with self.assertRaises(ValueError):
            reason(vehicle_speed_kmh=-5)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "driver_state.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_gives_default_not_fresh(self):
        state = ds.get_driver_state(self.path)
        self.assertFalse(ds.is_state_fresh(state))

    def test_fresh_when_running_and_recent(self):
        ds.update_driver_state(True, "open", path=self.path, calibrated=True, features={"perclos": 0.1})
        state = ds.get_driver_state(self.path)
        self.assertTrue(ds.is_state_fresh(state))
        self.assertEqual(state["features"]["perclos"], 0.1)

    def test_stopped_monitor_is_not_fresh(self):
        ds.update_driver_state(False, monitor_running=False, path=self.path)
        self.assertFalse(ds.is_state_fresh(ds.get_driver_state(self.path)))

    def test_old_timestamp_is_stale(self):
        ds.update_driver_state(True, path=self.path)
        state = ds.get_driver_state(self.path)
        later = datetime.now(timezone.utc) + timedelta(seconds=10)
        self.assertFalse(ds.is_state_fresh(state, now=later))

    def test_corrupt_file_falls_back(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertFalse(ds.is_state_fresh(ds.get_driver_state(self.path)))

    def test_advisory_roundtrip_and_expiry(self):
        adv_path = os.path.join(self.tmp.name, "adv.json")
        ds.write_advisory("1-5", "HIGH", "slow down", speak=True, path=adv_path)
        adv = ds.read_advisory(adv_path)
        self.assertEqual(adv["level"], "HIGH")
        self.assertIsNone(ds.read_advisory(adv_path, max_age_seconds=-1))


if __name__ == "__main__":
    unittest.main()
