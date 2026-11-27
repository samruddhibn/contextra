import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import fusion_model as fm
from context_provider import DrivingContext

ALERT = dict(perclos=0.03, blink_rate=17, longest_closure_s=0.25, yawns_5min=0, head_nod=0, distraction_fraction=0.02)
DROWSY = dict(perclos=0.28, blink_rate=27, longest_closure_s=2.2, yawns_5min=4, head_nod=1, distraction_fraction=0.05)
CALM = DrivingContext(speed_kmh=40, road_curvature=0.1, visibility=1.0, hour_of_day=14, drive_minutes=20)
BAD = DrivingContext(speed_kmh=95, road_curvature=0.85, road_roughness=0.5, visibility=0.25,
                     traffic_density=0.3, hour_of_day=23, drive_minutes=200)


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bundle, cls.metrics = fm.train_and_evaluate(
            n=3000, save=True,
            model_file=os.path.join(cls.tmp.name, "m.joblib"),
            metrics_file=os.path.join(cls.tmp.name, "metrics.json"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_is_deterministic(self):
        X1, y1 = fm.generate_dataset(300, seed=1)
        X2, y2 = fm.generate_dataset(300, seed=1)
        self.assertTrue(np.array_equal(X1, X2) and np.array_equal(y1, y2))
        self.assertEqual(set(y1), {0, 1, 2})

    def test_alert_calm_is_low(self):
        self.assertEqual(fm.predict(self.bundle, ALERT, CALM)["level"], "LOW")

    def test_drowsy_bad_conditions_is_high(self):
        self.assertEqual(fm.predict(self.bundle, DROWSY, BAD)["level"], "HIGH")

    def test_same_drowsy_driver_is_riskier_in_bad_context(self):
        calm = fm.predict(self.bundle, DROWSY, CALM)["expected_risk"]
        bad = fm.predict(self.bundle, DROWSY, BAD)["expected_risk"]
        self.assertGreater(bad, calm)

    def test_distraction_at_speed_raises_risk(self):
        away = dict(ALERT, distraction_fraction=0.5)
        highway = DrivingContext(speed_kmh=110, hour_of_day=14)
        self.assertGreater(fm.predict(self.bundle, away, highway)["expected_risk"],
                           fm.predict(self.bundle, ALERT, highway)["expected_risk"])

    def test_probabilities_sum_to_one_and_explanation_present(self):
        r = fm.predict(self.bundle, DROWSY, BAD)
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=3)
        self.assertTrue(r["top_factors"])
        self.assertTrue(all(f["delta"] > 0 for f in r["top_factors"]))

    def test_metrics_structure_and_fusion_beats_single_signal(self):
        comp = self.metrics["comparisons"]
        self.assertGreater(comp["Fusion model (driver + context)"]["macro_f1"],
                           comp["PERCLOS-only threshold"]["macro_f1"])
        self.assertIn("SYNTHETIC", self.metrics["data_note"])

    def test_rule_baseline_runs_on_dataset_rows(self):
        X, _ = fm.generate_dataset(50, seed=3)
        self.assertEqual(len(fm.rule_baseline_predict(X)), 50)


if __name__ == "__main__":
    unittest.main()
