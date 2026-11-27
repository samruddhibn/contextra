import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fatigue_features import (FatigueTracker, eye_aspect_ratio, head_pose_from_matrix,
                              mouth_aspect_ratio)


def eye(open_amount):
    # corner, top, top, corner, bottom, bottom
    return [(0, 0), (1, -open_amount), (2, -open_amount), (3, 0), (2, open_amount), (1, open_amount)]


class GeometryTests(unittest.TestCase):
    def test_ear_open_greater_than_closed(self):
        self.assertGreater(eye_aspect_ratio(eye(0.5)), eye_aspect_ratio(eye(0.05)))

    def test_ear_degenerate_eye_is_zero(self):
        self.assertEqual(eye_aspect_ratio([(0, 0)] * 6), 0.0)

    def test_mar(self):
        self.assertAlmostEqual(mouth_aspect_ratio((0, 0), (0, 1), (-1, 0.5), (1, 0.5)), 0.5)

    def test_head_pose_identity_and_yaw(self):
        ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        yaw, pitch, roll = head_pose_from_matrix(ident)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        a = math.radians(30)
        rot = [[math.cos(a), 0, math.sin(a), 0], [0, 1, 0, 0], [-math.sin(a), 0, math.cos(a), 0], [0, 0, 0, 1]]
        yaw, _, _ = head_pose_from_matrix(rot)
        self.assertAlmostEqual(abs(yaw), 30.0, places=3)


def calibrated_tracker():
    t = FatigueTracker()
    for _ in range(30):
        t.add_calibration_sample(0.30, 0.0, 0.0)
    assert t.finish_calibration()
    return t


def run(tracker, start, seconds, ear=0.30, mar=0.05, yaw=0.0, pitch=0.0, fps=30, face=True):
    snap, t = None, start
    for _ in range(int(seconds * fps)):
        t += 1.0 / fps
        snap = tracker.update(t, face, ear, mar, yaw, pitch)
    return snap, t


class TrackerTests(unittest.TestCase):
    def test_not_calibrated_until_enough_samples(self):
        t = FatigueTracker()
        t.add_calibration_sample(0.3)
        self.assertFalse(t.finish_calibration())
        self.assertFalse(t.calibrated)

    def test_open_eyes_zero_perclos(self):
        snap, _ = run(calibrated_tracker(), 0, 10)
        self.assertEqual(snap["perclos"], 0.0)
        self.assertFalse(snap["eye_closed"])

    def test_long_closure_detected(self):
        t = calibrated_tracker()
        _, now = run(t, 0, 5)
        snap, now = run(t, now, 2.0, ear=0.05)
        self.assertGreaterEqual(snap["closed_duration_s"], 1.9)
        self.assertTrue(snap["eye_closed"])
        snap, _ = run(t, now, 1.0)
        self.assertGreaterEqual(snap["longest_closure_s"], 1.9)
        self.assertGreater(snap["perclos"], 0.15)

    def test_blinks_are_counted_not_flagged_as_long(self):
        t = calibrated_tracker()
        now = 0.0
        for _ in range(6):
            _, now = run(t, now, 2.0)
            _, now = run(t, now, 0.15, ear=0.05)
        snap, _ = run(t, now, 1.0)
        self.assertGreater(snap["blink_rate"], 5)
        self.assertLess(snap["longest_closure_s"], 0.5)
        self.assertEqual(snap["long_closures_60s"], 0)

    def test_yawn_needs_sustained_open_mouth(self):
        t = calibrated_tracker()
        _, now = run(t, 0, 2)
        snap, now = run(t, now, 0.5, mar=0.8)       # brief opening = talking
        snap, now = run(t, now, 1.0)
        self.assertEqual(snap["yawns_5min"], 0)
        snap, now = run(t, now, 2.0, mar=0.8)       # sustained = yawn
        self.assertEqual(snap["yawns_5min"], 1)
        self.assertTrue(snap["yawning"])

    def test_looking_away_flag_and_fraction(self):
        t = calibrated_tracker()
        _, now = run(t, 0, 5)
        snap, now = run(t, now, 2.0, yaw=40.0)
        self.assertTrue(snap["looking_away"])
        self.assertGreater(snap["distraction_fraction"], 0.1)

    def test_head_nod_flag(self):
        t = calibrated_tracker()
        _, now = run(t, 0, 3)
        snap, _ = run(t, now, 2.0, pitch=30.0)
        self.assertEqual(snap["head_nod"], 1.0)

    def test_face_lost_is_not_closed_eyes(self):
        t = calibrated_tracker()
        _, now = run(t, 0, 5)
        snap, now = run(t, now, 4.0, face=False, ear=None)
        self.assertFalse(snap["face_detected"])
        self.assertEqual(snap["closed_duration_s"], 0.0)
        self.assertEqual(snap["perclos"], 0.0)

    def test_uncalibrated_snapshot_is_flagged(self):
        snap = FatigueTracker().update(1.0, True, 0.3, 0.05, 0, 0)
        self.assertFalse(snap["calibrated"])


if __name__ == "__main__":
    unittest.main()
