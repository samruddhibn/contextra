"""
Driver fatigue / distraction features computed from MediaPipe face landmarks.

Pure Python + math (no OpenCV / MediaPipe import) so it can be unit-tested
with synthetic geometry. All time-based logic takes an explicit timestamp.

Features produced per update():
    perclos            fraction of the last window the eyes were closed
                       (PERCLOS-style; closure = below a per-person threshold)
    blink_rate_per_min short closures per minute
    longest_closure_s  longest eye closure in the window (or current)
    yawns_5min         sustained wide-open-mouth events in last 5 minutes
    head_nod           1.0 if head pitch deviated from neutral recently
    distraction_fraction  fraction of window with head turned away from road

IMPORTANT: Thresholds are calibrated per person (first seconds of monitoring)
but are NOT validated against a labelled drowsiness dataset yet.
"""

import math
from collections import deque
from statistics import median

# MediaPipe FaceLandmarker (478-point) indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(p):
    """p = six (x, y) eye points in MediaPipe order [corner, top, top, corner, bottom, bottom]."""
    horizontal = _dist(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horizontal)


def mouth_aspect_ratio(top, bottom, left, right):
    width = _dist(left, right)
    if width == 0:
        return 0.0
    return _dist(top, bottom) / width


def head_pose_from_matrix(m):
    """
    Yaw, pitch, roll in degrees from MediaPipe's 4x4 facial transformation
    matrix. Frontal face is close to (0, 0, 0). Only the MAGNITUDE of the
    deviation from the person's neutral pose is used by the tracker, so the
    sign convention does not matter.
    """
    r = [[float(m[i][j]) for j in range(3)] for i in range(3)]
    yaw = math.degrees(math.atan2(-r[2][0], math.hypot(r[2][1], r[2][2])))
    pitch = math.degrees(math.atan2(r[2][1], r[2][2]))
    roll = math.degrees(math.atan2(r[1][0], r[0][0]))
    return yaw, pitch, roll


class FatigueTracker:
    def __init__(
        self,
        window_s=60.0,
        yawn_window_s=300.0,
        closed_ratio=0.72,      # eye is "closed" below 72% of this person's open-eye EAR
        blink_max_s=0.5,        # closures up to this long are normal blinks
        blink_min_s=0.05,       # shorter than this is landmark noise
        yawn_mar=0.55,          # mouth aspect ratio that counts as wide open
        yawn_min_s=1.5,         # mouth must stay open this long to count as a yawn
        yaw_limit_deg=25.0,     # head turned this far from neutral = looking away
        pitch_limit_deg=18.0,   # head tilted this far from neutral = nod
        pose_hold_s=0.8,        # pose must persist this long to raise a flag
        nod_memory_s=10.0,      # a nod stays "recent" for this long
        min_calibration_samples=20,
        max_dt=0.5,
    ):
        self.window_s = window_s
        self.yawn_window_s = yawn_window_s
        self.closed_ratio = closed_ratio
        self.blink_max_s = blink_max_s
        self.blink_min_s = blink_min_s
        self.yawn_mar = yawn_mar
        self.yawn_min_s = yawn_min_s
        self.yaw_limit = yaw_limit_deg
        self.pitch_limit = pitch_limit_deg
        self.pose_hold_s = pose_hold_s
        self.nod_memory_s = nod_memory_s
        self.min_cal = min_calibration_samples
        self.max_dt = max_dt

        self.reset_calibration()

    # ---------------------------------------------------------- calibration
    def reset_calibration(self):
        self._cal_ear, self._cal_yaw, self._cal_pitch = [], [], []
        self.baseline_ear = None
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self._reset_runtime()

    def _reset_runtime(self):
        self._samples = deque()      # (t, dt, closed, away)
        self._episodes = deque()     # (t_end, duration)
        self._yawns = deque()        # t of each yawn
        self._last_t = None
        self._closure_start = None
        self._mouth_start = None
        self._yawn_counted = False
        self._away_start = None
        self._nod_start = None
        self._last_nod_t = None
        self._face_lost_since = None

    @property
    def calibrated(self):
        return self.baseline_ear is not None

    def add_calibration_sample(self, ear, yaw=0.0, pitch=0.0):
        self._cal_ear.append(float(ear))
        self._cal_yaw.append(float(yaw))
        self._cal_pitch.append(float(pitch))

    @property
    def calibration_samples(self):
        return len(self._cal_ear)

    def finish_calibration(self):
        if len(self._cal_ear) < self.min_cal:
            return False
        self.baseline_ear = median(self._cal_ear)
        self.baseline_yaw = median(self._cal_yaw)
        self.baseline_pitch = median(self._cal_pitch)
        self._reset_runtime()
        return True

    # ---------------------------------------------------------------- update
    def update(self, t, face_detected, ear=None, mar=None, yaw=None, pitch=None):
        if not self.calibrated:
            return self._snapshot(t, face_detected, ear, mar, 0.0, 0.0)

        if not face_detected:
            # A lost face is NOT treated as closed eyes (no evidence either way).
            if self._face_lost_since is None:
                self._face_lost_since = t
            self._last_t = t
            self._closure_start = None
            self._mouth_start = None
            self._yawn_counted = False
            self._away_start = None
            self._nod_start = None
            return self._snapshot(t, False, None, None, 0.0, 0.0)

        self._face_lost_since = None
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), self.max_dt)
        self._last_t = t

        yaw_dev = abs((yaw if yaw is not None else self.baseline_yaw) - self.baseline_yaw)
        pitch_dev = abs((pitch if pitch is not None else self.baseline_pitch) - self.baseline_pitch)

        # ---- eye closure
        closed = ear is not None and ear < self.closed_ratio * self.baseline_ear
        if closed:
            if self._closure_start is None:
                self._closure_start = t
        elif self._closure_start is not None:
            duration = t - self._closure_start
            if duration >= self.blink_min_s:
                self._episodes.append((t, duration))
            self._closure_start = None

        # ---- yawning (sustained wide-open mouth)
        if mar is not None and mar >= self.yawn_mar:
            if self._mouth_start is None:
                self._mouth_start = t
            if (t - self._mouth_start) >= self.yawn_min_s and not self._yawn_counted:
                self._yawns.append(t)
                self._yawn_counted = True
        else:
            self._mouth_start = None
            self._yawn_counted = False

        # ---- head pose
        away_now = yaw_dev >= self.yaw_limit
        if away_now:
            if self._away_start is None:
                self._away_start = t
        else:
            self._away_start = None

        if pitch_dev >= self.pitch_limit:
            if self._nod_start is None:
                self._nod_start = t
            if (t - self._nod_start) >= self.pose_hold_s:
                self._last_nod_t = t
        else:
            self._nod_start = None

        self._samples.append((t, dt, closed, away_now))
        self._trim(t)
        return self._snapshot(t, True, ear, mar, yaw_dev, pitch_dev)

    def _trim(self, t):
        while self._samples and t - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        while self._episodes and t - self._episodes[0][0] > self.window_s:
            self._episodes.popleft()
        while self._yawns and t - self._yawns[0] > self.yawn_window_s:
            self._yawns.popleft()

    def _snapshot(self, t, face, ear, mar, yaw_dev, pitch_dev):
        total = sum(s[1] for s in self._samples)
        closed_t = sum(s[1] for s in self._samples if s[2])
        away_t = sum(s[1] for s in self._samples if s[3])
        perclos = closed_t / total if total > 0 else 0.0
        distraction = away_t / total if total > 0 else 0.0

        blinks = sum(1 for _, d in self._episodes if d <= self.blink_max_s)
        covered = max(total, 15.0)  # avoid inflated rates in the first seconds
        blink_rate = blinks * 60.0 / covered

        closed_now = 0.0
        if self._closure_start is not None:
            closed_now = max(0.0, t - self._closure_start)
        longest = max([closed_now] + [d for _, d in self._episodes])
        long_closures = sum(1 for _, d in self._episodes if d >= 1.0)

        yawning = (
            self._mouth_start is not None
            and (t - self._mouth_start) >= self.yawn_min_s
        )
        looking_away = (
            self._away_start is not None
            and (t - self._away_start) >= self.pose_hold_s
        )
        head_nod = (
            self._last_nod_t is not None
            and (t - self._last_nod_t) <= self.nod_memory_s
        )

        return {
            "calibrated": self.calibrated,
            "face_detected": bool(face),
            "eye_closed": self._closure_start is not None,
            "closed_duration_s": round(closed_now, 3),
            "perclos": round(perclos, 4),
            "blink_rate": round(blink_rate, 2),
            "longest_closure_s": round(longest, 3),
            "long_closures_60s": long_closures,
            "yawning": bool(yawning),
            "yawns_5min": len(self._yawns),
            "looking_away": bool(looking_away),
            "head_nod": 1.0 if head_nod else 0.0,
            "distraction_fraction": round(distraction, 4),
            "yaw_dev_deg": round(yaw_dev, 1),
            "pitch_dev_deg": round(pitch_dev, 1),
            "ear": None if ear is None else round(ear, 4),
            "baseline_ear": None if self.baseline_ear is None else round(self.baseline_ear, 4),
            "mar": None if mar is None else round(mar, 4),
        }


# Features the fusion model consumes (from a snapshot)
DRIVER_MODEL_KEYS = [
    "perclos", "blink_rate", "longest_closure_s",
    "yawns_5min", "head_nod", "distraction_fraction",
]


def frame_metrics(landmarks, matrix, width, height):
    """
    landmarks: sequence of objects with .x/.y (normalised), as returned by MediaPipe.
    matrix: 4x4 facial transformation matrix or None.
    Returns (ear, mar, yaw_deg, pitch_deg).
    """
    def pt(i):
        p = landmarks[i]
        return (p.x * width, p.y * height)

    left = eye_aspect_ratio([pt(i) for i in LEFT_EYE])
    right = eye_aspect_ratio([pt(i) for i in RIGHT_EYE])
    ear = (left + right) / 2.0
    mar = mouth_aspect_ratio(pt(MOUTH_TOP), pt(MOUTH_BOTTOM), pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
    if matrix is not None:
        yaw, pitch, _ = head_pose_from_matrix(matrix)
    else:
        yaw, pitch = 0.0, 0.0
    return ear, mar, yaw, pitch
