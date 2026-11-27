"""
CONTEXTRA contextual fusion model (the "AI reasoning" layer).

What it is
----------
A RandomForest classifier that fuses DRIVER fatigue/distraction features with
DRIVING CONTEXT (speed, road, visibility, traffic, time of day, drive length)
into a 3-level risk (LOW / ELEVATED / HIGH) with class probabilities.

What it is NOT  (be upfront about this in the presentation)
-----------------------------------------------------------
There is no public dataset that links these inputs to real accident outcomes.
The training labels therefore come from an expert-knowledge scenario simulator
(`expert_risk` below): fatigue x speed x hazard interactions plus noise. All
reported metrics are on that SYNTHETIC scenario data. They show that the
pipeline works and that continuous multi-signal fusion beats single-signal and
threshold rules ON THIS DATA - not that it predicts real-world crashes.

Per-prediction explanations use occlusion: each feature is replaced by a
"safe" reference value and the drop in expected risk is that feature's
contribution. This is deterministic and easy to audit.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "fusion_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

LABELS = ["LOW", "ELEVATED", "HIGH"]
DRIVER_FEATURES = [
    "perclos", "blink_rate", "longest_closure_s",
    "yawns_5min", "head_nod", "distraction_fraction",
]
CONTEXT_FEATURES = [
    "speed_kmh", "road_curvature", "road_roughness", "visibility",
    "traffic_density", "is_night", "drive_minutes",
]
FEATURES = DRIVER_FEATURES + CONTEXT_FEATURES
SEED = 42


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# --------------------------------------------------------------- feature rows
def build_row(driver, ctx):
    """driver: dict with DRIVER_FEATURES; ctx: DrivingContext -> dict of FEATURES."""
    row = {k: float(driver.get(k, 0.0) or 0.0) for k in DRIVER_FEATURES}
    row.update({
        "speed_kmh": float(ctx.speed_kmh),
        "road_curvature": float(ctx.road_curvature),
        "road_roughness": float(ctx.road_roughness),
        "visibility": float(ctx.visibility),
        "traffic_density": float(ctx.traffic_density),
        "is_night": float(ctx.is_night),
        "drive_minutes": float(ctx.drive_minutes),
    })
    return row


def to_vector(row):
    return [row[k] for k in FEATURES]


# ---------------------------------------------------- expert scenario simulator
def expert_risk(r):
    """Latent risk in roughly 0..1+ from an expert-knowledge scenario model."""
    fatigue = (
        0.50 * _clip(r["perclos"] / 0.25)
        + 0.30 * _clip(r["longest_closure_s"] / 2.0)
        + 0.15 * _clip(r["yawns_5min"] / 4.0)
        + 0.15 * r["head_nod"]
        + 0.05 * _clip(abs(r["blink_rate"] - 17.0) / 15.0)
    )
    fatigue = _clip(fatigue)
    distraction = _clip(r["distraction_fraction"] / 0.35)
    hazard = (
        0.35 * r["road_curvature"]
        + 0.15 * r["road_roughness"]
        + 0.30 * (1.0 - r["visibility"])
        + 0.10 * r["traffic_density"]
        + 0.10 * r["is_night"]
    )
    exposure = _clip(r["speed_kmh"] / 120.0)
    time_on_task = _clip(r["drive_minutes"] / 240.0)

    return (
        0.60 * fatigue * (0.5 + 0.9 * exposure) * (1.0 + hazard)
        + 0.35 * distraction * (0.3 + 0.9 * exposure) * (1.0 + hazard)
        + 0.12 * hazard * exposure
        + 0.10 * time_on_task * fatigue
    )


LOW_MAX = 0.20
ELEVATED_MAX = 0.42


def label_from_risk(x):
    if x < LOW_MAX:
        return 0
    if x < ELEVATED_MAX:
        return 1
    return 2


def generate_dataset(n=6000, seed=SEED, noise=0.04):
    """Synthetic scenario dataset (clearly labelled synthetic)."""
    rng = random.Random(seed)
    rows, labels = [], []

    def gauss(m, s, lo, hi):
        return _clip(rng.gauss(m, s), lo, hi)

    def poisson(lam):
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rng.random()
            if p <= L:
                return k
            k += 1

    for _ in range(n):
        state = rng.choices(["alert", "mild", "drowsy", "distracted"], [50, 20, 12, 18])[0]
        if state == "alert":
            d = dict(perclos=gauss(0.04, 0.02, 0, 1), blink_rate=gauss(17, 4, 4, 40),
                     longest_closure_s=gauss(0.25, 0.10, 0.05, 5), yawns_5min=poisson(0.2),
                     head_nod=1.0 if rng.random() < 0.02 else 0.0,
                     distraction_fraction=_clip(rng.expovariate(1 / 0.05)))
        elif state == "mild":
            d = dict(perclos=gauss(0.10, 0.04, 0, 1), blink_rate=gauss(21, 5, 4, 45),
                     longest_closure_s=gauss(0.7, 0.3, 0.05, 5), yawns_5min=poisson(1.2),
                     head_nod=1.0 if rng.random() < 0.10 else 0.0,
                     distraction_fraction=_clip(rng.expovariate(1 / 0.08)))
        elif state == "drowsy":
            d = dict(perclos=gauss(0.22, 0.08, 0, 1), blink_rate=gauss(26, 7, 4, 50),
                     longest_closure_s=gauss(1.8, 0.8, 0.1, 6), yawns_5min=poisson(3.0),
                     head_nod=1.0 if rng.random() < 0.40 else 0.0,
                     distraction_fraction=_clip(rng.expovariate(1 / 0.10)))
        else:  # alert but looking away from the road
            d = dict(perclos=gauss(0.04, 0.02, 0, 1), blink_rate=gauss(17, 4, 4, 40),
                     longest_closure_s=gauss(0.25, 0.10, 0.05, 5), yawns_5min=poisson(0.2),
                     head_nod=0.0, distraction_fraction=gauss(0.35, 0.15, 0.05, 1))

        scene = rng.choices(["city", "highway", "rural"], [35, 35, 30])[0]
        speed = {"city": rng.gauss(30, 12), "highway": rng.gauss(95, 15), "rural": rng.gauss(60, 15)}[scene]
        d.update(
            speed_kmh=max(0.0, min(160.0, speed)),
            road_curvature=_clip(rng.betavariate(1.5, 4)),
            road_roughness=_clip(rng.betavariate(1.5, 5)),
            visibility=_clip(1.0 - rng.betavariate(1.2, 5)),
            traffic_density=_clip(rng.betavariate(2, 3)),
            is_night=1.0 if rng.random() < 0.30 else 0.0,
            drive_minutes=min(480.0, rng.expovariate(1 / 90.0)),
        )
        risk = expert_risk(d) + rng.gauss(0, noise)
        rows.append(to_vector(d))
        labels.append(label_from_risk(risk))
    return np.array(rows, dtype=float), np.array(labels, dtype=int)


# ----------------------------------------------------------- baselines (rules)
def rule_baseline_predict(X):
    """The original transparent rule engine, applied to the same rows."""
    from contextra_engine import reason
    idx = {k: i for i, k in enumerate(FEATURES)}
    out = []
    for x in X:
        road = "sharp_turn" if x[idx["road_curvature"]] >= 0.6 else (
            "rough_road" if x[idx["road_roughness"]] >= 0.6 else "normal")
        vis = "poor" if x[idx["visibility"]] < 0.5 else "good"
        res = reason(
            driver_eye_closed=x[idx["longest_closure_s"]] >= 1.5,
            road_condition=road, visibility=vis,
            vehicle_speed_kmh=float(x[idx["speed_kmh"]]),
        )
        out.append(LABELS.index(res["risk_level"]))
    return np.array(out)


def perclos_only_predict(X):
    """Single-signal baseline: PERCLOS thresholds only."""
    p = X[:, FEATURES.index("perclos")]
    return np.where(p < 0.10, 0, np.where(p < 0.20, 1, 2))


# ------------------------------------------------------------------- training
def _new_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1)


def _scores(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            LABELS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                        "f1": round(float(f[i]), 4)} for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def train_and_evaluate(n=6000, seed=SEED, save=True, model_file=None, metrics_file=None):
    import joblib
    import sklearn
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    X, y = generate_dataset(n=n, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)

    model = _new_forest().fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # latency of one live prediction
    t0 = time.perf_counter()
    for i in range(200):
        model.predict_proba(X_te[i % len(X_te)].reshape(1, -1))
    latency_ms = (time.perf_counter() - t0) / 200 * 1000

    # baselines + ablations (all on the same held-out rows)
    comparisons = {
        "Rule engine (original, hand-weighted)": _scores(y_te, rule_baseline_predict(X_te)),
        "PERCLOS-only threshold": _scores(y_te, perclos_only_predict(X_te)),
        "Fusion model (driver + context)": _scores(y_te, pred),
    }
    di = [FEATURES.index(k) for k in DRIVER_FEATURES]
    ci = [FEATURES.index(k) for k in CONTEXT_FEATURES]
    m_drv = _new_forest().fit(X_tr[:, di], y_tr)
    m_ctx = _new_forest().fit(X_tr[:, ci], y_tr)
    comparisons["Ablation: driver features only (no context)"] = _scores(y_te, m_drv.predict(X_te[:, di]))
    comparisons["Ablation: context features only (no driver)"] = _scores(y_te, m_ctx.predict(X_te[:, ci]))

    cv = cross_val_score(_new_forest(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                         scoring="f1_macro")

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])

    low_ref = np.median(X_tr[y_tr == 0], axis=0)
    reference = {k: float(v) for k, v in zip(FEATURES, low_ref)}

    metrics = {
        "data_note": "SYNTHETIC scenario data from an expert-knowledge simulator. "
                     "Not real-world accident data.",
        "samples": int(n), "train": int(len(X_tr)), "test": int(len(X_te)), "seed": seed,
        "class_balance": {LABELS[i]: int((y == i).sum()) for i in range(3)},
        "comparisons": comparisons,
        "fusion_cv_macro_f1_mean": round(float(cv.mean()), 4),
        "fusion_cv_macro_f1_std": round(float(cv.std()), 4),
        "inference_latency_ms": round(latency_ms, 2),
        "feature_importance": [[k, round(float(v), 4)] for k, v in importances],
        "sklearn_version": sklearn.__version__,
    }

    bundle = {"model": model, "features": FEATURES, "labels": LABELS,
              "reference": reference, "sklearn_version": sklearn.__version__}
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_file or MODEL_FILE)
        with open(metrics_file or METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return bundle, metrics


def ensure_model(force=False):
    """Load the saved model, or (re)train if missing / incompatible. Takes a few seconds."""
    import joblib
    import sklearn
    if not force and MODEL_FILE.exists():
        try:
            bundle = joblib.load(MODEL_FILE)
            if bundle.get("sklearn_version") == sklearn.__version__ and bundle.get("features") == FEATURES:
                return bundle
        except Exception:
            pass
    bundle, _ = train_and_evaluate(save=True)
    return bundle


def load_metrics():
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ----------------------------------------------------------------- prediction
def _expected_risk(probs):
    return float(probs[1] * 0.5 + probs[2] * 1.0)


def describe_feature(name, value):
    v = value
    return {
        "perclos": f"eyes were closed {v * 100:.0f}% of the last minute (PERCLOS)",
        "blink_rate": f"blink rate is {v:.0f} per minute",
        "longest_closure_s": f"longest eye closure was {v:.1f} s",
        "yawns_5min": f"{int(round(v))} yawn(s) in the last 5 minutes",
        "head_nod": "head nodding / tilting away from neutral",
        "distraction_fraction": f"head turned away from the road {v * 100:.0f}% of the last minute",
        "speed_kmh": f"vehicle speed is {v:.0f} km/h",
        "road_curvature": f"road curvature ahead is high ({v:.2f})",
        "road_roughness": f"road is rough ({v:.2f})",
        "visibility": f"visibility is reduced ({v:.2f}, 1.0 = clear)",
        "traffic_density": f"traffic is dense ({v:.2f})",
        "is_night": "it is night-time",
        "drive_minutes": f"driving for {v:.0f} minutes without a reset",
    }[name]


def predict(bundle, driver, ctx, top_k=4):
    model, reference = bundle["model"], bundle["reference"]
    row = build_row(driver, ctx)
    x = np.array([to_vector(row)], dtype=float)
    probs = model.predict_proba(x)[0]
    # RandomForest classes are 0,1,2; guard against missing classes
    full = np.zeros(3)
    for cls, p in zip(model.classes_, probs):
        full[int(cls)] = p
    probs = full
    base = _expected_risk(probs)
    level_idx = int(np.argmax(probs))

    # one batched call: the actual row plus one "feature replaced by safe value" row per feature
    variants, idxs = [], []
    for i, name in enumerate(FEATURES):
        if x[0, i] == reference[name]:
            continue
        x2 = x[0].copy()
        x2[i] = reference[name]
        variants.append(x2)
        idxs.append(i)

    contributions = []
    if variants:
        P = model.predict_proba(np.array(variants))
        full_p = np.zeros((len(variants), 3))
        for col, cls in enumerate(model.classes_):
            full_p[:, int(cls)] = P[:, col]
        for i, p2 in zip(idxs, full_p):
            name = FEATURES[i]
            delta = base - _expected_risk(p2)
            contributions.append({
                "feature": name,
                "value": float(x[0, i]),
                "delta": round(float(delta), 4),
                "text": describe_feature(name, float(x[0, i])),
            })
    contributions.sort(key=lambda c: -c["delta"])
    top = [c for c in contributions if c["delta"] >= 0.02][:top_k]

    return {
        "level": LABELS[level_idx],
        "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(3)},
        "expected_risk": round(base, 4),
        "top_factors": top,
        "all_contributions": contributions,
        "inputs": row,
    }
