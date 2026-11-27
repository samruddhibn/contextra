"""
CONTEXTRA live demo: webcam + driving simulation + AI reasoning in ONE window.

    python live_demo.py                              # built-in simulator (runs anywhere)
    python live_demo.py --scenario "Night drive, fog, winding road"
    python live_demo.py --sim carla --weather HardRainNight   # needs a CARLA server (untested)
    python live_demo.py --video clip.mp4             # recorded clip instead of the webcam

Keys: V test voice | C recalibrate | D toggle SIMULATED drowsy driver |
      R restart simulation | Q or ESC quit

What happens: the webcam gives fatigue/distraction features, the simulated drive
gives the context, the trained fusion model decides LOW / ELEVATED / HIGH, and
the SIMULATED ego car slows, pulls over to the shoulder and stops if HIGH risk
persists; it resumes only after the driver has been clearly alert for a while.
Advisory research prototype - it never controls a real vehicle.
All video is processed locally; frames are never stored.
"""

import argparse
import os
import random
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

import contextra_monitor as cm
import fusion_model as fm
from context_provider import SCENARIOS, write_context_file
from driver_state import update_driver_state
from fatigue_features import FatigueTracker, frame_metrics
from responder import PULLING_OVER, STOPPED, Responder

W, H = 640, 480
HEADER, BANNER = 44, 56
GREEN, AMBER, RED, WHITE, CYAN = (80, 200, 80), (0, 190, 255), (50, 50, 230), (255, 255, 255), (255, 255, 0)
LEVEL_COLOR = {"LOW": GREEN, "ELEVATED": AMBER, "HIGH": RED, None: (180, 180, 180)}

DROWSY_FEATURES = dict(perclos=0.28, blink_rate=27.0, longest_closure_s=2.2, yawns_5min=3,
                       head_nod=1.0, distraction_fraction=0.05)


def text(img, s, x, y, color=WHITE, scale=0.55, thick=1):
    # outline = same font and thickness drawn 1 px away in black (keeps glyph spacing identical)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w]
    cv2.addWeighted(np.zeros_like(sub), alpha, sub, 1 - alpha, 0, sub)


def fit(frame, w, h):
    """Centre-crop and resize to w x h."""
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    resized = cv2.resize(frame, (int(fw * scale) + 1, int(fh * scale) + 1))
    x0, y0 = (resized.shape[1] - w) // 2, (resized.shape[0] - h) // 2
    return resized[y0:y0 + h, x0:x0 + w].copy()


def compose(cam, sim_img, snap, result, ctx, responder, sim, simulated_driver, status_note=""):
    level = result["level"] if result else None
    canvas = np.zeros((HEADER + H + BANNER, W * 2, 3), dtype=np.uint8)
    canvas[HEADER:HEADER + H, :W] = cam
    canvas[HEADER:HEADER + H, W:] = sim_img
    top = HEADER

    # header bar
    canvas[:HEADER] = (30, 30, 30)
    text(canvas, "SIMULATED DRIVER" if simulated_driver else "LIVE WEBCAM", 12, 29,
         AMBER if simulated_driver else GREEN, 0.75, 2)
    text(canvas, f"RISK: {level or 'NO DATA'}", 430, 29, LEVEL_COLOR[level], 0.75, 2)
    text(canvas, sim.label[:52], W + 12, 29, WHITE, 0.5)

    # driver panel (left)
    panel(canvas, 8, top + 8, 300, 168)
    text(canvas, "DRIVER", 16, top + 28, CYAN, 0.55, 1)
    if result:
        p = result["probabilities"]
        text(canvas, f"{level}  (P high {p['HIGH']:.2f})", 16, top + 56, LEVEL_COLOR[level], 0.75, 2)
    else:
        text(canvas, status_note or "waiting...", 16, top + 56, (180, 180, 180), 0.6, 1)
    if snap and snap["calibrated"] and snap["face_detected"]:
        eyes = "CLOSED" if snap["eye_closed"] else "open"
        text(canvas, f"Eyes: {eyes}  closed {snap['closed_duration_s']:.1f}s", 16, top + 82)
        text(canvas, f"PERCLOS {snap['perclos'] * 100:.0f}%   blinks/min {snap['blink_rate']:.0f}", 16, top + 104)
        text(canvas, f"Yawning: {'YES' if snap['yawning'] else 'no'}   yawns(5m) {snap['yawns_5min']}", 16, top + 126)
        text(canvas, f"Head turn {snap['yaw_dev_deg']:.0f} deg  tilt {snap['pitch_dev_deg']:.0f} deg", 16, top + 148)
        text(canvas, "Nodding" if snap["head_nod"] else "", 16, top + 168, AMBER)

    # vehicle panel (right)
    px = W + 8
    panel(canvas, px, top + 8, 300, 168)
    text(canvas, "VEHICLE (simulated)", px + 8, top + 28, CYAN)
    text(canvas, f"Speed: {sim.speed_kmh:.0f} km/h", px + 8, top + 54, WHITE, 0.65, 2)
    mode_color = RED if responder.mode in (PULLING_OVER, STOPPED) else AMBER if responder.mode == "SLOWING" else GREEN
    text(canvas, f"Mode: {responder.mode}", px + 8, top + 80, mode_color, 0.65, 2)
    text(canvas, f"Visibility {ctx.visibility:.2f}  curve {ctx.road_curvature:.2f}", px + 8, top + 104)
    text(canvas, f"Hour {ctx.hour_of_day:.1f}  driving {ctx.drive_minutes:.0f} min", px + 8, top + 126)
    text(canvas, f"Traffic {ctx.traffic_density:.2f}  roughness {ctx.road_roughness:.2f}", px + 8, top + 148)

    # why (top factors) under the driver panel
    if result and result["top_factors"]:
        panel(canvas, 8, top + 182, 420, 24 + 20 * len(result["top_factors"][:3]))
        text(canvas, "WHY:", 16, top + 200, CYAN)
        for i, f in enumerate(result["top_factors"][:3]):
            text(canvas, f"+{f['delta']:.2f} {f['text']}"[:62], 16, top + 220 + 20 * i, WHITE, 0.45)

    # bottom banner
    by = HEADER + H
    msg, color = "", None
    if responder.mode == PULLING_OVER:
        msg, color = "CRITICAL RISK - PULLING OVER SAFELY", (0, 0, 180)
    elif responder.mode == STOPPED:
        msg, color = f"STOPPED - resumes when you are alert ({responder.resume_progress * 100:.0f}%)", (0, 0, 140)
    elif snap and snap.get("closed_duration_s", 0) >= cm.MICROSLEEP_SECONDS:
        msg, color = "CRITICAL DROWSINESS - WAKE UP!", (0, 0, 200)
    elif snap and snap.get("looking_away"):
        msg, color = "EYES ON THE ROAD!", (0, 90, 200)
    elif level == "HIGH":
        msg, color = "HIGH RISK - take a break soon", (0, 0, 180)
    elif level == "ELEVATED":
        msg, color = "CAUTION - stay alert", (0, 120, 200)
    if color:
        canvas[by:by + BANNER] = color
        text(canvas, msg, 16, by + 38, WHITE, 0.9, 2)
    else:
        canvas[by:by + BANNER] = (40, 40, 40)
        text(canvas, "V voice | C recalibrate | D simulated drowsy driver | R restart sim | Q quit",
             16, by + 34, (200, 200, 200), 0.5)
    return canvas


def make_sim(args):
    if args.sim == "carla":
        from carla_sim import CarlaSim
        return CarlaSim(args.host, args.port, args.weather, args.minutes_offset)
    from sim_world import BuiltinSim
    return BuiltinSim(args.scenario)


def main():
    p = argparse.ArgumentParser(description="CONTEXTRA live demo")
    p.add_argument("--sim", choices=["builtin", "carla"], default="builtin")
    p.add_argument("--scenario", default="Night drive, fog, winding road", choices=list(SCENARIOS))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--weather", default=None)
    p.add_argument("--minutes-offset", type=float, default=0.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--inject-drowsy-after", type=float, default=None,
                   help="test aid: switch to the SIMULATED drowsy driver after N seconds")
    p.add_argument("--save-frame", default=None, help="save the composite window to this PNG at --save-at")
    p.add_argument("--save-at", type=float, default=None)
    args = p.parse_args()

    bundle = fm.ensure_model()
    sim = make_sim(args)
    responder = Responder()
    tracker = FatigueTracker()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=cm.MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
        output_facial_transformation_matrixes=True)

    threading.Thread(target=cm.voice_worker, daemon=True).start()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("ERROR: webcam / video could not be opened.")
        cm.shutdown.set()
        raise SystemExit(1)

    start = time.monotonic()
    prev_ts, last_t = -1, start
    cal_start = None
    snap = None
    result = None
    last_predict = last_publish = last_ctx_write = 0.0
    simulated_driver = False
    last_said = {k: -999.0 for k in cm.COOLDOWNS}
    last_yawn_count = 0
    prev_mode = responder.mode
    saved = False

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            print("\n=== CONTEXTRA LIVE DEMO ===")
            print("Calibrating: look at the road, eyes open, mouth closed (5 s).")
            cm.speak("CONTEXTRA live demo. Please look straight ahead.")
            while True:
                now = time.monotonic()
                elapsed = now - start
                if args.max_seconds and elapsed > args.max_seconds:
                    break
                if args.inject_drowsy_after is not None and elapsed >= args.inject_drowsy_after:
                    simulated_driver = True

                ok, frame = cap.read()
                if not ok and args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    print("ERROR: cannot read frame.")
                    break
                if args.video:
                    time.sleep(1 / 30)
                else:
                    frame = cv2.flip(frame, 1)

                h0, w0 = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(int((now - start) * 1000), prev_ts + 1)
                prev_ts = ts
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if res.face_landmarks:
                    matrix = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else None
                    ear, mar, yaw, pitch = frame_metrics(res.face_landmarks[0], matrix, w0, h0)
                    if not tracker.calibrated:
                        cal_start = cal_start or now
                        tracker.add_calibration_sample(ear, yaw, pitch)
                        if now - cal_start >= cm.CALIBRATION_SECONDS and tracker.finish_calibration():
                            print(f"Calibrated. Baseline EAR {tracker.baseline_ear:.3f}")
                            cm.speak("Calibration complete.")
                    snap = tracker.update(now, True, ear, mar, yaw, pitch)
                else:
                    snap = tracker.update(now, False)
                    if not tracker.calibrated:
                        cal_start = None

                dt = now - last_t
                last_t = now

                # driver evidence: real webcam features, or the clearly-labelled simulated driver
                feats, status_note = None, ""
                if simulated_driver:
                    feats = dict(DROWSY_FEATURES)
                elif not tracker.calibrated:
                    status_note = "calibrating - look at the road"
                elif not snap["face_detected"]:
                    status_note = "no face - no driver evidence"
                else:
                    feats = {k: snap[k] for k in fm.DRIVER_FEATURES}

                ctx = sim.context()
                if feats is not None and now - last_predict >= 0.5:
                    result = fm.predict(bundle, feats, ctx)
                    last_predict = now
                elif feats is None:
                    result = None

                alert_ok = bool(feats) and not simulated_driver and snap["perclos"] < 0.08 \
                    and snap["closed_duration_s"] < 0.5 and not snap["head_nod"]
                responder.update(dt, result["level"] if result else None, alert_ok, sim.speed_kmh)
                sim.step(dt, responder)

                # ---------------- voice
                if not simulated_driver and tracker.calibrated and snap["face_detected"]:
                    if snap["closed_duration_s"] >= cm.MICROSLEEP_SECONDS and now - last_said["sleep"] >= cm.COOLDOWNS["sleep"]:
                        cm.speak(random.choice(cm.SLEEP_ALERTS)); last_said["sleep"] = now
                    if snap["yawns_5min"] > last_yawn_count and now - last_said["yawn"] >= cm.COOLDOWNS["yawn"]:
                        cm.speak(random.choice(cm.YAWN_ALERTS)); last_said["yawn"] = now
                    last_yawn_count = snap["yawns_5min"]
                    if snap["head_nod"] and now - last_said["nod"] >= cm.COOLDOWNS["nod"]:
                        cm.speak(random.choice(cm.NOD_ALERTS)); last_said["nod"] = now
                    if snap["looking_away"] and now - last_said["away"] >= cm.COOLDOWNS["away"]:
                        cm.speak(random.choice(cm.AWAY_ALERTS)); last_said["away"] = now
                if responder.mode != prev_mode:
                    if responder.mode == PULLING_OVER:
                        cm.speak("Critical risk. Pulling over safely.")
                    elif responder.mode == "NORMAL" and prev_mode == STOPPED:
                        cm.speak("Driver alert again. Resuming.")
                    prev_mode = responder.mode

                # ---------------- share state with the dashboard
                if now - last_publish >= cm.PUBLISH_INTERVAL and snap:
                    update_driver_state(
                        face_detected=snap["face_detected"],
                        eye_state="unknown" if not snap["face_detected"] else "closed" if snap["eye_closed"] else "open",
                        eye_closed_duration_seconds=snap["closed_duration_s"],
                        prolonged_eye_closure=snap["closed_duration_s"] >= cm.MICROSLEEP_SECONDS,
                        calibrated=tracker.calibrated,
                        features={k: snap[k] for k in fm.DRIVER_FEATURES} if snap["calibrated"] else {})
                    last_publish = now
                if now - last_ctx_write >= 0.5:
                    write_context_file(ctx)
                    last_ctx_write = now

                canvas = compose(fit(frame, W, H), fit(sim.render(W, H), W, H), snap, result, ctx,
                                 responder, sim, simulated_driver, status_note)
                if args.save_frame and args.save_at is not None and elapsed >= args.save_at and not saved:
                    cv2.imwrite(args.save_frame, canvas)
                    saved = True
                if args.headless:
                    key = 255
                else:
                    cv2.imshow("CONTEXTRA - live demo", canvas)
                    key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("v"):
                    cm.speak(random.choice(cm.SLEEP_ALERTS))
                elif key == ord("c"):
                    tracker.reset_calibration(); cal_start = None; last_yawn_count = 0
                    cm.speak("Recalibrating. Please look straight ahead.")
                elif key == ord("d"):
                    simulated_driver = not simulated_driver
                elif key == ord("r"):
                    sim = make_sim(args) if args.sim == "builtin" else sim
                    responder = Responder()
    finally:
        try:
            update_driver_state(face_detected=False, monitor_running=False)
        except Exception as error:
            print("Could not publish shutdown state:", error)
        if hasattr(sim, "close"):
            sim.close()
        cap.release()
        cv2.destroyAllWindows()
        cm.shutdown.set()
        cm.voice_event.set()


if __name__ == "__main__":
    main()
