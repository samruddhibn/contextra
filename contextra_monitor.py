"""
CONTEXTRA webcam driver monitor.

    python contextra_monitor.py

* Calibrates to the driver for ~5 s (look at the road, eyes open, mouth closed).
* Extracts fatigue / distraction features (PERCLOS, blinks, long closures,
  yawns, head nod, head turned away) from MediaPipe face landmarks.
* Publishes them continuously to driver_state.json (this is also the heartbeat).
* Speaks fast-path warnings through the existing PowerShell voice, and speaks
  the context-aware advisories that the dashboard writes to advisory_state.json.

Keys: V = test voice | C = recalibrate | Q = quit
All video processing is local; frames are never saved or sent anywhere.
"""

import argparse
import os
import random
import threading
import time

import cv2
import mediapipe as mp

from driver_state import read_advisory, update_driver_state
from fatigue_features import FatigueTracker, frame_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

CALIBRATION_SECONDS = 5.0
MICROSLEEP_SECONDS = 1.5      # continuous closure that triggers the fast-path alert
PUBLISH_INTERVAL = 0.25       # seconds between state writes (also the heartbeat)

COOLDOWNS = {"sleep": 5.0, "yawn": 20.0, "nod": 12.0, "away": 4.0, "advisory": 15.0}

# No apostrophes in these lines: the text is passed to PowerShell inside quotes.
SLEEP_ALERTS = [
    "Bro, your eyes are closed. This is a car, not a bed. Wake up!",
    "No cap, you are falling asleep. Eyes open, bestie!",
    "Hello? Anyone in there? Stay awake, main character.",
    "Sleeping at the wheel is not a vibe. Open your eyes!",
    "Rise and shine, legend. The road needs you awake.",
]
YAWN_ALERTS = [
    "That was a big yawn. Consider a break soon.",
    "Yawning detected. A short rest stop might help.",
]
NOD_ALERTS = ["Head nod detected. Are you getting sleepy?"]
AWAY_ALERTS = ["Eyes on the road!", "Eyes on the road, please!"]

voice_queue = []
voice_lock = threading.Lock()
voice_event = threading.Event()
shutdown = threading.Event()


def voice_worker():
    import subprocess

    while not shutdown.is_set():
        voice_event.wait(timeout=0.2)
        if shutdown.is_set():
            break

        with voice_lock:
            message = voice_queue.pop(0) if voice_queue else None
            if not voice_queue:
                voice_event.clear()

        if message:
            print("PLAYING VOICE:", message)
            escaped_message = message.replace("'", "''")
            try:
                subprocess.run(
                    [
                        "powershell", "-NoProfile", "-Command",
                        (
                            "Add-Type -AssemblyName System.Speech; "
                            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                            "$s.Volume = 100; "
                            f"$s.Speak('{escaped_message}'); "
                            "$s.Dispose()"
                        ),
                    ],
                    check=True, timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception as error:
                print("VOICE ERROR:", error)


def speak(message):
    with voice_lock:
        if len(voice_queue) >= 2:      # never build up a backlog of stale warnings
            voice_queue.pop(0)
        voice_queue.append(message)
        voice_event.set()


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="CONTEXTRA webcam driver monitor")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--video", default=None, help="use a recorded video file instead of the webcam (backup for demos)")
    parser.add_argument("--headless", action="store_true", help="no preview window (for tests)")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop automatically after this many seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Face landmarker model not found: {MODEL_PATH}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )

    threading.Thread(target=voice_worker, daemon=True).start()

    camera = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        print("ERROR: Webcam / video could not be opened.")
        shutdown.set()
        raise SystemExit(1)

    tracker = FatigueTracker()
    start_time = time.monotonic()
    previous_timestamp = -1
    calibration_start = None
    last_publish = 0.0
    last_said = {k: -999.0 for k in COOLDOWNS}
    last_yawn_count = 0
    last_advisory_id = None
    last_advisory_check = 0.0
    snap = tracker.update(0.0, False)

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            print("\n=== CONTEXTRA DRIVER MONITOR ===")
            print("Calibrating: look at the road, eyes open, mouth closed.")
            print("V = test voice | C = recalibrate | Q = quit")
            speak("CONTEXTRA monitoring started. Please look straight ahead.")

            while True:
                if args.max_seconds and time.monotonic() - start_time > args.max_seconds:
                    break

                success, frame = camera.read()
                if not success and args.video:
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)     # loop the recorded video
                    success, frame = camera.read()
                if not success:
                    print("ERROR: Cannot read webcam frame.")
                    break
                if args.video:
                    time.sleep(1 / 30)                         # play at roughly real time

                if not args.video:
                    frame = cv2.flip(frame, 1)                 # mirror view for a live webcam
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp = int((time.monotonic() - start_time) * 1000)
                timestamp = max(timestamp, previous_timestamp + 1)
                previous_timestamp = timestamp

                result = landmarker.detect_for_video(image, timestamp)
                now = time.monotonic()

                ear = mar = None
                if result.face_landmarks:
                    matrix = (result.facial_transformation_matrixes[0]
                              if result.facial_transformation_matrixes else None)
                    ear, mar, yaw, pitch = frame_metrics(
                        result.face_landmarks[0], matrix, width, height)

                    if not tracker.calibrated:
                        if calibration_start is None:
                            calibration_start = now
                        tracker.add_calibration_sample(ear, yaw, pitch)
                        if now - calibration_start >= CALIBRATION_SECONDS:
                            if tracker.finish_calibration():
                                print(f"Calibrated. Open-eye baseline EAR = {tracker.baseline_ear:.3f}")
                                speak("Calibration complete.")
                            else:
                                calibration_start = None   # not enough samples, retry
                        snap = tracker.update(now, True, ear, mar, yaw, pitch)
                    else:
                        snap = tracker.update(now, True, ear, mar, yaw, pitch)
                else:
                    snap = tracker.update(now, False)
                    if not tracker.calibrated:
                        calibration_start = None

                # ---------------- fast-path voice warnings (driver signals only)
                if tracker.calibrated and snap["face_detected"]:
                    if snap["closed_duration_s"] >= MICROSLEEP_SECONDS and now - last_said["sleep"] >= COOLDOWNS["sleep"]:
                        print("WARNING: prolonged eye closure.")
                        speak(random.choice(SLEEP_ALERTS))
                        last_said["sleep"] = now
                    if snap["yawns_5min"] > last_yawn_count and now - last_said["yawn"] >= COOLDOWNS["yawn"]:
                        speak(random.choice(YAWN_ALERTS))
                        last_said["yawn"] = now
                    last_yawn_count = snap["yawns_5min"]
                    if snap["head_nod"] and now - last_said["nod"] >= COOLDOWNS["nod"]:
                        speak(random.choice(NOD_ALERTS))
                        last_said["nod"] = now
                    if snap["looking_away"] and now - last_said["away"] >= COOLDOWNS["away"]:
                        speak(random.choice(AWAY_ALERTS))
                        last_said["away"] = now

                # ---------------- context-aware advisory written by the dashboard
                if now - last_advisory_check >= 1.0:
                    last_advisory_check = now
                    adv = read_advisory()
                    if (adv and adv.get("speak") and adv.get("level") == "HIGH"
                            and adv.get("id") != last_advisory_id
                            and now - last_said["advisory"] >= COOLDOWNS["advisory"]):
                        speak(adv.get("message", "High risk detected."))
                        last_advisory_id = adv.get("id")
                        last_said["advisory"] = now

                # ---------------- publish state / heartbeat
                if now - last_publish >= PUBLISH_INTERVAL:
                    eye_state = ("unknown" if not snap["face_detected"]
                                 else "closed" if snap["eye_closed"] else "open")
                    update_driver_state(
                        face_detected=snap["face_detected"],
                        eye_state=eye_state,
                        eye_closed_duration_seconds=snap["closed_duration_s"],
                        prolonged_eye_closure=snap["closed_duration_s"] >= MICROSLEEP_SECONDS,
                        calibrated=tracker.calibrated,
                        features={k: snap[k] for k in (
                            "perclos", "blink_rate", "longest_closure_s", "yawns_5min",
                            "head_nod", "distraction_fraction", "yawning", "looking_away",
                            "closed_duration_s", "long_closures_60s", "yaw_dev_deg",
                            "pitch_dev_deg", "ear", "baseline_ear", "mar")},
                    )
                    last_publish = now

                # ---------------- on-screen overlay
                if not tracker.calibrated:
                    secs = 0.0 if calibration_start is None else now - calibration_start
                    put(frame, f"CALIBRATING {secs:0.1f}/{CALIBRATION_SECONDS:0.0f}s - look at the road", 35, (0, 255, 255))
                    put(frame, "Eyes open, mouth closed", 65, (0, 255, 255))
                elif not snap["face_detected"]:
                    put(frame, "NO FACE DETECTED", 35, (0, 0, 255))
                else:
                    warn = snap["closed_duration_s"] >= MICROSLEEP_SECONDS
                    put(frame, f"Eye ratio {snap['ear']:.2f} (baseline {snap['baseline_ear']:.2f})", 30)
                    put(frame, f"PERCLOS {snap['perclos'] * 100:4.1f}%   blinks/min {snap['blink_rate']:.0f}", 58, (0, 255, 255))
                    put(frame, f"Longest closure {snap['longest_closure_s']:.1f}s   Yawns(5m) {snap['yawns_5min']}", 86, (0, 255, 255))
                    put(frame, f"Head turn {snap['yaw_dev_deg']:.0f} deg   tilt {snap['pitch_dev_deg']:.0f} deg", 114, (0, 255, 255))
                    status = ("EYE CLOSURE WARNING!" if warn else "LOOKING AWAY" if snap["looking_away"]
                              else "YAWNING" if snap["yawning"] else "HEAD NOD" if snap["head_nod"] else "ALERT")
                    put(frame, status, 148, (0, 0, 255) if status != "ALERT" else (0, 255, 0), 0.8)
                put(frame, "V: voice | C: recalibrate | Q: quit", height - 15, (255, 255, 0), 0.5)

                if args.headless:
                    key = 255
                else:
                    cv2.imshow("CONTEXTRA - DRIVER MONITOR", frame)
                    key = cv2.waitKey(1) & 0xFF
                if key == ord("v"):
                    speak(random.choice(SLEEP_ALERTS))
                elif key == ord("c"):
                    tracker.reset_calibration()
                    calibration_start = None
                    last_yawn_count = 0
                    speak("Recalibrating. Please look straight ahead.")
                elif key == ord("q"):
                    break
    finally:
        try:
            update_driver_state(face_detected=False, monitor_running=False)
        except Exception as error:
            print("Could not publish shutdown state:", error)
        camera.release()
        cv2.destroyAllWindows()
        shutdown.set()
        voice_event.set()


if __name__ == "__main__":
    main()
