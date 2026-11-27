# CONTEXTRA - AI-Driven Contextual Reasoning for Automotive Software Engineering

Pipeline: **Perceive -> Contextualize -> Fuse -> Reason -> Explain -> Respond**

A research prototype (advisory only - it never controls a vehicle) that fuses
**driver fatigue / distraction signals from a webcam** with **driving context**
(speed, road, visibility, traffic, time of day, drive length) using a **trained
model**, explains its decision, and speaks context-aware warnings.

## What is in this version

| Piece | File | What it does |
|---|---|---|
| Perception | `contextra_monitor.py`, `fatigue_features.py` | MediaPipe face landmarks -> per-person calibration, PERCLOS, blink rate, long closures, yawns, head nod, head turned away |
| Shared state | `driver_state.py` | Atomic JSON hand-off + heartbeat; stale or stopped monitor is never trusted |
| Context | `context_provider.py` | Built-in simulator (4 scripted drives), manual sliders, or CARLA via a bridge |
| CARLA | `carla_bridge.py` | Reads speed / weather / curvature / traffic from a CARLA server (see limits below) |
| AI reasoning | `fusion_model.py`, `train_fusion_model.py` | RandomForest fusing 13 signals -> LOW / ELEVATED / HIGH + probabilities + factor explanations |
| Baseline | `contextra_engine.py` | The original transparent rule engine, kept for comparison |
| Explain | `rag.py`, `knowledge/`, `llm_explainer.py` | TF-IDF retrieval over project notes + local Ollama LLM (falls back to a template) |
| **Live demo (one window)** | `live_demo.py`, `responder.py`, `sim_world.py`, `carla_sim.py` | Webcam + driving simulation + AI decision side by side; the simulated car slows, pulls over and stops on HIGH risk |
| Dashboard | `app.py` | Live status, AI vs rule baseline, factors, risk-over-time, explanation, evaluation tab |
| Tests | `tests/` | 48 unit tests (features, state, engine scenarios, context, model) |
| Old experiments | `legacy/` | Earlier synthetic eye-model files, kept for reference |

## Setup (Windows PowerShell)

Copy the contents of this folder over `C:\Contextra` (keep your existing `.venv`), then:

```powershell
cd C:\Contextra
.\.venv\Scripts\Activate.ps1
python train_fusion_model.py        # trains the fusion model, prints the metrics (about 30-60 s)
python -m unittest discover -s tests # optional: run the tests
```

## Run - the one-window live demo (recommended for the presentation)

```powershell
python live_demo.py                                     # built-in simulator, runs on any laptop
python live_demo.py --scenario "Highway, daytime, clear"
python live_demo.py --video yourclip.mp4                # recorded clip instead of the webcam
python live_demo.py --sim carla --weather HardRainNight # needs a CARLA server (untested)
```

Left: your webcam with fatigue / distraction readouts. Right: the simulated drive.
Top-left shows the AI risk level and why. If HIGH risk persists for about 3 seconds the
**simulated** car slows down, moves to the shoulder and stops, and it resumes only after you
have been clearly alert for about 8 seconds. Keys: **V** voice, **C** recalibrate,
**D** toggle a clearly-labelled SIMULATED drowsy driver (so you can show the full response
even if live eye-closure detection misbehaves), **R** restart the simulation, **Q** quit.
While the simulated drowsy driver is on, the car will not resume - press **D** again to go back to your real webcam.

## Run - separate monitor and dashboard

Two windows (or `.\start_contextra.ps1` to open both):

```powershell
python contextra_monitor.py          # window 1: webcam monitor (look at the road for 5 s to calibrate)
streamlit run app.py                 # window 2: dashboard at http://localhost:8501
```

Monitor keys: **V** test voice, **C** recalibrate, **Q** quit.
No webcam on stage? Record a short video of yourself and run
`python contextra_monitor.py --video yourclip.mp4` (it loops the video).

In the dashboard sidebar you can switch the **driver source** (live webcam or
manual sliders) and the **context source** (built-in simulator, manual sliders,
or CARLA bridge). Sliders make the model's behaviour easy to demonstrate.

## CARLA

`carla_bridge.py` is written against the CARLA 0.9.x Python API and writes
`context_state.json`, which the dashboard reads in "CARLA bridge" mode.

```powershell
# in a separate Python environment that has the carla package for your CARLA version
python carla_bridge.py --spawn --weather HardRainNight
```

**Not tested on a real CARLA server.** CARLA needs a dedicated GPU and a lot of
disk space; an Intel Iris Xe laptop is not suitable. Run it on a GPU machine, or
present the built-in simulator and describe the bridge as implemented but not yet
demonstrated. The mapping functions (weather -> visibility, waypoint heading ->
curvature, etc.) are unit-tested without CARLA.

## Honest limitations

- The fusion model is trained on **synthetic scenario data** from an expert-knowledge
  simulator (`expert_risk` in `fusion_model.py`). No public dataset links these combined
  inputs to real accident outcomes. Reported metrics show the pipeline works and that
  multi-signal fusion beats single-signal rules **on that data** - not real-world accuracy.
- The old rule engine scores low in the comparison partly because it cannot see yawning,
  head pose or distraction. Do not present the gap as real-world superiority.
- Eye closure, yawning and head pose are **indicators**, not proof of drowsiness.
- Thresholds (`fatigue_features.py`) are calibrated per person but **not validated** on a
  labelled drowsiness dataset (e.g. NTHU-DDD). Glasses, lighting and head angle affect them.
- Webcam behaviour was verified on a real face image / video and with synthetic signals; it still
  needs tuning with live closed eyes / yawns on the demo laptop.
- The built-in simulator is a lightweight 2-D road view, not CARLA. The car "pull over" behaviour is
  simulated only.
- Advisory prototype - not a certified safety system.

## Next steps (future work)

1. Validate / retrain the eye-closure and yawn detectors on NTHU-DDD or YawDD (subject-independent split).
2. Replace the simulator labels with a better-founded risk target if suitable labelled data is found.
3. Run the CARLA bridge on a GPU machine and record a demo.
4. Compare the explanation quality of the local LLM with and without retrieval.
