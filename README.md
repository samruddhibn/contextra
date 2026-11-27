# CONTEXTRA
**AI-Driven Contextual Reasoning for Automotive Software Engineering**
*An explainable AI co-pilot that understands the driver and the road together.*

## Overview
Most in-car warnings react to one signal at a time. CONTEXTRA fuses the driver's state (fatigue and distraction from a webcam) with the driving context (speed, road curvature, visibility, traffic, time of day, drive length). A trained model judges the overall risk, explains why, and triggers a response. It is an advisory research prototype that runs entirely on a laptop CPU and never controls a real vehicle.

## How it works
1. **Perceive:** MediaPipe FaceLandmarker gives 478 face landmarks per frame, processed locally. Frames are never stored or sent anywhere.
2. **Measure:** per-person calibration, then PERCLOS, blink rate, long eye closures, yawns, head nodding and head turned away.
3. **Contextualize:** speed, road, visibility, traffic, time of day and drive length from a built-in simulator, manual sliders or a CARLA bridge.
4. **Fuse and reason:** a RandomForest fuses 13 signals into LOW / ELEVATED / HIGH with probabilities.
5. **Explain:** occlusion-based factor contributions, TF-IDF retrieval over project notes, and an optional local LLM (Ollama) that only explains the result.
6. **Respond:** voice warnings, and in the simulation the car slows, pulls over to the shoulder, stops, and resumes only when the driver is alert.

## Features
- Live webcam fatigue and distraction detection with voice alerts
- Trained contextual fusion model compared against the original rule engine
- One-window live demo: webcam plus simulated drive (`python live_demo.py`)
- Streamlit dashboard with an evaluation tab (`streamlit run app.py`)
- 48 automated tests (`python -m unittest discover -s tests`)

## Results (held-out synthetic scenarios)
- Fusion model: accuracy 0.849, macro-F1 0.854, 5-fold CV 0.851 +/- 0.010, HIGH recall 0.901
- Rule engine baseline: accuracy 0.381, macro-F1 0.212
- PERCLOS-only baseline: accuracy 0.497, macro-F1 0.428
- Inference latency: about 87 ms per prediction on a laptop CPU

## Limitations (please read)
- The model is trained on **synthetic scenarios** from an expert-knowledge simulator. No public dataset links these inputs to real accidents. The metrics show the pipeline works, not real-world accuracy.
- The rule engine scores low partly because it cannot see yawning, head pose or distraction.
- Eye closure, yawning and head pose are indicators, not proof of drowsiness.
- Thresholds are calibrated per person but have not been validated on a labelled dataset such as NTHU-DDD.
- The car response is simulated only. The CARLA bridge is written but has not been tested on a CARLA server.
- Advisory prototype, not a certified safety system.

## Important setup note
`face_landmarker.task` is not included in this repository. Download it from Google's MediaPipe Face Landmarker page and place it in the project folder. Then run `python train_fusion_model.py`, followed by `python live_demo.py`.

## Tech stack
Python, OpenCV, MediaPipe, scikit-learn, Streamlit, NumPy, pandas, Ollama (optional), CARLA .
