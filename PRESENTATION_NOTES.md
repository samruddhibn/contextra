# Presentation notes - CONTEXTRA (read once, keep it honest)

## 30-second pitch
Today's in-car warnings react to one signal at a time. CONTEXTRA fuses the **driver's state**
(fatigue and distraction from a camera) with the **driving context** (speed, road, visibility,
time of day, how long they have been driving), uses a **trained model** to judge the overall risk,
**explains why**, and **speaks a warning that matches the situation**. It is an advisory research
prototype that runs fully locally.

## Night-before checklist
1. `python train_fusion_model.py` - note the numbers it prints on YOUR laptop.
2. Run the monitor in the real room lighting. Calibrate. Close your eyes 2 s -> alert? Hold a yawn 2 s -> counted?
   If it triggers too easily or not at all, change `closed_ratio` (default 0.72) in `fatigue_features.py`.
3. Record a 30-60 s backup video of yourself (eyes open, then a 2 s closure, a yawn, a head turn) and test
   `python contextra_monitor.py --video clip.mp4`.
4. Turn the laptop volume up. Close other apps. Plug in the charger.
5. Start Ollama if you want the LLM explanation (`phi3:3.8b`); on CPU it can take a while, so click the button early.

## Demo flow A - one window (about 3 minutes, recommended)
`python live_demo.py`
1. Look at the road for 5 s while it calibrates (voice says "Calibration complete").
2. Everything normal: risk LOW, car cruising in the simulated night / fog drive.
3. Close your eyes about 2 s -> voice alert, "CRITICAL DROWSINESS - WAKE UP!" banner, risk rises, the "WHY" box lists the reasons.
4. Keep them closed (or press **D** for the labelled simulated drowsy driver): the car slows, pulls over to the shoulder and stops.
5. Open your eyes / press **D** again and stay alert about 8 s: the car resumes.
6. Say: same driver state, different context gives different risk (restart with `--scenario "Highway, daytime, clear"`).

## Demo flow B - dashboard (about 5 minutes)
1. **Dashboard, Live tab, manual driver + simulator.** Pick "Night drive, fog, winding road". Alert driver -> LOW.
2. Move the sliders to a drowsy driver (PERCLOS 0.30, closure 2.5 s, 4 yawns) -> **HIGH**, with the factors listed.
3. Show the **same drowsy driver** on "Highway, daytime, clear" -> lower risk. Point: *same driver, different context, different risk.*
4. Switch to **Live webcam**. Calibrate, then close your eyes ~2 s -> voice alert + features change on the dashboard.
5. Click **Explain the current risk** -> local LLM / retrieved notes explanation.
6. Open **Model evaluation** tab -> fusion vs rule baseline vs single signal vs ablations (say clearly: synthetic data).
7. Show `carla_bridge.py` / `carla_sim.py` and say what it maps; be clear whether you ran it in CARLA.

Note: looking at the audience triggers "Eyes on the road" - that is the distraction detector working.

## What you can honestly claim
- Perception uses a trained neural network (MediaPipe FaceLandmarker) running locally on CPU.
- A trained RandomForest does the contextual fusion; the old rule engine is kept as a baseline.
- Decisions are explainable (factor contributions) and stale / missing data is never used.
- Real-time on a laptop: model inference is a few milliseconds; the dashboard refreshes every 2 s.
- Modular: the context source can be swapped (simulator / manual / CARLA) without touching the reasoning.

## What NOT to claim
- That it predicts real accidents or "prevents accidents" - it is an advisory prototype.
- That the fusion model's accuracy is real-world accuracy - labels are synthetic.
- That CARLA was demonstrated, unless you actually ran it.
- That eye closure = drowsiness - it is an indicator.
- That the LLM makes the decision - it only explains the model's output.

## Likely questions - honest answers
**Is the data real?** The camera signals are real and live. The fusion model is trained on synthetic scenarios
because no public dataset links these combined signals to accident outcomes. We report that clearly. Next step is
validating the detectors on NTHU-DDD.

**Why does the rule engine do so badly in your table?** It only sees eye closure, road and visibility, so it cannot
see yawning or head pose. The table shows the value of fusing more signals on this data, not real-world superiority.

**What is new compared with commercial driver-monitoring?** We do not claim to beat commercial systems. The project
contribution is an open, explainable, modular reasoning pipeline (fusion + explanation + context-aware response)
that runs locally on a student laptop.

**Where is the AI?** Face-landmark neural network, trained fusion classifier, TF-IDF retrieval, and a local LLM for explanations.

**Privacy?** Frames are processed locally and never stored or sent; only small numeric features are exchanged.

**What if the camera loses the face?** No evidence is not treated as closed eyes: the dashboard shows "no face" and does not compute risk.

**Why not control the car?** Out of scope and unsafe without certified validation; the system is advisory.
