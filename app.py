"""
CONTEXTRA dashboard.   streamlit run app.py

Reads the driver's fatigue features (webcam monitor, or manual sliders for a
demo), reads / simulates the driving context, runs the contextual fusion model,
compares it with the rule baseline, explains the result, and writes a
context-aware advisory that the monitor speaks.
"""

import time

import pandas as pd
import streamlit as st

import fusion_model as fm
import llm_explainer
import rag
from context_provider import (SCENARIOS, DrivingContext, ScenarioProvider,
                              read_context_file)
from contextra_engine import reason
from driver_state import get_driver_state, is_state_fresh, write_advisory

st.set_page_config(page_title="CONTEXTRA | Automotive AI", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="Preparing the fusion model (first run trains it, ~30 s)...")
def load_bundle():
    return fm.ensure_model()


bundle = load_bundle()

st.title("CONTEXTRA")
st.subheader("AI-Driven Contextual Reasoning for Automotive Software Engineering")
st.caption("Perceive - Contextualize - Fuse - Reason - Explain - Respond   |   "
           "Research prototype. Advisory only; never controls a vehicle.")

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Inputs")
    driver_source = st.radio("Driver source", ["Live webcam", "Manual sliders (demo)"], key="driver_source")
    context_source = st.radio("Context source",
                              ["Built-in simulator", "Manual sliders", "CARLA bridge (file)"],
                              key="context_source")

    if context_source == "Built-in simulator":
        st.selectbox("Scenario", list(SCENARIOS), key="scenario")
        if st.button("Restart scenario"):
            st.session_state["sim_start"] = time.time()
    st.checkbox("Speak HIGH-risk advisories (needs monitor running)", value=True, key="voice_on")

    if driver_source.startswith("Manual"):
        st.markdown("**Simulated driver**")
        st.slider("PERCLOS (share of last minute eyes closed)", 0.0, 0.6, 0.04, 0.01, key="m_perclos")
        st.slider("Longest eye closure (s)", 0.0, 5.0, 0.3, 0.1, key="m_longest")
        st.slider("Blink rate (per min)", 5, 40, 17, key="m_blink")
        st.slider("Yawns in last 5 min", 0, 8, 0, key="m_yawns")
        st.checkbox("Head nodding", key="m_nod")
        st.slider("Head turned away (share of last minute)", 0.0, 1.0, 0.0, 0.05, key="m_distract")

    if context_source == "Manual sliders":
        st.markdown("**Simulated context**")
        st.slider("Speed (km/h)", 0, 160, 80, key="c_speed")
        st.slider("Road curvature", 0.0, 1.0, 0.1, 0.05, key="c_curve")
        st.slider("Road roughness", 0.0, 1.0, 0.1, 0.05, key="c_rough")
        st.slider("Visibility (1 = clear)", 0.0, 1.0, 1.0, 0.05, key="c_vis")
        st.slider("Traffic density", 0.0, 1.0, 0.2, 0.05, key="c_traffic")
        st.slider("Hour of day", 0.0, 24.0, 14.0, 0.5, key="c_hour")
        st.slider("Minutes already driving", 0, 480, 30, 10, key="c_minutes")

if "sim_start" not in st.session_state:
    st.session_state["sim_start"] = time.time()
st.session_state.setdefault("history", [])
st.session_state.setdefault("adv_counter", 0)
st.session_state.setdefault("last_level", None)


# ------------------------------------------------------------------- helpers
def get_driver():
    """Returns (features_dict or None, status_text, is_simulated)."""
    if st.session_state["driver_source"].startswith("Manual"):
        return ({
            "perclos": st.session_state["m_perclos"],
            "longest_closure_s": st.session_state["m_longest"],
            "blink_rate": st.session_state["m_blink"],
            "yawns_5min": st.session_state["m_yawns"],
            "head_nod": 1.0 if st.session_state["m_nod"] else 0.0,
            "distraction_fraction": st.session_state["m_distract"],
        }, "SIMULATED driver (sliders)", True)

    state = get_driver_state()
    if not is_state_fresh(state):
        return None, "Waiting for webcam monitor (run contextra_monitor.py)", False
    if not state.get("calibrated"):
        return None, "Monitor is calibrating - look at the road for ~5 s", False
    if not state.get("face_detected"):
        return None, "No face detected - no driver evidence, risk not computed", False
    return state.get("features", {}), "Live webcam", False


def get_context():
    """Returns (DrivingContext or None, status_text)."""
    src = st.session_state["context_source"]
    if src == "Built-in simulator":
        provider = ScenarioProvider(st.session_state.get("scenario", list(SCENARIOS)[0]))
        elapsed = time.time() - st.session_state["sim_start"]
        return provider.get(elapsed), f"SIMULATED: {provider.name} ({elapsed % provider.spec['duration']:.0f}s)"
    if src == "Manual sliders":
        return DrivingContext(
            speed_kmh=st.session_state["c_speed"], road_curvature=st.session_state["c_curve"],
            road_roughness=st.session_state["c_rough"], visibility=st.session_state["c_vis"],
            traffic_density=st.session_state["c_traffic"], hour_of_day=st.session_state["c_hour"],
            drive_minutes=st.session_state["c_minutes"], source="manual sliders"), "SIMULATED (manual sliders)"
    ctx = read_context_file()
    if ctx is None:
        return None, "CARLA bridge not running (no recent context_state.json)"
    return ctx, "CARLA bridge (live context file)"


def advisory_text(level, factors):
    lead = factors[0]["text"] if factors else "the combination of signals"
    if level == "HIGH":
        return (f"High risk: {lead}. Slow down and, when safe, stop at a suitable place to rest. "
                "Do not stop in a live traffic lane.")
    if level == "ELEVATED":
        return f"Elevated risk: {lead}. Stay alert and consider a break soon."
    return "Driver state and driving context look normal. Continue monitoring."


def voice_text(level, factors):
    lead = factors[0]["text"] if factors else "several signals together"
    return f"High risk. {lead}. Please slow down and find a safe place to rest."


def rule_level(driver, ctx):
    r = reason(
        driver_eye_closed=float(driver.get("longest_closure_s", 0)) >= 1.5,
        road_condition=ctx.road_category(), visibility=ctx.visibility_category(),
        vehicle_speed_kmh=ctx.speed_kmh)
    return r["risk_level"], r["risk_score"]


# --------------------------------------------------------------- live fragment
@st.fragment(run_every="2s")
def live_panel():
    driver, driver_status, driver_sim = get_driver()
    ctx, ctx_status = get_context()

    s1, s2 = st.columns(2)
    s1.markdown(f"**Driver:** {driver_status}")
    s2.markdown(f"**Context:** {ctx_status}")

    if driver is None or ctx is None:
        st.session_state["last_result"] = None
        st.info("Waiting for both a driver signal and a driving context. "
                "Nothing is inferred from stale or missing data.")
        return

    result = fm.predict(bundle, driver, ctx)
    rule_lvl, rule_score = rule_level(driver, ctx)
    st.session_state["last_result"] = {**result, "context": ctx.to_dict(), "driver": driver,
                                       "driver_simulated": driver_sim}

    level = result["level"]
    probs = result["probabilities"]
    st.session_state["history"].append(result["expected_risk"])
    st.session_state["history"] = st.session_state["history"][-90:]

    # advisory + hand-off to the monitor for voice
    if level != st.session_state["last_level"]:
        st.session_state["adv_counter"] += 1
        st.session_state["last_level"] = level
    adv_id = f"{st.session_state['adv_counter']}-{int(time.time() // 20)}"
    write_advisory(adv_id, level, voice_text(level, result["top_factors"]),
                   speak=st.session_state.get("voice_on", True))

    st.divider()
    a, b, c = st.columns([1.1, 1.1, 1.4])
    a.metric("AI fusion model - risk", level)
    b.metric("Rule baseline (old engine)", f"{rule_lvl}", f"score {rule_score}", delta_color="off")
    c.metric("Expected risk (0 - 1)", f"{result['expected_risk']:.2f}")

    msg = advisory_text(level, result["top_factors"])
    (st.error if level == "HIGH" else st.warning if level == "ELEVATED" else st.success)(msg)

    p1, p2, p3 = st.columns(3)
    for col, name in zip((p1, p2, p3), fm.LABELS):
        col.progress(min(1.0, probs[name]), text=f"{name}: {probs[name] * 100:.0f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Why (contributing factors)**")
        if result["top_factors"]:
            for f in result["top_factors"]:
                st.write(f"- {f['text']}  (+{f['delta']:.2f})")
        else:
            st.write("- No single factor stands out.")
        st.caption("Occlusion explanation: each factor is replaced by a safe value and the drop in "
                   "expected risk is its contribution.")
    with right:
        st.markdown("**Live driver features**")
        st.write(
            f"- PERCLOS: {driver.get('perclos', 0) * 100:.1f}%\n"
            f"- Longest closure: {driver.get('longest_closure_s', 0):.1f} s\n"
            f"- Blinks/min: {driver.get('blink_rate', 0):.0f}\n"
            f"- Yawns (5 min): {int(driver.get('yawns_5min', 0))}\n"
            f"- Head nod: {'yes' if driver.get('head_nod') else 'no'}\n"
            f"- Looking away: {driver.get('distraction_fraction', 0) * 100:.0f}% of last minute")

    st.markdown("**Driving context**")
    st.write(f"{ctx.speed_kmh:.0f} km/h | curvature {ctx.road_curvature:.2f} | roughness {ctx.road_roughness:.2f} | "
             f"visibility {ctx.visibility:.2f} | traffic {ctx.traffic_density:.2f} | "
             f"hour {ctx.hour_of_day:.1f} | {ctx.drive_minutes:.0f} min driving")

    st.markdown("**Risk over time**")
    st.line_chart(pd.DataFrame({"expected risk": st.session_state["history"]}), height=160)


tab_live, tab_eval, tab_arch = st.tabs(["Live", "Model evaluation", "Architecture and limits"])

with tab_live:
    live_panel()

    st.divider()
    st.markdown("### AI explanation (local LLM + retrieved knowledge)")
    models = llm_explainer.ollama_models()
    if models is None:
        st.caption("Ollama is not running - a template explanation will be used. "
                   "Start Ollama to use the local LLM.")
    else:
        st.caption(f"Ollama reachable. Models: {', '.join(models) or 'none'}")
    if st.button("Explain the current risk"):
        last = st.session_state.get("last_result")
        if not last:
            st.warning("No current risk result yet.")
        else:
            notes = rag.get_kb().retrieve(llm_explainer.retrieval_query(last), k=3)
            with st.spinner("Generating explanation (the local model can take a while on CPU)..."):
                out = llm_explainer.explain(last, notes)
            st.write(out["text"])
            st.caption(f"Source: {out['source']} | {out['seconds']} s. The LLM only explains the "
                       "model's output; it does not decide the risk level.")
            if notes:
                with st.expander("Retrieved knowledge used"):
                    for n in notes:
                        st.markdown(f"**{n['source']}** (score {n['score']})")
                        st.write(n["text"])

with tab_eval:
    st.markdown("### Fusion model vs baselines")
    metrics = fm.load_metrics()
    if not metrics:
        st.info("Run `python train_fusion_model.py` to generate evaluation results.")
    else:
        st.warning(metrics["data_note"] + " The rule engine scores low partly because it cannot see "
                   "yawning, head pose or distraction. These numbers show the benefit of multi-signal "
                   "fusion on this data, not real-world accuracy.")
        rows = [{"Model": k, "Accuracy": v["accuracy"], "Macro F1": v["macro_f1"],
                 "HIGH precision": v["per_class"]["HIGH"]["precision"],
                 "HIGH recall": v["per_class"]["HIGH"]["recall"]}
                for k, v in metrics["comparisons"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("5-fold CV macro-F1", f"{metrics['fusion_cv_macro_f1_mean']:.3f}",
                  f"+/- {metrics['fusion_cv_macro_f1_std']:.3f}", delta_color="off")
        c2.metric("Inference latency", f"{metrics['inference_latency_ms']:.1f} ms")
        c3.metric("Samples (train / test)", f"{metrics['train']} / {metrics['test']}")
        cm = metrics["comparisons"]["Fusion model (driver + context)"]["confusion_matrix"]
        st.markdown("**Fusion confusion matrix** (rows = true, columns = predicted)")
        st.dataframe(pd.DataFrame(cm, index=[f"true {l}" for l in fm.LABELS],
                                  columns=[f"pred {l}" for l in fm.LABELS]))
        st.markdown("**Feature importance**")
        imp = pd.DataFrame(metrics["feature_importance"], columns=["feature", "importance"]).set_index("feature")
        st.bar_chart(imp)

with tab_arch:
    st.markdown("""
**Pipeline**

1. **Perceive** - webcam, MediaPipe FaceLandmarker (trained neural network) gives 478 face landmarks, computed locally.
2. **Contextualize** - driver features (PERCLOS, blinks, long closures, yawns, head nod, head turned away) plus driving context (speed, curve, roughness, visibility, traffic, time of day, drive length).
3. **Fuse + Reason** - a trained RandomForest fuses all 13 signals into LOW / ELEVATED / HIGH with probabilities.
4. **Explain** - occlusion-based factor contributions, retrieved notes (TF-IDF) and a local LLM (Ollama) that only explains the result.
5. **Respond** - dashboard advisory, fast-path voice warnings, and a context-aware voice advisory for HIGH risk.

**Context sources:** built-in simulator, manual sliders, or a CARLA bridge (separate process, JSON file hand-off).

**Honest limitations**
- The fusion model is trained on synthetic scenario data from an expert-knowledge simulator, not real accident data.
- Eye-closure, yawning and head-pose are indicators, not proof of drowsiness.
- Thresholds are calibrated per person but not validated on a labelled dataset (e.g. NTHU-DDD) yet.
- Prototype advisory system - not certified, never controls the vehicle.
""")
