
from datetime import datetime, timezone


VALID_ROAD_CONDITIONS = {
    "normal",
    "sharp_turn",
    "rough_road",
    "unknown",
}

VALID_VISIBILITY = {
    "good",
    "poor",
    "unknown",
}


def reason(
    driver_eye_closed=False,
    road_condition="unknown",
    visibility="unknown",
    vehicle_speed_kmh=None,
    driver_signal_confidence=None,
):
    """
    Transparent contextual advisory engine for CONTEXTRA.

    This is a research prototype, not a safety-certified system.
    Its risk score is a heuristic, not a calibrated probability.
    """

    if road_condition not in VALID_ROAD_CONDITIONS:
        raise ValueError(f"Unsupported road condition: {road_condition}")

    if visibility not in VALID_VISIBILITY:
        raise ValueError(f"Unsupported visibility: {visibility}")

    if vehicle_speed_kmh is not None and vehicle_speed_kmh < 0:
        raise ValueError("Vehicle speed cannot be negative.")

    risk_score = 0
    evidence = []

    # Driver evidence
    if driver_eye_closed:
        risk_score += 3
        evidence.append({
            "signal": "prolonged_eye_closure",
            "contribution": 3,
            "observation": "Eye closure persisted beyond the configured threshold.",
        })

    # Road context
    road_weights = {
        "sharp_turn": 2,
        "rough_road": 1,
        "normal": 0,
        "unknown": 0,
    }

    if road_weights[road_condition]:
        risk_score += road_weights[road_condition]
        evidence.append({
            "signal": road_condition,
            "contribution": road_weights[road_condition],
            "observation": {
                "sharp_turn": "A sharp turn was reported in the road context.",
                "rough_road": "A rough road was reported in the road context.",
            }[road_condition],
        })

    # Visibility context
    if visibility == "poor":
        risk_score += 2
        evidence.append({
            "signal": "poor_visibility",
            "contribution": 2,
            "observation": "Poor visibility was reported.",
        })

    # Speed is recorded, but not weighted without a validated context-specific rule.
    context = {
        "road_condition": road_condition,
        "visibility": visibility,
        "vehicle_speed_kmh": vehicle_speed_kmh,
    }

    # Conservative advisory policy for the prototype.
    if risk_score >= 5:
        level = "HIGH"
        advisory = (
            "High-priority advisory: when safe, pull over at a suitable "
            "location and rest. Do not stop in a live traffic lane."
        )
    elif risk_score >= 3:
        level = "ELEVATED"
        advisory = (
            "Elevated advisory: pay attention to the combined signals "
            "and assess whether it is safe to continue."
        )
    else:
        level = "LOW"
        advisory = "Continue monitoring; no elevated combination was identified."

    explanation = (
        "Combined evidence: "
        + "; ".join(item["observation"] for item in evidence)
        if evidence
        else "No elevated signals were reported by the available inputs."
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "risk_score": risk_score,
        "risk_level": level,
        "explanation": explanation,
        "advisory": advisory,
        "evidence": evidence,
        "context": context,
        "driver_signal_confidence": driver_signal_confidence,
        "limitations": [
            "Heuristic score; not a calibrated probability.",
            "Inputs may be simulated or unvalidated.",
            "Not suitable for controlling a real vehicle.",
        ],
    }


if __name__ == "__main__":
    result = reason(
        driver_eye_closed=True,
        road_condition="sharp_turn",
        visibility="poor",
        vehicle_speed_kmh=60,
    )

    print("=== CONTEXTRA REASONING ENGINE ===")
    print("Risk score:", result["risk_score"])
    print("Risk level:", result["risk_level"])
    print("Explanation:", result["explanation"])
    print("Advisory:", result["advisory"])
    print("Evidence:", result["evidence"])
