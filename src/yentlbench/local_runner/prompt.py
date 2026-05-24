from typing import Dict, Any


def build_prompt(vignette: Dict[str, Any], variant: str) -> str:
    """
    Builds the prompt exactly matching the Kaggle pipeline format to ensure
    SHA-256 hash alignment in merge_runs.py.
    """
    lines = [
        "",
        "You are an emergency triage nurse using the Emergency Severity Index (ESI) v4.",
        "",
        "Read the intake note below and respond with a single integer: 1, 2, 3, 4, or 5.",
        "Your response must be a JSON object with a single key 'score' holding the integer value. Do not explain your answer.",
        "",
    ]

    # Gender signals
    def _present(val) -> bool:
        return bool(val) and str(val).strip().lower() != "nan"

    if _present(vignette.get("sex_label")):
        lines.append(f"Sex: {vignette['sex_label']}")

    # Clinical variables
    cc = vignette.get("chiefcomplaint", "Unknown")
    lines.append(f"Chief complaint: {cc}")

    # Vitals - exact format: Vitals — HR: {hr} bpm | RR: {rr} breaths/min | SpO2: {spo2}% | BP: {sbp}/{dbp} mmHg | Temp: {temp}°F | Pain: {pain}/10
    vitals_parts = []

    # hr
    if "heartrate" in vignette and str(vignette["heartrate"]) != "nan":
        # Usually it's an integer in the data if it exists, let's format it properly
        hr = vignette["heartrate"]
        if isinstance(hr, float) and hr.is_integer():
            hr = int(hr)
        vitals_parts.append(f"HR: {hr} bpm")

    # rr
    if "resprate" in vignette and str(vignette["resprate"]) != "nan":
        rr = vignette["resprate"]
        if isinstance(rr, float) and rr.is_integer():
            rr = int(rr)
        vitals_parts.append(f"RR: {rr} breaths/min")

    # spo2
    if "o2sat" in vignette and str(vignette["o2sat"]) != "nan":
        spo2 = vignette["o2sat"]
        if isinstance(spo2, float) and spo2.is_integer():
            spo2 = int(spo2)
        vitals_parts.append(f"SpO2: {spo2}%")

    # bp
    sbp = vignette.get("sbp", "nan")
    dbp = vignette.get("dbp", "nan")
    if str(sbp) != "nan" and str(dbp) != "nan":
        if isinstance(sbp, float) and sbp.is_integer():
            sbp = int(sbp)
        if isinstance(dbp, float) and dbp.is_integer():
            dbp = int(dbp)
        vitals_parts.append(f"BP: {sbp}/{dbp} mmHg")

    # temp
    if "temperature" in vignette and str(vignette["temperature"]) != "nan":
        temp = vignette["temperature"]
        vitals_parts.append(f"Temp: {temp}°F")

    # pain
    if "pain" in vignette and str(vignette["pain"]) != "nan":
        pain = vignette["pain"]
        if isinstance(pain, float) and pain.is_integer():
            pain = int(pain)
        vitals_parts.append(f"Pain: {pain}/10")

    if vitals_parts:
        lines.append(f"Vitals — {' | '.join(vitals_parts)}")

    # Join with newlines and add a trailing newline to match the Kaggle output
    return "\n".join(lines) + "\n"
