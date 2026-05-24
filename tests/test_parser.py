import json
import os
from yentlbench.local_runner.parser import parse_esi


def process_runner_output(text):
    """
    Simulates the logic inside ollama_runner.py
    to process the output of parse_esi.
    """
    parsed = parse_esi(text)
    if parsed is None:
        return {"predicted_score": -1.0, "parse_failed": True}
    else:
        return {"predicted_score": float(parsed), "parse_failed": False}


def test_esi_parser():
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "esi_parser_cases.json"
    )
    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        result = process_runner_output(case["text"])

        if case["valid"]:
            assert result["parse_failed"] is False, (
                f"Failed to parse valid case: {case['text']}"
            )
            assert result["predicted_score"] == float(case["expected"]), (
                f"Parsed wrong score for: {case['text']}"
            )
        else:
            assert result["parse_failed"] is True, (
                f"Incorrectly parsed invalid case: {case['text']}"
            )
            assert result["predicted_score"] == -1.0, (
                f"Incorrect sentinel for invalid case: {case['text']}"
            )
