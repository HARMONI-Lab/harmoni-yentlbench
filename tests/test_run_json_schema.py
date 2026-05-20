import json
import os
import tempfile
import pytest
from yentlbench.merge_runs import validate_run_file, parse_run_results
from yentlbench.local_runner.ollama_runner import OllamaRunner
from unittest.mock import patch

def validate_run_json(path):
    """
    Validates a .run.json file against the schema merge_runs.py expects.
    This acts as a contract test to keep Kaggle and local output aligned.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # First, test the basic top-level structure required
    is_valid = validate_run_file(data, path)
    if not is_valid:
        return False
        
    # Second, test if it can actually be parsed correctly by merge_runs
    try:
        df = parse_run_results(path, include_metrics=True)
        if df.empty:
            return False
            
        # Ensure it extracted the necessary elements
        expected_cols = [
            "prompt_hash", "prompt", "actual_score", 
            "predicted_score", "input_tokens", "output_tokens", "total_latency_ms"
        ]
        for col in expected_cols:
            if col not in df.columns:
                return False
                
        return True
    except Exception:
        return False

def test_kaggle_schema():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "kaggle_sample.run.json")
    assert validate_run_json(fixture_path) is True

@patch("requests.post")
@patch("requests.get")
def test_local_schema(mock_get, mock_post):
    # Mock healthcheck
    mock_get.return_value.status_code = 200
    
    # Mock Ollama generation
    mock_post.return_value.json.return_value = {
        "response": '{"score": 3}',
        "eval_count": 15,
        "prompt_eval_count": 200,
        "total_duration": 1500000000 # nanoseconds
    }
    mock_post.return_value.status_code = 200

    vignettes = [
        {
            "gender_variant": "female",
            "acuity": 3,
            "chiefcomplaint": "Chest pain",
            "heartrate": 80,
            "resprate": 16,
            "o2sat": 98,
            "sbp": 120,
            "dbp": 80,
            "temperature": 98.6,
            "pain": 5
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = OllamaRunner("test_model")
        out_path = runner.run(vignettes, "female", run_number=1, output_dir=tmpdir)
        
        assert validate_run_json(out_path) is True

