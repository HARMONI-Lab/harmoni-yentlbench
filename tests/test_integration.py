import os
import sys
import tempfile
import pytest
import pandas as pd
from unittest.mock import patch

from yentlbench.__main__ import main

@pytest.fixture
def mock_ollama():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
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
        yield mock_get, mock_post

def test_full_pipeline_integration(mock_ollama):
    mock_get, mock_post = mock_ollama
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Prepare mock dataset
        data_path = os.path.join(tmpdir, "dataset_quintets.csv")
        results_dir = os.path.join(tmpdir, "results")
        eval_dir = os.path.join(tmpdir, "eval")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        vignettes = []
        for i in range(3):
            for variant, name, sex_label, pronoun in [
                ("male", "John", "Male", "he/him"),
                ("female", "Jane", "Female", "she/her"),
                ("nb_ambiguous", "J.", "", ""),
                ("nb_label_only", "J", "Non-binary", ""),
            ]:
                vignettes.append({
                    "quintet_id": i,
                    "source_stay_id": 1000 + i,
                    "gender_variant": variant,
                    "patient_name": name,
                    "sex_label": sex_label,
                    "pronoun": pronoun,
                    "chiefcomplaint": f"Complaint {i}",
                    "heartrate": 80,
                    "resprate": 16,
                    "o2sat": 98,
                    "sbp": 120,
                    "dbp": 80,
                    "temperature": 98.6,
                    "pain": 5,
                    "acuity": 3
                })
        pd.DataFrame(vignettes).to_csv(data_path, index=False)
        
        # 2. Run: yentlbench run
        test_args_run = [
            "yentlbench", "run",
            "--model", "mock_model",
            "--variants", "nb_ambiguous", "female", "male", "nb_label_only",
            "--data", data_path,
            "--results-dir", results_dir,
            "--run-id", "1"
        ]
        
        with patch.object(sys, 'argv', test_args_run):
            main()
            
        # Assert: results/ contains 4 .run.json files with correct filenames
        run_files = sorted(os.listdir(results_dir))
        assert len(run_files) == 4
        expected_files = [
            "batch_esi_triage_scorer_female-run_id_Run_1_mock_model.run.json",
            "batch_esi_triage_scorer_male-run_id_Run_1_mock_model.run.json",
            "batch_esi_triage_scorer_nb_ambiguous-run_id_Run_1_mock_model.run.json",
            "batch_esi_triage_scorer_nb_label_only-run_id_Run_1_mock_model.run.json",
        ]
        assert run_files == expected_files
        
        # 3. Merge: yentlbench merge
        merged_csv = os.path.join(eval_dir, "merged_evaluations.csv")
        test_args_merge = [
            "yentlbench", "merge",
            "--results-dir", results_dir,
            "--output", merged_csv
        ]
        
        with patch.object(sys, 'argv', test_args_merge):
            main()
            
        # Assert: eval/merged_evaluations.csv has correct column structure ({variant}__{model} format)
        assert os.path.exists(merged_csv)
        df_merged = pd.read_csv(merged_csv)
        
        # It should have 3 rows (because 3 quintets = 3 distinct clinical prompts)
        assert len(df_merged) == 3
        
        expected_cols = [
            "prompt_hash", "prompt", "actual_score",
            "predicted_score__female__mock_model",
            "predicted_score__male__mock_model",
            "predicted_score__nb_ambiguous__mock_model",
            "predicted_score__nb_label_only__mock_model"
        ]
        for col in expected_cols:
            assert col in df_merged.columns
            
        # 4. Analyze: yentlbench analyze
        stats_csv = os.path.join(eval_dir, "benchmark_stats.csv")
        attention_dir = os.path.join(eval_dir, "attention")
        test_args_analyze = [
            "yentlbench", "analyze",
            "--input", merged_csv,
            "--output-stats", stats_csv,
            "--output-attention", attention_dir
        ]
        
        with patch.object(sys, 'argv', test_args_analyze):
            with patch('seaborn.heatmap'), patch('seaborn.pointplot'), patch('seaborn.barplot'), patch('seaborn.violinplot'), patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
                main()
        
        # Assert: eval/benchmark_stats.csv contains expected rows
        assert os.path.exists(stats_csv)
        df_stats = pd.read_csv(stats_csv)
        
        # Should have 4 rows (one for each variant run)
        assert len(df_stats) == 4
        assert "run" in df_stats.columns
        assert "accuracy" in df_stats.columns
        
        # Verify no live Ollama endpoint was called by checking call counts on our mocks
        # 4 variants, each does a health_check (get) and generates 3 vignettes (post)
        assert mock_get.call_count == 4
        assert mock_post.call_count == 12

