import json
import uuid
import os
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any

from yentlbench.local_runner.prompt import build_prompt
from yentlbench.local_runner.parser import parse_esi


class OllamaNotRunningError(Exception):
    """Raised when the Ollama service is unreachable."""

    pass


class OllamaRunner:
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip("/")

    def health_check(self) -> None:
        """Pings the Ollama endpoint to ensure it is running."""
        try:
            response = requests.get(f"{self.host}/api/version", timeout=5)
            response.raise_for_status()
        except (requests.exceptions.RequestException, ConnectionError) as e:
            raise OllamaNotRunningError(
                f"Could not connect to Ollama at {self.host}. "
                "Please ensure Ollama is installed and running locally. "
                f"Underlying error: {e}"
            )

    def list_models(self) -> List[str]:
        """Fetches available model names from the Ollama service."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model.get("name") for model in data.get("models", [])]
        except (requests.exceptions.RequestException, ConnectionError) as e:
            raise OllamaNotRunningError(
                f"Could not connect to Ollama at {self.host} to list models. "
                "Please ensure Ollama is installed and running locally. "
                f"Underlying error: {e}"
            )

    def run(
        self,
        vignettes: List[Dict[str, Any]],
        target_variant: str,
        run_number: int = 1,
        output_dir: str = "results",
        dry_run: bool = False,
    ) -> str:
        """
        Runs the Ollama model against the given vignettes for the target_variant.
        """
        model_slug = self.model_name.replace(":", "_").replace("/", "_")
        filename = f"batch_esi_triage_scorer_{target_variant}-run_id_Run_{run_number}_{model_slug}.run.json"
        out_path = os.path.join(output_dir, filename)

        matching_vignettes = [
            v for v in vignettes if v.get("gender_variant") == target_variant
        ]

        if dry_run:
            print("--- Dry Run ---")
            print(f"Model: {self.model_name}")
            print(f"Variant: {target_variant}")
            print(f"Vignette count: {len(matching_vignettes)}")
            print(f"Expected output file: {out_path}")
            return out_path

        if os.path.exists(out_path):
            print(
                f"[{self.model_name}] Run {run_number} for variant '{target_variant}' already exists. Skipping."
            )
            return out_path

        self.health_check()

        os.makedirs(output_dir, exist_ok=True)

        subruns = []

        for vignette in vignettes:
            # Only process matching variant
            if vignette.get("gender_variant") != target_variant:
                continue

            prompt_text = build_prompt(vignette, target_variant)
            run_id = str(uuid.uuid4())
            start_time = datetime.now(timezone.utc).isoformat()

            try:
                response = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt_text,
                        "stream": False,
                        "options": {"temperature": 0.0},
                    },
                    timeout=120,
                )
                response.raise_for_status()
                res_data = response.json()

                predicted_text = res_data.get("response", "")
                eval_count = res_data.get("eval_count", 0)
                prompt_eval_count = res_data.get("prompt_eval_count", 0)

                # total_duration is in nanoseconds in Ollama
                total_duration = res_data.get("total_duration", 0)
                latency_ms = total_duration / 1_000_000.0

            except Exception as e:
                print(f"Error calling Ollama for {run_id}: {e}")
                predicted_text = ""
                eval_count = 0
                prompt_eval_count = 0
                latency_ms = 0.0

            end_time = datetime.now(timezone.utc).isoformat()

            parsed = parse_esi(predicted_text)
            if parsed is None:
                predicted_score = -1.0
                parse_failed = True
            else:
                predicted_score = parsed
                parse_failed = False

            acuity = vignette.get("acuity")
            actual_score = (
                float(acuity) if acuity is not None and str(acuity) != "nan" else None
            )

            dict_result = {
                "actual_score": actual_score,
                "predicted_score": predicted_score,
            }
            if parse_failed:
                dict_result["parse_failed"] = True

            subrun = {
                "pyRunId": run_id,
                "state": "COMPLETED" if not parse_failed else "FAILED",
                "startTime": start_time,
                "endTime": end_time,
                "conversations": [
                    {
                        "requests": [
                            {"contents": [{"parts": [{"text": prompt_text}]}]}
                        ],
                        "metrics": {
                            "inputTokens": prompt_eval_count,
                            "outputTokens": eval_count,
                            "totalBackendLatencyMs": latency_ms,
                        },
                    }
                ],
                "results": [{"dictResult": dict_result}],
            }
            subruns.append(subrun)

        output_data = {"subruns": subruns}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(
            f"[{self.model_name}] Completed {len(subruns)} evaluations for variant '{target_variant}'."
        )
        print(f"Results saved to {out_path}")
        return out_path
