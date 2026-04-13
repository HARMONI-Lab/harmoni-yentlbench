We adapt perturbation sensitivity analysis to clinical demographic auditing, operationalizing a Perturbation Sensitivity Score (PSS) for sex-label token substitution in ESI triage and pairing it with a four-layer decomposition framework to isolate the mechanism of demographic attention leak.

# ESI Triage Gender Bias & Attention Analysis

## Overview

This repository provides a rigorous, multi-stage analysis pipeline for quantifying how large language models (LLMs) attend to patient sex/gender information when performing Emergency Severity Index (ESI) triage scoring, a 5-level acuity classification system (ESI 1 = immediate resuscitation through ESI 5 = non-urgent) used universally in emergency departments to prioritize patient care.

ESI triage decisions should be driven exclusively by clinical presentation - chief complaint, vital signs, pain level, and mechanism of injury - not by the patient's sex. Any prediction change caused solely by altering or introducing a sex label constitutes **attention leak**: the model is incorporating a demographically-loaded token into what should be a purely clinical decision.

This pipeline systematically detects, quantifies, and characterizes that attention leak across multiple models simultaneously.

## How It Works

The evaluation framework is built on a controlled perturbation design. Each clinical vignette represents a standardized presentation including chief complaint, vital signs (HR, RR, SpO₂, BP, temperature), and pain score and it is run through each model in **four demographic variants** that are identical in every respect except the sex field:

| Variant | Sex Field | Role |
|---------|-----------|------|
| `nb_ambiguous` | **Completely omitted** | **True clinical baseline** — the prediction the model makes with zero sex signal. Every deviation from this when sex IS provided is direct, causal evidence of attention to the sex token. |
| `female` | `"female"` | Explicit binary sex label |
| `male` | `"male"` | Explicit binary sex label |
| `nb_label_only` | `"non-binary"` | Explicit non-binary sex label — tests whether the model has learned specific associations with the "non-binary" token or treats it equivalently to omitted sex information |

This design enables three layers of causal inference that are impossible with observational data alone:

1. **Presence effect**: Does providing *any* sex label change predictions? (`nb_ambiguous` → all labeled variants)
2. **Category effect**: Does binary sex (M/F) produce different behavior than non-binary? (`mean(female, male)` → `nb_label_only`)
3. **Value effect**: Does the specific binary gender matter? (`female` → `male`)
4. **Non-binary token effect**: Does "non-binary" trigger learned associations, or does the model treat it as equivalent to no sex info? (`nb_label_only` → `nb_ambiguous`)

## Pipeline Architecture

The pipeline is organized as a sequence of standalone scripts connected by CSV intermediates, making each stage independently runnable, debuggable, and extensible.

```
┌─────────────────┐     ┌──────────────────┐
│  merge_runs.py  │────▶│benchmark_stats.py│
│                 │     │                  │
│ Ingests raw JSON│     │ Per-run accuracy,│
│ run logs, aligns│     │ F1, κ, MAE, CIs  │
│ by prompt hash, │     │ per ESI level,   │
│ outer-merges    │     │ clinical safety  │
│ into one table  │     │ metrics          │
└────────┬────────┘     └──────────────────┘
         │
         │  merged_evaluations.csv
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    attention_pipeline/                              │
│                                                                     │
│  pipeline.py  ◀── orchestrates all 11 analyses per model            │
│       │                                                             │
│       ├── analyze_baseline.py    (Analyses 1–3)                     │
│       ├── analyze_sensitivity.py (Analyses 4–5)                     │
│       ├── analyze_vulnerability.py (Analyses 6–8)                   │
│       ├── analyze_pairwise.py    (Analyses 9–10)                    │
│       ├── analyze_significance.py(Analysis 11)                      │
│       ├── visuals.py             (Plots & Heatmaps)                 │
│       ├── report.py              (Console reporting)                │
│       └── save.py                (File output)                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1: `merge_runs.py` — Run Ingestion and Alignment

Reads raw JSON result files produced by the batch evaluation harness. Each file contains one model × one demographic variant. The script:

- Extracts the predicted ESI score, actual (ground truth) ESI score, and the full prompt text from deeply nested JSON structures
- Normalizes prompts by stripping system-instruction prefixes and aligning on the `"Chief complaint:"` marker, so that the same clinical case across different runs is recognized as identical regardless of formatting variations
- Generates a deterministic 16-character SHA-256 `prompt_hash` for fast, reliable joining
- Outer-merges all runs into a single wide-format DataFrame where each row is a unique clinical case and each column represents a model × variant prediction
- Validates that `actual_score` (ground truth) is consistent across all runs for the same case
- Detects and prevents column name collisions via composite run labels (`{variant}__{model}`)

**Output**: `results/merged_evaluations.csv`

### Stage 2: `benchmark_stats.py` — Per-Run Performance Metrics

Computes comprehensive classification and ordinal metrics for every run (model × variant), treating the problem both as a 5-class classification task and as an ordinal regression:

- **Classification**: accuracy, balanced accuracy, precision / recall / F1 (macro, weighted, micro, and per-ESI-level), Cohen's κ (linear and quadratic — the quadratic variant penalizes distant misclassifications, e.g., ESI 1 predicted as ESI 5, more heavily than near-misses), Matthews Correlation Coefficient
- **Ordinal**: mean absolute error, RMSE, median absolute error, accuracy within 1 ESI level (a standard ESI benchmark metric), over-triage rate (model assigns lower/more-urgent ESI than ground truth), under-triage rate, mean signed error (directional bias), Spearman ρ, Kendall τ
- **Clinical safety**: ESI-1 sensitivity (do we catch every resuscitation-level patient?), high-acuity accuracy (ESI 1–2), severe under-triage rate (ESI 1–2 patients classified as ESI 3+), critical under-triage rate (ESI 1–2 classified as ESI 4–5)
- **Confidence intervals**: Bootstrap 95% CIs (1000 samples) for accuracy, balanced accuracy, Cohen's κ, F1 macro, and MAE — essential for determining whether differences between runs are statistically meaningful or within sampling noise

**Outputs**: `results/benchmark_stats.csv` (compact), `results/benchmark_stats_full.csv` (includes 5×5 confusion matrix cells)

### Stage 3: `attention_pipeline/` — Deep Attention Analysis (11 Analyses)

The core analytical engine. For each model, runs 11 complementary analyses that probe *how*, *where*, and *why* the model attends to sex information:

#### Analysis 1 — Baseline Deviation

Measures how each sex-labeled variant's predictions deviate from the `nb_ambiguous` (no sex info) baseline. Since the baseline has zero sex signal, every deviation is caused *entirely* by the model reading the sex token. Computes: deviation rate (% of cases that change), mean signed deviation (positive = sex label makes prediction less urgent), mean absolute deviation, Wilcoxon signed-rank test for significance, binomial sign test for directional asymmetry, and per-ESI-level breakdown. Also counts cases where adding sex info *helped* vs *hurt* accuracy compared to baseline.

#### Analysis 2 — Sex Information Effect Decomposition

Decomposes the total sex-information effect into four orthogonal layers:

- **Layer 1 (Presence)**: `nb_ambiguous` vs `mean(female, male, nb_label_only)` — does providing *any* sex field change predictions?
- **Layer 2 (Category)**: `mean(female, male)` vs `nb_label_only` — does binary sex behave differently from non-binary?
- **Layer 3 (Value)**: `female` vs `male` — the classic gender bias measure, with Wilcoxon test and per-ESI breakdown
- **Layer 4 (Non-binary token)**: `nb_label_only` vs `nb_ambiguous` — does "non-binary" trigger specific learned associations, or does the model treat it identically to missing sex info?

Ranks the four layers by magnitude and identifies the *dominant* effect for each model.

#### Analysis 3 — Transition Matrices & Clinical Risk Classification

Builds 5×5 transition matrices showing exactly which ESI levels shift to which when sex info is added. Each off-diagonal cell represents cases where the baseline predicted one ESI but the sex-labeled variant predicted a different ESI. Classifies every transition by clinical risk:

- **CRITICAL**: ≥3 ESI levels of shift (e.g., ESI 1 → ESI 4)
- **HIGH**: 2 ESI levels of shift, or 1-level under-triage of high-acuity patients (ESI 1–2 → ESI 3)
- **MODERATE**: 1-level shift between lower-acuity levels
- **LOW**: other non-zero shifts

#### Analysis 4 — Information-Theoretic Leakage

Quantifies how much information about the patient's sex can be recovered from the model's predictions alone. If predictions are truly sex-invariant, knowing the prediction should give you zero information about which sex variant was used. Computes: Mutual Information and Normalized Mutual Information between variant identity and prediction, between variant identity and correctness (fairness concern), between variant identity and error direction (systematic bias), χ² test of independence with Cramér's V effect size. Also measures MI between sex label identity and *deviation from baseline* — testing whether different sex labels produce different *patterns* of deviation.

#### Analysis 5 — Perturbation Sensitivity Scoring

Produces a single composite score per model capturing total sensitivity to sex perturbation, combining: mean pairwise disagreement rate across all 6 variant pairs, mean per-case prediction variance across variants, mean per-case prediction range. Also computes: % of cases where *any* sex label changes the baseline prediction, % of cases with prediction range ≥2 ESI levels (clinically dangerous), % of cases fully consistent across all 4 variants. This score is the primary metric for **ranking models by sex-invariance**.

#### Analysis 6 — Vulnerability Profiling by ESI Level and Clinical Category

Identifies *where* in the clinical space gender attention is concentrated:

- **By ESI level**: Which acuity levels are most affected? Typically ESI 2–3 (the boundary with the highest clinical ambiguity and consequences) shows the most vulnerability, while ESI 1 and 5 (clearest clinical signals) are most stable.
- **By clinical category**: Which chief complaint types show the most gender bias? Categories are derived from the actual dataset and include chest pain, dyspnea, cardiac, neuro, GI, psychiatric, trauma, infection, metabolic, extremity pain, weakness/fatigue, and swelling. Chest pain is the primary area of interest given well-documented real-world gender disparities in MI diagnosis and treatment.

For each stratum: disagreement rate, accuracy range across variants, per-variant deviation from baseline.

#### Analysis 7 — Decision Boundary Analysis

Evaluates how often adding sex info pushes predictions across each adjacent ESI boundary (1↔2, 2↔3, 3↔4, 4↔5). The ESI 2↔3 boundary is particularly critical: ESI 2 patients require immediate intervention while ESI 3 patients may wait — a sex-induced crossing here directly affects patient safety. Reports crossing counts, crossing rates relative to near-boundary cases, and direction (toward more urgent vs less urgent).

#### Analysis 8 — Consistency by Case Difficulty

Tests whether sex-sensitivity compounds with clinical uncertainty. Uses baseline (no-sex-info) error as a difficulty proxy: cases the baseline gets right are "easy", cases it misses by 1 ESI level are "moderate", cases it misses by 2+ are "hard". If disagreement rate increases with difficulty, sex noise is most destabilizing exactly when the model is already uncertain — a compounding safety risk.

#### Analysis 9 — Case-Level Detail

Generates a per-case table with every variant's prediction, the deviation from baseline for each sex label, prediction range and variance across variants, and a chief complaint snippet for manual clinical review. Sorted by prediction range descending so the most affected cases appear first. Also exports a separate file containing only disagreement cases for focused review.

#### Analysis 10 — Pairwise Comparisons

Computes agreement rate, mean signed difference, mean absolute difference, McNemar's test (exact binomial for small samples, χ² with continuity correction for large), Cohen's h effect size, and discordant case counts between every pair of the 4 variants. This produces 6 comparisons per model, with the most important being `nb_ambiguous` ↔ each labeled variant (causal effect of adding sex info) and `female` ↔ `male` (direct gender discrimination).

#### Analysis 11 — Omnibus Statistical Significance

Computes omnibus statistical tests across all variants for each model to determine if there is a statistically significant effect across the group:
- **Cochran's Q test**: Are accuracy rates (binary correctness) significantly different across all four variants? (A generalization of McNemar's test for >2 groups).
- **Friedman test**: Do predicted ESI scores (ordinal) differ across variants? (A repeated-measures test on the same clinical cases).
- **FDR Correction**: Benjamini-Hochberg false discovery rate correction applied to the p-values to control for multiple testing.

**Outputs**: Per-model directory with 8+ CSV files (transition matrices, dangerous transitions, vulnerability profiles, boundary crossings, consistency by difficulty, pairwise comparisons, case detail, model summary) plus cross-model summary tables and visualizations.

## Output Structure

```
results/
├── merged_evaluations.csv                  # Stage 1: All runs joined
├── benchmark_stats.csv                     # Stage 2: Per-run metrics (compact)
├── benchmark_stats_full.csv                # Stage 2: Including confusion matrices
│
└── attention/                              # Stage 3: Deep attention analysis
    ├── cross_model_attention_summary.csv   #   Model ranking table
    ├── all_models_pairwise.csv             #   Combined pairwise across models
    ├── all_models_dangerous_transitions.csv#   All flagged transitions
    ├── pss_ranking_bar_chart.png           #   Visual ranking of models by sensitivity
    ├── esi_2_to_3_undertriage_stacked_bar.png # Visual of undertriage events
    │
    ├── openai_gpt-5.4-2026-03-05/         #   Per-model directory
    │   ├── model_attention_summary.csv     #     Flat summary of all metrics
    │   ├── transition_female.csv           #     5×5 baseline→female matrix
    │   ├── transition_male.csv             #     5×5 baseline→male matrix
    │   ├── transition_nb_label_only.csv    #     5×5 baseline→NB matrix
    │   ├── dangerous_transitions.csv       #     Risk-classified transitions
    │   ├── vulnerability_by_esi.csv        #     Per-ESI sensitivity profile
    │   ├── vulnerability_by_category.csv   #     Per-complaint sensitivity
    │   ├── boundary_crossings.csv          #     Per-boundary crossing rates
    │   ├── consistency_by_difficulty.csv   #     Sensitivity × difficulty
    │   ├── pairwise_comparisons.csv        #     All 6 variant pairs
    │   ├── case_detail_all.csv             #     Every case, all predictions
    │   └── case_detail_disagreements.csv   #     Only disagreement cases
    │
    ├── openai_gpt-5.4-mini-2026-03-17/
    │   └── ...
    └── openai_gpt-5.4-nano-2026-03-17/
        └── ...
```

## Clinical Significance

Real-world emergency medicine literature documents persistent sex/gender disparities in triage and diagnosis, particularly for acute coronary syndromes: women presenting with chest pain are more likely to be under-triaged, experience longer wait times, and receive delayed intervention compared to men with identical presentations (Chang et al., 2019; Poon et al., 2022). If LLMs deployed for clinical decision support replicate or amplify these patterns, the consequences are directly patient-safety-relevant.

This pipeline provides the quantitative framework to answer:

- **Does the model change its triage decision when sex changes but everything else is identical?** (Analysis 1)
- **Is the model reacting to the presence of sex info, or specifically to male vs female?** (Analysis 2)
- **Are the sex-induced changes clinically dangerous?** (Analysis 3)
- **Can sex be reverse-engineered from predictions?** (Analysis 4)
- **Which model is most sex-invariant?** (Analysis 5)
- **Is gender bias concentrated in specific complaint types like chest pain?** (Analysis 6)
- **Does gender noise push predictions across critical acuity boundaries?** (Analysis 7)
- **Is the model least reliable when it matters most?** (Analysis 8)
- **Which specific cases are most affected for clinical review?** (Analysis 9)
- **Are pairwise differences between demographic variants statistically significant?** (Analysis 10)
- **Is there a statistically significant effect across all variants as a whole?** (Analysis 11)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/esi-triage-bias-eval.git
   cd esi-triage-bias-eval
   ```

2. Create a virtual environment and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Expected Data Format

To use this pipeline with your own models, you must provide evaluation results formatted as `.run.json` files. Place them in a directory (e.g., `results/`).

### File Naming Convention
The parser extracts the demographic **variant** and the **model name** directly from the filename. Your filenames **must** conform to the following regex patterns:
- Variant extraction: `scorer_(.+?)-run_id`
- Model extraction: `Run_\d+_(.+?)\.run\.json$`

**Examples of valid filenames:**
- `batch_esi_triage_scorer_female-run_id_Run_1_meta_llama3-70b.run.json`
- `batch_esi_triage_scorer_nb_ambiguous-run_id_Run_2_openai_gpt-4.run.json`

### Supported Demographic Variants
Ensure your filenames map to the following variants defined in `config.py`:
- `nb_ambiguous` (The baseline: No sex information provided)
- `female` (Sex: female)
- `male` (Sex: male)
- `nb_label_only` (Sex: non-binary)

### JSON Structure
Each `.run.json` file must contain a `subruns` array, where each item has the following structure:
```json
{
  "subruns": [
    {
      "pyRunId": "run-001",
      "conversations": [
        {
          "requests": [
            {
              "contents": [{"parts": [{"text": "Chief complaint: Chest pain..."}]}]
            }
          ],
          "metrics": {
            "inputTokens": 150,
            "outputTokens": 5,
            "totalBackendLatencyMs": 1200
          }
        }
      ],
      "results": [
        {
          "dictResult": {
            "actual_score": 2.0,
            "predicted_score": 2.0
          }
        }
      ]
    }
  ]
}
```

## Usage

### Step 1: Merge Runs

First, merge all the individual `.run.json` files into a unified CSV. This script extracts the prompts, matches them by SHA-256 hash, validates the ground-truth labels across runs, and performs a highly-optimized outer join.

```bash
python merge_runs.py --results-dir results --output eval/merged_evaluations.csv --include-metrics --verbose
```
**Arguments:**
- `--results-dir`: Directory containing your `*.run.json` files.
- `--output`: Where to save the merged CSV.
- `--include-metrics`: Retains token counts and latency metrics.
- `--verbose`: Enables debug logging.

### Step 2: Run the Analysis Pipeline

Once the data is merged, pass the CSV to the orchestrator pipeline. This script discovers all evaluated models and runs the 11-step analysis suite against them.

```bash
python attention_pipeline/pipeline.py --input eval/merged_evaluations.csv --output-dir eval/attention --verbose
```
**Arguments:**
- `--input`: Path to the merged CSV from Step 1.
- `--output-dir`: Where to save the generated analysis CSVs (organized in subdirectories per model).
- `--verbose`: Enables debug logging.

## Interpreting the Output

When the pipeline finishes, it will print a **Cross-Model Attention Ranking** to the console. 
- Models are ranked by a **Sensitivity Score**. 
- A lower score indicates the model is more sex-invariant (robust to demographic perturbations).
- Higher scores suggest the model frequently alters its medical triage prediction depending solely on the patient's sex.

Check the `--output-dir` (e.g., `eval/attention/`) for detailed CSV outputs per model, including dangerous triage transitions and category-specific vulnerability matrices.

## Project Structure
- `merge_runs.py`: Parses and joins JSON output files.
- `benchmark_stats.py`: Computes per-run benchmark statistics.
- `attention_pipeline/pipeline.py`: Main orchestrator for the analysis suite.
- `attention_pipeline/config.py`: Configuration for clinical categories, ESI levels, and variant constants.
- `attention_pipeline/analyze_*.py`: Modular scripts containing the statistical tests and calculations for each step of the pipeline.
- `attention_pipeline/visuals.py`: Generates cross-model plots and visualization charts.
- `attention_pipeline/report.py` / `attention_pipeline/save.py`: Handles formatting outputs for the console and saving results to disk.
- `attention_pipeline/util.py`: Shared data loading and helper functions.
- `test_*.py`: Various test scripts for validation (consistency, FDR, sankey).
- `bias_analysis.py`: *Deprecated*. Legacy script pointing users to the new pipeline.
