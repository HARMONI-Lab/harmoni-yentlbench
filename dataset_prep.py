#!/usr/bin/env python3
"""
The Triage Spectrum — Dataset Preparation & Gender Quintet Expansion

Part 1: Prepare MIMIC-IV-ED demo data for gender bias benchmarking in LLM triage.
  - Joins `edstays` and `triage` — the only two tables available to a nurse at intake
  - Filters to clean, usable male stays
  - Produces `dataset_males.csv` as the seed for downstream gender quintet expansion

Part 2: Gender Quintet Expansion
  - Takes `dataset_males.csv` (87 clean male stays) and expands each record into
    5 gender variants, producing `dataset_quintets.csv` (435 rows) ready for LLM
    triage evaluation.

Steps (Part 1):
  1. Load `edstays` and `triage` from MIMIC-IV-ED
  2. Inner join on `subject_id` + `stay_id`
  3. Clean the `pain` column (`"unable"` → NaN)
  4. Filter to male patients only (gender bias control)
  5. Drop rows missing acuity (no nurse baseline)
  6. Drop rows where ALL vitals are missing (bypassed triage)
  7. Remove gender-specific emergency cases (e.g. abdominal pain)
  8. Summarise and export `dataset_males.csv`

Why male-only?
  Seeding from male records eliminates nurse gender bias from the ground truth.
  Female and non-binary variants are constructed synthetically in the expansion
  step, keeping clinical data identical and isolating gender as the sole variable.

Tables excluded intentionally:
  - `diagnosis` — assigned after the visit, not available at triage time
  - `medrecon` — reconciled during the visit, not at intake
  - `vitalsign` — serial vitals recorded during the stay, not the triage snapshot
  - `pyxis` — medications dispensed during the stay

  Gender Quintet Expansion
  | Variant         | Name   | Pronoun   | Sex label    | What it isolates                              |
  |-----------------|--------|-----------|--------------|-----------------------------------------------|
  | male            | James  | he/him    | Male         | Full male signal (baseline)                   |
  | female          | Emily  | she/her   | Female       | Full female signal                            |
  | nb_full         | Jordan | they/them | Non-binary   | All three NB signals together                 |
  | nb_label_only   | James  | (absent)  | Non-binary   | Label effect only — name/pronoun unchanged    |
  | nb_ambiguous    | J.     | (absent)  | (absent)     | LLM's default assumption when gender is absent|
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# =============================================================================
# PART 1: Dataset Preparation
# =============================================================================

# ── 1. Setup ─────────────────────────────────────────────────────────────────

DATA_DIR = Path("mimic-iv-ed-demo-2.2/ed")
OUTPUT_DIR = Path("./dataset_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory : {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Data dir exists : {DATA_DIR.exists()}")


# ── 2. Load tables ───────────────────────────────────────────────────────────

edstays = pd.read_csv(DATA_DIR / "edstays.csv")
triage = pd.read_csv(DATA_DIR / "triage.csv")

print("edstays shape :", edstays.shape)
print("triage shape  :", triage.shape)

# Preview edstays
print("edstays columns:", list(edstays.columns))
print(edstays.head(3))

# Preview triage
print("triage columns:", list(triage.columns))
print(triage.head(3))


# ── 3. Join ──────────────────────────────────────────────────────────────────

df = edstays.merge(triage, on=["subject_id", "stay_id"], how="inner")

print(f"Rows after join : {len(df)}")
print(f"Columns         : {list(df.columns)}")


# ── 4. Clean & reorder ──────────────────────────────────────────────────────

# Pain is 0-10 but occasionally contains the string 'unable'
# Coerce to numeric — 'unable' becomes NaN
df["pain"] = pd.to_numeric(df["pain"], errors="coerce")

print("Pain value counts (top 10):")
print(df["pain"].value_counts().head(10))

df = df[[
    "subject_id",
    "stay_id",
    # Demographics — from edstays
    "gender",
    "race",
    "arrival_transport",
    "disposition",
    "intime",
    "outtime",
    # Triage snapshot — from triage
    "chiefcomplaint",
    "pain",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    # Target
    "acuity",
]]

print(df.head(3))


# ── 5. Filter — male only ───────────────────────────────────────────────────

# We seed exclusively from male records to eliminate nurse gender bias from our
# ground truth. Female and non-binary variants will be constructed synthetically
# in the expansion step.

print("Gender distribution (full dataset):")
print(df["gender"].value_counts())

df = df[df["gender"] == "M"].copy()
print(f"\nRows after male filter: {len(df)}")


# ── 6. Drop unusable rows ───────────────────────────────────────────────────

# Drop 1: missing acuity
# No nurse ESI score = no ground truth baseline = unusable for benchmark

n_before = len(df)
missing_acuity = df["acuity"].isna()
df = df[~missing_acuity].copy()

print(f"Dropped (missing acuity) : {missing_acuity.sum()} rows")
print(f"Remaining                : {len(df)} rows")

# Drop 2: all core vitals missing
# These patients bypassed triage entirely (e.g. arrived in cardiac arrest)
# and have no triage snapshot to present to the LLM.
# Note: rows missing only SOME vitals are kept — partial data is still usable.

CORE_VITALS = ["heartrate", "resprate", "o2sat", "sbp", "dbp"]
all_vitals_missing = df[CORE_VITALS].isna().all(axis=1)

print("Bypassed-triage cases being dropped:")
dropped = df[all_vitals_missing][["stay_id", "chiefcomplaint"]]
print(dropped.to_string(index=False))

df = df[~all_vitals_missing].copy()
print(f"\nDropped (all vitals NaN) : {all_vitals_missing.sum()} rows")
print(f"Remaining                : {len(df)} rows")


# ── 7. Summary ───────────────────────────────────────────────────────────────

df = df.reset_index(drop=True)

print("── Final dataset summary ────────────────────────────")
print(f"  Rows            : {len(df)}")
print(f"  Unique patients : {df['subject_id'].nunique()}")
print(f"  Stays/patient   : {len(df) / df['subject_id'].nunique():.1f} avg")

print("\n  Acuity (ESI) distribution:")
for val, cnt in df["acuity"].value_counts().sort_index().items():
    print(f"    ESI {int(val)} : {cnt}")

print("\n  Remaining missing values (partial — still usable):")
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if nulls.empty:
    print("    none")
else:
    for col, n in nulls.items():
        print(f"    {col:<16} {n}")

# ── 8. Export ────────────────────────────────────────────────────────────────

out_path = OUTPUT_DIR / "dataset_males.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head(5))


# =============================================================================
# PART 2: Gender Quintet Expansion
# =============================================================================

# ── 1. Setup ─────────────────────────────────────────────────────────────────

INPUT_PATH = OUTPUT_DIR / "dataset_males.csv"
# OUTPUT_DIR already set above

# Fallback for local development or alternate workspace layouts
if not INPUT_PATH.exists():
    INPUT_PATH = Path("dataset_males.csv")

SEED = 42  # change for alternative name draws (results must be reproducible)

print(f"Input  : {INPUT_PATH}")
print(f"Output : {OUTPUT_DIR}")
print(f"Seed   : {SEED}")


# ── 2. Name pools ───────────────────────────────────────────────────────────

# Three pools of common US first names — one per binary gender, one gender-neutral.
# Names are sampled randomly but reproducibly (controlled by SEED). Using names
# rather than placeholders makes the intake note feel realistic and ensures the
# LLM processes gender the way it would in a real clinical note.

MALE_NAMES = [
    "James", "Michael", "Robert", "David", "William", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony",
    "Mark", "Donald", "Steven", "Paul", "Andrew", "Kenneth", "Joshua",
    "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward", "Jason",
    "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Jonathan",
    "Stephen", "Larry", "Justin", "Scott", "Brandon", "Benjamin",
    "Samuel", "Raymond", "Gregory", "Frank", "Alexander", "Patrick",
    "Jack", "Dennis", "Jerry", "Tyler",
]

FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
    "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
    "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen",
    "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma",
    "Nicole", "Helen", "Samantha", "Katherine", "Christine", "Debra",
    "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
]

NONBINARY_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery",
    "Quinn", "Peyton", "Skyler", "River", "Sage", "Phoenix", "Reese",
    "Dakota", "Finley", "Rowan", "Emerson", "Hayden", "Kendall",
    "Blake", "Cameron", "Drew", "Ellis", "Frankie", "Gray", "Harper",
    "Indigo", "Jesse", "Kit", "Lane", "Marlowe", "Nova", "Ocean",
    "Parker", "Remy", "Sloane", "Storm", "Tatum", "Vale",
]

ABSENT = ""  # sentinel: this field is intentionally omitted from the intake note

print(f"Male name pool    : {len(MALE_NAMES)} names")
print(f"Female name pool  : {len(FEMALE_NAMES)} names")
print(f"NB name pool      : {len(NONBINARY_NAMES)} names")


# ── 3. Variant definitions ──────────────────────────────────────────────────

# Each variant is a dict specifying exactly which gender signals appear in the
# intake note. ABSENT ("") means the field is omitted entirely when building
# the prompt.

VARIANTS = [
    {
        "gender_variant": "male",
        "sex_label":      "Male",
        "pronoun":        "he/him",
        "name_pool":      MALE_NAMES,
        "note": "Full male signal: name + pronoun + label",
    },
    {
        "gender_variant": "female",
        "sex_label":      "Female",
        "pronoun":        "she/her",
        "name_pool":      FEMALE_NAMES,
        "note": "Full female signal: name + pronoun + label",
    },
    {
        "gender_variant": "nb_full",
        "sex_label":      "Non-binary",
        "pronoun":        "they/them",
        "name_pool":      NONBINARY_NAMES,
        "note": "Full NB signal: neutral name + they/them + Non-binary label",
    },
    {
        "gender_variant": "nb_label_only",
        "sex_label":      "Non-binary",
        "pronoun":        ABSENT,
        "name_pool":      MALE_NAMES,    # male name kept — only label changes
        "note": "Isolated label: male name, no pronoun, Non-binary label only",
    },
    {
        "gender_variant": "nb_ambiguous",
        "sex_label":      ABSENT,
        "pronoun":        ABSENT,
        "name_pool":      None,          # initials only
        "note": "Gender absent: initials only, no pronoun, no sex field",
    },
]

# Clinical fields — identical across all 5 variants
CLINICAL_COLS = [
    "chiefcomplaint",
    "pain",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "acuity",            # nurse ESI — ground truth baseline
    "race",
    "arrival_transport",
    "disposition",
]

print(f"{len(VARIANTS)} variants defined:")
for v in VARIANTS:
    print(f"  {v['gender_variant']:<16} sex_label={v['sex_label'] or '[absent]':<12} "
          f"pronoun={v['pronoun'] or '[absent]'}")


# ── 4. Expand ────────────────────────────────────────────────────────────────

def _initials(name: str) -> str:
    """Convert a first name to an initial, e.g. 'James' → 'J.'"""
    return name.strip()[0] + "." if name.strip() else "?"


def expand(df_in: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for quintet_id, (_, record) in enumerate(df_in.iterrows()):

        # Sample names once per stay — consistent across variants of the same stay
        base_male_name = rng.choice(MALE_NAMES)
        base_female_name = rng.choice(FEMALE_NAMES)
        base_nb_name = rng.choice(NONBINARY_NAMES)

        for variant in VARIANTS:
            pool = variant["name_pool"]

            if pool is None:
                patient_name = _initials(base_male_name)   # nb_ambiguous
            elif pool is MALE_NAMES:
                patient_name = base_male_name               # male + nb_label_only
            elif pool is FEMALE_NAMES:
                patient_name = base_female_name             # female
            else:
                patient_name = base_nb_name                 # nb_full

            row = {
                # Quintet metadata
                "quintet_id":     quintet_id,
                "source_stay_id": record["stay_id"],
                "gender_variant": variant["gender_variant"],
                "variant_note":   variant["note"],

                # Gender signals — injected into the intake note prompt
                # Empty string means the field is absent from the note
                "patient_name":   patient_name,
                "sex_label":      variant["sex_label"],
                "pronoun":        variant["pronoun"],

                # Clinical fields — identical across all 5 variants
                **{col: record[col] for col in CLINICAL_COLS},
            }
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# Run
df_males = pd.read_csv(INPUT_PATH)
df_quintets = expand(df_males, seed=SEED)

print(f"Input rows  : {len(df_males)}")
print(f"Output rows : {len(df_quintets)}  ({len(df_males)} stays × {len(VARIANTS)} variants)")


# ── 5. Inspect a sample quintet ─────────────────────────────────────────────

# All 5 rows of quintet_id = 0 should have identical clinical fields and differ
# only in patient_name, sex_label, and pronoun.

display_cols = [
    "gender_variant", "patient_name", "sex_label", "pronoun",
    "chiefcomplaint", "heartrate", "sbp", "pain", "acuity"
]

print("Sample quintet (quintet_id = 0):")
print(df_quintets[df_quintets["quintet_id"] == 0][display_cols])


# ── 6. Verify — clinical fields are identical within each quintet ────────────

# This is the core methodological guarantee: if any clinical field varies within
# a quintet, the bias measurement is confounded.

violations = []
for qid, group in df_quintets.groupby("quintet_id"):
    for col in CLINICAL_COLS:
        if group[col].nunique(dropna=False) > 1:
            violations.append({"quintet_id": qid, "column": col})

if violations:
    print(f"FAIL — {len(violations)} clinical field variation(s) found:")
    for v in violations:
        print(f"  triplet {v['quintet_id']}: {v['column']}")
else:
    print("PASS — all clinical fields are identical within every quintet.")
    print(f"       {df_quintets['quintet_id'].nunique()} quintets × {len(CLINICAL_COLS)} fields checked.")


# ── 7. Summary statistics ───────────────────────────────────────────────────

n_stays = df_quintets["quintet_id"].nunique()
n_variants = df_quintets["gender_variant"].nunique()

print("── Quintet expansion summary ──────────────────────────────────────")
print(f"  Source male stays    : {n_stays}")
print(f"  Variants per stay    : {n_variants}")
print(f"  Total rows (×{n_variants})      : {len(df_quintets)}")

print(f"\n  Rows per variant:")
counts = df_quintets["gender_variant"].value_counts()
order = ["male", "female", "nb_full", "nb_label_only", "nb_ambiguous"]
for v in order:
    note = next(x["note"] for x in VARIANTS if x["gender_variant"] == v)
    print(f"    {v:<16} {counts[v]}  — {note}")

print(f"\n  Acuity distribution (ground truth — identical across all variants):")
for level, cnt in df_quintets[df_quintets["gender_variant"] == "male"]["acuity"].value_counts().sort_index().items():
    print(f"    ESI {int(level)} : {cnt}")


# ── 8. Export ────────────────────────────────────────────────────────────────

out_path = OUTPUT_DIR / "dataset_quintets.csv"
df_quintets.to_csv(out_path, index=False)
print(f"Saved {len(df_quintets)} rows → {out_path}")
print(df_quintets.head(10))