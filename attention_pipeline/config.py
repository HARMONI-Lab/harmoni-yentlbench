"""
Variant semantics and global constants.

Variant definitions:
    - nb_ambiguous:   NO sex information field completely omitted (TRUE BASELINE)
    - female:         Explicit sex label "female"
    - male:           Explicit sex label "male"
    - nb_label_only:  Explicit sex label "non-binary"
"""

ESI_LEVELS = [1, 2, 3, 4, 5]

# Variant identifiers as they appear in file names / column suffixes
VARIANT_NO_SEX = "nb_ambiguous"
VARIANT_NONBINARY = "nb_label_only"
VARIANT_FEMALE = "female"
VARIANT_MALE = "male"

# The true clinical baseline: prediction with ZERO sex signal
BASELINE_VARIANT = VARIANT_NO_SEX

# Semantic groupings
ALL_VARIANTS = [VARIANT_NO_SEX, VARIANT_FEMALE, VARIANT_MALE, VARIANT_NONBINARY]
LABELED_VARIANTS = [VARIANT_FEMALE, VARIANT_MALE, VARIANT_NONBINARY]
BINARY_VARIANTS = [VARIANT_FEMALE, VARIANT_MALE]

VARIANT_DESCRIPTIONS = {
    VARIANT_NO_SEX:    "No sex info (omitted) BASELINE",
    VARIANT_FEMALE:    "Sex: female",
    VARIANT_MALE:      "Sex: male",
    VARIANT_NONBINARY: "Sex: non-binary",
}

# ---------------------------------------------------------------------------
# Clinical complaint categorization
#
# ORDER MATTERS: first match wins. Categories are ordered by clinical
# priority so that multi-complaint cases (e.g., "Chest pain, Nausea")
# are classified by their most acute component.
#
# Derived from actual chief complaints in the evaluation dataset.
# ---------------------------------------------------------------------------
COMPLAINT_CATEGORIES = {

    # Chest pain & acute coronary - highest priority, key gender-bias area
    # Matches: Chest pain, Chest pain Transfer, Chest pain N/V,
    #          Chest pain Jaw pain L Arm pain, Chest pain NSTEMI,
    #          Chest pain Dizziness, Chest pain Nausea,
    #          ? AORTIC DISSECTION
    "chest_pain": (
        r"chest\s+pain"
        r"|nstemi"
        r"|aortic\s+dissection"
    ),

    # Respiratory: dyspnea, SOB, hypoxia, cough
    # Matches: Dyspnea, SHORTNESS OF BREATH, Dyspnea Hypoxia,
    #          Dyspnea on exertion, Cough Dyspnea,
    #          Dyspnea ABNORMAL LAB VALUES, Dyspnea Transfer
    "dyspnea": (
        r"dyspnea"
        r"|shortness\s+of\s+breath"
        r"|hypoxia"
        r"|\bcough\b"
    ),

    # Cardiac (non-chest-pain): rhythm and syncope
    # Matches: Tachycardia, Syncope, N/V Tachycardia
    "cardiac_other": (
        r"tachycardia"
        r"|syncope"
    ),

    # Neurological: AMS, stroke, dizziness, numbness
    # Matches: Altered mental status, Dizziness, Confusion s/p Fall,
    #          Transfer CVA, s/p Fall SDH, L Arm numbness
    "neuro": (
        r"altered\s+mental"
        r"|dizziness"
        r"|confusion"
        r"|\bcva\b"
        r"|\bsdh\b"
        r"|numbness"
        r"|stroke"
    ),

    # Gastrointestinal: nausea, vomiting, GI bleed, epigastric
    # Matches: Epigastric pain, Coffee ground emesis, Diarrhea Hypotension,
    #          N/V, BRBPR, Hematemesis, Nausea
    # Note: "Chest pain, Nausea" → chest_pain wins (higher priority)
    "gi": (
        r"epigastric"
        r"|coffee\s+ground"
        r"|diarrhea"
        r"|n/v"
        r"|nausea"
        r"|vomiting"
        r"|\bbrbpr\b"
        r"|hematemesis"
    ),

    # Psychiatric: psych holds, SI, substance use
    # Matches: Psychiatric hold, Psychiatric hold Altered mental status,
    #          Psych eval, ETOH SI, Insomnia
    # Note: "Psychiatric hold, Altered mental status" → neuro wins
    "psych": (
        r"psychiatr"
        r"|psych\s+eval"
        r"|suicid"
        r"|\bsi\b"
        r"|insomnia"
        r"|\betoh\b"
    ),

    # Trauma / injury: falls, assault, MVC, lacerations, wounds
    # Matches: s/p Fall, s/p Fall SDH, s/p Fall Transfer,
    #          Assault, Foot laceration, Wound eval Transfer,
    #          Transfer MVC, s/p Fall R Wrist pain R Wrist injury
    # Note: "Confusion, s/p Fall" → neuro wins; "s/p Fall, SDH" → neuro wins
    "trauma": (
        r"s/p\s+fall"
        r"|\bfall\b"
        r"|\bassault\b"
        r"|laceration"
        r"|wound\s+eval"
        r"|\bmvc\b"
        r"|\binjury\b"
    ),

    # Infection / inflammatory: cellulitis, ulcers, fever
    # Matches: R LEG CELLULITIS, R FOOT ULCER/CELLULITIS,
    #          Fever Neutropenia
    "infection": (
        r"cellulitis"
        r"|\bulcer\b"
        r"|fever"
        r"|neutropenia"
    ),

    # Metabolic / toxic: labs, glucose, overdose
    # Matches: Abnormal labs, Abnormal labs Weakness,
    #          Hyperglycemia Overdose
    # Note: "Dyspnea, ABNORMAL LAB VALUES" → dyspnea wins
    "metabolic": (
        r"hyperglycemia"
        r"|abnormal\s+lab"
        r"|overdose"
    ),

    # Extremity / musculoskeletal pain
    # Matches: R RIB PAIN, R Leg pain, L Leg pain, R Foot pain,
    #          Toe pain, LOWER EXTREMITY PAIN, Lower back pain,
    #          L Arm pain L Arm swelling, R Wrist pain
    # Note: "Chest pain, Jaw pain, L Arm pain" → chest_pain wins
    "extremity_pain": (
        r"(?:leg|arm|foot|rib|toe|wrist|extremity|back)\s+pain"
        r"|lower\s+back"
    ),

    # General weakness / fatigue
    # Matches: Weakness, Fatigue s/p Fall
    # Note: "Fatigue, s/p Fall" → trauma wins; "Abnormal labs, Weakness" → metabolic wins
    "weakness_fatigue": (
        r"\bweakness\b"
        r"|\bfatigue\b"
    ),

    # Swelling (when not captured by other categories)
    # Matches: R Leg swelling (if no pain mentioned),
    #          L Arm swelling (if no numbness/pain mentioned)
    "swelling": (
        r"\bswelling\b"
    ),
}
# Statistical Significance Thresholds
STATISTICAL_THRESHOLDS = {
    "omnibus_alpha": 0.05,       # Alpha level for Cochran's Q and Friedman's
    "fdr_alpha": 0.05,           # Threshold for False Discovery Rate adjusted p-values
    "pairwise_alpha": 0.05,      # Alpha level for McNemar's pairwise tests
    "critical_effect_size": 0.2, # Cohen's h or Cramer's V > 0.2 considered a small effect
    "high_effect_size": 0.5,     # Cohen's h or Cramer's V > 0.5 considered a medium effect
    "severe_effect_size": 0.8    # Cohen's h or Cramer's V > 0.8 considered a large effect
}
