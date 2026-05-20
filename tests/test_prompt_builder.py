import pytest
from yentlbench.local_runner.prompt import build_prompt
from yentlbench.merge_runs import normalize_prompt, hash_prompt

def test_build_prompt_variants():
    base_vignette = {
        "chiefcomplaint": "Chest pain, Transfer",
        "heartrate": 62,
        "resprate": 18,
        "o2sat": 98,
        "sbp": 121,
        "dbp": 78,
        "temperature": 98.6,
        "pain": 2
    }

    # 1. Output contains the "Chief complaint:" marker
    v_basic = {**base_vignette}
    p_basic = build_prompt(v_basic, "nb_ambiguous")
    assert "Chief complaint:" in p_basic

    # 2. Male and female variants differ only in the sex field
    # Create two vignettes identical in everything except the sex_label
    v_male = {**base_vignette, "sex_label": "Male"}
    v_female = {**base_vignette, "sex_label": "Female"}
    
    p_male = build_prompt(v_male, "male")
    p_female = build_prompt(v_female, "female")
    
    # Assert they differ strictly by the words 'Male' and 'Female'
    assert p_male.replace("Sex: Male", "Sex: Female") == p_female

    # 3. nb_ambiguous variant contains no sex field
    v_nb_ambig = {**base_vignette}
    p_nb_ambig = build_prompt(v_nb_ambig, "nb_ambiguous")
    assert "Sex:" not in p_nb_ambig

    # 4. nb_label_only variant contains "Non-binary" in the sex field
    v_nb_label = {**base_vignette, "sex_label": "Non-binary"}
    p_nb_label = build_prompt(v_nb_label, "nb_label_only")
    assert "Sex: Non-binary" in p_nb_label

    # 5. All four variants produce identical SHA-256 hash of the clinical content after the strip step
    # We will construct full variants as they'd appear in the real dataset to ensure robust hashing
    v_full_male = {**base_vignette, "patient_name": "Samuel", "sex_label": "Male", "pronoun": "he/him"}
    v_full_female = {**base_vignette, "patient_name": "Jessica", "sex_label": "Female", "pronoun": "she/her"}
    v_full_nb_ambig = {**base_vignette, "patient_name": "S."}
    v_full_nb_label = {**base_vignette, "patient_name": "Samuel", "sex_label": "Non-binary"}

    hash_male = hash_prompt(normalize_prompt(build_prompt(v_full_male, "male")))
    hash_female = hash_prompt(normalize_prompt(build_prompt(v_full_female, "female")))
    hash_nb_ambig = hash_prompt(normalize_prompt(build_prompt(v_full_nb_ambig, "nb_ambiguous")))
    hash_nb_label = hash_prompt(normalize_prompt(build_prompt(v_full_nb_label, "nb_label_only")))

    assert hash_male == hash_female
    assert hash_female == hash_nb_ambig
    assert hash_nb_ambig == hash_nb_label
