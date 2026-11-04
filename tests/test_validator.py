import pandas as pd

from validator import required_cols, validate_dataframe


def make_row(**overrides):
    row = {
        "user_id": "u1",
        "image_paths": "images/a.jpg",
        "user_claim": "dent on the door",
        "claim_object": "car",
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "clear",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "door",
        "claim_status": "supported",
        "claim_status_justification": "visible",
        "supporting_image_ids": "a",
        "valid_image": "true",
        "severity": "medium",
    }
    row.update(overrides)
    return row


def validate(**overrides):
    return validate_dataframe(pd.DataFrame([make_row(**overrides)])).iloc[0]


def test_severity_synonyms_are_mapped():
    assert validate(severity="Moderate")["severity"] == "medium"
    assert validate(severity="severe")["severity"] == "high"


def test_invalid_severity_becomes_unknown():
    assert validate(severity="catastrophic")["severity"] == "unknown"


def test_free_text_issue_type_is_mapped_to_the_allowed_list():
    assert validate(issue_type="windshield_crack")["issue_type"] == "crack"
    assert validate(issue_type="liquid_damage")["issue_type"] == "water_damage"


def test_unmapped_issue_type_becomes_unknown():
    assert validate(issue_type="something_else")["issue_type"] == "unknown"


def test_object_part_is_mapped_or_unknown():
    assert validate(object_part="front bumper")["object_part"] == "front_bumper"
    assert validate(object_part="left wing mirror cover")["object_part"] == "unknown"


def test_risk_flags_are_translated_deduplicated_and_sorted():
    result = validate(risk_flags="stock_image_detected; blurry_image, blurry_image")
    assert result["risk_flags"] == "blurry_image;non_original_image"


def test_unknown_risk_flags_are_dropped():
    assert validate(risk_flags="made_up_flag")["risk_flags"] == "none"


def test_supporting_image_ids_are_filename_stems():
    assert validate(supporting_image_ids="img/a1.jpg, a2.png")["supporting_image_ids"] == "a1;a2"
    assert validate(supporting_image_ids="none")["supporting_image_ids"] == "none"


def test_booleans_are_normalised():
    assert validate(evidence_standard_met="supported")["evidence_standard_met"] == "true"
    assert validate(valid_image="maybe")["valid_image"] == "false"


def test_output_has_exactly_the_required_columns_in_order():
    result = validate_dataframe(pd.DataFrame([make_row(extra_column="x")]))
    assert list(result.columns) == required_cols


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame([make_row(severity="Moderate")])
    validate_dataframe(df)
    assert df.iloc[0]["severity"] == "Moderate"