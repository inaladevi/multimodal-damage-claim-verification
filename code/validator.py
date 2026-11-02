import os
import re
import pandas as pd

severity_map = {
    "minor": "low", "moderate": "medium", "major": "high",
    "severe": "high", "negligible": "low", "not applicable": "unknown",
}

valid_severities = {"none", "low", "medium", "high", "unknown"}

evidence_map = {"true": "true", "false": "false", "supported": "true"}
valid_booleans = ["true", "false"]

def sanitize_text(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"(_input_token_count|quotaId|generativelanguage|RPC).*$", "", text).strip()

allowed_risk_enums = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
}

risk_translation_map = {
    "stock_image_detected": "non_original_image",
    "suspicious_instruction_attempt": "text_instruction_present",
    "attempt to bypass manual review": "text_instruction_present",
    "potential_fraud": "possible_manipulation",
    "high_risk_mismatch": "claim_mismatch",
    "mismatch_between_claim_and_image": "claim_mismatch",
    "inconsistent_evidence": "manual_review_required",
    "inconsistent_claim": "manual_review_required",
    "high_frequency_claimant": "user_history_risk",
    "multiple previous claims": "user_history_risk"
}

def strict_normalize_risk_flags(val):
    if pd.isna(val) or str(val).lower() in ["none", "nan", "", "unknown"]:
        return "none"
    delimiters = r'[;,]'
    raw_tokens = [t.strip().lower() for t in re.split(delimiters, str(val)) if t.strip()]
    translated_tokens = []
    for token in raw_tokens:
        if token in risk_translation_map:
            token = risk_translation_map[token]
        if token in allowed_risk_enums:
            translated_tokens.append(token)
    final_tokens = sorted(list(set(translated_tokens)))
    if not final_tokens or final_tokens == ["none"]:
        return "none"
    return ";".join(final_tokens)

issue_type_map = {
    "windshield_crack": "crack",
    "cracked_screen": "crack",
    "cracked_lid": "crack",
    "broken_side_mirror": "broken_part",
    "headlight_damage": "broken_part",
    "taillight_damage": "broken_part",
    "hinge_damage": "broken_part",
    "keyboard_damage": "broken_part",
    "screen_damage": "broken_part",
    "side_mirror": "broken_part",
    "bumper_damage": "dent",
    "collision_damage": "dent",
    "hail_damage": "dent",
    "liquid_damage": "water_damage",
    "packaging_damage": "crushed_packaging",
    "damaged_packaging": "crushed_packaging",
    "package_damage": "crushed_packaging",
    "torn_open_package": "torn_packaging",
    "package_stain": "stain",
    "shipping_label_damage": "stain",
    "missing_item": "missing_part",
    "missing_key": "missing_part",
    "package_damage_missing_contents": "missing_part",
    "claim_mismatch": "unknown",
    "wrong_object": "unknown",
    "damage_not_visible": "unknown",
    "inconsistent_evidence": "unknown",
    "physical_damage": "unknown",
    "damaged_item": "unknown"
}

allowed_issue_types = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown"
}

object_part_map = {
    "front bumper": "front_bumper",
    "rear bumper": "rear_bumper",
    "display": "screen",
    "display_panel": "screen",
    "display_screen": "screen",
    "laptop body": "body",
    "car body": "body",
    "package_seal": "seal",
    "shipping label": "label",
    "shipping_label": "label",
    "outer_box": "box",
    "cardboard box exterior": "box",
    "box contents": "contents",
    "interior_contents": "contents",
    "door panel": "door",
    "left side door": "door",
    "outer_packaging": "box",
    "packaging": "box",
    "cardboard_packaging": "box",
    "exterior cardboard surface": "box",
    "tamper_evident_tape": "seal",
    "front_bumper_left_headlight": "unknown",
    "front_headlight_and_hood": "unknown",
    "door;bumper": "unknown",
    "hinge and left chassis": "unknown",
    "scooter side panel": "unknown",
}

allowed_object_parts = {
    "front_bumper","rear_bumper","door","hood","windshield",
    "side_mirror","headlight","taillight","fender","quarter_panel",
    "body","screen","keyboard","trackpad","hinge","lid","corner",
    "port","base","box","package_corner","package_side","seal",
    "label","contents","item","unknown"
}

def clean_supporting_image_ids(val):
    if pd.isna(val) or str(val).lower() in ["none", "nan", "", "unknown"]:
        return "none"
    delimiters = r'[;,]'
    raw_parts = [p.strip() for p in re.split(delimiters, str(val)) if p.strip()]
    clean_ids = []
    for part in raw_parts:
        base_name = os.path.basename(part)
        img_id = os.path.splitext(base_name)[0]
        if img_id and img_id.lower() != "none":
            clean_ids.append(img_id)
    final_ids = sorted(list(set(clean_ids)))
    return ";".join(final_ids) if final_ids else "none"

required_cols = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity"
]

def validate_dataframe(df):
    """Normalize raw model output into the final schema (same rules for the pipeline and the evaluation)."""
    df = df.copy()
    df["severity"] = df["severity"].astype(str).str.lower().str.strip().replace(severity_map)
    df.loc[~df["severity"].isin(valid_severities), "severity"] = "unknown"
    df["evidence_standard_met"] = df["evidence_standard_met"].astype(str).str.lower().str.strip()
    df["evidence_standard_met"] = df["evidence_standard_met"].replace(evidence_map)
    df.loc[~df["evidence_standard_met"].isin(valid_booleans), "evidence_standard_met"] = "false"
    df["valid_image"] = df["valid_image"].astype(str).str.lower().str.strip().replace(evidence_map)
    df.loc[~df["valid_image"].isin(valid_booleans), "valid_image"] = "false"
    df["user_claim"] = df["user_claim"].apply(sanitize_text)
    df["risk_flags"] = df["risk_flags"].apply(strict_normalize_risk_flags)
    df["issue_type"] = (
        df["issue_type"]
        .astype(str)
        .str.strip()
        .replace(issue_type_map)
    )
    df.loc[
        ~df["issue_type"].isin(allowed_issue_types),
        "issue_type"
    ] = "unknown"
    df["object_part"] = (
        df["object_part"]
        .astype(str)
        .str.strip()
        .replace(object_part_map)
    )
    df.loc[
        ~df["object_part"].isin(allowed_object_parts),
        "object_part"
    ] = "unknown"
    df["supporting_image_ids"] = df["supporting_image_ids"].apply(clean_supporting_image_ids)
    df = df[required_cols]
    return df


def main():
    print("Initializing post-processing validation...")
    claims_input_df = pd.read_csv("dataset/claims.csv")
    df = pd.read_csv("dataset/output.csv")
    df = validate_dataframe(df)

    df["user_claim_clean"] = df["user_claim"].str.strip()
    claims_input_df["user_claim_clean"] = claims_input_df["user_claim"].str.strip()

    df['user_claim_clean'] = pd.Categorical(df['user_claim_clean'], categories=claims_input_df['user_claim_clean'], ordered=True)
    df = df.sort_values('user_claim_clean').reset_index(drop=True)
    df = df.drop(columns=["user_claim_clean"])

    df.to_csv("dataset/output.csv", index=False)
    print("Validation complete: output schema verified.")


if __name__ == "__main__":
    main()