"""Evaluate the production pipeline on the labeled sample claims.

This uses the SAME prompt, schema, image loading, model fallback and validator
as code/main.py + code/validator.py. It reports accuracy twice: on the raw model
output, and after the validator (which is what the production pipeline ships).
"""

import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai

# Make code/ importable (shared.py, validator.py) when running this file directly.
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from shared import build_prompt, load_images, review_claim  # noqa: E402
from validator import validate_dataframe  # noqa: E402

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

sample_df = pd.read_csv(os.path.join(DATASET_DIR, "sample_claims.csv"))
history_df = pd.read_csv(os.path.join(DATASET_DIR, "user_history.csv"))
requirements_df = pd.read_csv(os.path.join(DATASET_DIR, "evidence_requirements.csv"))

LABELS = {
    "claim_status": "claim_status",
    "severity": "severity",
    "issue_type": "issue_type",
}

print(f"Running evaluation on {len(sample_df)} sample claims...\n")

results = []
truth = []

for idx, claim in sample_df.iterrows():
    prompt = build_prompt(claim, history_df, requirements_df)
    contents = [prompt] + load_images(claim, DATASET_DIR)
    response_text = review_claim(client, contents)

    if response_text:
        try:
            result = json.loads(response_text)
            result["user_id"] = claim["user_id"]
            result["image_paths"] = claim["image_paths"]
            result["user_claim"] = claim["user_claim"]
            result["claim_object"] = claim["claim_object"]
            results.append(result)
            truth.append({name: claim[name] for name in LABELS})
            mark = "OK " if result["claim_status"] == claim["claim_status"] else "BAD"
            print(f"Row {idx + 1}: {mark} predicted={result['claim_status']} | actual={claim['claim_status']}")
        except Exception as e:
            print(f"Row {idx + 1}: could not parse response - {e}")
    else:
        print(f"Row {idx + 1}: no model response")

    time.sleep(5)

if not results:
    sys.exit("No predictions were produced, nothing to evaluate.")

raw_df = pd.DataFrame(results)
truth_df = pd.DataFrame(truth)
validated_df = validate_dataframe(raw_df)


def accuracy_report(title, pred_df):
    print(f"\n=== {title} ===")
    for name in LABELS:
        acc = (pred_df[name].astype(str).values == truth_df[name].astype(str).values).mean()
        print(f"{name:<13} accuracy: {acc:.0%}")


accuracy_report("RAW MODEL OUTPUT (before validator)", raw_df)
accuracy_report("AFTER VALIDATOR (what the pipeline ships)", validated_df)
print(f"\nEvaluated {len(validated_df)}/{len(sample_df)} claims")

out = validated_df.copy()
for name in LABELS:
    out[f"true_{name}"] = truth_df[name].values
out.to_csv(os.path.join(CODE_DIR, "evaluation", "sample_predictions.csv"), index=False)
print("Saved to code/evaluation/sample_predictions.csv")