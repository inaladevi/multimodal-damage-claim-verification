import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

from shared import build_prompt, load_images, review_claim

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

claims_df = pd.read_csv("dataset/claims.csv")
history_df = pd.read_csv("dataset/user_history.csv")
requirements_df = pd.read_csv("dataset/evidence_requirements.csv")

results_master = []

print(f"Initializing pipeline across {len(claims_df)} rows...\n")
start_time = time.time()

for idx, claim in claims_df.iterrows():
    print(f"PROCESSING ROW {idx + 1} of {len(claims_df)} (User ID: {claim['user_id']})")

    prompt = build_prompt(claim, history_df, requirements_df)
    contents = [prompt] + load_images(claim, "dataset")
    response_text = review_claim(client, contents)

    if response_text:
        try:
            result = json.loads(response_text)

            result["user_id"] = claim["user_id"]
            result["image_paths"] = claim["image_paths"]
            result["user_claim"] = claim["user_claim"]
            result["claim_object"] = claim["claim_object"]

            results_master.append(result)
            print("  Row response compiled into raw matrix payload.")
        except Exception as json_err:
            print(f"  JSON structural payload mismatch: {json_err}")
    else:
        print(f"  Skipping Row {idx + 1}: Model pipeline unavailable.")

    time.sleep(5)

if results_master:
    raw_output_df = pd.DataFrame(results_master)
    raw_output_df.to_csv("dataset/output.csv", index=False)
    print("\nStage 1 complete! Raw output generated at dataset/output.csv.")
else:
    print("\nPipeline Error: Intermediate results buffer empty.")

print(f"\nRuntime: {(time.time() - start_time) / 60:.2f} minutes")