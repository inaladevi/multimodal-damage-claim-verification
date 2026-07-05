# Multi-Modal Evidence Review: Operational Analysis Report

## Pipeline Execution

The solution is executed in two stages:

```bash
python code/main.py
python code/validator.py
```

* `main.py` performs multimodal claim evaluation using Gemini Flash Lite.
* `validator.py` performs schema normalization, enum enforcement, row alignment, and final output validation.

---

# 1. Solution Overview

This solution evaluates damage claims using four information sources:

* User claim conversation
* Submitted images
* User claim history
* Evidence requirements

The objective is to determine whether the submitted visual evidence:

* Supports the claim
* Contradicts the claim
* Does not provide enough information

Images are treated as the primary source of truth. User history contributes risk context but never overrides visible evidence.

---

# 2. Data Sources Used

The solution consumes all provided datasets.

## claims.csv

Provides:

* user_id
* image_paths
* user_claim
* claim_object

## user_history.csv

Provides:

* past claim counts
* claim outcomes
* historical risk indicators

User history is incorporated into risk assessment and prompt context.

## evidence_requirements.csv

Provides minimum visual evidence requirements for different object and damage categories.

Relevant evidence requirements are retrieved for each claim and injected into the multimodal prompt to guide evidence evaluation.

---

# 3. System Architecture

## Step 1 – Load Claim Context

For each claim:

* Claim text is loaded
* Object category is identified
* User history is retrieved
* Relevant evidence requirements are retrieved

## Step 2 – Image Processing

Images are loaded locally.

To reduce request size and improve throughput:

* Images larger than 1024 pixels are resized
* LANCZOS interpolation is used for quality-preserving downscaling
* Multiple images are grouped into a single multimodal request

## Step 3 – Multimodal Evidence Review

Gemini receives:

* User claim text
* Claim object
* User history context
* Evidence requirements
* Submitted images

The model returns structured JSON containing:

* evidence_standard_met
* risk_flags
* issue_type
* object_part
* claim_status
* severity
* supporting_image_ids

## Step 4 – Schema Validation

A dedicated validator layer normalizes and validates outputs.

Validation includes:

* Severity normalization
* Risk flag normalization
* Object part normalization
* Issue type normalization
* Supporting image ID cleanup
* Column ordering enforcement
* Row alignment with claims.csv

## Step 5 – Output Generation

Validated predictions are written to:

`dataset/output.csv`

One output row is generated for every claim row.

---

# 4. Processing Metrics

## Test Dataset Statistics

| Metric                | Value |
| --------------------- | ----- |
| Claims Processed      | 44    |
| Images Processed      | 82    |
| Output Rows Generated | 44    |
| Missing Rows          | 0     |
| Null Values           | 0     |

## Runtime

| Metric           | Value        |
| ---------------- | ------------ |
| Measured Runtime | 6.10 minutes |

---

# 5. Model Usage

## Models Configured

Primary model:

* gemini-3.1-flash-lite

Fallback models:

* gemini-2.5-flash
* gemini-2.0-flash

## Request Statistics

| Metric                | Value |
| --------------------- | ----- |
| Test Processing Calls | 44    |
| Evaluation Calls      | 20    |
| Total Calls           | 64    |

## Fallback Performance

| Metric                  | Value |
| ----------------------- | ----- |
| Primary Model Successes | 44    |
| Fallback Activations    | 0     |
| Primary Success Rate    | 100%  |

Although fallback routing was implemented, all claims were successfully processed using the primary model.

---

# 6. Evaluation Methodology

The solution was evaluated using:

`dataset/sample_claims.csv`

Ground-truth labels were available for:

* claim_status
* issue_type
* severity

Three metrics were measured:

1. Claim Status Accuracy
2. Severity Accuracy
3. Issue Type Accuracy

## Evaluation Results

| Metric                | Result      |
| --------------------- | ----------- |
| Claim Status Accuracy | 85% (17/20) |
| Severity Accuracy     | 35%         |
| Issue Type Accuracy   | 40%         |

Claim Status Accuracy was treated as the primary metric because it directly determines the final claim decision.

---

# 7. Validator Impact

Large language models occasionally generate values outside the allowed competition schema.

The validator layer was introduced to guarantee compliance.

## Validation Improvements

| Validation Metric          | Before Validation | After Validation |
| -------------------------- | ----------------- | ---------------- |
| Invalid object_part values | 26                | 0                |
| Invalid issue_type values  | 30                | 0                |
| Missing values             | 0                 | 0                |
| Output rows                | 44                | 44               |

The validator reduced invalid object_part values from 26 to 0 and invalid issue_type values from 30 to 0 while preserving row count and schema structure.

This validation layer ensured that the final output.csv fully complied with the competition schema requirements.
---

# 8. Risk Detection Coverage

The following risk categories were detected in the final validated output:

* blurry_image
* claim_mismatch
* damage_not_visible
* manual_review_required
* non_original_image
* possible_manipulation
* text_instruction_present
* user_history_risk
* wrong_object
* wrong_object_part

Total unique risk categories detected:

**10**

---

# 9. Output Distribution

## Claim Status Distribution

| Decision               | Count |
| ---------------------- | ----- |
| Contradicted           | 26    |
| Supported              | 15    |
| Not Enough Information | 3     |

The distribution indicates the model is making differentiated decisions rather than defaulting to a single class.

---

# 10. Edge Cases Addressed

The system explicitly handles:

* Prompt injection attempts
* Wrong-object submissions
* Wrong-object-part submissions
* Blurry images
* Missing evidence scenarios
* Non-original image indicators
* High-risk user histories
* Multi-image evidence aggregation
* Contradictory evidence
* Insufficient visual evidence

When evidence is insufficient, the system returns:

`not_enough_information`

instead of forcing a supported or contradicted decision.

---

# 11. Design Tradeoffs

### Strategy A: Direct Multimodal Ingestion (Baseline)
The model was prompted to directly generate output fields from images and claim text without using strict API validation configurations.
* **Constraints**: This method led to structural failures, introducing 30 invalid issue types and 26 invalid object parts that broke closed vocabulary limits.

### Strategy B: Native Structured Schema + Validator Layer (Selected)
The final system used structured Gemini JSON outputs followed by deterministic schema validation.
* **Advantages**: This strategy successfully corrected all LLM variations down to 0 invalid entries, forced clean semicolon delimiters (`;`), and automatically re-aligned out-of-order records to guarantee absolute readiness.

---

# 12. Limitations

Known limitations include:

* Severity estimation remains subjective and achieved lower accuracy than claim-status prediction.
* Some issue categories overlap visually and can lead to semantic classification disagreement.
* Damage that is not externally visible may be difficult to verify from images alone.
* Token usage and cost figures were not directly measured and therefore were not reported.

---

## Operational Cost Estimates

Approximate model calls:

* **Test claims:** 44
* **Evaluation claims:** 20
* **Total calls:** 64

Images processed:

* **Test dataset:** 82 images

*Note: Exact token usage was not logged.*

Because Gemini Flash Lite was used and each request contained a small prompt and 1–3 resized images, overall cost is expected to remain low and well within typical free-tier or low-cost usage limits.

Runtime for the full test dataset was measured at **6.10 minutes**.

