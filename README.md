# Multi-Modal Damage Claim Verification System

An AI-powered multimodal system for verifying insurance damage claims using images, claim conversations, user history, and evidence requirements.

## Overview

This project automates the review of damage claims for:

- Cars
- Laptops
- Packages

The system analyzes uploaded images together with claim descriptions to determine whether the visual evidence supports, contradicts, or is insufficient to verify the claim.

## Features

- Image-based damage verification using Google's Gemini multimodal model
- Structured JSON output generation
- Automated schema validation and normalization
- Support for multiple uploaded images per claim
- Risk detection for:
  - Wrong object
  - Wrong object part
  - Prompt injection attempts
  - Non-original images
  - User history risk
  - Missing or insufficient evidence
- Evaluation pipeline for measuring prediction accuracy

## Tech Stack

- Python
- Gemini API
- Pandas
- Pillow
- CSV Processing

## Project Structure

```
code/
├── main.py
├── validator.py
└── evaluation/
    ├── main.py
    └── evaluation_report.md
```

## Workflow

1. Read claim data and image paths.
2. Load user history and evidence requirements.
3. Process one or more uploaded images.
4. Send multimodal prompt to Gemini.
5. Generate structured claim decisions.
6. Validate outputs against the required schema.
7. Export predictions.

## Validation

The validator performs:

- Schema enforcement
- Enum normalization
- Output consistency checks
- Supporting image ID validation
- Final CSV formatting

## Highlights

- Built an end-to-end multimodal AI pipeline
- Automated post-processing using rule-based validation
- Evaluated on labeled sample claims before inference
- Designed to handle multiple object categories and edge cases

## Disclaimer

This repository contains only the implementation code. Competition datasets, images, and proprietary evaluation files are not included.