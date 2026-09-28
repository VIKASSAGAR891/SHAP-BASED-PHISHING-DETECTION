# Implementation notes for Assessment 2 Weeks 3–4

## Scope locked for this prototype

The submitted proposal accepts URLs, email content or other relevant features at a broad requirements level. For the first working implementation, this repository uses a **URL-only, pre-fetch classifier** because the selected PhiUSIIL dataset contains a strong URL feature set and the dashboard is designed around direct URL submission.

## Why webpage fields are not used for the first model

PhiUSIIL also contains features derived from webpage source code. Those fields require a page to be fetched and inspected. Using them in training while accepting only a raw URL at inference time would create a training/inference mismatch. It would also introduce avoidable security risks because the system would be asked to visit arbitrary potentially malicious sites.

## What counts as real progress

After running training, keep evidence of:

1. Dataset dimensions and class distribution.
2. URL-only feature list.
3. Train/test split.
4. Feature-selection output.
5. XGBoost and CatBoost metrics.
6. Confusion matrix.
7. SHAP explanation example.
8. Running Flask dashboard.

Do not write invented performance values into the report. Use the values produced by `artifacts/metrics.json`.
