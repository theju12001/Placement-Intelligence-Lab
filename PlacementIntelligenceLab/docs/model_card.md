# Model Card — Logistic Regression (calibrated)

**Project:** Statistical Modeling of Student Placement Outcomes Using Academic and Socioeconomic Factors
**Version:** v20261019-033516  ·  **Trained:** 2026-10-19T03:35:28+00:00

## Intended use
Educational research on academic and socioeconomic correlates of placement. Not for hiring, admission or other consequential individual decisions.

## Data
Source file: `synthetic_placement_demo.csv` (1500 rows). **SYNTHETIC demo data — not real students.**
Predictors: cgpa, ssc_percentage, hsc_percentage, aptitude_score, projects, internships, certifications, backlogs, household_income_band, parental_education, internet_access.
Audit-only (not predictors): gender.

## Performance (held-out test set)
ROC-AUC 0.808 · PR-AUC 0.808 · Accuracy 0.713 · F1 0.713 · Brier 0.179

## Explainability
SHAP (permutation explainer, mean |value|, on uncalibrated best pipeline). Top: internships, cgpa, backlogs, aptitude_score, projects.

## Limitations
- Single split, no external validation. Synthetic data.
- Probabilities are calibrated on training data only; they are not guarantees.
- Associations learned by the model are not causal.

## Ethical considerations
- Protected attributes excluded from predictors; audited only at subgroup level.
- Keep a human in the loop; do not use as the sole basis for any decision.