# Methodology

**Title:** Statistical Modeling of Student Placement Outcomes Using Academic and Socioeconomic Factors

## Data
Real CSVs in `data/raw/` take priority. Otherwise a synthetic demo set is generated (`generate_synthetic`, seed 42). Synthetic relationships, including socioeconomic effects, are developer assumptions and must not be reported as findings.

## Cleaning
Rows with a missing target and exact duplicates are removed at training time (counts stored in `metadata.json`). Missing feature values are imputed inside the pipeline (median / most frequent) so imputation statistics come from training folds only.

## Splitting and selection
Stratified 80/20 train/test split (seed 42). Each candidate is scored by 5-fold stratified CV ROC-AUC on the training split; the top mean CV ROC-AUC is selected. The test set is used only to report metrics. Candidate differences below ~0.01 AUC are within CV noise.

## Calibration
Sigmoid (Platt) calibration via 5-fold `CalibratedClassifierCV` on the training split. Test-set calibration curve and Brier score are reported. Fold-model spread is shown as an empirical range, not a confidence interval.

## Explainability
SHAP permutation explainer on the transformed feature matrix of the uncalibrated selected pipeline; one-hot columns are summed per original feature (SHAP is additive). Fallbacks are labelled. Importance ≠ causal effect.

## Statistics
Numeric vs outcome: Welch t-test (Cohen's d) or Mann–Whitney U (rank-biserial r) when small samples fail Shapiro–Wilk. Categorical vs outcome: chi-square (or Fisher exact for 2×2 with small expected counts) with Cramér's V. Benjamini–Hochberg FDR across all tests. Wilson 95% intervals for rates; groups with n < 30 are flagged. Multivariable logistic regression (statsmodels) with standardised numeric predictors gives adjusted odds ratios. None of this establishes causation.

## Monitoring
PSI (10 quantile bins / category shares), KS test (numeric), chi-square (categorical), missing-rate deltas, prediction-score drift, and AUC/accuracy when a label column exists. Thresholds are configurable. Applies to uploaded files only.

## Fairness
Protected attributes are never predictors. If present they support subgroup rate / TPR / FPR / AUC audits on the held-out set; small groups are flagged. Parity gaps are descriptive, not proof of fairness or bias.
