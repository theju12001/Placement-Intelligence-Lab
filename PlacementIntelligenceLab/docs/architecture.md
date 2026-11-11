# Architecture

```
app.py ── sidebar nav + top bar ──► views/<page>.py ──► src/*
                                                         │
 data/raw/*.csv ─┐                                       ├─ data/loader.py       load, target detection, schema inference, validation
 (else synthetic)├─► loader ─► train.py ─► models/*      ├─ models/train.py      zoo, CV selection, calibration, persistence
                 │            │                          ├─ models/predict.py    validation, single/batch inference, fold spread
                 │            └─► tracker (JSONL[+MLflow])├─ explainability/      SHAP (+ permutation / occlusion fallbacks)
                 │                                        ├─ analytics/stats.py   tests, effect sizes, odds ratios
 uploaded CSV ──► monitoring/monitor.py (PSI/KS/χ²)       ├─ analytics/viz3d.py   Plotly dark-neon figures
                                                          ├─ reporting/reports.py HTML / PDF / model card / CSV
                                                          └─ analytics/assistant.py deterministic Q&A
```

* **Schema-driven UI.** Input forms, batch validation, statistics and monitoring read the schema inferred from the actual dataset (`infer_schema`); no field is hard-coded or invented.
* **Single source of truth for artifacts:** `models/trained/best_model.joblib` (calibrated model, uncalibrated pipeline, schema, version), `models/metrics/metrics.json`, `models/metadata/metadata.json`, `data/processed/{train_reference,test_predictions}.csv`.
* **Graceful degradation:** SHAP → permutation importance → occlusion; MLflow → JSONL; reportlab missing → PDF button shows a message.
* **Paths** are centralised in `config.py`; tests redirect them to a temp directory.
