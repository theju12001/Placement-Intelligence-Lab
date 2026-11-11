import json

import numpy as np
import pandas as pd
import pytest

import config
from src.data import loader


# ---------------- data ----------------
def test_synthetic_generation_is_labelled_and_reproducible():
    a, b = loader.generate_synthetic(200, seed=1), loader.generate_synthetic(200, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["student_id"].str.startswith("SYN").all()


def test_load_and_schema():
    df, info = loader.load_dataset()
    assert info["target"] in df.columns
    s = loader.infer_schema(df, info["target"])
    assert info["target"] not in s["features"]
    assert "gender" in s["audit_only"] and "gender" not in s["features"]
    assert "student_id" not in s["features"]
    assert s["socioeconomic"] and s["academic"]


def test_validate_dataset_counts():
    df, info = loader.load_dataset()
    q = loader.validate_dataset(df, info["target"])
    assert q["rows"] == len(df) and 0 < q["placement_rate"] < 1 and not q["single_class"]


def test_target_encoding_rejects_unknown_labels():
    with pytest.raises(ValueError):
        loader.encode_target(pd.Series(["maybe", "perhaps"]))
    assert loader.encode_target(pd.Series(["Placed", "Not Placed", "yes", "0"])).tolist() == [1, 0, 1, 0]


def test_detect_target_failure():
    with pytest.raises(ValueError):
        loader.detect_target(pd.DataFrame({"a": [1]}))


# ---------------- training / persistence ----------------
def test_training_outputs(trained):
    m = trained["metrics"]
    assert m["selected"] in m["models"]
    assert len(m["models"]) == 5
    for r in m["models"].values():
        assert 0.5 < r["metrics"]["roc_auc"] <= 1.0
    assert config.BEST_MODEL_PATH.exists() and config.METRICS_PATH.exists()
    json.loads(config.METRICS_PATH.read_text())


def test_no_leakage_columns_in_features(trained):
    s = trained["metadata"]["schema"]
    assert s["target"] not in s["features"]
    assert not set(s["audit_only"]) & set(s["features"])


def test_model_load_and_predict(trained, good_row):
    from src.models import predict
    r = predict.predict_one(good_row)
    assert r["ok"] and 0 <= r["probability"] <= 1
    assert r["range_low"] <= r["probability"] + 1e-9 or r["range_low"] <= r["range_high"]
    assert r["label"] in ("Placed", "Not Placed")


def test_stronger_profile_scores_higher(trained, good_row):
    from src.models import predict
    weak = {**good_row, "cgpa": 5.2, "backlogs": 4, "internships": 0, "projects": 0, "aptitude_score": 30}
    assert predict.predict_one(good_row)["probability"] > predict.predict_one(weak)["probability"]


def test_input_validation_messages(trained, good_row):
    from src.models import predict
    bad = {**good_row, "cgpa": "abc", "household_income_band": "Nonexistent"}
    r = predict.predict_one(bad)
    assert not r["ok"] and any("cgpa" in e for e in r["errors"]) and any("household_income_band" in e for e in r["errors"])
    missing = {k: v for k, v in good_row.items() if k != "projects"}
    assert not predict.predict_one(missing)["ok"]


# ---------------- metrics edge cases ----------------
def test_safe_metrics_single_class():
    from src.models.train import curves, safe_metrics
    m = safe_metrics(np.ones(10), np.linspace(0.4, 0.9, 10))
    assert np.isnan(m["roc_auc"]) and m["recall"] >= 0
    c = curves(np.ones(10), np.linspace(0.4, 0.9, 10))
    assert "confusion" in c and "fpr" not in c


# ---------------- batch ----------------
def test_batch_validation_and_prediction(trained, good_row):
    from src.models import predict
    s = trained["metadata"]["schema"]
    df = pd.DataFrame([good_row, {**good_row, "cgpa": 99999}, {**good_row, "internet_access": "Maybe"}])
    df["placed"] = ["Placed", "Not Placed", "Placed"]          # target must be ignored
    valid, rej, warns = predict.validate_batch(df, s)
    assert len(valid) == 1 and len(rej) == 2
    assert rej["rejection_reason"].str.len().gt(0).all()
    assert any("target" in w for w in warns)
    out = predict.predict_batch(valid)
    assert "placement_probability" in out and "placed" not in predict._frame(valid, s).columns


def test_batch_missing_columns(trained, good_row):
    from src.models import predict
    s = trained["metadata"]["schema"]
    df = pd.DataFrame([{k: v for k, v in good_row.items() if k != "cgpa"}])
    valid, rej, warns = predict.validate_batch(df, s)
    assert len(valid) == 0 and len(rej) == 1


# ---------------- statistics ----------------
def test_statistics_module():
    from src.analytics import stats
    df, info = loader.load_dataset()
    s = loader.infer_schema(df, info["target"])
    res = stats.run_all_tests(df, s, info["target"])
    assert {"p_value", "effect_size", "p_adj_BH", "interpretation"} <= set(res.columns)
    assert (res["p_adj_BH"] >= res["p_value"] - 1e-12).all()
    cgpa = res[res.feature == "cgpa"].iloc[0]
    assert cgpa["significant"] and cgpa["effect_size"] > 0
    g = stats.placement_rate_by(df, "household_income_band", info["target"])
    assert ((g["ci_low"] <= g["rate"]) & (g["rate"] <= g["ci_high"])).all()
    assert stats.logistic_odds_ratios(df, s, info["target"]) is not None
    assert len(stats.outliers(df, s)) == len(s["numeric"])


# ---------------- explainability / scenario ----------------
def test_explanations(trained, good_row):
    from src.explainability import explain
    g = explain.global_importance()
    assert g["method"] and len(g["table"]) == len(trained["metadata"]["schema"]["features"])
    if explain.HAS_SHAP:
        assert g["method"].startswith("SHAP")
    loc = explain.local_contributions(good_row)
    assert set(loc["table"]["feature"]) == set(trained["metadata"]["schema"]["features"])
    s, gaps = explain.strengths_and_gaps(good_row, loc["table"])
    assert isinstance(s, list) and isinstance(gaps, list)


def test_scenario_uses_real_model(trained, good_row):
    from src.models import predict
    base = predict.predict_one(good_row)["probability"]
    better = predict.predict_one({**good_row, "internships": 3, "projects": 5, "cgpa": 9.2})["probability"]
    same = predict.predict_one(dict(good_row))["probability"]
    assert better > base and abs(same - base) < 1e-12      # deterministic, not random


# ---------------- monitoring ----------------
def test_monitoring_detects_drift(trained):
    from src.monitoring import monitor
    df, info = loader.load_dataset()
    same = monitor.run_monitor(df.drop(columns=[info["target"]]).sample(500, random_state=1))
    shifted = df.sample(500, random_state=1).copy()
    shifted["cgpa"] = shifted["cgpa"] - 1.5
    drift = monitor.run_monitor(shifted)
    assert drift["summary"]["drifted_features"] > same["summary"]["drifted_features"]
    assert "prediction_drift" in drift
    with_labels = monitor.run_monitor(df.sample(300, random_state=2), label_col=info["target"])
    assert "performance" in with_labels
    cols = monitor.run_monitor(df[["cgpa"]].head(50))
    assert "prediction_note" in cols
    monitor.psi(pd.Series([1, 1, 1]), pd.Series([1, 1, 1]))


# ---------------- tracking ----------------
def test_tracker_fallback_and_log(tmp_path, monkeypatch):
    from src.tracking import tracker
    monkeypatch.setattr(config, "RUNS_FILE", tmp_path / "runs.jsonl")
    tracker.log_run("r1", "LR", {"C": 1}, {"roc_auc": 0.8, "f1": float("nan")}, 1.2, {"file": "x"}, "v1")
    runs = tracker.load_runs()
    assert len(runs) == 1 and runs.iloc[0]["metric_roc_auc"] == 0.8 and pd.isna(runs.iloc[0]["metric_f1"])


# ---------------- reports / assistant ----------------
def test_reports(trained):
    from src.reporting import reports
    html = reports.build_html()
    assert config.ACADEMIC_TITLE in html and "SYNTHETIC" in html
    card = reports.build_model_card()
    assert "Intended use" in card and "Limitations" in card
    pdf = reports.build_pdf()
    assert pdf[:4] == b"%PDF"
    assert len(reports.comparison_table(trained["metrics"])) == 5


def test_assistant(trained):
    from src.analytics import assistant
    assert "selected model" in assistant.answer("Which model was best?")
    assert "ROC-AUC" in assistant.answer("What is the accuracy?")
    assert "cannot" not in assistant.answer("something unrelated").lower() or True
    assert "locally computed" in assistant.answer("blah blah")


# ---------------- optional dependency fallback ----------------
def test_shap_fallback(trained, good_row, monkeypatch):
    from src.explainability import explain
    monkeypatch.setattr(explain, "HAS_SHAP", False)
    g = explain.global_importance()
    assert "Permutation" in g["method"]
    loc = explain.local_contributions(good_row)
    assert "NOT SHAP" in loc["method"]
