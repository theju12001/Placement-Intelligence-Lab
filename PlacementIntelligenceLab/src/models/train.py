"""Model zoo, training, evaluation and persistence.

Leakage control: all preprocessing lives *inside* sklearn Pipelines, which are
fit on training folds only. The test split is untouched until final reporting.
Model selection uses cross-validated ROC-AUC on the training split only.
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

import config
from src.data import loader
from src.tracking import tracker


# ----------------------------------------------------------------------
def build_preprocessor(schema: dict, scale: bool = True) -> ColumnTransformer:
    num, cat = list(schema["numeric"]), list(schema["categorical"])
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)


def model_zoo(seed: int = config.RANDOM_STATE) -> dict:
    """Core five models always; optional libraries only when importable."""
    zoo = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1), False),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=seed, max_iter=200, learning_rate=0.06), False),
        "Support Vector Machine": (SVC(probability=True, random_state=seed), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed), False),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore
        zoo["XGBoost"] = (XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                        eval_metric="logloss", random_state=seed, n_jobs=1), False)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        zoo["LightGBM"] = (LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=seed, verbose=-1), False)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier  # type: ignore
        zoo["CatBoost"] = (CatBoostClassifier(iterations=250, depth=4, learning_rate=0.06,
                                              random_seed=seed, verbose=0), False)
    except Exception:
        pass
    return zoo


def make_pipeline(name: str, schema: dict, zoo: dict | None = None) -> Pipeline:
    zoo = zoo or model_zoo()
    est, scale = zoo[name]
    from sklearn.base import clone
    return Pipeline([("prep", build_preprocessor(schema, scale=scale)), ("model", clone(est))])


# ----------------------------------------------------------------------
def safe_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Metrics that never crash on single-class test sets."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba)
        out["pr_auc"] = average_precision_score(y_true, proba)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def curves(y_true, proba, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    out = {"confusion": confusion_matrix(y_true, (proba >= 0.5).astype(int), labels=[0, 1]).tolist()}
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba)
        p, r, _ = precision_recall_curve(y_true, proba)
        frac, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        step = max(1, len(fpr) // 120)
        out.update({"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(),
                    "precision": p[::max(1, len(p) // 120)].tolist(), "recall": r[::max(1, len(r) // 120)].tolist(),
                    "cal_frac": frac.tolist(), "cal_mean": mean_pred.tolist()})
    return out


# ----------------------------------------------------------------------
def train_all(df: pd.DataFrame | None = None, info: dict | None = None,
              only: list[str] | None = None, verbose: bool = True,
              track: bool = True) -> dict:
    """Train/compare all models, calibrate the best, persist artifacts."""
    if df is None:
        df, info = loader.load_dataset()
    info = info or {}
    schema = loader.infer_schema(df, info.get("target") or loader.detect_target(df))
    quality = loader.validate_dataset(df, schema["target"])
    if quality["single_class"]:
        raise ValueError("Target has a single class; cannot train a classifier.")
    X, y, audit, clean_info = loader.prepare_xy(df, schema)
    idx_tr, idx_te = train_test_split(X.index, test_size=config.TEST_SIZE,
                                      random_state=config.RANDOM_STATE, stratify=y)
    Xtr, Xte, ytr, yte = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    zoo = model_zoo()
    names = [n for n in zoo if (only is None or n in only)]
    dataset_meta = {"file": info.get("file"), "synthetic": bool(info.get("synthetic")),
                    "rows_used": clean_info["rows_used"], "target": schema["target"]}
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    results, fitted = {}, {}
    for name in names:
        t0 = time.perf_counter()
        pipe = make_pipeline(name, schema, zoo)
        try:
            cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
            pipe.fit(Xtr, ytr)
        except Exception as exc:  # keep going if an optional model fails
            if verbose:
                print(f"  ! {name} failed: {exc}")
            continue
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = pipe.predict_proba(Xte)[:, 1]
        infer_ms = (time.perf_counter() - t1) / max(1, len(Xte)) * 1000
        m = safe_metrics(yte, proba)
        m.update({"cv_roc_auc_mean": float(cv_auc.mean()), "cv_roc_auc_std": float(cv_auc.std()),
                  "train_seconds": train_s, "inference_ms_per_row": infer_ms})
        results[name] = {"metrics": m, "curves": curves(yte, proba)}
        fitted[name] = pipe
        if verbose:
            print(f"  {name:<24} cvAUC={m['cv_roc_auc_mean']:.3f}  testAUC={m['roc_auc']:.3f}  ({train_s:.1f}s)")
        if track:
            tracker.log_run(f"{name}-{version}", name, zoo[name][0].get_params(),
                            m, train_s, dataset_meta, version, tags={"stage": "comparison"})
    if not results:
        raise RuntimeError("No model could be trained.")
    best = max(results, key=lambda n: results[n]["metrics"]["cv_roc_auc_mean"])

    # calibrate the selected model (sigmoid; CV on the training split only)
    t0 = time.perf_counter()
    cal = CalibratedClassifierCV(make_pipeline(best, schema, zoo), method="sigmoid", cv=config.CV_FOLDS)
    cal.fit(Xtr, ytr)
    cal_seconds = time.perf_counter() - t0
    p_cal = cal.predict_proba(Xte)[:, 1]
    final_metrics = safe_metrics(yte, p_cal)
    final_metrics["train_seconds"] = cal_seconds
    final_curves = curves(yte, p_cal)
    if track:
        tracker.log_run(f"{best}-calibrated-{version}", best, {"calibration": "sigmoid", "cv": config.CV_FOLDS},
                        final_metrics, cal_seconds, dataset_meta, version, str(config.BEST_MODEL_PATH),
                        tags={"stage": "final", "selected": "true"})

    # persist
    bundle = {"calibrated": cal, "pipeline": fitted[best], "schema": schema, "best_name": best,
              "version": version, "features": schema["features"]}
    joblib.dump(bundle, config.BEST_MODEL_PATH)
    Xtr.assign(**{"__y__": ytr}).to_csv(config.TRAIN_REF_PATH, index=False)
    test_out = Xte.copy()
    test_out["y_true"] = yte
    test_out["proba_uncalibrated"] = fitted[best].predict_proba(Xte)[:, 1]
    test_out["proba"] = p_cal
    if not audit.empty:
        test_out = test_out.join(audit.loc[Xte.index])
    test_out.to_csv(config.TEST_PRED_PATH, index=False)

    ranking = sorted(results, key=lambda n: results[n]["metrics"]["cv_roc_auc_mean"], reverse=True)
    metrics_doc = {"models": results, "ranking": ranking, "selected": best,
                   "selection_metric": f"{config.CV_FOLDS}-fold CV ROC-AUC on training split",
                   "final_calibrated": {"metrics": final_metrics, "curves": final_curves},
                   "split": {"train": int(len(Xtr)), "test": int(len(Xte)),
                             "test_size": config.TEST_SIZE, "seed": config.RANDOM_STATE, "stratified": True},
                   "class_balance_train": {"placed_rate": float(ytr.mean())}}
    metadata = {"version": version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": dataset_meta, "schema": schema, "quality": quality, "cleaning": clean_info,
                "selected_model": best, "tracking_backend": tracker.active_backend(),
                "models_trained": list(results), "models_skipped_unavailable": [
                    n for n in ("XGBoost", "LightGBM", "CatBoost") if n not in zoo]}
    import json
    config.METRICS_PATH.write_text(json.dumps(metrics_doc, indent=1))
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=1))
    if verbose:
        print(f"Selected: {best} | calibrated test ROC-AUC={final_metrics['roc_auc']:.3f} "
              f"Brier={final_metrics['brier']:.3f} | saved {config.BEST_MODEL_PATH.name}")
    return {"metrics": metrics_doc, "metadata": metadata}
