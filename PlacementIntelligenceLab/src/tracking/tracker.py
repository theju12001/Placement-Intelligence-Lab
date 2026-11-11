"""Experiment tracking.

Active backend:
  * MLflow (local ``mlruns/`` file store) when the ``mlflow`` package is importable
  * otherwise a local JSON-lines fallback (``models/metadata/experiment_runs.jsonl``)

The JSONL log is ALWAYS written (it is what the Experiments page reads), and
MLflow is mirrored on top when available.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

import config

try:  # optional dependency
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:  # pragma: no cover
    mlflow = None
    HAS_MLFLOW = False


def active_backend() -> str:
    return "MLflow (local file store) + JSONL log" if HAS_MLFLOW else "Local JSONL fallback (MLflow not installed)"


def log_run(run_name: str, model_name: str, params: dict, metrics: dict,
            train_seconds: float, dataset: dict, model_version: str,
            artifact_path: str | None = None, tags: dict | None = None) -> dict:
    rec = {
        "run_id": uuid.uuid4().hex[:12],
        "run_name": run_name,
        "model": model_name,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(float(train_seconds), 4),
        "params": {k: str(v) for k, v in params.items()},
        "metrics": {k: (None if v is None or v != v else round(float(v), 6)) for k, v in metrics.items()},
        "dataset": dataset,
        "artifact": artifact_path,
        "tags": tags or {},
    }
    config.RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if HAS_MLFLOW:  # pragma: no cover - exercised only when mlflow is installed
        try:
            mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
            mlflow.set_experiment("placement-intelligence-lab")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({k: v[:250] for k, v in rec["params"].items()})
                mlflow.log_metrics({k: v for k, v in rec["metrics"].items() if v is not None})
                mlflow.log_metric("train_seconds", rec["train_seconds"])
                mlflow.set_tags({"model": model_name, "model_version": model_version})
        except Exception:
            pass
    return rec


def load_runs() -> pd.DataFrame:
    if not config.RUNS_FILE.exists():
        return pd.DataFrame()
    rows = []
    for line in config.RUNS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            flat = {k: r[k] for k in ("run_id", "run_name", "model", "model_version",
                                      "timestamp", "train_seconds", "artifact")}
            flat.update({f"metric_{k}": v for k, v in r["metrics"].items()})
            flat["params"] = json.dumps(r["params"])
            flat["dataset"] = json.dumps(r["dataset"])
            rows.append(flat)
    return pd.DataFrame(rows)


def reset_runs() -> None:
    if config.RUNS_FILE.exists():
        config.RUNS_FILE.unlink()
