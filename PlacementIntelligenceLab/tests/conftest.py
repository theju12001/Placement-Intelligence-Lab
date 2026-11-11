import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session", autouse=True)
def isolated_artifacts(tmp_path_factory):
    """Redirect every artifact path to a temp dir so tests never overwrite the
    real trained model, experiment log or monitoring history."""
    import config
    tmp = tmp_path_factory.mktemp("artifacts")
    names = {"BEST_MODEL_PATH": "best_model.joblib", "METRICS_PATH": "metrics.json", "METADATA_PATH": "metadata.json",
             "TEST_PRED_PATH": "test_predictions.csv", "TRAIN_REF_PATH": "train_reference.csv",
             "RUNS_FILE": "runs.jsonl", "MONITOR_HISTORY": "monitor.jsonl"}
    mp = pytest.MonkeyPatch()
    for attr, fname in names.items():
        mp.setattr(config, attr, tmp / fname)
    mp.setattr(config, "REPORTS_DIR", tmp)
    mp.setattr(config, "EXPORTS_DIR", tmp)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def trained():
    """Train once on the synthetic demo data (fast: core models only)."""
    from src.data import loader
    from src.models.train import train_all
    df, info = loader.load_dataset()
    out = train_all(df, info, only=["Logistic Regression", "Random Forest", "Extra Trees",
                                    "Gradient Boosting", "HistGradientBoosting"], verbose=False, track=False)
    return out


@pytest.fixture(scope="session")
def good_row():
    return {"cgpa": 8.1, "ssc_percentage": 78, "hsc_percentage": 75, "aptitude_score": 70, "projects": 3,
            "internships": 1, "certifications": 2, "backlogs": 0, "household_income_band": "Middle",
            "parental_education": "Graduate", "internet_access": "Yes"}
