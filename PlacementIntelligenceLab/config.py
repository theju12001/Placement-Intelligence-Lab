"""Central configuration for Placement Intelligence Lab."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

APP_NAME = "Placement Intelligence Lab"
TAGLINE = "Predict • Analyze • Empower"
ACADEMIC_TITLE = (
    "Statistical Modeling of Student Placement Outcomes Using "
    "Academic and Socioeconomic Factors"
)
EXECUTORS = ["Theju Gangadhar", "Karthik TG"]
APP_VERSION = "1.0.0"

# --- paths -------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "trained"
METRICS_DIR = ROOT / "models" / "metrics"
METADATA_DIR = ROOT / "models" / "metadata"
OUTPUTS = ROOT / "outputs"
REPORTS_DIR = OUTPUTS / "reports"
EXPORTS_DIR = OUTPUTS / "exports"
ASSETS = ROOT / "assets"
RUNS_FILE = METADATA_DIR / "experiment_runs.jsonl"
MONITOR_HISTORY = OUTPUTS / "monitoring_history.jsonl"
SYNTHETIC_CSV = DATA_SAMPLE / "synthetic_placement_demo.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = METRICS_DIR / "metrics.json"
METADATA_PATH = METADATA_DIR / "metadata.json"
TEST_PRED_PATH = DATA_PROCESSED / "test_predictions.csv"
TRAIN_REF_PATH = DATA_PROCESSED / "train_reference.csv"

# --- modelling ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"

TARGET_CANDIDATES = [
    "placed", "placement", "placement_status", "status", "is_placed",
    "placement_outcome", "hired",
]
POSITIVE_LABELS = {"placed", "yes", "1", "true", "hired", "selected", "y"}

# Columns that are never used as predictors; they are kept for fairness
# auditing only (see Responsible AI page).
SENSITIVE_COLUMNS = [
    "gender", "sex", "caste", "religion", "disability", "category",
    "community", "ethnicity", "race", "marital_status",
]
ID_LIKE = ["id", "student_id", "usn", "roll_no", "rollno", "sl_no", "name"]

# Known socioeconomic columns (used only to tag features; never invented)
SOCIOECONOMIC_HINTS = [
    "income", "parental", "parent", "father", "mother", "internet",
    "resource", "scholarship", "hostel", "residence", "locality",
    "first_gen", "family",
]

NAV_ITEMS = [
    ("Overview", "overview", "◈"),
    ("Predict", "predict", "◎"),
    ("Batch Analysis", "batch_analysis", "▦"),
    ("Scenario Simulator", "scenario_simulator", "⟁"),
    ("Model Lab", "model_lab", "⚗"),
    ("Experiments (MLflow)", "experiments", "⌬"),
    ("Model Monitoring", "monitoring", "◉"),
    ("Statistical Evidence", "statistical_evidence", "∑"),
    ("Research Reports", "research_reports", "▤"),
    ("Responsible AI", "responsible_ai", "⚖"),
    ("Research Assistant", "research_assistant", "✦"),
    ("Settings", "settings", "⚙"),
]

# Plot palette (dark neon)
COLORS = {
    "bg": "#050816", "panel": "rgba(16,20,48,0.72)", "cyan": "#22d3ee",
    "blue": "#3b82f6", "purple": "#a855f7", "magenta": "#ec4899",
    "green": "#34d399", "amber": "#fbbf24", "text": "#e5e7ff",
    "muted": "#8b93c7",
}

for _p in (DATA_RAW, DATA_SAMPLE, DATA_PROCESSED, MODELS_DIR, METRICS_DIR,
           METADATA_DIR, REPORTS_DIR, EXPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
