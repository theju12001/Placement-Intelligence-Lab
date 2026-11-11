"""Train, compare and persist placement models.

Usage:
    python train_model.py                 # uses data/raw/*.csv, else synthetic demo data
    python train_model.py --data my.csv   # explicit CSV
    python train_model.py --fresh         # clear previous experiment log first
"""
import argparse

from src.data import loader
from src.tracking import tracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to a CSV with a placement target column")
    ap.add_argument("--fresh", action="store_true", help="clear the experiment log before training")
    args = ap.parse_args()
    if args.fresh:
        tracker.reset_runs()
    df, info = loader.load_dataset(args.data)
    tag = "SYNTHETIC DEMO DATA" if info["synthetic"] else "real data"
    print(f"Dataset: {info['file']} ({info['n_rows']} rows, {tag}); target = {info['target']}")
    from src.models.train import train_all
    train_all(df, info)
    print("Done. Start the app with:  python -m streamlit run app.py")


if __name__ == "__main__":
    main()
