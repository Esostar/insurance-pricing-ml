"""Feature engineering for insurance pricing.

Input-only transforms (no target leakage) — safe to run before train/test split.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/insurance.csv")
PROCESSED = Path("data/processed/insurance_clean.csv")

TARGET = "charges"
TARGET_LOG = "log_charges"


def load_clean(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_category"] = df["bmi"].apply(_bmi_category)
    df["age_bin"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-64"],
    )
    df["is_smoker"] = (df["smoker"] == "yes").astype(int)
    df["smoker_bmi"] = df["is_smoker"] * df["bmi"]
    df["smoker_age"] = df["is_smoker"] * df["age"]
    df[TARGET_LOG] = np.log1p(df[TARGET])
    return df


if __name__ == "__main__":
    df = add_features(load_clean())
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    new_cols = [c for c in df.columns if c not in
                ["age","sex","bmi","children","smoker","region","charges"]]
    print(df.head(3).to_string())
    print(f"\nRows: {len(df)}")
    print(f"New features: {new_cols}")
    print(f"Saved -> {PROCESSED}")
