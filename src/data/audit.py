"""Dataset audit: shape, dtypes, missing, duplicates, target stats, leakage check."""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/insurance.csv")

def load_raw(path: Path = RAW) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path}")
    return pd.read_csv(path)

def audit(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "nunique": df.nunique().to_dict(),
        "target_stats": df["charges"].describe().round(2).to_dict(),
        "categoricals": {
            c: df[c].value_counts().to_dict()
            for c in ["sex", "smoker", "region"]
        },
    }

if __name__ == "__main__":
    import json
    print(json.dumps(audit(load_raw()), indent=2, default=str))
