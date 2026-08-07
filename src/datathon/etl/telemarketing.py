"""ETL for comparison dataset: aguado/telemarketing-jyb-dataset -> data/processed/"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/kaggle/telemarketing-aguado")
OUT_PATH = Path("data/processed")
LEAKAGE_COLS = ["duration"]


def _find_csv(directory: Path) -> Path:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {directory}. Run scripts/download_data.sh first.")
    return csvs[0]


def load_raw() -> pd.DataFrame:
    path = _find_csv(RAW_PATH)
    try:
        return pd.read_csv(path, sep=",")
    except Exception:
        return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    target_col = next((c for c in df.columns if c.lower() in ("y", "subscribed", "target", "response")), None)
    if target_col and df[target_col].dtype == object:
        df[target_col] = (df[target_col].str.strip().str.lower() == "yes").astype(int)
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def run():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df = clean(df)
    out = OUT_PATH / "telemarketing.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    run()
