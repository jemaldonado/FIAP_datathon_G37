"""ETL for comparison dataset: dharmik34/bank-term-deposit-subscription -> data/processed/"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/kaggle/bank-term-deposit-dharmik34")
OUT_PATH = Path("data/processed")
LEAKAGE_COLS = ["duration"]


def _find_csv(directory: Path) -> Path:
    # bank-full.csv is the complete real dataset (45,211 rows); bank.csv is a
    # 10% subsample of it. Prefer the full file when both are present.
    preferred = directory / "bank-full.csv"
    if preferred.exists():
        return preferred
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {directory}. Run scripts/download_data.sh first.")
    return csvs[0]


def load_raw() -> pd.DataFrame:
    path = _find_csv(RAW_PATH)
    # This file is ";"-quoted. pd.read_csv(sep=",") does NOT raise on a
    # semicolon-delimited file — it silently parses everything into a single
    # column, so a try/except around the separator can't detect the mismatch.
    # Detect it by checking the resulting column count instead.
    df = pd.read_csv(path, sep=",")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    # handle various target column names
    target_col = next((c for c in df.columns if c.lower() in ("y", "subscribed", "target")), None)
    if target_col and df[target_col].dtype == object:
        df[target_col] = (df[target_col].str.strip().str.lower() == "yes").astype(int)
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def run():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df = clean(df)
    out = OUT_PATH / "bank_term_deposit.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    run()
