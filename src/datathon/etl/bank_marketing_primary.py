"""ETL for primary dataset: henriqueyamahata/bank-marketing -> data/processed/"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/kaggle/bank-marketing")
OUT_PATH = Path("data/processed")
LEAKAGE_COLS = ["duration"]


def _find_csv(directory: Path) -> Path:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {directory}. Run scripts/download_data.sh first.")
    # prefer the full dataset if multiple CSVs exist
    full = [f for f in csvs if "full" in f.name.lower()]
    return full[0] if full else csvs[0]


def load_raw() -> pd.DataFrame:
    path = _find_csv(RAW_PATH)
    return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # drop leakage columns
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    # normalise target to binary int
    df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)
    # strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def run():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df = clean(df)
    out = OUT_PATH / "bank_marketing_primary.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    run()
