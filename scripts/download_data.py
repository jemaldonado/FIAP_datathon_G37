#!/usr/bin/env python
"""
Download the Bank Marketing dataset from Kaggle and build the processed parquet.

Usage:
    python scripts/download_data.py           # skip the download if the CSV is already there
    python scripts/download_data.py --force   # re-download even when the CSV is present

Prerequisites:
    - Copy .env.example to .env and fill in KAGGLE_USERNAME and KAGGLE_KEY
      (both come from the kaggle.json that Kaggle > Settings > API > Create New Token gives you).
    - Dependencies installed: pip install -r requirements.txt

The credentials are checked before the kaggle package is imported on purpose: importing it
authenticates immediately and terminates the process on failure, which would make every error
message below unreachable.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """A Kaggle dataset and the ETL module that turns it into a parquet."""

    slug: str
    directory: str
    etl_module: str

    @property
    def raw_path(self) -> Path:
        return REPO_ROOT / "data" / "kaggle" / self.directory


PRIMARY = Dataset(
    slug='henriqueyamahata/bank-marketing',
    directory='bank-marketing',
    etl_module='datathon.etl.bank_marketing_primary',
)

CREDENTIALS_HELP = """
Kaggle credentials not found.

1. Sign in at https://www.kaggle.com and open Settings > API > Create New Token.
   The browser downloads a kaggle.json holding a "username" and a "key".
2. Copy .env.example to .env in the repository root.
3. Put those two values in .env:

       KAGGLE_USERNAME=your_kaggle_username
       KAGGLE_KEY=your_kaggle_api_key

4. Run this script again.

A KAGGLE_API_TOKEN access token in .env is accepted as an alternative to the pair above.
Never commit .env - it is listed in .gitignore.
"""


def has_credentials() -> bool:
    """True when the environment carries credentials the Kaggle client can use."""
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        return True
    return bool(os.environ.get('KAGGLE_API_TOKEN'))


def already_downloaded(dataset: Dataset) -> bool:
    """True when the target directory already holds a CSV, manually placed or not."""
    return any(dataset.raw_path.glob('*.csv'))


def build_api():
    """Import and authenticate the Kaggle client. Only call this after has_credentials()."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(api, dataset: Dataset, force: bool) -> bool:
    """Download one dataset. Returns False when it could not be made available."""
    if already_downloaded(dataset) and not force:
        logger.info(f"Skipping {dataset.slug} - CSV already in {dataset.raw_path.relative_to(REPO_ROOT)}")
        return True

    dataset.raw_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {dataset.slug} -> {dataset.raw_path.relative_to(REPO_ROOT)}")

    try:
        api.dataset_download_files(dataset.slug, path=str(dataset.raw_path), unzip=True)
    except Exception as exc:
        logger.error(f"Download of {dataset.slug} failed: {exc}")
        logger.error(
            f"Manual alternative: download from https://www.kaggle.com/datasets/{dataset.slug} "
            f"and extract the CSV into {dataset.raw_path.relative_to(REPO_ROOT)}"
        )
        return False

    if not already_downloaded(dataset):
        logger.error(f"{dataset.slug} downloaded but no CSV appeared in {dataset.raw_path}")
        return False

    logger.info(f"OK {dataset.slug}")
    return True


def run_etl(dataset: Dataset) -> bool:
    """Run the ETL module of one dataset. Returns False on failure."""
    import importlib

    try:
        module = importlib.import_module(dataset.etl_module)
        module.run()
        return True
    except Exception as exc:
        logger.error(f"ETL {dataset.etl_module} failed: {exc}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        '--force',
        action='store_true',
        help='re-download the dataset even when the CSV is already present',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Every ETL module resolves data/ relative to the working directory.
    os.chdir(REPO_ROOT)

    logger.info("=" * 70)
    logger.info("DATA DOWNLOAD AND SETUP")
    logger.info("=" * 70)

    if not has_credentials():
        logger.error(CREDENTIALS_HELP)
        sys.exit(1)

    api = build_api()

    if not (download_dataset(api, PRIMARY, args.force) and run_etl(PRIMARY)):
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("SETUP COMPLETE")
    logger.info("=" * 70)

    logger.info("Data ready at: data/processed/bank_marketing_primary.parquet")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. python scripts/train_simple.py")
    logger.info("2. python src/datathon/api/app.py")
    logger.info("3. Open http://localhost:5000/apidocs")


if __name__ == '__main__':
    main()
