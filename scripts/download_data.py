#!/usr/bin/env python
"""
Automated data download from Kaggle.

Downloads Bank Marketing Dataset and processes it into parquet format.

Usage:
    python scripts/download_data.py

Prerequisites:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - kaggle Python package installed (pip install kaggle)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def download_from_kaggle():
    """Download dataset from Kaggle using Kaggle API"""
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False

    kaggle_dir = Path(__file__).parent.parent / "data" / "kaggle" / "bank-marketing"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Bank Marketing Dataset (primary: henriqueyamahata/bank-marketing) from Kaggle...")
    logger.info(f"Destination: {kaggle_dir}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            'henriqueyamahata/bank-marketing',
            path=str(kaggle_dir),
            unzip=True
        )

        logger.info("✓ Download successful")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("\nManual setup:")
        logger.error("1. Go to: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing")
        logger.error("2. Click 'Download'")
        logger.error(f"3. Extract to: {kaggle_dir}")
        logger.error("4. Run: python -m datathon.etl.bank_marketing_primary")
        return False


def process_data():
    """Process raw data into parquet format"""
    logger.info("\nProcessing data...")

    from datathon.etl.bank_marketing_primary import run

    try:
        run()
        logger.info("✓ Data processing complete")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def main():
    logger.info("=" * 70)
    logger.info("DATA DOWNLOAD AND SETUP")
    logger.info("=" * 70)

    # Step 1: Download
    if not download_from_kaggle():
        sys.exit(1)

    # Step 2: Process
    if not process_data():
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("SETUP COMPLETE")
    logger.info("=" * 70)
    logger.info("Data ready at: data/processed/bank_marketing_primary.parquet")
    logger.info("\nNext steps:")
    logger.info("1. python scripts/train_simple.py")
    logger.info("2. python src/datathon/api/app.py")
    logger.info("3. Open http://localhost:5000/apidocs")


if __name__ == '__main__':
    main()
