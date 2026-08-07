"""
Data Quality Module

Implements comprehensive data quality checks with logging.
Validates data ranges, null values, distributions, and data drift.

Usage:
    from datathon.quality import validate_data_quality, get_quality_report

    df = pd.read_parquet('data.parquet')
    report = validate_data_quality(df, dataset_name='bank_marketing')
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Single quality metric result"""
    name: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    value: any
    threshold: any
    message: str


@dataclass
class QualityReport:
    """Complete data quality report"""
    dataset_name: str
    total_records: int
    null_values: int
    metrics: List[QualityMetric]
    overall_status: str
    timestamp: str

    def to_dict(self):
        """Convert to dictionary for JSON/dashboard"""
        return {
            'dataset_name': self.dataset_name,
            'total_records': self.total_records,
            'null_values': self.null_values,
            'overall_status': self.overall_status,
            'metrics': [
                {
                    'name': m.name,
                    'status': m.status,
                    'value': float(m.value) if isinstance(m.value, (np.number, float)) else m.value,
                    'message': m.message
                }
                for m in self.metrics
            ],
            'timestamp': self.timestamp
        }


def validate_data_quality(
    df: pd.DataFrame,
    dataset_name: str = 'unknown',
    check_age_range: Tuple[int, int] = (18, 150),
    check_conversions: bool = True,
    check_duplicates: bool = True,
    verbose: bool = True
) -> QualityReport:
    """
    Comprehensive data quality validation.

    Args:
        df: Input dataframe
        dataset_name: Name of dataset for logging
        check_age_range: Valid age range (min, max)
        check_conversions: Whether to check target distribution
        check_duplicates: Whether to check for duplicates
        verbose: Whether to log all checks

    Returns:
        QualityReport with validation results
    """
    from datetime import datetime

    logger.info(f"Starting data quality validation for '{dataset_name}'")
    logger.info(f"Dataset shape: {df.shape}")

    metrics = []

    # 1. Check for nulls
    logger.info("Checking for null values...")
    null_count = df.isnull().sum().sum()
    null_metric = QualityMetric(
        name='Null Values',
        status='PASS' if null_count == 0 else 'FAIL',
        value=null_count,
        threshold=0,
        message=f'Found {null_count} null values (threshold: 0)'
    )
    metrics.append(null_metric)

    # 2. Check record count
    logger.info(f"Checking record count: {len(df)} records")
    if len(df) == 0:
        logger.error("CRITICAL: Dataset is empty!")
        return QualityReport(
            dataset_name=dataset_name,
            total_records=0,
            null_values=0,
            metrics=metrics,
            overall_status='FAIL',
            timestamp=datetime.now().isoformat()
        )

    record_metric = QualityMetric(
        name='Record Count',
        status='PASS',
        value=len(df),
        threshold='>0',
        message=f'{len(df)} records loaded successfully'
    )
    metrics.append(record_metric)

    # 3. Check age ranges
    if 'age' in df.columns:
        logger.info(f"Checking age range: {check_age_range}")
        invalid_ages = df[(df['age'] < check_age_range[0]) | (df['age'] > check_age_range[1])]
        age_metric = QualityMetric(
            name='Age Range',
            status='PASS' if len(invalid_ages) == 0 else 'WARN',
            value=len(invalid_ages),
            threshold=0,
            message=f'Found {len(invalid_ages)} out-of-range ages (range: {check_age_range})'
        )
        metrics.append(age_metric)
        if len(invalid_ages) > 0:
            logger.warning(f"Found {len(invalid_ages)} invalid ages outside {check_age_range}")

    # 4. Check target distribution
    if 'y' in df.columns and check_conversions:
        logger.info("Checking target distribution...")
        pos_count = (df['y'] == 1).sum()
        neg_count = (df['y'] == 0).sum()
        pos_pct = pos_count / len(df) * 100
        neg_pct = neg_count / len(df) * 100

        # Severe class imbalance (<1% or >99% positive) leaves too few
        # examples of the minority class to estimate a per-arm conversion
        # rate reliably — this metric used to always report PASS
        # regardless of value (Achado Baixo, auditoria de re-verificação
        # 2026-08-05), the same "check that can't fail" pattern as the
        # original golden-set gate bug.
        severe_imbalance = pos_pct < 1.0 or pos_pct > 99.0
        dist_metric = QualityMetric(
            name='Target Distribution',
            status='WARN' if severe_imbalance else 'PASS',
            value=pos_pct,
            threshold='1-99%',
            message=f'Positive: {pos_pct:.2f}% ({pos_count}), Negative: {neg_pct:.2f}% ({neg_count})'
        )
        metrics.append(dist_metric)
        logger.info(f"Target distribution - Positive: {pos_pct:.2f}%, Negative: {neg_pct:.2f}%")
        if severe_imbalance:
            logger.warning(f"Severe target imbalance: {pos_pct:.2f}% positive")

    # 5. Check duplicates
    if check_duplicates:
        logger.info("Checking for duplicate records...")
        dup_count = df.duplicated().sum()
        dup_metric = QualityMetric(
            name='Duplicates',
            status='PASS' if dup_count == 0 else 'WARN',
            value=dup_count,
            threshold=0,
            message=f'Found {dup_count} duplicate records'
        )
        metrics.append(dup_metric)
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate records")

    # 6. Check for expected columns
    expected_cols = ['age', 'job', 'y']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    col_metric = QualityMetric(
        name='Required Columns',
        status='FAIL' if missing_cols else 'PASS',
        value=len(df.columns),
        threshold=f'>= {len(expected_cols)}',
        message=f'Found {len(df.columns)} columns. Missing: {missing_cols if missing_cols else "None"}'
    )
    metrics.append(col_metric)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")

    # Determine overall status
    overall_status = 'PASS' if all(m.status == 'PASS' for m in metrics) else 'FAIL'
    if any(m.status == 'FAIL' for m in metrics):
        overall_status = 'FAIL'
    elif any(m.status == 'WARN' for m in metrics):
        overall_status = 'WARN'

    report = QualityReport(
        dataset_name=dataset_name,
        total_records=len(df),
        null_values=null_count,
        metrics=metrics,
        overall_status=overall_status,
        timestamp=datetime.now().isoformat()
    )

    logger.info(f"Data quality validation complete. Status: {overall_status}")

    if verbose:
        logger.info("Quality Report:")
        for metric in metrics:
            logger.info(f"  {metric.name}: {metric.status} - {metric.message}")

    return report


def get_quality_report(df: pd.DataFrame, dataset_name: str = 'unknown') -> Dict:
    """
    Get quality report as dictionary (for dashboard integration).

    Returns:
        Dict with quality metrics and status
    """
    report = validate_data_quality(df, dataset_name=dataset_name, verbose=False)
    return report.to_dict()


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
    """
    logger_root = logging.getLogger()
    logger_root.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger_root.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger_root.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
