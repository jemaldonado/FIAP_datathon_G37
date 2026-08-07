#!/usr/bin/env python
"""
Model retraining with versioning and MLflow tracking.

Trains a new Thompson Sampling model on updated data,
versions it with timestamp, and logs metrics.

Usage:
    python scripts/retrain_model.py [--sample-ratio 0.8] [--env local|aws|azure]
"""

import sys
import argparse
import shutil
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling, compute_baseline_vs_thompson, assign_arm
from datathon.config import Config
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manages model versioning and metadata"""

    def __init__(self, config: Config):
        self.config = config
        self.model_dir = config.model_path
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_version_suffix(self) -> str:
        """Generate version suffix: v{timestamp}"""
        return datetime.now().strftime("v%Y%m%d_%H%M%S")

    def save_versioned_model(self, model: ContextualThompsonSampling,
                            version: str) -> Path:
        """Save model with version suffix"""
        model_data = {}
        for context, bandit in model.bandits.items():
            age_group, job_cat = context
            model_data[f"{age_group}_{job_cat}"] = {
                "context": context,
                "alpha": bandit.alpha.tolist(),
                "beta": bandit.beta.tolist(),
                "trials": bandit.trials.tolist(),
                "successes": bandit.successes.tolist()
            }

        filename = f"thompson_model_{version}.json"
        filepath = self.model_dir / filename
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Model saved: {filename} ({filepath.stat().st_size / 1024:.1f} KB)")
        return filepath

    def save_production_model(self, version_path: Path):
        """Promote versioned model to production"""
        production_path = self.model_dir / "thompson_model.json"
        shutil.copy(version_path, production_path)
        logger.info(f"Promoted to production: {production_path}")


class ModelTrainer:
    """Trains Thompson Sampling model"""

    def __init__(self, config: Config, mlflow_enabled: bool = True):
        self.config = config
        self.mlflow_enabled = mlflow_enabled
        self.version_manager = ModelVersionManager(config)

    def setup_mlflow(self):
        """Configure MLflow tracking"""
        db_path = self.config.project_root / ".mlflow" / "mlflow.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("Datathon-Thompson-Sampling")
        logger.info(f"MLflow tracking configured at {db_path}")

    def train(self, df: pd.DataFrame) -> ContextualThompsonSampling:
        """Train model on dataframe"""
        logger.info("Initializing Thompson Sampling model")
        np.random.seed(42)
        model = ContextualThompsonSampling()

        # Map contexts and arms
        df['age_group'] = df['age'].apply(model._get_age_group)
        df['job_category'] = df['job'].apply(model._get_job_category)
        df['arm'] = assign_arm(df)

        # Train
        logger.info(f"Training on {len(df):,} customers")
        for idx, row in df.iterrows():
            context = (row['age_group'], row['job_category'])
            arm = int(row['arm'])
            reward = int(row['y'])
            model.update(context, arm, reward)

            if (idx + 1) % 10000 == 0:
                logger.info(f"  Processed {idx + 1:,} customers")

        return model

    def calculate_metrics(self, model: ContextualThompsonSampling,
                         df: pd.DataFrame) -> dict:
        """Calculate model performance metrics"""
        metrics = {
            'overall_conversion': float(df['y'].mean()),
            'best_context_conversion': 0.0,
            'worst_context_conversion': 1.0,
            'contexts_with_data': 0,
            'total_trials': int(df.shape[0])
        }

        context_results = []
        for context in sorted(model.bandits.keys()):
            stats = model.get_context_stats(context)
            total_trials = sum(s['trials'] for s in stats['arms'].values())
            total_successes = sum(s['successes'] for s in stats['arms'].values())

            if total_trials > 0:
                conv_rate = total_successes / total_trials
                context_results.append({
                    'context': context,
                    'trials': total_trials,
                    'conversions': int(total_successes),
                    'rate': conv_rate
                })
                metrics['contexts_with_data'] += 1

                if conv_rate > metrics['best_context_conversion']:
                    metrics['best_context_conversion'] = conv_rate
                if conv_rate < metrics['worst_context_conversion']:
                    metrics['worst_context_conversion'] = conv_rate

        spread = metrics['best_context_conversion'] - metrics['worst_context_conversion']
        metrics['spread_pp'] = float(spread * 100)

        return metrics, context_results

    def run(self, sample_ratio: float = 1.0, version: str = None,
            temporal_cutoff: float = None) -> dict:
        """Full training pipeline.

        temporal_cutoff, if set, trains on the first N% of rows in their
        original (chronological) order instead of a random sample. The
        bank-marketing rows are ordered by real contact date (day/month;
        see UCI docs), so this genuinely reproduces "the data available
        when the model was deployed" — unlike --sample-ratio, which
        removes a random 1-(sample_ratio) of rows from the SAME full
        history and therefore isn't new data, just noise from
        re-sampling it (Achado Crítico 2 da auditoria 2026-08-05).
        """
        if self.mlflow_enabled:
            self.setup_mlflow()

        version = version or self.version_manager.get_version_suffix()

        logger.info("\n" + "=" * 70)
        logger.info(f"RETRAINING THOMPSON SAMPLING MODEL - {version}")
        logger.info("=" * 70)

        data_path = self.config.get_data_path('bank_marketing_primary.parquet')
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)

        if temporal_cutoff is not None and temporal_cutoff < 1.0:
            cutoff_idx = int(len(df) * temporal_cutoff)
            df = df.iloc[:cutoff_idx].copy()
            logger.info(
                f"Using first {len(df):,} rows in chronological order "
                f"({temporal_cutoff:.0%} temporal cutoff — data available at that point in time)"
            )
        elif sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
            logger.info(
                f"Using {len(df):,} random samples ({sample_ratio:.0%} of data) — "
                "NOTE: this resamples the SAME historical data, it is not new data "
                "(use --temporal-cutoff for a genuine 'retrain with newer data' scenario)"
            )
        else:
            logger.info(f"Using all {len(df):,} samples")

        # Train model
        model = self.train(df)
        metrics, context_results = self.calculate_metrics(model, df)

        # Etapa 3: baseline (regra fixa) vs Thompson Sampling não-contextual,
        # mesma metodologia do notebooks/datathon_main.ipynb
        etapa3 = compute_baseline_vs_thompson(df)
        logger.info(f"Etapa 3 - Baseline: {etapa3['baseline_rate']:.2%}, "
                    f"Thompson: {etapa3['thompson_rate']:.2%}, "
                    f"Improvement: {etapa3['improvement']:+.2%}")

        # Save versioned model
        model_path = self.version_manager.save_versioned_model(model, version)

        # Log to MLflow
        if self.mlflow_enabled:
            with mlflow.start_run():
                mlflow.log_param("version", version)
                mlflow.log_param("sample_ratio", sample_ratio)
                mlflow.log_param("temporal_cutoff", temporal_cutoff)
                mlflow.log_param("n_customers", len(df))

                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Etapa 3 metrics (baseline vs Thompson), required by Etapa 7
                mlflow.log_metric("baseline_rate", etapa3['baseline_rate'])
                mlflow.log_metric("thompson_rate_etapa3", etapa3['thompson_rate'])
                mlflow.log_metric("improvement_vs_baseline", etapa3['improvement'])
                mlflow.log_metric("beat_baseline", etapa3['beat_baseline'])
                for arm, rate in etapa3['arm_true_rates'].items():
                    mlflow.log_metric(f"arm_rate_{arm}", rate)
                for arm, trials in etapa3['arm_trials'].items():
                    mlflow.log_metric(f"arm_trials_{arm}", trials)

                for result in context_results:
                    age_group, job_cat = result['context']
                    mlflow.log_metric(
                        f"conversion_{age_group}_{job_cat}",
                        result['rate']
                    )

                mlflow.log_artifact(model_path, artifact_path="models")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Version: {version}")
        logger.info(f"Overall conversion: {metrics['overall_conversion']:.2%}")
        logger.info(f"Best context: {metrics['best_context_conversion']:.2%}")
        logger.info(f"Worst context: {metrics['worst_context_conversion']:.2%}")
        logger.info(f"Spread: {metrics['spread_pp']:.1f} pp")
        logger.info(f"Contexts with data: {metrics['contexts_with_data']}/12")
        logger.info(f"Model path: {model_path}")

        return {
            'version': version,
            'model_path': str(model_path),
            'metrics': metrics,
            'contexts': context_results
        }


def main():
    parser = argparse.ArgumentParser(
        description='Retrain Thompson Sampling model'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=1.0,
        help='Fraction of data to use as a RANDOM sample (default: 1.0 = all data). '
             'Resamples the same historical data — not new data; use --temporal-cutoff for that.'
    )
    parser.add_argument(
        '--temporal-cutoff',
        type=float,
        default=None,
        help='Fraction of data to use, taking the first N rows in chronological order '
             '(data available at that point in time) instead of a random sample. '
             'Overrides --sample-ratio when set.'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Custom version suffix (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--env',
        choices=['local', 'aws', 'azure'],
        default=None,
        help='Environment (auto-detected if not specified)'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )

    args = parser.parse_args()
    config = Config(env=args.env)

    trainer = ModelTrainer(config, mlflow_enabled=not args.no_mlflow)
    result = trainer.run(
        sample_ratio=args.sample_ratio,
        version=args.version,
        temporal_cutoff=args.temporal_cutoff
    )

    logger.info(f"\nTo view metrics: mlflow ui --backend-store-uri sqlite:///{config.project_root}/.mlflow/mlflow.db")


if __name__ == '__main__':
    main()
