#!/usr/bin/env python
"""
Training script with MLflow experiment tracking.

Supports:
- LOCAL: MLflow runs on http://localhost:5000 (run 'mlflow ui' in another terminal)
- AWS: Uses SageMaker Experiments + S3
- AZURE: Uses Azure ML Experiments + Blob Storage

Usage (Local):
    1. Start MLflow server in another terminal:
        mlflow ui --host 127.0.0.1 --port 5000

    2. Run training:
        python scripts/train_with_mlflow.py

Usage (AWS):
    python scripts/train_with_mlflow.py --env aws

Usage (Azure):
    python scripts/train_with_mlflow.py --env azure
"""

import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling, compute_baseline_vs_thompson
from datathon.config import get_config, is_local, is_aws, is_azure
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def setup_mlflow(config):
    """Setup MLflow tracking based on environment"""
    if is_local():
        import mlflow
        # Use SQLite for tracking (modern MLflow default)
        db_dir = config.project_root / ".mlflow"
        db_dir.mkdir(exist_ok=True)
        db_path = db_dir / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("Datathon-Thompson-Sampling")
        logger.info("MLflow configured for LOCAL (SQLite tracking)")
        logger.info(f"Database: {db_path}")
        logger.info(f"To view results: mlflow ui --backend-store-uri sqlite:///{db_path}")
        return mlflow

    elif is_aws():
        try:
            import mlflow
            import sagemaker
            session = sagemaker.Session()
            mlflow.set_tracking_uri("s3://{}/mlflow".format(config.s3_bucket))
            mlflow.set_experiment("Datathon-Thompson-Sampling")
            logger.info(f"MLflow configured for AWS (S3: {config.s3_bucket})")
            return mlflow
        except ImportError:
            logger.error("AWS mode requires: pip install sagemaker")
            sys.exit(1)

    elif is_azure():
        try:
            import mlflow
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential

            # Autenticação automática com a sessão do 'az login'
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential,
                config.tracking_config['subscription_id'],
                config.tracking_config['resource_group'],
                config.tracking_config['workspace']
            )
            
            # Pedido formal (forma correta e atualizada do SDK)
            workspace = ml_client.workspaces.get(name=config.tracking_config['workspace'])
            azure_mlflow_uri = workspace.mlflow_tracking_uri
            
            mlflow.set_tracking_uri(azure_mlflow_uri)
            mlflow.set_experiment("Datathon-Thompson-Sampling")
            
            logger.info(f"MLflow configured for AZURE")
            logger.info(f"Tracking URI: {azure_mlflow_uri}")
            return mlflow
            
        except ImportError:
            logger.error("Azure mode requires: pip install azure-ai-ml azure-identity")
            sys.exit(1)


def train_model(config):
    """Train Thompson Sampling model with MLflow tracking"""
    mlflow = setup_mlflow(config)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING CONTEXTUAL THOMPSON SAMPLING WITH MLFLOW")
    logger.info(f"Environment: {config.env.value.upper()}")
    logger.info("=" * 70)

    # Load data
    data_file = config.get_data_path('bank_marketing_primary.parquet')
    logger.info(f"\nLoading data from {data_file}...")

    try:
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(df):,} customers")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)

    # Etapa 3: baseline (regra fixa) vs Thompson Sampling não-contextual,
    # mesma metodologia do notebooks/datathon_main.ipynb
    logger.info("\nCalculating Etapa 3 baseline vs Thompson comparison...")
    etapa3 = compute_baseline_vs_thompson(df)
    logger.info(f"  Baseline:  {etapa3['baseline_rate']:.2%}")
    logger.info(f"  Thompson:  {etapa3['thompson_rate']:.2%}")
    logger.info(f"  Improvement: {etapa3['improvement']:+.2%}")

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_customers", len(df))
        mlflow.log_param("n_arms", 4)
        mlflow.log_param("n_contexts", 12)
        mlflow.log_param("algorithm", "BetaBernoulliBandit")

        # Etapa 3 metrics (baseline vs Thompson), required by Etapa 7
        mlflow.log_metric("baseline_rate", etapa3['baseline_rate'])
        mlflow.log_metric("thompson_rate_etapa3", etapa3['thompson_rate'])
        mlflow.log_metric("improvement_vs_baseline", etapa3['improvement'])
        mlflow.log_metric("beat_baseline", etapa3['beat_baseline'])
        for arm, rate in etapa3['arm_true_rates'].items():
            mlflow.log_metric(f"arm_rate_{arm}", rate)
        for arm, trials in etapa3['arm_trials'].items():
            mlflow.log_metric(f"arm_trials_{arm}", trials)

        # Initialize model
        logger.info("\nInitializing Contextual Thompson Sampling...")
        np.random.seed(42)
        model = ContextualThompsonSampling()

        # Map contexts
        df['age_group'] = df['age'].apply(model._get_age_group)
        df['job_category'] = df['job'].apply(model._get_job_category)
        df['arm'] = df.apply(
            lambda row: (0 if row['contact'] == 'cellular' else 2) if row['campaign'] == 1 else (1 if row['contact'] == 'cellular' else 3),
            axis=1
        )

        # Train
        logger.info(f"Training on {len(df):,} customers...")
        for idx, row in df.iterrows():
            context = (row['age_group'], row['job_category'])
            arm = int(row['arm'])
            reward = int(row['y'])
            model.update(context, arm, reward)

            if (idx + 1) % 10000 == 0:
                logger.info(f"  Processed {idx + 1:,} customers")

        # Calculate metrics
        logger.info("\nCalculating metrics...")
        metrics = {
            'overall_conversion': float(df['y'].mean()),
            'best_context_conversion': 0.0,
            'worst_context_conversion': 1.0,
            'contexts_with_data': 0
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

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log context metrics
        logger.info("\nContext-level metrics:")
        for result in sorted(context_results, key=lambda x: x['rate'], reverse=True):
            age_group, job_cat = result['context']
            conv_pct = result['rate'] * 100
            logger.info(f"  {age_group:8} + {job_cat:12} = {conv_pct:.1f}% ({result['conversions']:,}/{result['trials']:,})")
            mlflow.log_metric(f"conversion_{age_group}_{job_cat}", result['rate'])

        # Save model
        logger.info(f"\nSaving model...")
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

        model_file = config.get_model_path('thompson_model.json')
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        file_size_kb = Path(model_file).stat().st_size / 1024
        logger.info(f"Model saved to {model_file} ({file_size_kb:.1f} KB)")

        # Log model as artifact
        try:
            mlflow.log_artifact(model_file, artifact_path="models")
            logger.info("Artifact logged to MLflow successfully.")
        except TypeError as e:
            logger.warning("Skipping artifact upload due to azureml-mlflow plugin version mismatch.")
            logger.warning("Metrics and parameters are safely recorded in Azure ML!")

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Overall conversion: {metrics['overall_conversion']:.2%}")
        logger.info(f"Best context: {metrics['best_context_conversion']:.2%}")
        logger.info(f"Worst context: {metrics['worst_context_conversion']:.2%}")
        logger.info(f"Spread: {(metrics['best_context_conversion'] - metrics['worst_context_conversion'])*100:.1f} pp")
        logger.info(f"Contexts with data: {metrics['contexts_with_data']}/12")

        if is_local():
            logger.info(f"\nView results: http://localhost:5000")
        elif is_aws():
            logger.info(f"\nView results in SageMaker Experiments")
        elif is_azure():
            logger.info(f"\nView results in Azure ML Studio")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train Thompson Sampling with MLflow tracking'
    )
    parser.add_argument(
        '--env',
        choices=['local', 'aws', 'azure'],
        default=None,
        help='Environment to run in (auto-detected if not specified)'
    )
    args = parser.parse_args()

    config = get_config(env=args.env)
    train_model(config)


if __name__ == '__main__':
    main()
