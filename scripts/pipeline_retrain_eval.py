#!/usr/bin/env python
"""
End-to-end pipeline: retrain → evaluate → promote.

Orchestrates the complete model update workflow:
1. Train new model
2. Evaluate against production model
3. If approved, promote to production
4. Log results

Usage:
    python scripts/pipeline_retrain_eval.py [--sample-ratio 1.0]
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import logging
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainEvalPipeline:
    """Orchestrates retrain → eval → promote workflow"""

    def __init__(self, config: Config):
        self.config = config
        self.scripts_dir = Path(__file__).parent
        self.results_dir = config.project_root / "outputs" / "retraining"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _run_script(self, script: str, args: list) -> bool:
        """Run a Python script and return success status"""
        cmd = [sys.executable, str(self.scripts_dir / script)] + args
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0

    def run(self, sample_ratio: float = 1.0) -> dict:
        """Execute full pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("RETRAIN → EVALUATE → PROMOTE PIPELINE")
        logger.info("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_log = self.results_dir / f"pipeline_{timestamp}.json"

        log_data = {
            'timestamp': timestamp,
            'status': 'running',
            'steps': {}
        }

        # Step 1: Retrain
        logger.info("\n[STEP 1/3] Training new model...")
        version = f"v{timestamp}"
        train_args = [
            '--sample-ratio', str(sample_ratio),
            '--version', version,
            '--env', self.config.env.value
        ]

        if not self._run_script('retrain_model.py', train_args):
            logger.error("Training failed!")
            log_data['status'] = 'failed'
            log_data['steps']['train'] = 'failed'
            return log_data

        new_model_path = self.config.model_path / f"thompson_model_{version}.json"
        if not new_model_path.exists():
            logger.error(f"Training output not found: {new_model_path}")
            log_data['status'] = 'failed'
            log_data['steps']['train'] = 'output_not_found'
            return log_data

        log_data['steps']['train'] = {
            'status': 'success',
            'model_path': str(new_model_path)
        }
        logger.info("✓ Training successful")

        # Step 2: Evaluate
        logger.info("\n[STEP 2/3] Evaluating model...")
        old_model_path = self.config.model_path / "thompson_model.json"
        eval_args = [
            '--new-model', str(new_model_path),
            '--env', self.config.env.value,
            '--save-report', str(self.results_dir / f"eval_report_{timestamp}.json")
        ]
        if old_model_path.exists():
            eval_args.extend(['--old-model', str(old_model_path)])

        eval_success = self._run_script('evaluate_model.py', eval_args)

        eval_report_path = self.results_dir / f"eval_report_{timestamp}.json"
        eval_report = {}
        if eval_report_path.exists():
            with open(eval_report_path, 'r') as f:
                eval_report = json.load(f)

        if not eval_success or not eval_report.get('approved'):
            logger.error("Model evaluation failed or not approved!")
            log_data['status'] = 'failed'
            log_data['steps']['evaluate'] = {
                'status': 'rejected',
                'report': eval_report
            }
            return log_data

        log_data['steps']['evaluate'] = {
            'status': 'approved',
            'report': eval_report
        }
        logger.info("✓ Evaluation successful - Model approved")

        # Step 3: Promote to production
        logger.info("\n[STEP 3/3] Promoting to production...")
        try:
            production_path = self.config.model_path / "thompson_model.json"
            shutil.copy(new_model_path, production_path)
            logger.info(f"✓ Model promoted to production: {production_path}")

            log_data['steps']['promote'] = {
                'status': 'success',
                'production_path': str(production_path)
            }
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            log_data['status'] = 'failed'
            log_data['steps']['promote'] = {
                'status': 'error',
                'error': str(e)
            }
            return log_data

        # Success!
        log_data['status'] = 'success'

        # Save pipeline log
        with open(pipeline_log, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE - SUCCESS")
        logger.info("=" * 70)
        logger.info(f"Pipeline log: {pipeline_log}")
        logger.info(f"New model: {new_model_path}")
        logger.info(f"Production model updated")

        return log_data


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end retrain → evaluate → promote pipeline'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=1.0,
        help='Fraction of data to use (default: 1.0 = all data)'
    )
    parser.add_argument(
        '--env',
        choices=['local', 'aws', 'azure'],
        default=None,
        help='Environment (auto-detected if not specified)'
    )

    args = parser.parse_args()
    config = Config(env=args.env)

    pipeline = RetrainEvalPipeline(config)
    result = pipeline.run(sample_ratio=args.sample_ratio)

    sys.exit(0 if result['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
