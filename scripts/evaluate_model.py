#!/usr/bin/env python
"""
Model evaluation and comparison.

Evaluates a new model against production model:
- Conversion rate by context
- Regression detection (contexts getting worse)
- Golden Set validation
- Overall quality metrics

Usage:
    python scripts/evaluate_model.py --new-model data/models/thompson_model_v20260707_123456.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling, BetaBernoulliBandit
from datathon.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and reconstruct models from JSON"""

    @staticmethod
    def load_from_json(filepath: Path) -> ContextualThompsonSampling:
        """Reconstruct model from JSON file"""
        logger.info(f"Loading model from {filepath}")

        with open(filepath, 'r') as f:
            model_data = json.load(f)

        model = ContextualThompsonSampling()

        for context_key, data in model_data.items():
            age_group, job_cat = data['context']
            context = (age_group, job_cat)

            bandit = BetaBernoulliBandit(
                n_arms=4,
                alpha_init=data['alpha'],
                beta_init=data['beta']
            )
            bandit.successes = np.array(data['successes'], dtype=float)
            bandit.trials = np.array(data['trials'], dtype=float)

            model.bandits[context] = bandit

        logger.info(f"Model loaded with {len(model.bandits)} contexts")
        return model


class ModelEvaluator:
    """Evaluates model quality and performance"""

    def __init__(self, config: Config):
        self.config = config
        self.loader = ModelLoader()

    def load_golden_set(self) -> List[Dict]:
        """Load golden set test cases"""
        golden_path = self.config.project_root / "data" / "golden_set" / "golden_set.json"
        if not golden_path.exists():
            logger.warning(f"Golden set not found at {golden_path}")
            return []

        with open(golden_path, 'r') as f:
            return json.load(f)

    def get_context_metrics(self, model: ContextualThompsonSampling) -> Dict:
        """Calculate metrics for all contexts"""
        metrics = {}

        for context in sorted(model.bandits.keys()):
            stats = model.get_context_stats(context)
            total_trials = sum(s['trials'] for s in stats['arms'].values())
            total_successes = sum(s['successes'] for s in stats['arms'].values())

            if total_trials > 0:
                rate = total_successes / total_trials
            else:
                rate = 0.0

            age_group, job_cat = context
            metrics[f"{age_group}_{job_cat}"] = {
                'context': context,
                'trials': total_trials,
                'successes': int(total_successes),
                'rate': rate
            }

        return metrics

    def compare_models(self, new_model: ContextualThompsonSampling,
                      old_model: ContextualThompsonSampling) -> Dict:
        """Compare new model against old model"""
        new_metrics = self.get_context_metrics(new_model)
        old_metrics = self.get_context_metrics(old_model)

        comparison = {}
        regressions = []

        for context_key in new_metrics:
            new = new_metrics[context_key]
            old = old_metrics.get(context_key, {'rate': 0.0})

            diff = new['rate'] - old['rate']
            pct_change = (diff / old['rate'] * 100) if old['rate'] > 0 else 0.0

            comparison[context_key] = {
                'new_rate': new['rate'],
                'old_rate': old['rate'],
                'diff': diff,
                'pct_change': pct_change,
                'regression': diff < -0.02  # Threshold: -2% regression
            }

            if comparison[context_key]['regression']:
                regressions.append({
                    'context': context_key,
                    'diff': diff,
                    'new_rate': new['rate'],
                    'old_rate': old['rate']
                })

        return {
            'comparison': comparison,
            'regressions': regressions,
            'has_regressions': len(regressions) > 0
        }

    def validate_golden_set(self, model: ContextualThompsonSampling) -> Dict:
        """Validate model against golden set.

        Checks the arm the model actually favors for each profile — not
        just which (age_group, job_category) context it maps to. Context
        mapping is a static lookup table, identical for any model, so a
        context-only check can never fail in practice (Achado Médio 4).

        Uses the arm with the highest posterior mean
        (get_conversion_rates().argmax()), not select_arm()'s stochastic
        Thompson draw — a promotion gate needs a deterministic answer, and
        Thompson Sampling explores by design, so a single stochastic draw
        would fail intermittently even for a perfectly good model.
        """
        golden_set = self.load_golden_set()
        if not golden_set:
            return {'status': 'skipped', 'reason': 'No golden set available'}

        validations = []
        for item in golden_set:
            profile = item.get('profile', item)
            age = profile.get('age')
            job = profile.get('job')
            expected_context_dict = item.get('context', {})
            age_group = expected_context_dict.get('age_group')
            job_category = expected_context_dict.get('job_category')
            expected_arm = item.get('recommendation', {}).get('arm_id')

            if age is None or job is None or age_group is None or job_category is None or expected_arm is None:
                continue

            expected_context = (age_group, job_category)
            actual_context = (model._get_age_group(age), model._get_job_category(job))

            bandit = model.bandits.get(actual_context)
            actual_arm = int(np.argmax(bandit.get_conversion_rates())) if bandit else None

            context_match = actual_context == expected_context
            arm_match = actual_arm == expected_arm

            validations.append({
                'profile': profile.get('name', 'Unknown'),
                'age': age,
                'job': job,
                'expected_context': expected_context,
                'actual_context': actual_context,
                'expected_arm': expected_arm,
                'actual_arm': actual_arm,
                'context_match': context_match,
                'arm_match': arm_match,
                'match': context_match and arm_match
            })

        passed = sum(1 for v in validations if v['match'])
        total = len(validations)

        return {
            'status': 'passed' if passed == total else 'partial',
            'passed': passed,
            'total': total,
            'validations': validations
        }

    def generate_report(self, new_model_path: Path,
                       old_model_path: Path = None) -> Dict:
        """Generate comprehensive evaluation report"""
        logger.info("\n" + "=" * 70)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 70)

        new_model = self.loader.load_from_json(new_model_path)
        new_metrics = self.get_context_metrics(new_model)

        report = {
            'timestamp': str(Path(new_model_path).stem),
            'new_model': str(new_model_path),
            'old_model': str(old_model_path),
            'new_metrics': new_metrics,
            'golden_set': self.validate_golden_set(new_model),
            'comparison': None,
            'approved': False
        }

        # Compare if old model provided
        if old_model_path and old_model_path.exists():
            logger.info(f"Comparing against {old_model_path}")
            old_model = self.loader.load_from_json(old_model_path)
            comparison = self.compare_models(new_model, old_model)
            report['comparison'] = comparison

            # Approval logic
            has_regressions = comparison['has_regressions']
            golden_set_pass = report['golden_set']['status'] in ['passed', 'partial']

            report['approved'] = not has_regressions and golden_set_pass
            logger.info(f"Regressions detected: {has_regressions}")
            logger.info(f"Golden Set status: {report['golden_set']['status']}")

        # Print summary
        logger.info("\nContext Performance:")
        for context_key, metrics in sorted(new_metrics.items()):
            logger.info(
                f"  {context_key}: {metrics['rate']:.1%} "
                f"({metrics['successes']:,}/{metrics['trials']:,})"
            )

        if report['comparison']:
            logger.info("\nRegression Analysis:")
            if report['comparison']['regressions']:
                for reg in report['comparison']['regressions']:
                    logger.info(
                        f"  REGRESSION: {reg['context']}: "
                        f"{reg['old_rate']:.1%} → {reg['new_rate']:.1%} "
                        f"({reg['diff']:+.1%})"
                    )
            else:
                logger.info("  No regressions detected ✓")

        logger.info("\n" + "=" * 70)
        if report['approved']:
            logger.info("APPROVAL: ✓ Model approved for production")
        else:
            logger.info("APPROVAL: ✗ Model NOT approved (regressions or issues detected)")
        logger.info("=" * 70)

        return report


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Thompson Sampling model'
    )
    parser.add_argument(
        '--new-model',
        type=Path,
        required=True,
        help='Path to new model to evaluate'
    )
    parser.add_argument(
        '--old-model',
        type=Path,
        default=None,
        help='Path to production model for comparison (optional)'
    )
    parser.add_argument(
        '--env',
        choices=['local', 'aws', 'azure'],
        default=None,
        help='Environment (auto-detected if not specified)'
    )
    parser.add_argument(
        '--save-report',
        type=Path,
        default=None,
        help='Save evaluation report as JSON'
    )

    args = parser.parse_args()
    config = Config(env=args.env)

    evaluator = ModelEvaluator(config)

    # Use production model if not specified
    old_model = args.old_model or (config.model_path / "thompson_model.json")

    report = evaluator.generate_report(args.new_model, old_model)

    # Save report if requested
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            report_serializable = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in report.items()
            }
            json.dump(report_serializable, f, indent=2)
        logger.info(f"Report saved to {args.save_report}")

    # Exit with appropriate code
    sys.exit(0 if report['approved'] else 1)


if __name__ == '__main__':
    main()
