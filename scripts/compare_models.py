#!/usr/bin/env python
"""
Compare two Thompson Sampling models side by side.

Shows conversion rates, differences, and highlights regressions.

Usage:
    python scripts/compare_models.py --model1 data/models/thompson_model.json --model2 data/models/thompson_model_v20260707_123456.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling, BetaBernoulliBandit
from datathon.config import Config
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Load models from JSON"""

    @staticmethod
    def load_from_json(filepath: Path) -> ContextualThompsonSampling:
        """Reconstruct model from JSON file"""
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

        return model


class ModelComparator:
    """Compare two models"""

    @staticmethod
    def get_metrics(model: ContextualThompsonSampling) -> Dict:
        """Extract metrics from model"""
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
                'rate': rate,
                'trials': total_trials,
                'successes': int(total_successes),
                'best_arm': ModelComparator._best_arm(model, context),
            }

        return metrics

    @staticmethod
    def _best_arm(model: ContextualThompsonSampling, context) -> Dict:
        """Arm with the highest posterior mean (alpha / (alpha + beta)) for a context.

        This is the arm Thompson Sampling favors on average — the same
        quantity get_conversion_rates() exposes — so it's what actually
        determines which offer a customer in this context tends to get.
        """
        bandit = model.bandits[context]
        rates = bandit.get_conversion_rates()
        arm_id = int(np.argmax(rates))
        return {
            'arm_id': arm_id,
            'arm_name': model.ARM_NAMES[arm_id],
            'rate': float(rates[arm_id]),
        }

    def compare(self, model1_path: Path, model2_path: Path) -> Dict:
        """Compare two models"""
        logger.info(f"Loading model 1: {model1_path.name}")
        model1 = ModelLoader.load_from_json(model1_path)
        metrics1 = self.get_metrics(model1)

        logger.info(f"Loading model 2: {model2_path.name}")
        model2 = ModelLoader.load_from_json(model2_path)
        metrics2 = self.get_metrics(model2)

        comparison = []

        for ctx_key in sorted(metrics1.keys()):
            m1 = metrics1[ctx_key]
            m2 = metrics2.get(ctx_key, {'rate': 0.0, 'best_arm': None})

            diff = m2['rate'] - m1['rate']
            pct_change = (diff / m1['rate'] * 100) if m1['rate'] > 0 else 0.0

            arm1 = m1['best_arm']
            arm2 = m2['best_arm']
            arm_changed = bool(arm2) and arm1['arm_id'] != arm2['arm_id']

            comparison.append({
                'context': ctx_key,
                'model1_rate': m1['rate'],
                'model2_rate': m2['rate'],
                'diff': diff,
                'pct_change': pct_change,
                'regression': diff < -0.02,
                'model1_best_arm': arm1,
                'model2_best_arm': arm2,
                'arm_changed': arm_changed,
            })

        return {
            'model1': str(model1_path),
            'model2': str(model2_path),
            'contexts': comparison,
            'summary': self._summarize(comparison)
        }

    @staticmethod
    def _summarize(comparison: list) -> Dict:
        """Summarize comparison"""
        regressions = [c for c in comparison if c['regression']]
        improvements = [c for c in comparison if c['diff'] > 0.02]
        no_change = [c for c in comparison if abs(c['diff']) <= 0.02]
        arm_changes = [c for c in comparison if c['arm_changed']]

        return {
            'total_contexts': len(comparison),
            'improvements': len(improvements),
            'regressions': len(regressions),
            'no_change': len(no_change),
            'avg_diff': sum(c['diff'] for c in comparison) / len(comparison),
            'arm_changes': len(arm_changes),
            'arm_changed_contexts': [c['context'] for c in arm_changes],
        }

    def print_report(self, comparison: Dict):
        """Print formatted comparison report"""
        print("\n" + "=" * 90)
        print("MODEL COMPARISON REPORT")
        print("=" * 90)
        print(f"Model 1: {Path(comparison['model1']).name}")
        print(f"Model 2: {Path(comparison['model2']).name}")
        print("=" * 90)

        # Header
        print(f"\n{'Context':<20} {'Model 1':<12} {'Model 2':<12} {'Diff':<12} {'Change %':<10} {'Status':<10}")
        print("-" * 90)

        # Data rows
        for ctx in comparison['contexts']:
            if ctx['arm_changed']:
                status = "[ARM!]"
            elif ctx['regression']:
                status = "[REG]"
            elif ctx['diff'] > 0.02:
                status = "[IMP]"
            else:
                status = "[OK]"

            print(
                f"{ctx['context']:<20} "
                f"{ctx['model1_rate']:>10.2%}  "
                f"{ctx['model2_rate']:>10.2%}  "
                f"{ctx['diff']:>+10.2%}  "
                f"{ctx['pct_change']:>+8.1f}%  "
                f"{status:<10}"
            )

        # Summary
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        summary = comparison['summary']
        print(f"Total contexts: {summary['total_contexts']}")
        print(f"Improvements:   {summary['improvements']}")
        print(f"Regressions:    {summary['regressions']}")
        print(f"No change:      {summary['no_change']}")
        print(f"Average diff:   {summary['avg_diff']:+.2%}")
        print(f"Arm changes:    {summary['arm_changes']}")
        print("=" * 90)

        if summary['arm_changes'] > 0:
            print(f"\nARM CHANGE: {summary['arm_changes']} context(s) where the winning offer changed after retraining:")
            for ctx in comparison['contexts']:
                if not ctx['arm_changed']:
                    continue
                a1, a2 = ctx['model1_best_arm'], ctx['model2_best_arm']
                print(
                    f"  - {ctx['context']}: "
                    f"{a1['arm_name']} ({a1['rate']:.2%}) -> {a2['arm_name']} ({a2['rate']:.2%})"
                )

        if summary['regressions'] > 0:
            print("\nWARNING: Model 2 has regressions in some contexts")
        else:
            print("\nOK: No regressions detected")

        print()


def main():
    parser = argparse.ArgumentParser(
        description='Compare two Thompson Sampling models'
    )
    parser.add_argument(
        '--model1',
        type=Path,
        required=True,
        help='Path to first model (baseline)'
    )
    parser.add_argument(
        '--model2',
        type=Path,
        required=True,
        help='Path to second model (candidate)'
    )
    parser.add_argument(
        '--save-report',
        type=Path,
        default=None,
        help='Save comparison report as JSON'
    )

    args = parser.parse_args()

    if not args.model1.exists():
        logger.error(f"Model 1 not found: {args.model1}")
        sys.exit(1)

    if not args.model2.exists():
        logger.error(f"Model 2 not found: {args.model2}")
        sys.exit(1)

    comparator = ModelComparator()
    comparison = comparator.compare(args.model1, args.model2)

    comparator.print_report(comparison)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Report saved to {args.save_report}")


if __name__ == '__main__':
    main()
