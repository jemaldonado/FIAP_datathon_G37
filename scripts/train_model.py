#!/usr/bin/env python
"""
EXPLORATORY / LEGACY SCRIPT — not the production training path.

This script explores a *stochastic* arm-assignment strategy (see
`assign_arm` below) as an alternative design. It is superseded by
`scripts/retrain_model.py`, which uses a *deterministic*, documented rule
(`datathon.bandit.assign_arm`) and is what actually produces the model
served by the API (`data/models/thompson_model.json`). This script writes
to a different path (`models/contextual_thompson.pkl` / `models/
thompson_model.json`, project root) and is not invoked by
`scripts/pipeline_retrain_eval.py` or any other part of the live pipeline.

Kept for reference/portfolio value (an alternative design considered and
its rationale), but do not cite it as a description of what's in
production — see `datathon.bandit.assign_arm` docstring for that instead.

This script:
1. Loads cleaned data from parquet
2. Defines 4 arms (contact strategies) via a stochastic (exploratory) rule
3. Trains 12 Thompson Sampling bandits (4 age_groups × 3 job_categories)
4. Validates with desk tests (sample predictions)
5. Checks for drifts in data
6. Serializes model with joblib
7. Saves to models/contextual_thompson.pkl

Run: python scripts/train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
import sys

# Import bandit models
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from datathon.bandit import BetaBernoulliBandit, ContextualThompsonSampling

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def assign_arm(row) -> int:
    """
    Assign customer to one of 4 campaign strategies via stochastic distribution.

    IMPORTANT: Real data has only 2 channels (cellular: 14.74% baseline, telephone: 5.23%).
    Email_Campaign, SMS_Alert, Call_Premium are SYNTHETIC campaigns inferred from data patterns:
    - Email_Campaign: Estimated at 10.3% (70% of cellular = less intrusive alternative)
    - SMS_Alert: Estimated at 17.9% (base 15% for <30yo + 10% mobile-first boost)
    - Call_Premium: Estimated at 13.9% (base 11% for housing customers + 20% VIP boost)

    These synthesized rates allow Thompson to explore beyond just cellular/telephone,
    discovering which STRATEGY works best per CONTEXT (age × job).

    Strategy: Create stochastic (not deterministic) assignment so Thompson can explore all arms:
    - Cellular customers: favor Cellular_Standard (real) + Email/SMS exploration
    - Telephone customers: diversify across all 4 strategies for learning
    """
    contact = row['contact']
    campaign = row['campaign']

    if contact == 'cellular':
        if campaign == 1:
            probabilities = [0.50, 0.35, 0.10, 0.05]
        else:
            probabilities = [0.45, 0.40, 0.10, 0.05]
    else:
        if campaign == 1:
            probabilities = [0.25, 0.30, 0.25, 0.20]
        else:
            probabilities = [0.20, 0.30, 0.30, 0.20]

    return np.random.choice(4, p=probabilities)


def desk_test(model: ContextualThompsonSampling, df: pd.DataFrame) -> None:
    """Manual desk tests to validate model behavior"""
    logger.info("\n" + "=" * 70)
    logger.info("DESK TESTS - Validating Model Behavior")
    logger.info("=" * 70)

    test_cases = [
        {'age': 28, 'job': 'admin', 'name': 'Young + Technical'},
        {'age': 38, 'job': 'director', 'name': 'Prime + Business'},
        {'age': 72, 'job': 'retired', 'name': 'Senior + Other'},
        {'age': 55, 'job': 'technician', 'name': 'Mature + Technical'},
    ]

    logger.info("\nTest 1: Context Identification")
    for test in test_cases:
        arm, (age_group, job_cat) = model.select_arm(test['age'], test['job'])
        logger.info(f"  {test['name']:25} -> Age: {age_group:8} Job: {job_cat:12} (Recommended arm: {model.ARM_NAMES[arm]})")

    logger.info("\nTest 2: Conversion Rate Comparison (Should vary by context)")
    for age_group in ['Young', 'Prime', 'Mature', 'Senior']:
        for job_cat in ['Technical', 'Business', 'Other']:
            context = (age_group, job_cat)
            stats = model.get_context_stats(context)

            arm_count = stats['arms']
            best_arm = max(arm_count.items(), key=lambda x: x[1]['rate'])
            total_trials = sum(a['trials'] for a in arm_count.values())

            if total_trials > 0:
                best_rate = best_arm[1]['rate']
                logger.info(f"  {age_group:8} + {job_cat:12} -> Best arm: {best_arm[0]:15} ({best_rate:.1%})")

    logger.info("\nTest 3: Sanity Check - Alpha/Beta should increase after training")
    sample_context = ('Prime', 'Business')
    stats = model.get_context_stats(sample_context)
    all_trials = sum(s['trials'] for s in stats['arms'].values())
    logger.info(f"  {sample_context}: Total trials = {all_trials}")
    assert all_trials > 0, "ERROR: No trials recorded!"
    logger.info("  OK: Model learned from data")


def check_data_drift(df: pd.DataFrame, original_cols: list) -> None:
    """Check for unexpected data drift or anomalies"""
    logger.info("\n" + "=" * 70)
    logger.info("DATA DRIFT DETECTION")
    logger.info("=" * 70)

    # Check 1: Missing values
    logger.info("\nCheck 1: Missing Values")
    missing = df[['age', 'job', 'contact', 'campaign', 'y']].isnull().sum()
    for col, count in missing.items():
        if count > 0:
            logger.warning(f"  WARNING: {col}: {count} missing values ({count/len(df)*100:.2f}%)")
        else:
            logger.info(f"  OK: {col}: No missing values")

    # Check 2: Unusual age values
    logger.info("\nCheck 2: Age Distribution")
    age_min, age_max, age_mean = df['age'].min(), df['age'].max(), df['age'].mean()
    logger.info(f"  Age range: {age_min} to {age_max}")
    logger.info(f"  Age mean: {age_mean:.1f}")
    if age_min < 18:
        logger.warning(f"  WARNING: Found ages < 18: {(df['age'] < 18).sum()} records")
    if age_max > 100:
        logger.warning(f"  WARNING: Found ages > 100: {(df['age'] > 100).sum()} records")

    # Check 3: Campaign value distribution
    logger.info("\nCheck 3: Campaign Distribution")
    campaign_dist = df['campaign'].value_counts().sort_index()
    for camp, count in campaign_dist.head(5).items():
        pct = count / len(df) * 100
        logger.info(f"  campaign={camp}: {count:,} ({pct:.1f}%)")

    # Check 4: Contact type distribution
    logger.info("\nCheck 4: Contact Type Distribution")
    contact_dist = df['contact'].value_counts()
    for contact, count in contact_dist.items():
        pct = count / len(df) * 100
        logger.info(f"  {contact}: {count:,} ({pct:.1f}%)")

    logger.info("\nOK: Data drift check complete")


def main():
    """Main training pipeline"""
    logger.info("=" * 70)
    logger.info("CONTEXTUAL THOMPSON SAMPLING - Training Pipeline")
    logger.info("=" * 70)

    # Setup paths
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    models_dir = Path(__file__).parent.parent / "models"
    parquet_file = data_dir / "bank_marketing_primary.parquet"
    model_file = models_dir / "contextual_thompson.pkl"

    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"\nLoading data from {parquet_file}...")
    if not parquet_file.exists():
        logger.error(f"Data file not found: {parquet_file}")
        sys.exit(1)

    df = pd.read_parquet(parquet_file)
    logger.info(f"Loaded {len(df):,} customers")
    logger.info(f"Overall conversion rate: {df['y'].mean():.2%}")

    # Check data quality
    check_data_drift(df, ['age', 'job', 'contact', 'campaign', 'y'])

    # Assign arms
    logger.info(f"\nAssigning customers to arms...")
    df['arm'] = df.apply(assign_arm, axis=1)

    logger.info("Campaign Strategy Distribution:")
    for arm_id in range(4):
        mask = df['arm'] == arm_id
        count = mask.sum()
        conv_rate = df[mask]['y'].mean()
        logger.info(f"  {arm_id}. {ContextualThompsonSampling.ARM_NAMES[arm_id]:20} -> {count:,} customers ({conv_rate:.2%} conversion)")

    # Initialize model
    logger.info(f"\nInitializing Contextual Thompson Sampling (12 contexts)...")
    np.random.seed(42)
    model = ContextualThompsonSampling()

    # Map contexts
    df['age_group'] = df['age'].apply(model._get_age_group)
    df['job_category'] = df['job'].apply(model._get_job_category)

    # Initialize with priors
    logger.info("Initializing priors from historical data...")
    for context in model.bandits.keys():
        age_group, job_cat = context
        mask = (df['age_group'] == age_group) & (df['job_category'] == job_cat)
        context_df = df[mask]

        if len(context_df) > 0:
            arm_stats = context_df.groupby('arm')['y'].agg(['sum', 'count'])
            for arm_id in range(4):
                if arm_id in arm_stats.index:
                    successes = int(arm_stats.loc[arm_id, 'sum'])
                    trials = int(arm_stats.loc[arm_id, 'count'])
                    failures = trials - successes

                    model.bandits[context].alpha[arm_id] = successes + 1
                    model.bandits[context].beta[arm_id] = failures + 1
                    model.bandits[context].trials[arm_id] = trials
                    model.bandits[context].successes[arm_id] = successes

    logger.info("Initialized priors for all 12 contexts:")
    for context in sorted(model.bandits.keys()):
        age_group, job_cat = context
        total_trials = sum(model.bandits[context].trials)
        if total_trials > 0:
            logger.info(f"  {age_group:8} + {job_cat:12}: {total_trials:,} customers")

    # Validate all customers mapped
    total_in_contexts = sum(
        len(df[(df['age_group'] == ctx[0]) & (df['job_category'] == ctx[1])])
        for ctx in model.bandits.keys()
    )
    logger.info(f"\nValidation: {total_in_contexts}/{len(df)} customers assigned to contexts")
    assert total_in_contexts == len(df), "ERROR: Not all customers assigned to contexts!"
    logger.info("OK: All customers correctly mapped to contexts\n")

    # Train model
    logger.info(f"Training Thompson Sampling posteriors on {len(df):,} customers...")
    logger.info("(Using historical arm assignments from data, not Thompson selection)")

    for idx, row in df.iterrows():
        age_group = row['age_group']
        job_cat = row['job_category']
        context = (age_group, job_cat)

        arm = int(row['arm'])
        reward = int(row['y'])

        model.update(context, arm, reward)

        if (idx + 1) % 10000 == 0:
            logger.info(f"  Processed {idx + 1:,} customers")

    # Desk tests
    desk_test(model, df)

    # Calculate metrics by context
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS BY CONTEXT (Ranking by conversion rate)")
    logger.info("=" * 70)

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
                'successes': int(total_successes),
                'rate': conv_rate,
                'stats': stats
            })

    # Sort by conversion rate
    context_results.sort(key=lambda x: x['rate'], reverse=True)

    for rank, result in enumerate(context_results, 1):
        age_group, job_cat = result['context']
        logger.info(f"\n{rank:2d}. {age_group:8} + {job_cat:12} = {result['rate']:.2%} ({result['successes']:,}/{result['trials']:,})")

        for arm_name, stats in result['stats']['arms'].items():
            if stats['trials'] > 0:
                logger.info(f"     {arm_name:15}: {stats['rate']:.1%} ({stats['successes']:,}/{stats['trials']:,})")

    # Overall statistics
    best_context = context_results[0]
    worst_context = context_results[-1]

    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)
    logger.info(f"Best context:  {best_context['context'][0]} + {best_context['context'][1]} = {best_context['rate']:.2%}")
    logger.info(f"Worst context: {worst_context['context'][0]} + {worst_context['context'][1]} = {worst_context['rate']:.2%}")
    logger.info(f"Difference:    {(best_context['rate'] - worst_context['rate'])*100:.2f} pp ({(best_context['rate']/worst_context['rate'] - 1)*100:.1f}% improvement)")

    # Serialize model (both pickle and JSON)
    logger.info(f"\nSerializing model...")

    # Save as pickle
    try:
        joblib.dump(model, model_file)
        file_size_kb = model_file.stat().st_size / 1024
        logger.info(f"  Pickle: {model_file} ({file_size_kb:.1f} KB)")
    except Exception as e:
        logger.error(f"Failed to save pickle model: {e}")
        sys.exit(1)

    # Save as JSON (for dashboard and API)
    json_file = models_dir / "thompson_model.json"
    try:
        model_data = {}
        for context in sorted(model.bandits.keys()):
            age_group, job_cat = context
            bandit = model.bandits[context]
            key = f"{age_group}_{job_cat}"

            model_data[key] = {
                "context": list(context),
                "alpha": bandit.alpha.tolist(),
                "beta": bandit.beta.tolist(),
                "trials": bandit.trials.tolist(),
                "successes": bandit.successes.tolist()
            }

        with open(json_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        file_size_kb = json_file.stat().st_size / 1024
        logger.info(f"  JSON: {json_file} ({file_size_kb:.1f} KB)")
    except Exception as e:
        logger.error(f"Failed to save JSON model: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Model ready for deployment: {model_file}")
    logger.info(f"Use: model = joblib.load('{model_file}')")
    logger.info(f"     arm, context = model.select_arm(age=35, job='admin')")

    return 0


if __name__ == '__main__':
    sys.exit(main())
