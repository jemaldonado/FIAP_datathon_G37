#!/usr/bin/env python
"""
Create Golden Set - 5 representative customer profiles with Thompson Sampling recommendations.

This script:
1. Loads the trained model
2. Selects 5 diverse customer profiles
3. Gets recommendations for each
4. Saves results to golden_set.json
"""

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling

def load_model():
    """Load the trained Thompson Sampling model from JSON"""
    model = ContextualThompsonSampling()

    model_file = Path(__file__).parent.parent / "data" / "models" / "thompson_model.json"

    with open(model_file, 'r') as f:
        model_data = json.load(f)

    # Reconstruct bandits from saved data
    for context_key, data in model_data.items():
        context = tuple(data['context'])
        bandit = model.bandits[context]
        bandit.alpha = np.array(data['alpha'], dtype=float)
        bandit.beta = np.array(data['beta'], dtype=float)
        bandit.trials = np.array(data['trials'], dtype=float)
        bandit.successes = np.array(data['successes'], dtype=float)

    return model

def create_golden_set():
    """Create 5 representative customer profiles"""

    print("=" * 70)
    print("CREATING GOLDEN SET - 5 Representative Profiles")
    print("=" * 70)

    model = load_model()

    # Define 5 diverse profiles
    profiles = [
        {
            "name": "Young Technical Professional",
            "age": 28,
            "job": "admin",
            "marital": "single",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 1,
            "description": "Entry-level tech professional, first contact attempt"
        },
        {
            "name": "Prime Business Executive",
            "age": 42,
            "job": "director",
            "marital": "married",
            "education": "university.degree",
            "contact": "cellular",
            "campaign": 2,
            "description": "Senior manager, multiple contact attempts"
        },
        {
            "name": "Senior Retired Citizen",
            "age": 68,
            "job": "retired",
            "marital": "married",
            "education": "basic.6y",
            "contact": "telephone",
            "campaign": 1,
            "description": "Retiree, prefers phone contact"
        },
        {
            "name": "Mature Technical Expert",
            "age": 52,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "contact": "cellular",
            "campaign": 3,
            "description": "Experienced technician, repeated contact"
        },
        {
            "name": "Prime Other Professional",
            "age": 35,
            "job": "services",
            "marital": "divorced",
            "education": "basic.9y",
            "contact": "telephone",
            "campaign": 1,
            "description": "Service worker, first phone contact"
        }
    ]

    golden_set = []

    print("\nGenerating recommendations...")
    for i, profile in enumerate(profiles, 1):
        # Get context
        age_group = model._get_age_group(profile['age'])
        job_cat = model._get_job_category(profile['job'])
        context = (age_group, job_cat)
        stats = model.get_context_stats(context)

        # Find BEST arm for this context (not random selection)
        best_arm = None
        best_rate = -1
        all_arm_stats = []

        for arm_id in range(4):
            arm_name = model.ARM_NAMES[arm_id]
            arm_stat = stats['arms'][arm_name]
            all_arm_stats.append({
                'arm_id': arm_id,
                'arm_name': arm_name,
                'rate': arm_stat['rate'],
                'trials': int(arm_stat['trials']),
                'successes': int(arm_stat['successes'])
            })

            if arm_stat['rate'] > best_rate:
                best_rate = arm_stat['rate']
                best_arm = arm_id

        best_arm_name = model.ARM_NAMES[best_arm]
        best_arm_stat = stats['arms'][best_arm_name]

        # Get overall context conversion rate
        all_stats = stats['arms']
        total_trials = sum(s['trials'] for s in all_stats.values())
        total_successes = sum(s['successes'] for s in all_stats.values())
        context_conversion = (total_successes / total_trials * 100) if total_trials > 0 else 0

        result = {
            "id": i,
            "profile": {
                "name": profile['name'],
                "description": profile['description'],
                "age": profile['age'],
                "job": profile['job'],
                "marital": profile['marital'],
                "education": profile['education'],
                "contact": profile['contact'],
                "campaign": profile['campaign']
            },
            "context": {
                "age_group": age_group,
                "job_category": job_cat
            },
            "recommendation": {
                "arm_id": int(best_arm),
                "arm_name": best_arm_name,
                "expected_conversion_rate": round(best_arm_stat['rate'] * 100, 2),
                "context_conversion_rate": round(context_conversion, 2),
                "rationale": f"Thompson Sampling recommends {best_arm_name} for {age_group} + {job_cat} segment (expected {best_arm_stat['rate']*100:.1f}% conversion)"
            },
            "arm_rankings": sorted(all_arm_stats, key=lambda x: x['rate'], reverse=True),
            "arm_details": {
                arm_name: {
                    "rate": round(arm_stat['rate'] * 100, 2),
                    "trials": int(arm_stat['trials']),
                    "successes": int(arm_stat['successes'])
                }
                for arm_name, arm_stat in stats['arms'].items()
            }
        }

        golden_set.append(result)

        print(f"\n{i}. {profile['name']}")
        print(f"   Context: {age_group} + {job_cat}")
        print(f"   BEST: {best_arm_name} ({best_rate*100:.1f}%)")
        print(f"   Rankings:")
        for rank, arm_info in enumerate(sorted(all_arm_stats, key=lambda x: x['rate'], reverse=True), 1):
            print(f"      {rank}. {arm_info['arm_name']:20} -> {arm_info['rate']*100:5.1f}% ({arm_info['successes']:,}/{arm_info['trials']:,})")

    # Save to JSON
    output_dir = Path(__file__).parent.parent / "data" / "golden_set"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "golden_set.json"

    with open(output_file, 'w') as f:
        json.dump(golden_set, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Golden Set saved to: {output_file}")
    print("=" * 70)
    print(f"\nSummary:")
    for result in golden_set:
        print(f"  - {result['profile']['name']:30} -> {result['recommendation']['arm_name']:15} ({result['recommendation']['expected_conversion_rate']:.1f}%)")

    return golden_set

if __name__ == '__main__':
    create_golden_set()
