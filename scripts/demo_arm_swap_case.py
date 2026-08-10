#!/usr/bin/env python
"""
Demo: a real customer segment where retraining flips the recommended offer.

Comparing data/models/thompson_model_baseline_temporal.json (a snapshot
trained on the first 70% of contacts, in real chronological order) with
data/models/thompson_model_v2.json (retrained candidate, trained on ALL
available data) shows one segment, Young_Technical (age 17-29, job in
admin./blue-collar/technician/services), where the winning arm
actually changes:

    baseline (70% cronológico): Email_Campaign    (~10.72% expected conversion)
    v2 (100%, todos os dados):  Cellular_Standard  (~19.66% expected conversion)

The bank-marketing dataset is ordered by real contact date (day/month; see
UCI docs), so the baseline snapshot is genuinely "what the model would have
learned with the data available earlier in the campaign" — not a random
resample of the same rows (an earlier version of this demo used
`df.sample(frac=0.7)`, which is noise from resampling, not new data; see
audit finding 2026-08-05).

This script first proves that with the two model files directly (no API
needed), then - if the API is running - fires real /canary/recommend calls
for the same customer profile and shows the live system routing to a
different offer depending on which model answered (using the optional
baseline_model_path override on /canary/start, so this comparison doesn't
touch the production model served by /recommend).

Usage:
    python scripts/demo_arm_swap_case.py                 # static comparison only
    python src/datathon/api/app.py                       # terminal 1
    python scripts/demo_arm_swap_case.py --live           # terminal 2
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datathon.bandit import ContextualThompsonSampling, BetaBernoulliBandit

REPO_ROOT = Path(__file__).parent.parent
API_URL = "http://localhost:5000"

CONTEXT = ("Young", "Technical")

# Real customer profile that maps to the Young_Technical context:
# age_group='Young' -> 17 <= age < 30, job_category='Technical' -> job in
# ['admin', 'technician', 'blue-collar', 'services'] (see
# ContextualThompsonSampling.AGE_GROUPS / JOB_CATEGORIES).
CUSTOMER = {
    "age": 28,
    "job": "admin",
    "marital": "single",
    "education": "university.degree",
    "contact": "cellular",
    "campaign": 1,
}


def load_model(path: Path) -> ContextualThompsonSampling:
    with open(path, "r") as f:
        model_data = json.load(f)

    model = ContextualThompsonSampling()
    for data in model_data.values():
        context = tuple(data["context"])
        bandit = BetaBernoulliBandit(
            n_arms=4, alpha_init=data["alpha"], beta_init=data["beta"]
        )
        bandit.trials = np.array(data["trials"], dtype=float)
        bandit.successes = np.array(data["successes"], dtype=float)
        model.bandits[context] = bandit

    return model


def best_arm(model: ContextualThompsonSampling, context) -> tuple[int, str, float]:
    bandit = model.bandits[context]
    rates = bandit.get_conversion_rates()
    arm_id = int(np.argmax(rates))
    return arm_id, model.ARM_NAMES[arm_id], float(rates[arm_id])


def static_comparison(baseline_path: Path, canary_path: Path) -> bool:
    print("=" * 70)
    print(f"STATIC COMPARISON - context {CONTEXT[0]}_{CONTEXT[1]}")
    print("=" * 70)

    baseline = load_model(baseline_path)
    canary = load_model(canary_path)

    b_id, b_name, b_rate = best_arm(baseline, CONTEXT)
    c_id, c_name, c_rate = best_arm(canary, CONTEXT)

    print(f"baseline ({baseline_path.name}): {b_name:18s} ({b_rate:.2%})")
    print(f"canary   ({canary_path.name}): {c_name:18s} ({c_rate:.2%})")

    swapped = b_id != c_id
    if swapped:
        print(f"\n>>> ARM SWAP: {b_name} -> {c_name} after retraining")
    else:
        print("\nNo swap between these two model files for this context.")
    print()
    return swapped


def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        print(f"API not reachable at {API_URL}: {exc}")
        print("Start it first: python src/datathon/api/app.py")
        sys.exit(1)


def start_canary(canary_percentage: int, model_path: str, baseline_model_path: str):
    r = requests.post(
        f"{API_URL}/canary/start",
        json={
            "canary_percentage": canary_percentage,
            "model_path": model_path,
            "baseline_model_path": baseline_model_path,
        },
        timeout=10,
    )
    r.raise_for_status()
    print(f"Canary started: {r.json()}\n")


def run_live_demo(n: int, canary_percentage: int, canary_model_path: str, baseline_model_path: str):
    print("=" * 70)
    print("LIVE DEMO - real requests against a running API")
    print("=" * 70)
    check_api()
    print(f"Customer profile (maps to {CONTEXT[0]}_{CONTEXT[1]}): {CUSTOMER}\n")

    start_canary(canary_percentage, canary_model_path, baseline_model_path)

    print(f"Firing {n} requests at /canary/recommend "
          f"({canary_percentage}% routed to canary)...\n")

    results = {"BASELINE": Counter(), "CANARY": Counter()}
    for _ in range(n):
        r = requests.post(f"{API_URL}/canary/recommend", json=CUSTOMER, timeout=10)
        r.raise_for_status()
        data = r.json()
        results[data["model_version"]][data["arm_name"]] += 1

    print("Offer recommended, by model version:\n")
    tops = {}
    for version in ("BASELINE", "CANARY"):
        counts = results[version]
        total = sum(counts.values())
        print(f"{version} ({total} requests):")
        for arm_name, count in counts.most_common():
            pct = count / total * 100 if total else 0
            print(f"  {arm_name:20s} {count:5d}  ({pct:5.1f}%)")
        print()
        if counts:
            tops[version] = counts.most_common(1)[0][0]

    print("=" * 70)
    if tops.get("BASELINE") and tops.get("CANARY") and tops["BASELINE"] != tops["CANARY"]:
        print(f"ARM SWAP CONFIRMED LIVE: baseline majority '{tops['BASELINE']}', "
              f"canary majority '{tops['CANARY']}' for the same customer.")
    else:
        print("No clear majority swap in this run (Thompson Sampling is stochastic "
              "when both arms are close) - try a larger --n.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline", default="data/models/thompson_model_baseline_temporal.json",
        help="path to the baseline model json (a snapshot trained on less data, "
             "chronologically earlier — does not affect the production model)"
    )
    parser.add_argument(
        "--canary-model", default="data/models/thompson_model_v2.json",
        help="path to the retrained candidate model json"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="also hit a running API (python src/datathon/api/app.py) with real requests"
    )
    parser.add_argument(
        "--n", type=int, default=400,
        help="number of /canary/recommend calls for --live (default: 400)"
    )
    parser.add_argument(
        "--canary-percentage", type=int, default=50,
        help="traffic split for this demo run - not the production 5%% default, "
             "raised here so both sides get enough samples quickly (default: 50)"
    )
    args = parser.parse_args()

    static_comparison(REPO_ROOT / args.baseline, REPO_ROOT / args.canary_model)

    if args.live:
        run_live_demo(args.n, args.canary_percentage, args.canary_model, args.baseline)


if __name__ == "__main__":
    main()
