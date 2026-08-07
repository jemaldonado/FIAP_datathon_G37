#!/usr/bin/env python
"""
Simple training script - saves only data, not classes.
This avoids pickle/joblib serialization issues with custom classes.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from datathon.bandit import ContextualThompsonSampling

# Create data directory
data_dir = Path(__file__).parent.parent / "data" / "models"
data_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TRAINING CONTEXTUAL THOMPSON SAMPLING - SAVE ONLY DATA")
print("=" * 70)

# Load data
parquet_file = Path(__file__).parent.parent / "data" / "processed" / "bank_marketing_primary.parquet"
print(f"\nLoading data from {parquet_file}...")
df = pd.read_parquet(parquet_file)
print(f"Loaded {len(df):,} customers")

# Create model for context mapping
model = ContextualThompsonSampling()

# Add age_group and job_category
df['age_group'] = df['age'].apply(model._get_age_group)
df['job_category'] = df['job'].apply(model._get_job_category)

# Assign arms
df['arm'] = df.apply(lambda row: (0 if row['contact'] == 'cellular' else 2) if row['campaign'] == 1 else (1 if row['contact'] == 'cellular' else 3), axis=1)

# Train Thompson Sampling
print(f"\nTraining on {len(df):,} customers...")
np.random.seed(42)
for idx, row in df.iterrows():
    age_group = row['age_group']
    job_cat = row['job_category']
    context = (age_group, job_cat)
    arm = int(row['arm'])
    reward = int(row['y'])

    model.update(context, arm, reward)

    if (idx + 1) % 10000 == 0:
        print(f"  Processed {idx + 1:,} customers")

# Extract data from all bandits and save as JSON
print(f"\nExtracting model data...")
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

# Save as JSON
model_file = data_dir / "thompson_model.json"
print(f"Saving to {model_file}...")
with open(model_file, 'w') as f:
    json.dump(model_data, f, indent=2)

print(f"OK: Model saved ({model_file.stat().st_size / 1024:.1f} KB)")
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
