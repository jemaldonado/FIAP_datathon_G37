"""
Contextual Thompson Sampling Implementation

Thompson Sampling with Beta-Bernoulli distribution.
Supports contextual bandits with context = (age_group, job_category).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def assign_arm(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct the 4 "campaign strategy" arms from real historical columns.

    The UCI bank-marketing data only records 2 real contact channels
    (`contact`: cellular/telephone) plus how many times this customer was
    contacted in the current campaign (`campaign`). There are no real
    Email/SMS/Premium channels in the data.

    To get a 4-arm bandit demo out of a 2-channel dataset, we split each
    channel by first-contact vs. repeat-contact and relabel the 4 resulting
    real segments with business-friendly names:

        contact=cellular,  campaign==1  -> arm 0 (Cellular_Standard)
        contact=cellular,  campaign>1   -> arm 1 (Email_Campaign)
        contact=telephone, campaign==1  -> arm 2 (SMS_Alert)
        contact=telephone, campaign>1   -> arm 3 (Call_Premium)

    Every conversion (`y`) counted per arm is a 100% real observed outcome
    for that exact row — nothing here is simulated or estimated. Only the
    ARM NAMES are a relabeling convenience for the demo narrative; treat
    Email_Campaign/SMS_Alert/Call_Premium as "cellular repeat-contact" /
    "telephone first-contact" / "telephone repeat-contact" segments, not as
    channels that were actually tested in production.

    This is the single source of truth for arm assignment — used by the
    training pipeline (`scripts/retrain_model.py`) and by anything that
    needs to reproduce real per-(context, arm) rates, such as the canary
    simulation ground truth in `src/datathon/api/app.py`.
    """
    cellular = df['contact'] == 'cellular'
    first_contact = df['campaign'] == 1
    return pd.Series(
        np.select(
            [cellular & first_contact, cellular & ~first_contact, ~cellular & first_contact],
            [0, 1, 2],
            default=3,
        ),
        index=df.index,
    )


class BetaBernoulliBandit:
    """Thompson Sampling with Beta-Bernoulli distribution"""

    def __init__(self, n_arms: int, alpha_init: Optional[np.ndarray] = None,
                 beta_init: Optional[np.ndarray] = None):
        self.n_arms = n_arms
        self.alpha = np.array(alpha_init, dtype=float) if alpha_init is not None else np.ones(n_arms)
        self.beta = np.array(beta_init, dtype=float) if beta_init is not None else np.ones(n_arms)
        self.successes = np.zeros(n_arms)
        self.trials = np.zeros(n_arms)

    def select_arm(self) -> int:
        """Thompson: sample from Beta distribution and choose best"""
        thetas = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return int(np.argmax(thetas))

    def update(self, arm: int, reward: int) -> None:
        """Update posterior after observing reward"""
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.trials[arm] += 1
        self.successes[arm] += reward

    def get_stats(self) -> Dict:
        """Return arm statistics"""
        return {
            'alpha': self.alpha.copy(),
            'beta': self.beta.copy(),
            'trials': self.trials.copy(),
            'successes': self.successes.copy()
        }

    def get_conversion_rates(self) -> np.ndarray:
        """Return estimated conversion rate for each arm"""
        return self.alpha / (self.alpha + self.beta)


class ContextualThompsonSampling:
    """
    Thompson Sampling with context awareness (age_group × job_category).

    Implements 4 campaign strategies (arms), reconstructed from the 2 real
    contact channels × first-contact vs. repeat-contact — see
    `assign_arm()` in this module for exactly how and why. Every
    conversion counted per arm is a 100% real observed `y`; only the arm
    names below are a relabeling convenience:
    - Arm 0: Cellular_Standard — cellular, first contact
    - Arm 1: Email_Campaign    — cellular, repeat contact
    - Arm 2: SMS_Alert         — telephone, first contact
    - Arm 3: Call_Premium      — telephone, repeat contact

    This contextual bandit (Thompson × context) allows discovery of which
    STRATEGY works best for each CUSTOMER SEGMENT (age_group × job_category).
    """

    AGE_GROUPS = {
        'Young': (18, 30),
        'Prime': (30, 45),
        'Mature': (45, 60),
        'Senior': (60, 150),
    }

    JOB_CATEGORIES = {
        'Technical': ['admin', 'technician', 'engineer', 'scientist', 'services'],
        'Business': ['management', 'director', 'entrepreneur', 'self-employed'],
        'Other': ['student', 'unemployed', 'retired', 'unknown', 'housemaid', 'operator'],
    }

    ARM_NAMES = {
        0: 'Cellular_Standard',
        1: 'Email_Campaign',
        2: 'SMS_Alert',
        3: 'Call_Premium'
    }

    def __init__(self):
        """Initialize 12 Thompson Samplings (4 age_groups × 3 job_categories)"""
        self.bandits = {}
        for age_group in self.AGE_GROUPS:
            for job_cat in self.JOB_CATEGORIES:
                context = (age_group, job_cat)
                self.bandits[context] = BetaBernoulliBandit(n_arms=4)

    def _get_age_group(self, age: int) -> str:
        """Map age to age group"""
        for group, (min_age, max_age) in self.AGE_GROUPS.items():
            if min_age <= age < max_age:
                return group
        return 'Senior'

    def _get_job_category(self, job: str) -> str:
        """Map job to job category"""
        job_lower = job.lower()
        for category, jobs in self.JOB_CATEGORIES.items():
            if any(j in job_lower for j in jobs):
                return category
        return 'Other'

    def select_arm(self, age: int, job: str) -> Tuple[int, Tuple[str, str]]:
        """Select arm for a customer, returns (arm_id, context)"""
        age_group = self._get_age_group(age)
        job_category = self._get_job_category(job)
        context = (age_group, job_category)

        bandit = self.bandits[context]
        arm = bandit.select_arm()

        return arm, context

    def update(self, context: Tuple[str, str], arm: int, reward: int) -> None:
        """Update Thompson Sampling for given context"""
        if context not in self.bandits:
            raise ValueError(f"Unknown context: {context}")
        self.bandits[context].update(arm, reward)

    def get_context_stats(self, context: Tuple[str, str]) -> Dict:
        """Get statistics for a specific context"""
        if context not in self.bandits:
            raise ValueError(f"Unknown context: {context}")

        bandit = self.bandits[context]
        return {
            'context': context,
            'arms': {
                self.ARM_NAMES[i]: {
                    'trials': int(bandit.trials[i]),
                    'successes': int(bandit.successes[i]),
                    'rate': float(bandit.successes[i] / (bandit.trials[i] + 1e-10))
                }
                for i in range(4)
            }
        }


def compute_baseline_vs_thompson(df: pd.DataFrame, target_col: str = 'y',
                                  arm_col: str = 'contact', seed: int = 42) -> Dict:
    """
    Etapa 3 comparison: fixed-rule baseline conversion rate vs a non-contextual
    Thompson Sampling simulation, on the real (unmodified) training data.

    Arms are the real channels observed in `arm_col` (cellular/telephone).
    Each arm's true conversion probability is measured directly from the data
    (no synthetic/estimated rates); Thompson Sampling picks an arm per round
    and the reward is drawn from that arm's measured rate, then used to update
    the posterior. This matches notebooks/datathon_main.ipynb, so the numbers
    logged to MLflow match the ones shown in the baseline notebook.
    """
    baseline_rate = float(df[target_col].mean())

    arms = sorted(df[arm_col].unique())
    arm_true_rates = df.groupby(arm_col)[target_col].mean().to_dict()
    arm_index = {arm: i for i, arm in enumerate(arms)}
    true_rates = [float(arm_true_rates[arm]) for arm in arms]

    np.random.seed(seed)
    bandit = BetaBernoulliBandit(len(arms))
    for _ in range(len(df)):
        arm = bandit.select_arm()
        reward = 1 if np.random.random() < true_rates[arm] else 0
        bandit.update(arm, reward)

    thompson_rate = float(bandit.successes.sum() / len(df))

    return {
        'baseline_rate': baseline_rate,
        'thompson_rate': thompson_rate,
        'improvement': thompson_rate - baseline_rate,
        'beat_baseline': int(thompson_rate > baseline_rate),
        'arm_true_rates': {str(arm): rate for arm, rate in arm_true_rates.items()},
        'arm_trials': {str(arm): int(bandit.trials[idx]) for arm, idx in arm_index.items()},
    }
