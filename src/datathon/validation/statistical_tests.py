"""
Statistical Tests Module

Implements rigorous statistical testing for model evaluation.
- Chi-square tests for context significance
- Confidence intervals for conversion rates
- Multiple comparison corrections (Bonferroni)
- Power analysis for sample size calculation

Usage:
    from datathon.validation import compare_conversions, get_confidence_interval

    # Compare two conversion rates
    result = compare_conversions(successes_thompson, trials_thompson,
                                 successes_baseline, trials_baseline)
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant at α=0.05: {result['is_significant']}")
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    message: str
    context: Optional[str] = None


def get_confidence_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for proportion.

    More robust than normal approximation, especially for small samples.

    Args:
        successes: Number of positive outcomes
        trials: Total number of trials
        confidence: Confidence level (default 0.95 = 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 1.0)

    p_hat = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / trials
    centre_adjusted_probability = (p_hat + z**2 / (2 * trials)) / denominator
    adjustment = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator

    lower = max(0, centre_adjusted_probability - adjustment)
    upper = min(1, centre_adjusted_probability + adjustment)

    return (lower, upper)


def compare_conversions(
    successes_group1: int,
    trials_group1: int,
    successes_group2: int,
    trials_group2: int,
    alpha: float = 0.05,
    use_bonferroni: bool = False,
    n_comparisons: int = 1
) -> StatisticalTest:
    """
    Chi-square test for comparing two conversion rates.

    H0: Conversion rates are equal
    H1: Conversion rates are different

    Args:
        successes_group1: Successes in group 1
        trials_group1: Total trials in group 1
        successes_group2: Successes in group 2
        trials_group2: Total trials in group 2
        alpha: Significance level (default 0.05)
        use_bonferroni: Whether to apply Bonferroni correction
        n_comparisons: Number of comparisons (for Bonferroni)

    Returns:
        StatisticalTest with test results
    """
    # Create contingency table
    failures_group1 = trials_group1 - successes_group1
    failures_group2 = trials_group2 - successes_group2

    contingency = np.array([
        [successes_group1, failures_group1],
        [successes_group2, failures_group2]
    ])

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    except Exception as e:
        # If chi-square fails (insufficient data), return conservative result
        logger.warning(f"Chi-square test failed (insufficient data): {e}")
        chi2 = 0.0
        p_value = 1.0  # No significance
        dof = 1
        expected = np.array([[0, 0], [0, 0]])  # Dummy array

    # Apply Bonferroni correction if requested
    effective_alpha = alpha / n_comparisons if use_bonferroni else alpha
    is_significant = p_value < effective_alpha

    rate1 = successes_group1 / trials_group1 if trials_group1 > 0 else 0
    rate2 = successes_group2 / trials_group2 if trials_group2 > 0 else 0

    correction_str = f" (Bonferroni corrected α={effective_alpha:.4f})" if use_bonferroni else ""
    message = (
        f"Group 1: {rate1:.2%} ({successes_group1}/{trials_group1}), "
        f"Group 2: {rate2:.2%} ({successes_group2}/{trials_group2}), "
        f"χ²={chi2:.4f}, p={p_value:.4f}, "
        f"Significant: {'YES' if is_significant else 'NO'}{correction_str}"
    )

    return StatisticalTest(
        test_name='Chi-Square Test',
        statistic=chi2,
        p_value=p_value,
        is_significant=is_significant,
        alpha=effective_alpha,
        message=message
    )


def compare_contexts(
    context_results: Dict[str, Tuple[int, int]],
    baseline_successes: int,
    baseline_trials: int,
    alpha: float = 0.05
) -> Dict[str, StatisticalTest]:
    """
    Compare each context against baseline with multiple comparison correction.

    Args:
        context_results: Dict of {context_name: (successes, trials)}
        baseline_successes: Baseline successes
        baseline_trials: Baseline total trials
        alpha: Significance level

    Returns:
        Dict of {context_name: StatisticalTest}
    """
    n_contexts = len(context_results)
    results = {}

    logger.info(f"Comparing {n_contexts} contexts against baseline")
    logger.info(f"Baseline: {baseline_successes}/{baseline_trials} = {baseline_successes/baseline_trials:.2%}")

    for context_name, (successes, trials) in context_results.items():
        test = compare_conversions(
            baseline_successes, baseline_trials,
            successes, trials,
            alpha=alpha,
            use_bonferroni=True,
            n_comparisons=n_contexts
        )
        test.context = context_name
        results[context_name] = test

        logger.info(f"  {context_name}: {test.message}")

    return results


def calculate_sample_size(
    baseline_rate: float,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size for A/B test.

    Uses the normal approximation for two-proportion tests
    (n = (z_alpha + z_beta)^2 * 2p(1-p) / effect_size^2), computed with
    scipy.stats — no statsmodels dependency required.

    Args:
        baseline_rate: Baseline conversion rate (0-1)
        effect_size: Absolute effect size to detect (e.g., 0.02 for +2pp)
        alpha: Type I error rate (default 0.05)
        power: Statistical power (default 0.80)

    Returns:
        Required sample size per group
    """
    treatment_rate = min(0.999, baseline_rate + effect_size)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p = (baseline_rate + treatment_rate) / 2
    n = ((z_alpha + z_beta) ** 2) * (2 * p * (1 - p)) / (effect_size ** 2)

    return int(np.ceil(n))


def generate_statistical_report(
    thompson_successes: int,
    thompson_trials: int,
    baseline_successes: int,
    baseline_trials: int,
    context_name: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive statistical report.

    Args:
        thompson_successes: Thompson Sampling successes
        thompson_trials: Thompson Sampling trials
        baseline_successes: Baseline successes
        baseline_trials: Baseline trials
        context_name: Name of context (optional)

    Returns:
        Dict with complete statistical analysis
    """
    thompson_rate = thompson_successes / thompson_trials if thompson_trials > 0 else 0
    baseline_rate = baseline_successes / baseline_trials if baseline_trials > 0 else 0

    thompson_ci = get_confidence_interval(thompson_successes, thompson_trials)
    baseline_ci = get_confidence_interval(baseline_successes, baseline_trials)

    comparison = compare_conversions(
        thompson_successes, thompson_trials,
        baseline_successes, baseline_trials
    )

    report = {
        'context': context_name or 'Overall',
        'thompson': {
            'successes': thompson_successes,
            'trials': thompson_trials,
            'rate': float(thompson_rate),
            'ci_lower': float(thompson_ci[0]),
            'ci_upper': float(thompson_ci[1]),
            'ci_width': float(thompson_ci[1] - thompson_ci[0])
        },
        'baseline': {
            'successes': baseline_successes,
            'trials': baseline_trials,
            'rate': float(baseline_rate),
            'ci_lower': float(baseline_ci[0]),
            'ci_upper': float(baseline_ci[1]),
            'ci_width': float(baseline_ci[1] - baseline_ci[0])
        },
        'comparison': {
            'difference_pp': float((thompson_rate - baseline_rate) * 100),
            'p_value': float(comparison.p_value),
            'is_significant': comparison.is_significant,
            'chi_square': float(comparison.statistic),
            'message': comparison.message
        }
    }

    return report
