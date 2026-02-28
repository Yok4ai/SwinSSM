"""
Statistical significance tests for model comparison

Provides paired t-tests to determine if performance differences between models
are statistically significant, particularly when Dice scores show minimal differences.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def paired_ttest(scores_a: List[float], scores_b: List[float],
                 alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Perform paired t-test between two sets of scores.

    Args:
        scores_a: Scores from model A (e.g., baseline)
        scores_b: Scores from model B (e.g., proposed method)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        dict with t_statistic, p_value, mean_diff, std_diff, significant (p<0.05)
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length for paired t-test")

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
    diff = scores_b - scores_a

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'significant': p_value < 0.05,
        'n_samples': len(scores_a)
    }


def wilcoxon_test(scores_a: List[float], scores_b: List[float],
                  alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Non-parametric Wilcoxon signed-rank test for paired samples.
    Use when normality assumption is violated.

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        dict with statistic, p_value, significant (p<0.05)
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    diff = scores_b - scores_a

    return {
        'statistic': stat,
        'p_value': p_value,
        'median_diff': np.median(diff),
        'significant': p_value < 0.05,
        'n_samples': len(scores_a)
    }


def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for mean score.

    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def compare_models_dice(model_results: Dict[str, Dict[str, List[float]]],
                        baseline_name: str = 'SwinUNETR') -> Dict[str, Dict]:
    """
    Compare multiple models against a baseline using paired t-tests on Dice scores.

    Args:
        model_results: Dict mapping model_name -> {'dice': [...], 'tc': [...], 'wt': [...], 'et': [...]}
        baseline_name: Name of baseline model

    Returns:
        Dict with comparison results for each model vs baseline
    """
    if baseline_name not in model_results:
        raise ValueError(f"Baseline {baseline_name} not found in results")

    baseline = model_results[baseline_name]
    comparisons = {}

    for model_name, scores in model_results.items():
        if model_name == baseline_name:
            continue

        comparisons[model_name] = {
            'mean_dice': {
                'ttest': paired_ttest(baseline['dice'], scores['dice']),
                'baseline_mean': np.mean(baseline['dice']),
                'model_mean': np.mean(scores['dice']),
                'improvement': np.mean(scores['dice']) - np.mean(baseline['dice'])
            }
        }

        # Per-region comparisons
        for region in ['tc', 'wt', 'et']:
            if region in baseline and region in scores:
                comparisons[model_name][f'{region}_dice'] = {
                    'ttest': paired_ttest(baseline[region], scores[region]),
                    'baseline_mean': np.mean(baseline[region]),
                    'model_mean': np.mean(scores[region]),
                    'improvement': np.mean(scores[region]) - np.mean(baseline[region])
                }

    return comparisons


def format_comparison_table(comparisons: Dict[str, Dict],
                           significance_level: float = 0.05) -> str:
    """
    Format comparison results as a table string.

    Args:
        comparisons: Output from compare_models_dice
        significance_level: p-value threshold for significance marker

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Statistical Significance Analysis (Paired t-test)")
    lines.append("=" * 80)
    lines.append(f"{'Model':<20} {'Metric':<12} {'Δ Mean':<10} {'p-value':<12} {'Sig?':<6}")
    lines.append("-" * 80)

    for model_name, metrics in comparisons.items():
        for metric_name, results in metrics.items():
            ttest = results['ttest']
            improvement = results['improvement']
            p_val = ttest['p_value']
            sig = '*' if p_val < significance_level else ''

            lines.append(
                f"{model_name:<20} {metric_name:<12} {improvement:+.4f}    "
                f"{p_val:.4e}    {sig:<6}"
            )

    lines.append("=" * 80)
    lines.append(f"* indicates p < {significance_level}")

    return '\n'.join(lines)


def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000,
                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for more robust estimation.

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        (lower_bound, upper_bound) tuple
    """
    scores = np.array(scores)
    n = len(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (lower, upper)


def effect_size_cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Calculate Cohen's d effect size for paired samples.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B

    Returns:
        Cohen's d value
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diff = scores_b - scores_a

    return np.mean(diff) / np.std(diff, ddof=1)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulated Dice scores from different models on same test set
    baseline_dice = np.random.normal(0.85, 0.05, 50)
    proposed_dice = np.random.normal(0.87, 0.05, 50)

    print("Paired t-test results:")
    result = paired_ttest(baseline_dice, proposed_dice)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print(f"\nCohen's d effect size: {effect_size_cohens_d(baseline_dice, proposed_dice):.4f}")

    print(f"\n95% CI for baseline: {compute_confidence_interval(baseline_dice)}")
    print(f"95% CI for proposed: {compute_confidence_interval(proposed_dice)}")
