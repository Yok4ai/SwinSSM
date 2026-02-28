"""
Run statistical significance tests on model results

Usage:
    python run_statistical_analysis.py --results_file results.json
    python run_statistical_analysis.py --baseline_csv baseline.csv --proposed_csv proposed.csv
"""

import argparse
import json
import numpy as np
from src.utils.statistical_tests import (
    paired_ttest,
    wilcoxon_test,
    compute_confidence_interval,
    compare_models_dice,
    format_comparison_table,
    bootstrap_ci,
    effect_size_cohens_d
)


def load_results_from_json(filepath):
    """Load model results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_results_from_csv(filepath):
    """Load per-sample Dice scores from CSV (one score per line)"""
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                scores.append(float(line))
    return scores


def run_comparison(baseline_scores, proposed_scores, baseline_name='Baseline', proposed_name='Proposed'):
    """Run full statistical comparison between two models"""

    print("=" * 70)
    print(f"STATISTICAL SIGNIFICANCE ANALYSIS")
    print(f"Comparing: {baseline_name} vs {proposed_name}")
    print("=" * 70)

    # Basic stats
    print(f"\n DESCRIPTIVE STATISTICS:")
    print(f"  {baseline_name}: mean={np.mean(baseline_scores):.4f}, std={np.std(baseline_scores):.4f}, n={len(baseline_scores)}")
    print(f"  {proposed_name}: mean={np.mean(proposed_scores):.4f}, std={np.std(proposed_scores):.4f}, n={len(proposed_scores)}")

    # Paired t-test
    print(f"\n PAIRED T-TEST:")
    ttest = paired_ttest(baseline_scores, proposed_scores)
    print(f"  t-statistic: {ttest['t_statistic']:.4f}")
    print(f"  p-value: {ttest['p_value']:.6f}")
    print(f"  Mean difference: {ttest['mean_diff']:.4f}  {ttest['std_diff']:.4f}")
    print(f"  Statistically significant (p<0.05): {'Yes' if ttest['significant'] else 'No'}")

    # Wilcoxon test (non-parametric)
    print(f"\n WILCOXON SIGNED-RANK TEST (non-parametric):")
    wilcox = wilcoxon_test(baseline_scores, proposed_scores)
    print(f"  Statistic: {wilcox['statistic']:.4f}")
    print(f"  p-value: {wilcox['p_value']:.6f}")
    print(f"  Median difference: {wilcox['median_diff']:.4f}")
    print(f"  Statistically significant (p<0.05): {'YES YES' if wilcox['significant'] else 'NO NO'}")

    # Effect size
    print(f"\n EFFECT SIZE:")
    cohens_d = effect_size_cohens_d(baseline_scores, proposed_scores)
    effect_interp = (
        "negligible" if abs(cohens_d) < 0.2 else
        "small" if abs(cohens_d) < 0.5 else
        "medium" if abs(cohens_d) < 0.8 else
        "large"
    )
    print(f"  Cohen's d: {cohens_d:.4f} ({effect_interp})")

    # Confidence intervals
    print(f"\n 95% CONFIDENCE INTERVALS:")
    ci_baseline = compute_confidence_interval(baseline_scores)
    ci_proposed = compute_confidence_interval(proposed_scores)
    print(f"  {baseline_name}: [{ci_baseline[0]:.4f}, {ci_baseline[1]:.4f}]")
    print(f"  {proposed_name}: [{ci_proposed[0]:.4f}, {ci_proposed[1]:.4f}]")

    # Bootstrap CI for difference
    print(f"\n BOOTSTRAP 95% CI FOR DIFFERENCE:")
    diff = np.array(proposed_scores) - np.array(baseline_scores)
    boot_ci = bootstrap_ci(diff.tolist())
    print(f"  Difference CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
    ci_excludes_zero = boot_ci[0] > 0 or boot_ci[1] < 0
    print(f"  CI excludes zero: {'Yes (significant)' if ci_excludes_zero else 'No'}")

    print("\n" + "=" * 70)

    return {
        'ttest': ttest,
        'wilcoxon': wilcox,
        'cohens_d': cohens_d,
        'ci_baseline': ci_baseline,
        'ci_proposed': ci_proposed,
        'bootstrap_ci_diff': boot_ci
    }


def main():
    parser = argparse.ArgumentParser(description='Statistical significance tests for model comparison')
    parser.add_argument('--results_file', type=str, help='JSON file with model results')
    parser.add_argument('--baseline_csv', type=str, help='CSV with baseline per-sample Dice scores')
    parser.add_argument('--proposed_csv', type=str, help='CSV with proposed model per-sample Dice scores')
    parser.add_argument('--baseline_name', type=str, default='SwinUNETR', help='Baseline model name')
    parser.add_argument('--proposed_name', type=str, default='SwinMamba', help='Proposed model name')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    args = parser.parse_args()

    if args.demo:
        # Demo with synthetic data
        print("\n DEMO MODE: Using synthetic data\n")
        np.random.seed(42)
        baseline_scores = np.random.normal(0.850, 0.045, 100).tolist()
        proposed_scores = np.random.normal(0.865, 0.040, 100).tolist()
        run_comparison(baseline_scores, proposed_scores, 'SwinUNETR', 'SwinMamba')

    elif args.results_file:
        # Load from JSON
        results = load_results_from_json(args.results_file)
        if args.baseline_name in results and args.proposed_name in results:
            baseline = results[args.baseline_name]['dice']
            proposed = results[args.proposed_name]['dice']
            run_comparison(baseline, proposed, args.baseline_name, args.proposed_name)
        else:
            print(f"Error: Could not find {args.baseline_name} or {args.proposed_name} in results file")

    elif args.baseline_csv and args.proposed_csv:
        # Load from CSVs
        baseline_scores = load_results_from_csv(args.baseline_csv)
        proposed_scores = load_results_from_csv(args.proposed_csv)
        run_comparison(baseline_scores, proposed_scores, args.baseline_name, args.proposed_name)

    else:
        print("Usage examples:")
        print("  python run_statistical_analysis.py --demo")
        print("  python run_statistical_analysis.py --baseline_csv baseline_dice.csv --proposed_csv proposed_dice.csv")
        print("  python run_statistical_analysis.py --results_file all_results.json")


if __name__ == "__main__":
    main()
