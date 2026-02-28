"""Paired t-tests comparing models given seed-level mean dice scores.

Usage:
  python paired_ttests.py \
    --model SwinMamba 0.8942 0.8930 0.8929 \
    --model MambaUNETR 0.8923 0.8895 0.8892 \
    --model SwinUNETRPlus 0.8900 0.8861 0.8842
"""

import argparse
import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", nargs="+", action="append", metavar=("NAME", "SCORE"),
        help="Model name followed by seed scores, e.g. --model SwinMamba 0.894 0.893 0.892",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    models = {}
    for entry in args.model:
        name, *scores = entry
        models[name] = np.array([float(s) for s in scores])

    names = list(models.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    print("=" * 50)
    print("Mean ± std across seeds")
    print("=" * 50)
    for name, arr in models.items():
        print(f"  {name:<20}: {arr.mean():.4f} ± {arr.std():.4f}")

    print(f"\nPaired t-test (df={len(list(models.values())[0]) - 1})")
    print(f"{'Comparison':<42} {'t':>8} {'p':>10}")
    print("-" * 62)
    for a, b in pairs:
        t, p = stats.ttest_rel(models[a], models[b])
        sig = "*" if p < 0.05 else ""
        print(f"{a} vs {b:<22} {t:>8.4f} {p:>10.6f}  {sig}")


if __name__ == "__main__":
    main()
