"""
analyze_feature_importance.py
================================
Aggregates feature_importance_all.csv (or any of the per-dataset files)
across every model that tested each feature, since any single model's
ranking is noisy (n_repeats=3, and correlated features split credit
unpredictably between siblings). Produces one verdict per feature:
KEEP / REMOVE / INVESTIGATE.

Usage:
    python analyze_feature_importance.py feature_importance_all.csv
"""
import sys
import pandas as pd
import numpy as np

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else 'feature_importance_all.csv'

# refYear is constant in the 2023-only eval set -> permutation importance
# can't measure it (shuffling a constant column does nothing). Exclude it
# from verdicts rather than let it masquerade as "definitely useless".
EXCLUDE_FROM_VERDICT = {'refYear', 'node:refYear'}

REMOVE_THRESHOLD = 0.01     # mean importance below this -> not pulling weight
NEGATIVE_FRACTION_FLAG = 0.4  # if this share of runs show negative importance -> flag


def clean_feature_name(f):
    return f.split(':', 1)[-1] if ':' in f else f


def main():
    df = pd.read_csv(IN_PATH)
    df = df[~df['feature'].isin(EXCLUDE_FROM_VERDICT)].copy()
    df['feature_clean'] = df['feature'].apply(clean_feature_name)

    agg = df.groupby('feature_clean').agg(
        n_runs=('importance_log_r2', 'size'),
        mean_log_r2=('importance_log_r2', 'mean'),
        std_log_r2=('importance_log_r2', 'std'),
        min_log_r2=('importance_log_r2', 'min'),
        max_log_r2=('importance_log_r2', 'max'),
        mean_spearman=('importance_spearman', 'mean'),
        frac_negative=('importance_log_r2', lambda s: (s < 0).mean()),
        frac_positive=('importance_log_r2', lambda s: (s > REMOVE_THRESHOLD).mean()),
    ).round(4)

    def verdict(row):
        if row['mean_log_r2'] <= REMOVE_THRESHOLD and row['frac_positive'] < 0.3:
            return 'REMOVE — consistently near-zero/negative across models'
        if row['std_log_r2'] > abs(row['mean_log_r2']) and row['frac_negative'] >= NEGATIVE_FRACTION_FLAG:
            return 'INVESTIGATE — sign flips a lot; likely correlated w/ another feature'
        if row['mean_log_r2'] > REMOVE_THRESHOLD:
            return 'KEEP — consistently positive contribution'
        return 'INVESTIGATE — weak/unclear signal'

    agg['verdict'] = agg.apply(verdict, axis=1)
    agg = agg.sort_values('mean_log_r2', ascending=False)

    pd.set_option('display.width', 160)
    print(f"\n=== Feature importance, aggregated across {df['model'].nunique()} models "
          f"x {df['dataset'].nunique()} dataset(s) ===\n")
    print(agg[['n_runs', 'mean_log_r2', 'std_log_r2', 'frac_negative', 'verdict']].to_string())

    print('\n=== By verdict ===')
    for v in ['KEEP — consistently positive contribution',
              'INVESTIGATE — sign flips a lot; likely correlated w/ another feature',
              'INVESTIGATE — weak/unclear signal',
              'REMOVE — consistently near-zero/negative across models']:
        feats = agg[agg.verdict == v].index.tolist()
        if feats:
            print(f'\n{v}:')
            for f in feats:
                print(f'  - {f}  (mean={agg.loc[f,"mean_log_r2"]:+.4f}, seen in {int(agg.loc[f,"n_runs"])} runs)')

    print(f"\nNote: '{EXCLUDE_FROM_VERDICT}' excluded — constant in the 2023-only eval set,")
    print("so permutation importance can't measure it (not evidence it's unused).")

    # per-dataset breakdown, to check whether a verdict holds in BOTH datasets
    # (a feature that's REMOVE in one dataset but KEEP in another is not safe to drop globally)
    if df['dataset'].nunique() > 1:
        print('\n=== Cross-dataset consistency check (mean importance per dataset) ===')
        pivot = df.groupby(['feature_clean', 'dataset'])['importance_log_r2'].mean().unstack()
        pivot = pivot.reindex(agg.index)
        print(pivot.round(4).to_string())


if __name__ == '__main__':
    main()
