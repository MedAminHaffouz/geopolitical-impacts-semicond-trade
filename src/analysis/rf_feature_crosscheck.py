#!/usr/bin/env python3
"""
rf_feature_crosscheck.py
===========================
Zero-cost cross-check: your RF models (already trained, sitting in
trained_models/) have a built-in feature_importances_ attribute
(impurity-based -- how much each feature reduces prediction error when
used as a tree split, averaged across all trees and all estimators). No
retraining, no permutation, no forward passes -- just reading an
attribute off a pickle you already have.

This is methodologically DIFFERENT from permutation importance (impurity
vs. perturbation), which is exactly why it's useful: if RF's built-in
ranking and your GNN's permutation-importance ranking agree, that's
convergent validity across two structurally unrelated model families --
a much stronger claim than either method alone, "for free."

Compares against feature_importance_shrinkage_summary.csv (or any of your
other feature_importance_*.csv files) via Spearman rank correlation
between the two rankings.

Usage:
    python rf_feature_crosscheck.py [trained_models_dir] [feature_importance_summary.csv]
"""
import os, sys, glob, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else 'trained_models'
PERM_IMPORTANCE_PATH = sys.argv[2] if len(sys.argv) > 2 else 'feature_importance_shrinkage_summary.csv'

BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']

# RF model filename -> whether it was trained with GDELT features (determines feat_cols)
# extend this dict if you have RF models under other names (e.g. agg_mode='spec' variants)
RF_FEATCOLS = {
    'RF_first_none':    BASE_COLS,
    'RF_first_keepall': BASE_COLS + GDELT_COLS,
    'RF_first_topk':    BASE_COLS + GDELT_COLS,
    'RF_spec_none':     BASE_COLS,
    'RF_spec_keepall':  BASE_COLS + GDELT_COLS,
    'RF_spec_topk':     BASE_COLS + GDELT_COLS,
}


def load_rf_importances():
    rows = []
    found = glob.glob(os.path.join(MODEL_DIR, 'RF_*.pkl'))
    if not found:
        print(f"No RF_*.pkl files found in {MODEL_DIR}/ -- nothing to cross-check.")
        return pd.DataFrame()

    print(f"Found {len(found)} RF model(s) in {MODEL_DIR}/:")
    for path in found:
        name = os.path.splitext(os.path.basename(path))[0]
        feat_cols = RF_FEATCOLS.get(name)
        if feat_cols is None:
            print(f"  [skip] {name}: not in RF_FEATCOLS mapping -- add it at the top of this script "
                  f"if this is a real model you want included.")
            continue
        with open(path, 'rb') as f:
            rf = pickle.load(f)
        if not hasattr(rf, 'feature_importances_'):
            print(f"  [skip] {name}: loaded object has no feature_importances_ (not a fitted RF?)")
            continue
        importances = rf.feature_importances_
        if len(importances) != len(feat_cols):
            print(f"  [skip] {name}: {len(importances)} importances but {len(feat_cols)} expected feat_cols "
                  f"-- RF_FEATCOLS mapping is probably wrong for this model, fix before trusting this run.")
            continue
        print(f"  [ok]   {name}: {len(feat_cols)} features")
        for feat, imp in zip(feat_cols, importances):
            rows.append({'model': name, 'feature': feat, 'rf_importance': imp})
    return pd.DataFrame(rows)


def main():
    rf_df = load_rf_importances()
    if rf_df.empty:
        return

    rf_summary = rf_df.groupby('feature')['rf_importance'].agg(['mean', 'std', 'count']).round(5)
    rf_summary.columns = ['rf_mean', 'rf_std', 'rf_n_models']
    rf_summary = rf_summary.sort_values('rf_mean', ascending=False)

    print("\n=== RF built-in (impurity-based) feature importance, aggregated across RF models ===")
    print(rf_summary.to_string())

    if not os.path.exists(PERM_IMPORTANCE_PATH):
        print(f"\n[note] {PERM_IMPORTANCE_PATH} not found -- showing RF importance alone, "
              f"no cross-check against permutation importance possible this run.")
        rf_summary.to_csv('rf_feature_importance.csv')
        print("saved -> rf_feature_importance.csv")
        return

    perm_df = pd.read_csv(PERM_IMPORTANCE_PATH)
    # this file's index column after read_csv is 'feature_clean' if it came straight from
    # feature_importance_shrinkage_summary.csv -- handle both that and a raw per-model file
    if 'feature_clean' in perm_df.columns:
        perm_summary = perm_df.set_index('feature_clean')[['mean_log_r2']]
    elif 'feature' in perm_df.columns:
        perm_df = perm_df.copy()
        perm_df['feature_clean'] = perm_df['feature'].apply(lambda f: f.split(':', 1)[-1] if ':' in f else f)
        perm_summary = perm_df.groupby('feature_clean')['importance_log_r2'].mean().to_frame('mean_log_r2')
    else:
        print(f"\n[warn] {PERM_IMPORTANCE_PATH} doesn't look like either expected format -- "
              f"showing RF importance alone.")
        rf_summary.to_csv('rf_feature_importance.csv')
        return

    combined = rf_summary.join(perm_summary, how='outer')
    combined = combined.sort_values('rf_mean', ascending=False)

    print("\n=== Side by side: RF (impurity) vs. permutation importance (log-R2 drop) ===")
    print(combined.to_string())

    both = combined.dropna(subset=['rf_mean', 'mean_log_r2'])
    if len(both) >= 3:
        rho, pval = spearmanr(both['rf_mean'], both['mean_log_r2'])
        print(f"\n=== Convergent validity check ===")
        print(f"Spearman rank correlation between RF ranking and permutation-importance ranking: "
              f"rho={rho:.3f} (p={pval:.4f}), n={len(both)} shared features")
        if rho > 0.6:
            print(">>> Strong agreement between two structurally unrelated model families -- "
                  "this materially strengthens any feature-importance claim in your writeup.")
        elif rho > 0.3:
            print(">>> Moderate agreement -- the two methods broadly agree on direction but diverge "
                  "on some features; worth naming which ones disagree rather than only reporting rho.")
        else:
            print(">>> Weak/no agreement -- worth investigating before citing either ranking as reliable; "
                  "the two methods may be picking up genuinely different things (RF impurity is biased "
                  "toward high-cardinality/continuous features, for instance).")

        print("\n=== Features where the two methods disagree most (by rank) ===")
        both_ranked = both.copy()
        both_ranked['rf_rank'] = both_ranked['rf_mean'].rank(ascending=False)
        both_ranked['perm_rank'] = both_ranked['mean_log_r2'].rank(ascending=False)
        both_ranked['rank_gap'] = (both_ranked['rf_rank'] - both_ranked['perm_rank']).abs()
        print(both_ranked.sort_values('rank_gap', ascending=False)[['rf_rank','perm_rank','rank_gap']].to_string())

    combined.to_csv('rf_vs_permutation_importance.csv')
    print("\nsaved -> rf_vs_permutation_importance.csv")


if __name__ == '__main__':
    main()
