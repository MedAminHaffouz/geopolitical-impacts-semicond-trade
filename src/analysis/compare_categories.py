"""
compare_categories.py
======================
Tests the "focusing on one product category helps" hypothesis by comparing
best-achieved log-R^2 / Spearman across your per-category results_v2_*.csv files.

Common ground across ALL your per-category runs is GAT (no-GDELT + concat
keepall/topk) and EdgeGAT_full (4 fusion/scorer combos) — those are what
your restricted driver trains everywhere, so they're the fair comparison.
RF/TabTF/GCN/GATv2 are reported too, but only where a given category
happened to also run the fuller grid (e.g. electronics).

Usage:
    python compare_categories.py
"""
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# Edit this: one entry per category you've trained
# --------------------------------------------------------------------------
CATEGORY_FILES = {
    'medical':     'results_v2_medical.csv',
    'vehicles':    'results_v2_vehicles.csv',
    'oil':         'results_v2_oil.csv',
    'electronics': 'results_v2_electronics.csv',
    # 'semiconductors': 'results_v2.csv',   # <- add once you have this file again
}

# Hardcoded fallback for semiconductors, read off the dashboard you already
# generated, since no raw CSV for it was included this round. Delete this
# block once you add the real file above.
MANUAL_REFERENCE = {
    'semiconductors': {'gat_no_gdelt': 0.261, 'gat_gdelt_best': 0.298, 'edgegat_full_best': 0.390},
}

# All-products pooled baseline (RF, agg_mode='spec', from your main dashboard) —
# the number the whole hypothesis is being measured against.
ALL_PRODUCTS_RF_BASELINE = 0.564


def load(path):
    r = pd.read_csv(path)
    r = r.drop_duplicates(
        ['model', 'fusion', 'scoring', 'gdelt', 'risk_level', 'agg_mode', 'missing', 'tranche'],
        keep='last'
    )
    return r[r.tranche == 'all']


def summarize(df):
    out = {}
    gat = df[df.model == 'GAT']
    if len(gat):
        no_g = gat[~gat.gdelt]['log_r2'].max()
        with_g = gat[gat.gdelt]['log_r2'].max()
        out['gat_no_gdelt'] = no_g
        out['gat_gdelt_best'] = with_g

    edge = df[df.model == 'EdgeGAT_full']
    if len(edge):
        out['edgegat_full_best'] = edge['log_r2'].max()

    if 'RF' in df.model.values:
        rf = df[df.model == 'RF']
        out['rf_no_gdelt'] = rf[~rf.gdelt]['log_r2'].max()
        out['rf_gdelt_best'] = rf[rf.gdelt]['log_r2'].max()

    return out


def main():
    rows = {}
    for cat, path in CATEGORY_FILES.items():
        try:
            df = load(path)
        except FileNotFoundError:
            print(f'  [skip] {path} not found')
            continue
        rows[cat] = summarize(df)

    for cat, vals in MANUAL_REFERENCE.items():
        if cat not in rows:
            rows[cat] = vals

    table = pd.DataFrame(rows).T
    table = table.sort_values('gat_gdelt_best', ascending=False)

    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print('\n=== Per-category comparison (log-R², "all" tranche) ===')
    print(table)

    print(f"\nReference — pooled all-products RF baseline: {ALL_PRODUCTS_RF_BASELINE:.3f} log-R²")
    print("(the number specialization needs to beat, for RF; for GAT/EdgeGAT_full")
    print(" use the category's own no-GDELT/GDELT numbers above as the within-model comparison)")

    print('\n=== Verdict per category (does specializing help vs. that category’s own no-GDELT GAT?) ===')
    for cat in table.index:
        no_g = table.loc[cat].get('gat_no_gdelt', np.nan)
        best = table.loc[cat].get('gat_gdelt_best', np.nan)
        edge_best = table.loc[cat].get('edgegat_full_best', np.nan)
        if pd.isna(no_g) or pd.isna(best):
            continue
        lift = best - no_g
        edge_lift = (edge_best - no_g) if pd.notna(edge_best) else np.nan
        tag = 'GDELT helps' if lift > 0.01 else ('GDELT flat/hurts' if lift < -0.01 else 'GDELT ~neutral')
        print(f'  {cat:14s}  GAT no-GDELT={no_g:.3f}  GAT+GDELT best={best:.3f} ({tag}, Δ={lift:+.3f})'
              + (f'  EdgeGAT_full={edge_best:.3f} (Δ vs no-GDELT={edge_lift:+.3f})' if pd.notna(edge_best) else ''))

    print('\n=== Does category-specific training beat the pooled all-products baseline? ===')
    for cat in table.index:
        best_score = np.nanmax([table.loc[cat].get(c, np.nan) for c in
                                 ['gat_no_gdelt', 'gat_gdelt_best', 'edgegat_full_best', 'rf_gdelt_best', 'rf_no_gdelt']])
        verdict = 'YES — specialization wins' if best_score > ALL_PRODUCTS_RF_BASELINE else 'no — pooled RF still wins'
        print(f'  {cat:14s}  best log-R² = {best_score:.3f}   ->  {verdict}')


if __name__ == '__main__':
    main()