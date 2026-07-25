"""
compare_perproduct_vs_pooled.py
==================================
Pulls the matching pooled baselines out of results_v2_all.csv (same
agg_mode/missing/tranche conditions the per-heading run used) and puts
them side by side with the per-heading experiment's numbers.

Usage:
    python compare_perproduct_vs_pooled.py [results_v2_all.csv]
"""
import sys
import pandas as pd

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else 'results_v2_all.csv'

# hardcode the per-heading run's own printed output -- update these 4 lines
# if you rerun the experiment and get different numbers
PERHEAD = {
    'GAT no-GDELT':             {'log_r2': 0.324593, 'spearman': 0.612200, 'n': 123777},
    'GAT concat/topk':          {'log_r2': 0.319752, 'spearman': 0.593359, 'n': 123777},
    'EdgeGAT_full attn/ka_ka':  {'log_r2': 0.341239, 'spearman': 0.604824, 'n': 123777},
    'EdgeGAT_full blend/tk_tk': {'log_r2': 0.337088, 'spearman': 0.592681, 'n': 123777},
}

# (label, filter function) -- each pulls the matching pooled row out of results_v2_all.csv
POOLED_FILTERS = {
    'GAT no-GDELT':             lambda r: (r.model=='GAT') & (~r.gdelt),
    'GAT concat/topk':          lambda r: (r.model=='GAT') & (r.gdelt) & (r.scoring=='topk') & (r.fusion=='concat'),
    'EdgeGAT_full attn/ka_ka':  lambda r: (r.model=='EdgeGAT_full') & (r.fusion=='attention') & (r.scoring=='ka_ka'),
    'EdgeGAT_full blend/tk_tk': lambda r: (r.model=='EdgeGAT_full') & (r.fusion=='blend') & (r.scoring=='tk_tk'),
}


def load_pooled(path):
    r = pd.read_csv(path)
    r = r.drop_duplicates(
        ['model', 'fusion', 'scoring', 'gdelt', 'risk_level', 'agg_mode', 'missing', 'tranche'],
        keep='last'
    )
    return r[(r.tranche == 'all') & (r.agg_mode == 'first') & (r.missing == 'rf')]


def main():
    pooled = load_pooled(IN_PATH)

    rows = []
    for label, filt in POOLED_FILTERS.items():
        sub = pooled[filt(pooled)]
        if len(sub) == 0:
            print(f'  [warn] no pooled match found for "{label}" -- check the filter / file')
            continue
        if len(sub) > 1:
            print(f'  [warn] {len(sub)} pooled rows matched "{label}", using the first one -- verify this is right')
        p = sub.iloc[0]
        h = PERHEAD[label]
        rows.append({
            'model': label,
            'pooled_log_r2': p['log_r2'], 'perhead_log_r2': h['log_r2'], 'delta_log_r2': h['log_r2'] - p['log_r2'],
            'pooled_spearman': p['spearman'], 'perhead_spearman': h['spearman'], 'delta_spearman': h['spearman'] - p['spearman'],
            'pooled_n': p['n'], 'perhead_n': h['n'],
        })

    out = pd.DataFrame(rows)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print('\n=== Pooled (all_products, no per-heading head) vs. per-heading experiment ===')
    print(out.to_string(index=False))

    out.to_csv('perproduct_vs_pooled.csv', index=False)
    print('\nsaved -> perproduct_vs_pooled.csv')


if __name__ == '__main__':
    main()
