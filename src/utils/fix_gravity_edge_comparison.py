#!/usr/bin/env python3
"""
fix_gravity_edge_comparison.py
=================================
Patches the original_log_r2 / delta_vs_original columns in an already-
trained results_gravity_edge.csv, WITHOUT retraining anything. The bug:
retrain_gravity_edge.py built name = f'{fusion}_{combo}' (e.g.
'concat_ka_ka') but ORIGINAL_OVERALL's keys were written as
'{combo}_{fusion}' (e.g. 'ka_ka_concat') -- a string mismatch that made
every .get(name) call return None, regardless of the real data.

This script joins on the row's own combo/fusion COLUMNS instead of a
reconstructed string, which can't have this class of bug.

Usage:
    python fix_gravity_edge_comparison.py
"""
import pandas as pd
import os

IN_CSV = 'results_gravity_edge.csv'
SIMPLIFIED_CSV = 'results_simplified_edgegat.csv'

# same numbers as before, just no longer string-matched -- indexed by (combo, fusion) tuple instead
ORIGINAL_OVERALL = {
    ('ka_ka','attention'): 0.272, ('ka_tk','attention'): 0.287, ('tk_ka','attention'): 0.339, ('tk_tk','attention'): 0.304,
    ('ka_ka','blend'):     0.319, ('ka_tk','blend'):     0.366, ('tk_ka','blend'):     0.321, ('tk_tk','blend'):     0.338,
    ('ka_ka','concat'):    0.211, ('ka_tk','concat'):    0.335, ('tk_ka','concat'):    0.331, ('tk_tk','concat'):    0.340,
}

def main():
    if not os.path.exists(IN_CSV):
        print(f"{IN_CSV} not found -- nothing to fix.")
        return

    df = pd.read_csv(IN_CSV)
    df['original_log_r2'] = df.apply(lambda r: ORIGINAL_OVERALL.get((r['combo'], r['fusion'])), axis=1)
    df['delta_vs_original'] = df['gravity_log_r2'] - df['original_log_r2']

    if os.path.exists(SIMPLIFIED_CSV):
        simp = pd.read_csv(SIMPLIFIED_CSV)
        simp_map = simp.set_index(['combo', 'fusion'])['simplified_log_r2'].to_dict()
        df['simplified_log_r2'] = df.apply(lambda r: simp_map.get((r['combo'], r['fusion'])), axis=1)
        df['delta_vs_simplified'] = df['gravity_log_r2'] - df['simplified_log_r2']
        print(f"loaded {SIMPLIFIED_CSV} -- full 3-way comparison available")
    else:
        print(f"[note] {SIMPLIFIED_CSV} not found -- run retrain_simplified_edgegat.py to unlock that column")

    df = df.sort_values('delta_vs_original', ascending=False)
    df.to_csv(IN_CSV, index=False)

    pd.set_option('display.width', 160)
    print(f"\nsaved -> {IN_CSV} (corrected in place)\n")
    print(df.to_string(index=False))

    n_beat = (df['delta_vs_original'] > 0).sum()
    print(f"\n{n_beat}/{len(df)} configs: gravity-edge beats the original (crude gdpcap_d/pop_d node features)")
    if 'delta_vs_simplified' in df.columns:
        beats_simplified = (df['delta_vs_simplified'] > 0.01).sum()
        print(f"{beats_simplified}/{len(df)} configs: gravity-edge beats simplified by >0.01 log-R²")


if __name__ == '__main__':
    main()
