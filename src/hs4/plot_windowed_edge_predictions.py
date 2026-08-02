#!/usr/bin/env python3
"""
plot_windowed_edge_predictions.py
====================================
Reads windowed_edge_predictions.csv (produced by the updated
train_windowed_edge_hs4.py -- NOT the old version, which never saved
actual/predicted pairs) and builds actual-vs-predicted charts, one per
(model, corridor), matching the log10-line style from earlier in this chat.

Each fold in the CSV only covers ONE test_year (that's what "windowed" means
-- train_end=2019 only ever produces a prediction FOR 2020, not the whole
range). So for a single continuous line across years, this script uses each
fold's prediction for ITS OWN test_year -- i.e. the 2020 point comes from the
train_end=2019 fold, the 2021 point comes from the train_end=2020 fold, etc.
That's the correct way to stitch a walk-forward multi-year line: each point
is that fold's genuine one-step-ahead prediction, not a single model's
extrapolation across many years.

Usage:
    python plot_windowed_edge_predictions.py [windowed_edge_predictions.csv]

Not runnable in this sandbox -- no real data here yet (the CSV doesn't exist
until you rerun the updated training script). Run locally once you have it.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff',
    'axes.edgecolor': '#b0b0b0', 'axes.labelcolor': '#000000',
    'text.color': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000',
    'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': 0.8, 'font.size': 11,
})

OUT_DIR = 'results/windowed_edge_plots'
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_LABELS = {
    'HS4Edge_GAT_none': 'GAT (no GDELT)',
    'HS4Edge_GAT_topk_concat': 'GAT + GDELT (concat)',
    'HS4Edge_EdgeGAT_attention_ka_tk': 'EdgeGAT + GDELT (attention)',
    'HS4Edge_EdgeGAT_blend_tk_ka': 'EdgeGAT + GDELT (blend)',
}


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results/evaluation/windowed_edge_predictions.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. This file is only produced by the UPDATED "
              f"train_windowed_edge_hs4.py -- if you're running the old version, rerun "
              f"the updated one first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} rows from {csv_path}")
    print(f"models: {sorted(df['model'].unique())}")
    print(f"corridors: {sorted(df['corridor'].unique())}")
    print(f"test_years covered: {sorted(df['test_year'].unique())}")

    for model in df['model'].unique():
        for corridor in df['corridor'].unique():
            sub = df[(df['model'] == model) & (df['corridor'] == corridor)]
            if sub.empty:
                continue
            # sum both HS codes per test_year -- one point per year, matching
            # the "combined trade value" framing used earlier in this chat
            agg = sub.groupby('test_year')[['actual', 'predicted']].sum().reset_index()
            agg = agg.sort_values('test_year')
            agg['actual_log'] = np.log10(agg['actual'].clip(lower=1))
            agg['predicted_log'] = np.log10(agg['predicted'].clip(lower=1))

            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(agg['test_year'], agg['actual_log'], marker='s', markersize=7, lw=2,
                    color='#1F4E78', label='actual')
            ax.plot(agg['test_year'], agg['predicted_log'], marker='D', markersize=6, lw=2,
                    color='#ED7D31', label='predicted')

            ax.set_ylim(0, 12)
            ax.set_ylabel('log-scale trade value')
            ax.set_xlabel('year (each point = that fold\'s one-step-ahead prediction)')
            ax.set_xticks(agg['test_year'].astype(int))
            label = MODEL_LABELS.get(model, model)
            ax.set_title(f'{corridor.replace("_", "-")} trade value \u2014 {label} (windowed)', fontsize=12)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fname = os.path.join(OUT_DIR, f'windowed_{model}_{corridor}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'saved {fname}')

    print(f"=== DONE -- plots in {OUT_DIR}/ ===")


if __name__ == "__main__":
    main()
