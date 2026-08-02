#!/usr/bin/env python3
"""
analyze_convergence_old_vs_new.py
=====================================
Supports the paper's Section 7.1 gradient-interference discussion with actual
per-epoch training-dynamics evidence, not just final-metric comparisons.

Reads the epoch traces already produced by:
    src/retraining/retrain_track_metrics_old_approach.py
        -> results/grids/results_reseed_epoch_trace_OLD_<model>_<chips|all_products_ready>.csv
    src/retraining/retrain_track_metrics.py
        -> results/grids/results_reseed_epoch_trace_<model>_shrinkage.csv

For each of the two model families (GAT, EdgeGAT_full), compares three conditions:
    1. old approach, chips-only          -- no cross-category gradient interference possible
    2. old approach, pooled all-products -- classical pooling, interference possible
    3. new approach, pooled + shrinkage  -- interference decoupled via per-heading alpha

If condition 2 shows more regressions / volatility than conditions 1 and 3, that's
checkable evidence for (not proof of) the gradient-interference mechanism the paper
proposes -- distinct from just "the new approach scores higher at the end."

Metrics computed per condition:
    convergence_epoch    -- first epoch reaching 90% of that run's own final log-R2
    regression_frac       -- fraction of epochs where log-MSE got WORSE than the previous epoch
    loss_volatility        -- std of epoch-to-epoch %-change in train_loss (smoothness)
    final_log_r2 / final_log_mse -- plateau quality, for context

Outputs:
    results/grids/results_convergence_summary.csv
    One independent figure per (model family x metric): train_loss, log_r2, log_mse
    curves across the three conditions.

Usage:
    python src/analysis/analyze_convergence_old_vs_new.py \\
        [--gat_model GAT_first_topk_blend] [--edge_model EdgeGAT_full_attention_ka_ka]
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff',
    'axes.edgecolor': '#000000', 'axes.labelcolor': '#000000',
    'text.color': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000',
    'axes.grid': True, 'grid.color': '#000000', 'grid.alpha': 0.15, 'font.size': 10,
})
BLUE, ORANGE, POS, NEG, PURPLE, GRAY = '#4c8fbd', '#c47f3e', '#2ca02c', '#d62728', '#9467bd', '#888888'
CONDITION_COLORS = {
    'old (chips-only)': GRAY,
    'old (pooled, classical)': NEG,
    'new (pooled + shrinkage)': POS,
}

GRID_DIR = 'results/grids'

def find_csv(fname):
    for d in (GRID_DIR, '.'):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

def condition_paths(model_name):
    return {
        'old (chips-only)':          find_csv(f'results_reseed_epoch_trace_OLD_{model_name}_chips.csv'),
        'old (pooled, classical)':   find_csv(f'results_reseed_epoch_trace_OLD_{model_name}_all_products_ready.csv'),
        'new (pooled + shrinkage)':  find_csv(f'results_reseed_epoch_trace_{model_name}_shrinkage.csv'),
    }

REQUIRED_COLUMNS = {'log_mse', 'log_r2', 'train_loss'}

def validate_trace(df, path):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"[skip] {path} is missing column(s) {sorted(missing)}.")
        print(f"       This is almost always a STALE file from an older version of the training "
              f"script (run before log_mse/log_r2 were added to its output). Delete this file and "
              f"re-run the training script that produces it to regenerate it with the current columns:")
        print(f"         rm {path}")
        return False
    return True

def convergence_metrics(df, model_name, condition):
    df = df.sort_index()
    final_log_r2 = df['log_r2'].iloc[-1]
    final_log_mse = df['log_mse'].iloc[-1]

    # convergence speed: first epoch reaching 90% of this run's own final log-R2
    # (only meaningful if the run actually ends up with positive log-R2)
    conv_epoch = np.nan
    if final_log_r2 > 0:
        target = 0.9 * final_log_r2
        hit = df.index[df['log_r2'] >= target]
        conv_epoch = int(hit.min()) if len(hit) else np.nan

    # regression frequency: fraction of epochs where log_mse got WORSE than the previous epoch
    log_mse_diff = df['log_mse'].diff().dropna()
    regression_frac = (log_mse_diff > 0).mean() if len(log_mse_diff) else np.nan

    # loss volatility: std of epoch-to-epoch %-change in train_loss
    loss_pct_change = df['train_loss'].pct_change().dropna()
    loss_volatility = loss_pct_change.std() if len(loss_pct_change) else np.nan

    return dict(model=model_name, condition=condition, n_epochs=len(df),
                convergence_epoch=conv_epoch, regression_frac=regression_frac,
                loss_volatility=loss_volatility, final_log_r2=final_log_r2, final_log_mse=final_log_mse)

FIGURES_DIR = 'figures'

def plot_family(model_name, traces):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    saved = []
    for metric, ylabel, better in [('train_loss', 'training loss (MSE, scaled log-space)', 'lower'),
                                    ('log_r2', 'log-R\u00b2 (test set)', 'higher'),
                                    ('log_mse', 'log-MSE (test set)', 'lower')]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for condition, df in traces.items():
            if metric not in df.columns:
                continue
            ax.plot(df.index, df[metric], label=condition, color=CONDITION_COLORS[condition], linewidth=1.6)
        ax.set_xlabel('epoch'); ax.set_ylabel(f'{ylabel} ({better} = better)' if metric != 'train_loss' else ylabel)
        ax.set_title(f'{model_name}: {metric} per epoch, old vs new approach', loc='left', fontweight='bold')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f'convergence_{model_name}_{metric}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gat_model', default='GAT_first_topk_blend')
    ap.add_argument('--edge_model', default='EdgeGAT_full_attention_ka_ka')
    args = ap.parse_args()

    all_rows = []
    for model_name in [args.gat_model, args.edge_model]:
        paths = condition_paths(model_name)
        traces = {}
        for condition, path in paths.items():
            if path is None:
                print(f"[skip] {model_name} / {condition} -- trace not found "
                      f"(expected results_reseed_epoch_trace_{'OLD_' + model_name + '_...' if 'old' in condition else model_name + '_shrinkage'}.csv)")
                continue
            df = pd.read_csv(path, index_col=0)
            if not validate_trace(df, path):
                continue
            traces[condition] = df
            all_rows.append(convergence_metrics(df, model_name, condition))

        if len(traces) >= 2:
            saved_figs = plot_family(model_name, traces)
            for p in saved_figs:
                print(f"  saved figure -> {p}")
        else:
            print(f"[skip plots] {model_name} -- need at least 2 of the 3 conditions to compare, found {len(traces)}")

    if not all_rows:
        print("No traces found at all -- run retrain_track_metrics.py and "
              "retrain_track_metrics_old_approach.py first."); return

    summary = pd.DataFrame(all_rows).set_index(['model', 'condition'])
    os.makedirs(GRID_DIR, exist_ok=True)
    out_path = os.path.join(GRID_DIR, 'results_convergence_summary.csv')
    summary.to_csv(out_path)
    print(f"\nsaved -> {out_path}")
    print(summary.round(4).to_string())

    print("\n=== reading for the paper ===")
    for model_name in summary.index.get_level_values('model').unique():
        sub = summary.loc[model_name]
        if 'old (pooled, classical)' in sub.index and 'new (pooled + shrinkage)' in sub.index:
            old_reg = sub.loc['old (pooled, classical)', 'regression_frac']
            new_reg = sub.loc['new (pooled + shrinkage)', 'regression_frac']
            old_vol = sub.loc['old (pooled, classical)', 'loss_volatility']
            new_vol = sub.loc['new (pooled + shrinkage)', 'loss_volatility']
            print(f"{model_name}: classical pooling regressed on {old_reg:.1%} of epochs vs "
                  f"{new_reg:.1%} for shrinkage; loss volatility {old_vol:.4f} vs {new_vol:.4f}.")

if __name__ == "__main__":
    main()