#!/usr/bin/env python3
"""
run_diagnostics.py
==================
Fills in the 4 numbers your supervisor asked for that aren't produced by
train_benchmark.py / run_shrinkage_head.py / run_per_product_head.py:

  1. Number of trainable parameters, per model
  2. Training time (wall-clock), per model
  3. Inference time (wall-clock), per model
  4. Paired Wilcoxon signed-rank test + bootstrap 95% CI comparing your
     final model (EdgeGAT_full + shrinkage head, best config) against the
     best gravity-only baseline (RF, no GDELT)

Design choices, so you can defend them if asked:

  - Wilcoxon signed-rank (not a t-test) on |log1p error| per row, PAIRED
    on (reporterCode, cmdCode). Chosen because trade-value errors are
    heavy-tailed / non-Gaussian (same reason you use log-R2 as primary
    metric elsewhere) -- a paired t-test's normality assumption doesn't
    hold here, but Wilcoxon only assumes symmetry of the *differences*,
    which is far more defensible.
  - Bootstrap CI: percentile method, 2000 resamples with replacement over
    rows, recomputing log_r2 / spearman each time. This is the standard
    non-parametric way to get a CI on metrics whose sampling distribution
    you don't know in closed form.
  - Alignment: RF and the shrinkage-head model are trained on different
    intermediate row sets (edge-model dropna on graph mapping vs RF's
    plain dropna), so predictions are merged on the (reporterCode,
    cmdCode) key before pairing -- only rows present in BOTH sets are
    compared. This is reported as n_pairs so you can see how much of the
    test set that covers.
  - Both models are ACTUALLY RETRAINED here (not loaded from disk) so the
    timing numbers reflect real training cost, using the exact same
    hyperparameters as your existing scripts (lr=0.01, epochs=100,
    mini-batch subgraph pattern). If you only want timing on an already-
    trained model, see the --skip-train-baseline / final-model load path
    noted in the comments below.

Usage (run from ~/electronic_prods_pred/ -- the repo root -- same conda env
as your other scripts):
    python run_diagnostics.py --combo ka_ka --fusion attention

  (data_file defaults to data/processed/all_products_ready.parquet; pass a
  different path positionally if you want to point at another processed file)

  --combo / --fusion should match whichever EdgeGAT_full config your
  results table says is best (i.e. the "final model" you're reporting in
  the paper). Check results/grids/results_shrinkage_head.csv for the
  winning config name before running this.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
import torch
import dgl
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Repo-layout wiring. Run this script from repo root (~/electronic_prods_pred/).
# Three real mismatches vs the actual tree.txt layout, fixed here rather than
# by editing your source files:
#
#   1. train_benchmark.py has a MODULE-LEVEL side effect (line ~170:
#      `train_data, val_data, data_2023 = load_and_split('all_products_ready.parquet')`
#      runs at import time, not inside main()). It looks for the bare filename
#      in the CWD, but the real file lives at data/processed/all_products_ready.parquet.
#      Fix: symlink the bare name into CWD before importing, so the import-time
#      call succeeds (the values it computes there are unused -- we call
#      load_or_split() ourselves afterward with the correct path).
#   2. train_benchmark.py/run_shrinkage_head.py's KEEPALL/TOPK/BILAT_KEEP/
#      BILAT_TOPK constants are bare filenames, but the actual files live
#      under data/interim/. Fixed by monkey-patching the constants (and
#      rebuilding SCORER_COMBOS from them) right after import.
#   3. SPLIT_DIR defaults to root-relative "split_cache", but your split
#      cache actually lives at data/cache/split_cache/ (flat layout, which
#      matches train_benchmark.py's non-tagged convention exactly).
#      Fixed by reassigning tb.SPLIT_DIR after import.
#   4. The scripts themselves live in src/training/, not repo root.
# ---------------------------------------------------------------------------
REPO_ROOT = os.getcwd()
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'training'))
sys.path.insert(0, REPO_ROOT)  # in case you've copied/symlinked the scripts to root instead

_BARE_DATA = 'all_products_ready.parquet'
_REAL_DATA = os.path.join('data', 'processed', 'all_products_ready.parquet')
if not os.path.exists(_BARE_DATA) and os.path.exists(_REAL_DATA):
    os.symlink(os.path.abspath(_REAL_DATA), _BARE_DATA)  # satisfies train_benchmark.py's import-time call

import train_benchmark as tb          # reuses build_agg, load_or_split, compute_metrics, BASE_COLS, MISSING
import run_shrinkage_head as rsh      # reuses run_edge_full_pp, _eval_edge_full_pp, SCORER_COMBOS

# ---- patch stale path constants to match the real repo tree ----
tb.SPLIT_DIR = os.path.join('data', 'cache', 'split_cache')
_INTERIM = os.path.join('data', 'interim')
for mod in (tb, rsh):
    mod.KEEPALL     = os.path.join(_INTERIM, 'gdelt_features_by_country_year.parquet')
    mod.TOPK        = os.path.join(_INTERIM, 'gdelt_features_topk.parquet')
    mod.BILAT_KEEP  = os.path.join(_INTERIM, 'gdelt_bilateral_by_pair_year.parquet')
    mod.BILAT_TOPK  = os.path.join(_INTERIM, 'gdelt_bilateral_topk.parquet')
rsh.SCORER_COMBOS = {
    'ka_ka': (rsh.KEEPALL, rsh.BILAT_KEEP), 'ka_tk': (rsh.KEEPALL, rsh.BILAT_TOPK),
    'tk_ka': (rsh.TOPK, rsh.BILAT_KEEP),    'tk_tk': (rsh.TOPK, rsh.BILAT_TOPK),
}
os.makedirs(os.path.join('results', 'grids'), exist_ok=True)  # for this script's own CSV outputs


# ---------------------------------------------------------------------------
# 1. parameter counting
# ---------------------------------------------------------------------------
def count_parameters(model):
    """Trainable parameter count for any torch.nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_rf(rf):
    """RF has no direct analogue to NN parameters. Reporting tree count and
    total node count instead, clearly labelled as a different kind of
    'size' measure -- do NOT put this in the same column as NN param
    counts in a table without a footnote."""
    n_nodes = sum(t.tree_.node_count for t in rf.estimators_)
    return {'n_trees': len(rf.estimators_), 'total_nodes': int(n_nodes)}


# ---------------------------------------------------------------------------
# 2/3. timing helper
# ---------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


# ---------------------------------------------------------------------------
# baseline: gravity-only RF (no GDELT) -- mirrors tb.run_rf but keeps keys
# and timing instead of just logging the result row.
# ---------------------------------------------------------------------------
def train_and_time_rf_baseline(train_data, data_2023, agg_mode='first'):
    feat_cols = tb.BASE_COLS
    tr = tb.build_agg(train_data, False, None, agg_mode).dropna(subset=feat_cols + ['primaryValue'])
    te = tb.build_agg(data_2023,  False, None, agg_mode).dropna(subset=feat_cols + ['primaryValue'])
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(tr[feat_cols]); Xte = sf.transform(te[feat_cols])
    ytr = st.fit_transform(tr[['y_log']])

    with Timer() as t_train:
        rf = RandomForestRegressor(n_estimators=200, max_depth=25, n_jobs=2, random_state=0)
        rf.fit(Xtr, ytr.ravel())
    with Timer() as t_infer:
        pred = rf.predict(Xte)

    yp = np.expm1(st.inverse_transform(pred.reshape(-1, 1)).flatten())
    yt = np.expm1(st.inverse_transform(st.transform(te[['y_log']])).flatten())
    keys = te[['reporterCode', 'cmdCode']].reset_index(drop=True)

    return dict(name='RF_gravity_baseline', model=rf, yt=yt, yp=yp, keys=keys,
                train_s=t_train.elapsed, infer_s=t_infer.elapsed,
                n_params_nn=None, n_params_rf=count_parameters_rf(rf))


# ---------------------------------------------------------------------------
# GAT-family models (no shrinkage head): covers GAT_first_none (no GDELT) and
# GAT_first_{scoring}_{fusion} (with GDELT). Mirrors tb.run_graph exactly so
# behavior/hyperparameters match your existing pipeline, but keeps keys +
# timing + param count instead of only logging via report_and_log.
# ---------------------------------------------------------------------------
def train_and_time_gat(train_data, data_2023, name, use_gdelt, gdelt_file=None,
                        fusion='concat', agg_mode='first', epochs=100, kind='GAT'):
    # Forced to CPU: tb.DEVICE resolves to 'cuda' whenever torch.cuda.is_available(),
    # but your installed dgl build is CPU-only (no CUDA runtime compiled in), which
    # crashes on graph.to('cuda') with "Device API cuda is not enabled". The rest of
    # this script's models (RF, EdgeGAT_full+shrinkage) never call .to(device) at all
    # -- they're implicitly CPU-only already -- so this keeps all 4 models consistent.
    device = torch.device('cpu')
    feat_cols = tb.BASE_COLS + (tb.GDELT_COLS if use_gdelt else [])
    agg = tb.build_agg(train_data, use_gdelt, gdelt_file, agg_mode)
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nodeID'] = td['reporterCode'].map(cmap)
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(agg[feat_cols]); ytr = st.fit_transform(agg[['y_log']])
    g = dgl.graph((td['nodeID'].to_numpy(), td['partnerCode'].map(cmap).to_numpy()))
    g.ndata['feat'] = torch.tensor(Xtr, dtype=torch.float32)
    g = dgl.add_self_loop(g).to(device)

    fused = fusion in ('blend', 'attention'); nt = len(tb.BASE_COLS)
    if fusion == 'blend':      model = tb.BlendGAT(nt, len(tb.GDELT_COLS))
    elif fusion == 'attention': model = tb.AttnGAT(nt, len(tb.GDELT_COLS))
    elif kind == 'GAT':        model = tb.GATRegressionModel(len(feat_cols))
    elif kind == 'GATv2':      model = tb.GATv2RegressionModel(len(feat_cols))
    else:                       model = tb.GCNRegressionModel(len(feat_cols))
    model = model.to(device)

    ytr_t = torch.tensor(ytr[:, 0], dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01); crit = torch.nn.MSELoss()
    N = g.num_nodes(); bs = 10000; nb = N // bs + (N % bs > 0)

    with Timer() as t_train:
        for ep in range(epochs):
            model.train()
            for i in range(nb):
                bn = list(range(i * bs, min((i + 1) * bs, N)))
                bg = g.subgraph(torch.tensor(bn)); ft = bg.ndata['feat']
                logit = model(bg, ft[:, :nt], ft[:, nt:]) if fused else model(bg, ft)
                loss = crit(logit.view(-1, 1), ytr_t[bn].view(-1, 1))
                opt.zero_grad(); loss.backward(); opt.step()

    with Timer() as t_infer:
        yt, yp = tb._eval_graph(model, data_2023, cmap, sf, st, feat_cols, use_gdelt,
                                 gdelt_file, agg_mode, fusion, nt, device=device)

    # rebuild the same aggregated key table _eval_graph derives internally
    d = data_2023.copy()
    for c in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        d[c] = d[c].fillna(d[c].mean())
    d['nID'] = d['reporterCode'].map(cmap); d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    agg_eval = tb.build_agg(d, use_gdelt, gdelt_file, agg_mode)
    keys = agg_eval[['reporterCode', 'cmdCode']].reset_index(drop=True).iloc[:len(yt)]

    return dict(name=name, model=model, yt=yt, yp=yp, keys=keys,
                train_s=t_train.elapsed, infer_s=t_infer.elapsed,
                n_params_nn=count_parameters(model), n_params_rf=None)



def train_and_time_final_model(train_data, data_2023, combo='ka_ka', fusion='attention',
                                agg_mode='first', epochs=100):
    gfile, bfile = rsh.SCORER_COMBOS[combo]

    with Timer() as t_train:
        model, cmap, sf, st, heading_counts = rsh.run_edge_full_pp(
            train_data, data_2023, combo, gfile, bfile, fusion, agg_mode=agg_mode, epochs=epochs)
    with Timer() as t_infer:
        yt, yp, heading = rsh._eval_edge_full_pp(model, data_2023, cmap, sf, st, gfile, bfile, agg_mode)

    # Reconstruct the same aggregated key table _eval_edge_full_pp built
    # internally, so predictions can be merged against the RF baseline on
    # (reporterCode, cmdCode).
    d = data_2023.copy()
    for c in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        d[c] = d[c].fillna(d[c].mean())
    d['nID'] = d['reporterCode'].map(cmap); d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    agg = tb.build_agg(d, True, gfile, agg_mode)
    keys = agg[['reporterCode', 'cmdCode']].reset_index(drop=True).iloc[:len(yt)]

    return dict(name=f'EdgeGAT_full_{fusion}_{combo}_shrinkage', model=model,
                yt=yt, yp=yp, keys=keys, train_s=t_train.elapsed, infer_s=t_infer.elapsed,
                n_params_nn=count_parameters(model), n_params_rf=None)


# ---------------------------------------------------------------------------
# 4. paired significance test + bootstrap CI
# ---------------------------------------------------------------------------
def align_by_keys(res_a, res_b):
    """Inner-merge two result dicts on (reporterCode, cmdCode) so per-row errors
    are legitimately paired -- only rows present in both prediction sets are
    compared. Returns a DataFrame with yt_a/yp_a/yt_b/yp_b columns."""
    a = res_a['keys'].copy(); a['yt'] = res_a['yt']; a['yp'] = res_a['yp']
    b = res_b['keys'].copy(); b['yt'] = res_b['yt']; b['yp'] = res_b['yp']
    return a.merge(b, on=['reporterCode', 'cmdCode'], suffixes=('_a', '_b'))


def paired_wilcoxon_log_error(merged, name_a, name_b):
    """Wilcoxon signed-rank test on |log1p error|, model a vs model b."""
    err_a = np.abs(np.log1p(merged['yt_a'].clip(lower=0)) - np.log1p(merged['yp_a'].clip(lower=0)))
    err_b = np.abs(np.log1p(merged['yt_b'].clip(lower=0)) - np.log1p(merged['yp_b'].clip(lower=0)))
    stat, p = stats.wilcoxon(err_a, err_b, alternative='two-sided')
    return dict(model_a=name_a, model_b=name_b, n_pairs=len(merged),
                median_abs_log_err_a=float(np.median(err_a)),
                median_abs_log_err_b=float(np.median(err_b)),
                wilcoxon_stat=float(stat), p_value=float(p))


def bootstrap_ci(yt, yp, metric='log_r2', n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI on log_r2 or spearman, resampling rows with
    replacement n_boot times."""
    rng = np.random.default_rng(seed)
    n = len(yt); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = tb.compute_metrics(yt[idx], yp[idx])
        vals.append(m[metric])
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = tb.compute_metrics(yt, yp)[metric]
    return dict(metric=metric, point=float(point), ci_lo=float(lo), ci_hi=float(hi), n_boot=len(vals))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_file', nargs='?', default=os.path.join('data', 'processed', 'all_products_ready.parquet'))
    ap.add_argument('--combo', default='ka_ka', choices=list(rsh.SCORER_COMBOS.keys()),
                     help="scorer combo for the FINAL EdgeGAT_full+shrinkage model")
    ap.add_argument('--fusion', default='attention', choices=['concat', 'blend', 'attention'],
                     help="node fusion for the FINAL EdgeGAT_full+shrinkage model")
    ap.add_argument('--gat-gdelt-scoring', default='keepall', choices=['keepall', 'topk'],
                     help="GDELT scoring for the simple GAT+GDELT comparison model")
    ap.add_argument('--gat-gdelt-fusion', default='concat', choices=['concat', 'blend', 'attention'],
                     help="fusion for the simple GAT+GDELT comparison model (concat = simplest)")
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    tb.log(f"loading {args.data_file}")
    train_data, val_data, data_2023 = tb.load_or_split(args.data_file, missing=tb.MISSING)

    tb.log("training gravity-only RF baseline (timed) ...")
    rf = train_and_time_rf_baseline(train_data, data_2023)

    tb.log("training GAT_first_none (no GDELT, timed) ...")
    gat_none = train_and_time_gat(train_data, data_2023, name='GAT_first_none',
                                   use_gdelt=False, epochs=args.epochs)

    scoring = args.gat_gdelt_scoring; gfusion = args.gat_gdelt_fusion
    gfile = tb.KEEPALL if scoring == 'keepall' else tb.TOPK
    gat_name = f'GAT_first_{scoring}_{gfusion}'
    tb.log(f"training {gat_name} (GAT + GDELT, timed) ...")
    gat_gdelt = train_and_time_gat(train_data, data_2023, name=gat_name,
                                    use_gdelt=True, gdelt_file=gfile, fusion=gfusion, epochs=args.epochs)

    tb.log(f"training final model: EdgeGAT_full_{args.fusion}_{args.combo}_shrinkage (timed) ...")
    final = train_and_time_final_model(train_data, data_2023, combo=args.combo,
                                        fusion=args.fusion, epochs=args.epochs)

    print("\n=== Parameter counts ===")
    print(f"  {rf['name']:32s} n_trees={rf['n_params_rf']['n_trees']}  "
          f"total_nodes={rf['n_params_rf']['total_nodes']:,}  (not comparable to NN param counts)")
    for res in (gat_none, gat_gdelt, final):
        print(f"  {res['name']:32s} trainable_params={res['n_params_nn']:,}")

    print("\n=== Timing (wall-clock, this machine) ===")
    for res in (rf, gat_none, gat_gdelt, final):
        print(f"  {res['name']:32s} train={res['train_s']:.2f}s   infer={res['infer_s']:.4f}s")

    print("\n=== Bootstrap 95% CIs (2000 resamples, percentile method) ===")
    boot_rows = []
    for res in (final, rf, gat_none, gat_gdelt):
        for metric in ['log_r2', 'spearman']:
            ci = bootstrap_ci(res['yt'], res['yp'], metric=metric)
            print(f"  {res['name']:32s} {metric:9s} = {ci['point']:.4f}   95% CI [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
            boot_rows.append(dict(model=res['name'], **ci))

    print("\n=== Paired Wilcoxon signed-rank test (final model vs each baseline, |log error|) ===")
    wilcoxon_rows = []
    for baseline in (rf, gat_none, gat_gdelt):
        merged = align_by_keys(final, baseline)
        wres = paired_wilcoxon_log_error(merged, final['name'], baseline['name'])
        print(f"\n  {final['name']}  vs  {baseline['name']}")
        print(f"    n paired rows              = {wres['n_pairs']:,}  (rows present in both prediction sets)")
        print(f"    median |log err| final     = {wres['median_abs_log_err_a']:.4f}")
        print(f"    median |log err| baseline  = {wres['median_abs_log_err_b']:.4f}")
        print(f"    Wilcoxon statistic         = {wres['wilcoxon_stat']:.1f}")
        print(f"    p-value                    = {wres['p_value']:.3e}")
        print("    -> significant at alpha=0.05" if wres['p_value'] < 0.05
              else "    -> NOT significant at alpha=0.05 -- report this honestly if it's the case.")
        wilcoxon_rows.append(wres)

    # ---- persist everything for the paper/report ----
    summary_rows = [
        dict(model=rf['name'], n_trees=rf['n_params_rf']['n_trees'],
             total_nodes=rf['n_params_rf']['total_nodes'], trainable_params=None,
             train_s=rf['train_s'], infer_s=rf['infer_s']),
    ]
    for res in (gat_none, gat_gdelt, final):
        summary_rows.append(dict(model=res['name'], n_trees=None, total_nodes=None,
                                  trainable_params=res['n_params_nn'],
                                  train_s=res['train_s'], infer_s=res['infer_s']))
    summary = pd.DataFrame(summary_rows)

    out_dir = os.path.join('results', 'grids')
    p1 = os.path.join(out_dir, 'results_diagnostics_size_timing.csv')
    p2 = os.path.join(out_dir, 'results_diagnostics_bootstrap_ci.csv')
    p3 = os.path.join(out_dir, 'results_diagnostics_wilcoxon.csv')
    summary.to_csv(p1, index=False)
    pd.DataFrame(boot_rows).to_csv(p2, index=False)
    pd.DataFrame(wilcoxon_rows).to_csv(p3, index=False)
    tb.log(f"saved -> {p1}, {p2}, {p3}")


if __name__ == "__main__":
    main()