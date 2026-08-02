#!/usr/bin/env python3
"""
predict_hs4_corridors.py
=========================
Loads the 4 models trained by train_hs4_selected.py and evaluates them on the
sparse-year files (2015/2016/2024/2025), filtered to EXACTLY the two HS4 rows
(8541, 8542) per reporter/partner/year -- consistent with how those models
were trained (HS4-collapsed, sum-aggregated).

CONFIG YOU MUST STILL VERIFY (I don't have test_bilateral_pairs.py's actual
COUNTRY_PAIRS dict in front of me, so CORRIDORS below is my best guess from
standard UN Comtrade M49 codes, not confirmed against your files):
  - CORRIDORS: reporterCode/partnerCode pairs. USA=842, China=156, and Taiwan
    is typically reported under code 490 ("Other Asia, nes") in UN Comtrade --
    if your earlier South_Korea_to_China corridor used reporter=410/partner=156
    (confirmed working in your existing scripts), these two should follow the
    same pattern; double check against whatever lookup you already have.

SPARSE_FILES below are confirmed against tree.txt -- these are the two real
files (not one combined file, which was my earlier wrong guess):
  data/interim/all_products_2015_2016_ready_gravity.parquet
  data/interim/all_products_2024_2025_ready_gravity.parquet

For each HS code (8541, 8542):
  - one CSV: hs{code}_predictions.csv, all corridors x all 4 models x all years
  - two plots: one per corridor (US_to_China, China_to_Taiwan), log-scale
    actual-vs-predicted, all 4 models overlaid as separate lines so you can
    compare them directly on one chart per corridor.

Usage:
    python predict_hs4_corridors.py

Not runnable in this sandbox (no real data, no DGL/torch environment with your
trained weights). Run locally after train_hs4_selected.py finishes.
"""
import os, sys, pickle
import numpy as np, pandas as pd
import torch
import dgl
import matplotlib.pyplot as plt

# reuse everything from the training script instead of duplicating it
from train_hs4_selected import (
    GATRegressionModel, FusedEdgeModel, MISSING_COLS, BASE_COLS, GDELT_COLS,
    FULL_NODE_COLS, KEEPALL, TOPK, BILAT_KEEP, BILAT_TOPK, build_agg,
    edge_features_full, MODEL_DIR, log,
)

# ---- CONFIG: verify these against your actual data before trusting output ----
SPARSE_FILES = [
    'data/processed/all_products_2015_2016_ready_gravity.parquet',
    'data/processed/all_products_2024_2025_ready_gravity.parquet',
]
HS_CODES = [8541, 8542]
CORRIDORS = {
    'US_to_China':      dict(reporter=842, partner=156),  # VERIFY -- see docstring
    'China_to_Taiwan':  dict(reporter=156, partner=490),   # VERIFY -- see docstring
}
MODELS = ['HS4_GAT_none', 'HS4_GAT_topk_concat', 'HS4_EdgeGAT_attention_ka_tk', 'HS4_EdgeGAT_blend_tk_ka']
MODEL_COLORS = {
    'HS4_GAT_none': '#4c8fbd', 'HS4_GAT_topk_concat': '#c47f3e',
    'HS4_EdgeGAT_attention_ka_tk': '#2ca02c', 'HS4_EdgeGAT_blend_tk_ka': '#888888',
}
OUT_DIR = 'results/hs4_corridor_outputs'
os.makedirs(OUT_DIR, exist_ok=True)


def load_sparse(paths):
    cols = ['refYear', 'reporterCode', 'partnerCode', 'cmdCode',
            'gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist', 'primaryValue']
    parts = []
    for p in paths:
        if not os.path.exists(p):
            log(f"  WARNING: {p} not found -- skipping (check the path if this is unexpected)")
            continue
        d = pd.read_parquet(p, columns=cols)
        d['_source_file'] = os.path.basename(p)
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"none of {paths} were found -- check SPARSE_FILES paths")
    df = pd.concat(parts, ignore_index=True)
    for c in ['gdpcap_o', 'gdpcap_d', 'dist', 'pop_o', 'pop_d', 'primaryValue',
              'refYear', 'cmdCode', 'reporterCode', 'partnerCode']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['gdpcap_o'] /= 1e6
    df['gdpcap_d'] /= 1e6
    df['dist'] /= 1e3
    df = df[df['cmdCode'].notna()].copy()
    df['cmdCode'] = df['cmdCode'].astype(int)
    # The HS4 collapse (// 100) only applies to genuinely 6-digit source codes.
    # The 2015/2016/2024/2025 sparse files are ALREADY 4-digit-only (confirmed
    # earlier in this project) -- applying // 100 to an already-4-digit code like
    # 8541 gives 85 (2-digit chapter level), which is why nothing matched. Check
    # the actual scale present and only collapse if it's really 6-digit.
    if df['cmdCode'].max() >= 100000:
        log("  cmdCode looks 6-digit (max >= 100000) -- applying HS4 collapse (// 100)")
        df['cmdCode'] = df['cmdCode'] // 100
    else:
        log(f"  cmdCode already HS4-scale (max={df['cmdCode'].max()}) -- leaving as-is, NOT dividing by 100")
    return df


def diagnose_available_pairs(df):
    """Prints every (reporterCode, partnerCode) pair that actually has HS 8541/8542
    trade in the sparse files, ranked by total trade value. Use this to find the
    REAL codes for US/China/Taiwan instead of guessing -- cross-check against
    test_bilateral_pairs.py's COUNTRY_PAIRS or src/utils/country_codes.py, which
    you already have locally and I don't."""
    hs = df[df['cmdCode'].isin(HS_CODES)]
    if hs.empty:
        log("  DIAGNOSTIC: no rows at all for cmdCode in [8541, 8542] in these files -- "
            "check the HS4 collapse or that these files actually contain chip trade.")
        return
    top = (hs.groupby(['reporterCode', 'partnerCode'])['primaryValue']
             .sum().sort_values(ascending=False).head(20))
    log("  DIAGNOSTIC: top 20 (reporterCode, partnerCode) pairs by HS8541/8542 trade value "
        "in the sparse files -- match these against your real country codes:")
    for (rep, part), val in top.items():
        log(f"    reporter={rep:.0f}  partner={part:.0f}  total_value=${val:,.0f}")


def filter_hs_corridor(df, hs_code, reporter, partner):
    """EXACTLY the two HS4 rows, per your instruction -- 'pass 2 rows: 8541 and 8542'."""
    m = ((df['reporterCode'] == reporter) & (df['partnerCode'] == partner) & (df['cmdCode'] == hs_code))
    return df[m].copy()


def load_bundle(name):
    scaler_path = os.path.join(MODEL_DIR, f'{name}_scalers.pkl')
    weight_path = os.path.join(MODEL_DIR, f'{name}.pt')
    with open(scaler_path, 'rb') as f:
        b = pickle.load(f)
    return b['sf'], b['st'], b['cmap'], weight_path


def build_model(name):
    if name == 'HS4_GAT_none':
        return GATRegressionModel(len(BASE_COLS))
    if name == 'HS4_GAT_topk_concat':
        return GATRegressionModel(len(BASE_COLS) + len(GDELT_COLS))
    n_trade = len(FULL_NODE_COLS) - len(GDELT_COLS)
    n_gdelt = len(GDELT_COLS)
    ei = 5  # BILAT_COLS(4) + dist(1)
    if name == 'HS4_EdgeGAT_attention_ka_tk':
        return FusedEdgeModel(n_trade, n_gdelt, ei, fusion='attention')
    if name == 'HS4_EdgeGAT_blend_tk_ka':
        return FusedEdgeModel(n_trade, n_gdelt, ei, fusion='blend')
    raise ValueError(name)


def predict_one(name, df_year, imp):
    """Predict a single model on a single (already HS/corridor-filtered) year slice."""
    sf, st, cmap, weight_path = load_bundle(name)
    model = build_model(name)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    d = df_year.copy()
    d[MISSING_COLS] = imp.transform(d[MISSING_COLS])  # SAME fitted imputer as training -- fixes Taiwan/etc gdpcap
    d['nID'] = d['reporterCode'].map(cmap)
    d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    if len(d) == 0:
        return None, None, 'reporter/partner not in this model\'s training node map (cmap) -- ' \
                            'model never saw this country as a node during training'
    d['nID'] = d['nID'].astype(int)
    d['pID'] = d['pID'].astype(int)

    if name in ('HS4_GAT_none', 'HS4_GAT_topk_concat'):
        use_gdelt = name == 'HS4_GAT_topk_concat'
        gdelt_file = TOPK if use_gdelt else None
        feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
        agg = build_agg(d, use_gdelt, gdelt_file)
        if len(agg) == 0:
            return None, None, 'empty after aggregation'
        Xe = sf.transform(agg[feat_cols])
        ye = st.transform(agg[['y_log']])
        g = dgl.graph((d['nID'].to_numpy(), d['pID'].to_numpy()), num_nodes=max(d['nID'].max(), d['pID'].max()) + 1)
        # NOTE: for a single-corridor slice this graph is trivially small (1-2 nodes) --
        # that's expected here since we deliberately filtered to one reporter/partner/HS
        # combo. GAT still runs fine on a tiny graph, it just has almost no neighborhood
        # to attend over -- worth remembering when interpreting these specific numbers.
        Xfull = np.zeros((g.num_nodes(), Xe.shape[1]), dtype='float32')
        for i, rc in enumerate(agg['reporterCode']):
            rows = d[d['reporterCode'] == rc]  # FIXED: compare against reporterCode, not the mapped
            if len(rows):                       # node-index column (that was always false, hence the
                Xfull[int(rows['nID'].iloc[0])] = Xe[i]   # constant zero-input predictions)
        g.ndata['feat'] = torch.tensor(Xfull, dtype=torch.float32)
        g = dgl.add_self_loop(g)
        with torch.no_grad():
            ft = g.ndata['feat']
            # Both HS4_GAT_none and HS4_GAT_topk_concat are plain GATRegressionModel
            # trained on ONE concatenated feature tensor (see train_gat_none /
            # train_gat_topk_concat in train_hs4_selected.py -- both call model(g, ft),
            # never a split xt/xg call). The split pattern is only for blend/attention
            # fusion variants, which don't exist among these 4 models. Always single-arg.
            out = model(g, ft)
        yp_all = np.expm1(st.inverse_transform(out.view(-1, 1).numpy()).flatten())
        yt = np.expm1(st.inverse_transform(ye).flatten())
        yp = yp_all[agg['reporterCode'].map(lambda rc: int(d[d['reporterCode'] == rc]['nID'].iloc[0])).to_numpy()]
        return float(yt[0]) if len(yt) else None, float(yp[0]) if len(yp) else None, None

    # EdgeGAT variants
    node_file = KEEPALL if 'ka' in name.split('_')[-2:][0] or name.endswith('ka_tk') else TOPK
    pair_file = BILAT_TOPK if name.endswith('ka_tk') else BILAT_KEEP
    agg = build_agg(d, True, node_file)
    if len(agg) == 0:
        return None, None, 'empty after aggregation'
    Xe = sf.transform(agg[FULL_NODE_COLS])
    ye = st.transform(agg[['y_log']])
    ef = MinMaxScaler_transform_or_fit(edge_features_full(d, pair_file))
    g = dgl.graph((d['nID'].to_numpy(), d['pID'].to_numpy()), num_nodes=max(d['nID'].max(), d['pID'].max()) + 1)
    Xfull = np.zeros((g.num_nodes(), Xe.shape[1]), dtype='float32')
    for i, rc in enumerate(agg['reporterCode']):
        rows = d[d['reporterCode'] == rc]
        if len(rows):
            Xfull[int(rows['nID'].iloc[0])] = Xe[i]
    g.ndata['feat'] = torch.tensor(Xfull, dtype=torch.float32)
    g.edata['ef'] = torch.tensor(ef, dtype=torch.float32)
    g = dgl.add_self_loop(g, fill_data=0.)
    with torch.no_grad():
        out = model(g, g.ndata['feat'], g.edata['ef'])
    yp_all = np.expm1(st.inverse_transform(out.view(-1, 1).numpy()).flatten())
    yt = np.expm1(st.inverse_transform(ye).flatten())
    idxs = agg['reporterCode'].map(lambda rc: int(d[d['reporterCode'] == rc]['nID'].iloc[0])).to_numpy()
    yp = yp_all[idxs]
    return float(yt[0]) if len(yt) else None, float(yp[0]) if len(yp) else None, None


def MinMaxScaler_transform_or_fit(arr):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit_transform(arr)


def main():
    log(f"loading sparse-year files: {SPARSE_FILES}")
    df = load_sparse(SPARSE_FILES)
    with open(os.path.join(MODEL_DIR, 'HS4_imputer.pkl'), 'rb') as f:
        imp = pickle.load(f)
    years = sorted(df['refYear'].dropna().unique().astype(int))
    log(f"years present in sparse file: {years}")

    diagnose_available_pairs(df)

    # sanity check: if NEITHER configured corridor exists in the data, stop here
    # instead of burning time producing 8 empty results again.
    any_found = False
    for corridor, cfg in CORRIDORS.items():
        for hs in HS_CODES:
            if len(filter_hs_corridor(df, hs, cfg['reporter'], cfg['partner'])) > 0:
                any_found = True
    if not any_found:
        log("  STOPPING: none of the configured CORRIDORS codes matched anything in the "
            "diagnostic above. Update CORRIDORS at the top of this file with the correct "
            "reporterCode/partnerCode values from the diagnostic list, then rerun.")
        return

    for hs in HS_CODES:
        rows = []
        for corridor, cfg in CORRIDORS.items():
            for yr in years:
                yr_df = filter_hs_corridor(df[df['refYear'] == yr], hs, cfg['reporter'], cfg['partner'])
                if len(yr_df) == 0:
                    log(f"  HS{hs} {corridor} {yr}: no rows after filter (reporter/partner/HS not present that year)")
                    continue
                for model_name in MODELS:
                    yt, yp, err = predict_one(model_name, yr_df, imp)
                    if err:
                        log(f"  HS{hs} {corridor} {yr} {model_name}: SKIPPED -- {err}")
                        continue
                    rows.append(dict(hs_code=hs, corridor=corridor, year=yr, model=model_name,
                                      actual=yt, predicted=yp))
        OUT_COLS = ['hs_code', 'corridor', 'year', 'model', 'actual', 'predicted']
        out = pd.DataFrame(rows, columns=OUT_COLS)
        csv_path = os.path.join(OUT_DIR, f'hs{hs}_predictions.csv')
        out.to_csv(csv_path, index=False)
        log(f"saved {csv_path} ({len(out)} rows)")

        for corridor in CORRIDORS:
            sub = out[out['corridor'] == corridor]
            if sub.empty:
                log(f"  HS{hs} {corridor}: nothing to plot (all rows skipped above)")
                continue
            all_years = sorted(df['refYear'].dropna().unique().astype(int))  # full sparse-file range: 2015/16/24/25
            fig, ax = plt.subplots(figsize=(9, 5.5))

            actual_line = sub.drop_duplicates('year').sort_values('year')
            actual_full = actual_line.set_index('year').reindex(all_years)['actual']  # NaN for absent years -> real gap
            ax.plot(all_years, actual_full, color='#555555', marker='o', lw=2, label='actual')

            for model_name in MODELS:
                d = sub[sub['model'] == model_name].sort_values('year')
                if d.empty:
                    continue
                d_full = d.set_index('year').reindex(all_years)['predicted']
                ax.plot(all_years, d_full, marker='o', lw=1.8,
                         color=MODEL_COLORS[model_name], label=model_name.replace('HS4_', ''))

            present_years = sorted(sub['year'].unique())
            missing_years = sorted(set(all_years) - set(present_years))
            gap_note = (f'no prediction for: {", ".join(map(str, missing_years))} '
                        f'(shown as a break, not interpolated)') if missing_years else 'all sparse years present'
            ax.text(0.01, -0.14, gap_note, transform=ax.transAxes, fontsize=7, color='#888888')
            ax.text(0.01, -0.19,
                    '2017-2023 not included here -- this script evaluates only the sparse post-hoc '
                    'files (2015/2016/2024/2025), not the training-year split.',
                    transform=ax.transAxes, fontsize=7, color='#888888')

            ax.set_yscale('log')
            ax.set_xlabel('year')
            ax.set_xticks(all_years)
            ax.set_xticklabels([str(y) for y in all_years])
            ax.set_xlim(min(all_years) - 0.5, max(all_years) + 0.5)
            ax.set_ylabel('trade value (USD, log scale)')
            ax.set_title(f'HS{hs} \u2014 {corridor} \u2014 all 4 HS4-retrained models', loc='left', fontweight='bold')
            ax.legend(fontsize=8)
            fig.tight_layout()
            png_path = os.path.join(OUT_DIR, f'hs{hs}_{corridor}.png')
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log(f"  saved {png_path}")

    log("=== DONE -- send back hs4_corridor_outputs/ contents (2 CSVs, 4 PNGs) ===")


if __name__ == "__main__":
    main()