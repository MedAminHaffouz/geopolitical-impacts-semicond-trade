#!/usr/bin/env python3
"""
test_bilateral_pairs.py
------------------------
Evaluates the already-trained models (saved by train_benchmark.py into
trained_models/) on two specific bilateral trade corridors:

    Use case 1 : USA  -> China   (reporterCode=842, partnerCode=156)
    Use case 2 : China -> Taiwan (reporterCode=156, partnerCode=490)

Why 490 for Taiwan: Comtrade does not code Taiwan as its own reporter/partner
entity — it folds it into code 490 ("Other Asia, nes"). This is the same
Taiwan carve-out used throughout the GDELT<->Comtrade ISO3->M49 bridge
(ISO3_2_M49_FIX = {**ISO3_2_M49, "TWN": "490"}). Code 158 is kept as a
fallback in case a given data extract also carries a residual 158 entry.

How this works
---------------
Every model in this pipeline is trained at the (refYear, reporterCode,
cmdCode) NODE level: the target is the reporter's aggregated trade value in
that product-year, aggregated across whichever partners are present in the
frame passed into build_agg(). That means if we hand the eval function a
frame that has ALREADY been filtered down to a single partner, the resulting
per-node aggregate IS the bilateral value for that corridor. So we don't need
new model architectures — we just need to filter the evaluation frame BEFORE
it goes into build_agg()/the existing _eval_* functions, exactly the way
train_benchmark.py already does internally.

Scalers (MinMaxScaler) and the reporter->index map (cmap) are refit here from
split_cache/train.parquet, identically to how train_benchmark.py built them
originally (MinMaxScaler.fit_transform is deterministic given the same data,
so this reproduces the exact scalers used at training time without needing
them to have been pickled separately).

Output
------
A single CSV, bilateral_test_metrics.csv, with one row per
(model, country_pair) combination and columns:
    timestamp, model, kind, use_gdelt, scoring, fusion, agg_mode,
    country_pair, n, mae, rmse, r2, log_r2, spearman, smape, notes

Usage
-----
    python test_bilateral_pairs.py [split_cache_dir] [trained_models_dir] [interim_dir]

    Defaults match the repo layout in tree_repo.txt:
        split_cache_dir    = data/cache/split_cache     (flat train/val/test2023.parquet
                              — the all_products_ready default split, per load_or_split())
        trained_models_dir = models/trained_models      (the base GAT_first_*/EdgeGAT_full_*/
                              RF_first_* set — NOT the *_shrinkage/*_gravity/*_pruned/etc.
                              experiment variants, which live in sibling models/trained_models_*
                              folders and use different feature sets)
        interim_dir        = data/interim               (where the GDELT keepall/topk/bilateral
                              parquet files actually live, not the repo root)

    Run this from the repo root, or pass explicit paths if running elsewhere.
"""

import os
import sys
import csv
import gc
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cpu')  # inference-only eval, CPU is plenty

# ===================== constants (mirrors train_benchmark.py) =====================

GDELT_COLS = ['events_total', 'score_mean', 'score_max', 'score_vol',
              'goldstein_wmean', 'tone_mean', 'active_months']
BASE_COLS  = ['refYear', 'cmdCode', 'dist', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d']
BILAT_COLS = ['pair_events', 'pair_score_mean', 'pair_score_max', 'pair_gold_mean']

# these are joined with interim_dir at runtime (see main()) — the repo puts
# them under data/interim/, not the repo root
KEEPALL_NAME     = 'gdelt_features_by_country_year.parquet'
TOPK_NAME        = 'gdelt_features_topk.parquet'
BILAT_KEEP_NAME  = 'gdelt_bilateral_by_pair_year.parquet'
BILAT_TOPK_NAME  = 'gdelt_bilateral_topk.parquet'

# populated in main() once interim_dir is known
KEEPALL = TOPK = BILAT_KEEP = BILAT_TOPK = None
SCORER_COMBOS = None  # built in main() after paths are resolved

FULL_NODE_COLS = ['refYear', 'cmdCode', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d'] + GDELT_COLS
FULL_EDGE_COLS = BILAT_COLS + ['dist']

# ===================== country pairs under test =====================

# M49 numeric codes: USA=842, China=156, Taiwan (Comtrade "Other Asia,nes")=490
COUNTRY_PAIRS = {
    'US_to_China':     {'reporterCode': 842, 'partnerCode': 156},
    'China_to_Taiwan': {'reporterCode': 156, 'partnerCode': 490},
}
# fallback partner code for Taiwan, tried if 490 yields no rows
TAIWAN_FALLBACK_CODE = 158

RESULTS_FILE = 'bilateral_test_metrics.csv'


def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


# ===================== metrics (mirrors train_benchmark.py) =====================

def _smape(yt, yp):
    d = (np.abs(yt) + np.abs(yp)) / 2
    m = d > 0
    return np.mean(np.abs(yt[m] - yp[m]) / d[m]) * 100 if m.sum() else np.nan


def _spearman(a, b):
    if len(a) < 3:
        return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')


def compute_metrics(yt, yp):
    ytl = np.log1p(np.clip(yt, 0, None))
    ypl = np.log1p(np.clip(yp, 0, None))
    return dict(
        mae=mean_absolute_error(yt, yp),
        rmse=mean_squared_error(yt, yp) ** 0.5,
        r2=r2_score(yt, yp),
        log_r2=(r2_score(ytl, ypl) if len(yt) > 2 else np.nan),
        spearman=_spearman(yt, yp),
        smape=_smape(yt, yp),
    )


def log_result(writer_state, model_name, kind, use_gdelt, scoring, fusion,
               agg_mode, pair_name, met, n, notes=''):
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model': model_name, 'kind': kind, 'use_gdelt': use_gdelt,
        'scoring': scoring, 'fusion': fusion, 'agg_mode': agg_mode,
        'country_pair': pair_name, 'n': int(n),
    }
    if met is not None:
        row.update({
            'mae': round(met['mae'], 2), 'rmse': round(met['rmse'], 2),
            'r2': round(met['r2'], 4), 'log_r2': round(float(met['log_r2']), 4),
            'spearman': round(float(met['spearman']), 4), 'smape': round(met['smape'], 2),
        })
    else:
        row.update({'mae': '', 'rmse': '', 'r2': '', 'log_r2': '', 'spearman': '', 'smape': ''})
    row['notes'] = notes

    new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
    return row


# ===================== feature building (mirrors train_benchmark.py) =====================

def add_gdelt(agg, gdelt_file):
    cy = pd.read_parquet(gdelt_file)
    out = agg.copy()
    out['_r'] = out['reporterCode'].astype('Int64').astype(str)
    out['_y'] = out['refYear'].astype('Int64').astype(str)
    cy['_r'] = cy['reporterCode'].astype('Int64').astype(str)
    cy['_y'] = cy['year'].astype('Int64').astype(str)
    out = out.merge(cy[['_r', '_y'] + GDELT_COLS], on=['_r', '_y'], how='left')
    out[GDELT_COLS] = out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_r', '_y'])


def build_agg(df, use_gdelt, gdelt_file, agg_mode='sum'):
    if agg_mode == 'spec':
        agg = df.groupby(['refYear', 'reporterCode', 'cmdCode']).agg(
            primaryValue=('primaryValue', 'sum'), dist=('dist', 'first'),
            gdpcap_d=('gdpcap_d', 'first'), gdpcap_o=('gdpcap_o', 'first'),
            pop_o=('pop_o', 'first'), pop_d=('pop_d', 'first')).reset_index()
    else:
        grav = 'first' if agg_mode == 'first' else 'sum'
        agg = (df.groupby(['refYear', 'reporterCode', 'cmdCode'])
                 .agg(primaryValue=('primaryValue', 'mean'), dist=('dist', 'first'),
                      gdpcap_d=('gdpcap_d', grav), gdpcap_o=('gdpcap_o', grav),
                      pop_o=('pop_o', grav), pop_d=('pop_d', grav)).reset_index())
    agg['y_log'] = np.log1p(agg['primaryValue'].clip(lower=0))
    if use_gdelt:
        agg = add_gdelt(agg, gdelt_file)
    return agg


def edge_features_full(df_rows, bilat_file):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode', 'partnerCode', 'refYear', 'dist']].copy()
    for c in ['reporterCode', 'partnerCode', 'refYear']:
        k[c] = k[c].astype('Int64')
    b['reporterCode'] = b['reporterCode'].astype('Int64')
    b['partnerCode'] = b['partnerCode'].astype('Int64')
    b['year'] = b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode', 'partnerCode', 'refYear'],
                right_on=['reporterCode', 'partnerCode', 'year'], how='left')
    m[BILAT_COLS] = m[BILAT_COLS].fillna(0)
    m['dist'] = m['dist'].fillna(m['dist'].median())
    return m[FULL_EDGE_COLS].to_numpy(dtype='float32')


# ===================== model classes (mirrors train_benchmark.py) =====================

class GATRegressionModel(nn.Module):
    def __init__(s, inf, h=32, heads=4):
        super().__init__()
        s.c1 = GATConv(inf, h, heads)
        s.c2 = GATConv(h * heads, 1, heads)

    def forward(s, g, x):
        return s.c2(g, torch.relu(s.c1(g, x).flatten(1))).mean(1)


class BlendGAT(nn.Module):
    def __init__(s, nt, ng, proj=16, h=32, heads=4):
        super().__init__()
        s.tp = nn.Linear(nt, proj)
        s.gp = nn.Linear(ng, proj)
        s.alpha = nn.Parameter(torch.tensor(0.5))
        s.c1 = GATConv(proj, h, heads)
        s.c2 = GATConv(h * heads, 1, heads)

    def forward(s, g, xt, xg):
        a = torch.sigmoid(s.alpha)
        f = a * torch.relu(s.gp(xg)) + (1 - a) * torch.relu(s.tp(xt))
        return s.c2(g, torch.relu(s.c1(g, f).flatten(1))).mean(1)


class AttnGAT(nn.Module):
    def __init__(s, nt, ng, proj=16, h=32, heads=4):
        super().__init__()
        s.tp = nn.Linear(nt, proj)
        s.gp = nn.Linear(ng, proj)
        s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.c1 = GATConv(proj, h, heads)
        s.c2 = GATConv(h * heads, 1, heads)

    def forward(s, g, xt, xg):
        st = torch.stack([torch.relu(s.tp(xt)), torch.relu(s.gp(xg))], dim=1)
        f = s.attn(st, st, st)[0].mean(1)
        return s.c2(g, torch.relu(s.c1(g, f).flatten(1))).mean(1)


class EdgeGAT(nn.Module):
    def __init__(s, inf, ef, h=32, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h * heads, ef, 1, heads, allow_zero_in_degree=True)

    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1).squeeze(-1)


class FusedEdgeModel(nn.Module):
    def __init__(s, kind, n_trade, n_gdelt, n_edge, fusion='concat', proj=16, h=32):
        super().__init__()
        s.fusion = fusion
        s.n_trade = n_trade
        if fusion == 'concat':
            inf = n_trade + n_gdelt
        else:
            s.tp = nn.Linear(n_trade, proj)
            s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend':
                s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention':
                s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        s.backbone = EdgeGAT(inf, n_edge, h)  # only EdgeGAT is used by the driver

    def forward(s, g, x, ef):
        if s.fusion == 'concat':
            node = x
        else:
            xt = x[:, :s.n_trade]
            xg = x[:, s.n_trade:]
            t = torch.relu(s.tp(xt))
            d = torch.relu(s.gp(xg))
            if s.fusion == 'blend':
                a = torch.sigmoid(s.alpha)
                node = a * d + (1 - a) * t
            else:
                st = torch.stack([t, d], dim=1)
                node = s.attn(st, st, st)[0].mean(1)
        return s.backbone(g, node, ef)


# ===================== eval functions (mirrors train_benchmark.py) =====================

def _eval_graph(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                agg_mode, fusion, nt, bs=10000):
    d = df_eval.copy()
    for c in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        d[c] = d[c].fillna(d[c].mean())
    d['nID'] = d['reporterCode'].map(cmap)
    d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    if len(d) == 0:
        return None, None
    agg = build_agg(d, use_gdelt, gdelt_file, agg_mode)
    if len(agg) == 0:
        return None, None
    emap = {c: i for i, c in enumerate(agg['reporterCode'])}
    Xe = sf.transform(agg[feat_cols])
    ye = st.transform(agg[['y_log']])
    d['eN'] = d['reporterCode'].map(emap)
    d['eP'] = d['partnerCode'].map(emap)
    d = d.dropna(subset=['eN', 'eP'])
    d['eN'] = d['eN'].astype(int)
    d['eP'] = d['eP'].astype(int)
    eg = dgl.graph((d['eN'].to_numpy(), d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat'] = torch.tensor(Xe, dtype=torch.float32)
    eg = dgl.add_self_loop(eg)
    fused = fusion in ('blend', 'attention')
    model.eval()
    preds = []
    N = eg.num_nodes()
    for i in range(0, N, bs):
        bn = list(range(i, min(i + bs, N)))
        bg = eg.subgraph(torch.tensor(bn))
        ft = bg.ndata['feat']
        with torch.no_grad():
            out = model(bg, ft[:, :nt], ft[:, nt:]) if fused else model(bg, ft)
            preds.append(out.unsqueeze(1))
    yp = np.expm1(st.inverse_transform(torch.cat(preds, 0).view(-1, 1).cpu().numpy()).flatten())
    yt = np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp


def _eval_edge_full(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode, bs=20000):
    d = df_eval.copy()
    for c in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        d[c] = d[c].fillna(d[c].mean())
    d['nID'] = d['reporterCode'].map(cmap)
    d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    if len(d) == 0:
        return None, None
    agg = build_agg(d, True, gdelt_file, agg_mode)
    if len(agg) == 0:
        return None, None
    emap = {c: i for i, c in enumerate(agg['reporterCode'])}
    Xe = sf.transform(agg[FULL_NODE_COLS])
    ye = st.transform(agg[['y_log']])
    d['eN'] = d['reporterCode'].map(emap)
    d['eP'] = d['partnerCode'].map(emap)
    d = d.dropna(subset=['eN', 'eP'])
    d['eN'] = d['eN'].astype(int)
    d['eP'] = d['eP'].astype(int)
    ef = MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg = dgl.graph((d['eN'].to_numpy(), d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat'] = torch.tensor(Xe, dtype=torch.float32)
    eg.edata['ef'] = torch.tensor(ef, dtype=torch.float32)
    eg = dgl.add_self_loop(eg, fill_data=0.)
    model.eval()
    preds = []
    N = eg.num_nodes()
    for i in range(0, N, bs):
        bn = list(range(i, min(i + bs, N)))
        bg = eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef']).unsqueeze(1))
    yp = np.expm1(st.inverse_transform(torch.cat(preds, 0).view(-1, 1).cpu().numpy()).flatten())
    yt = np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp


def _eval_edge_full_bilateral(model, df_eval, sf, st, gdelt_file, bilat_file, agg_mode):
    """
    Bilateral-corridor-specific replacement for _eval_edge_full().

    _eval_edge_full() (mirroring train_benchmark.py) resolves BOTH the reporter
    and partner to a node index via `emap`, a lookup built from the (already
    corridor-filtered) aggregated frame. Since the partner never appears as a
    "reporter" in a frame that's been filtered down to one fixed partner, that
    lookup always misses -> empty edge-feature array -> MinMaxScaler crashes
    with "Found array with 0 sample(s)".

    Fix: for a single fixed corridor, the partner's identity only matters
    through the bilateral GDELT edge feature itself (pair_events, pair_score_*,
    etc.) -- it doesn't need its own graph node. So we build self-loop edges
    (reporter -> itself) carrying that corridor's bilateral risk feature
    directly, instead of routing through a partner node that can't exist in
    this filtered view.
    """
    d = df_eval.copy()
    for c in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        d[c] = d[c].fillna(d[c].mean())
    if len(d) == 0:
        return None, None
    agg = build_agg(d, True, gdelt_file, agg_mode)
    if len(agg) == 0:
        return None, None

    Xe = sf.transform(agg[FULL_NODE_COLS])
    ye = st.transform(agg[['y_log']])

    fixed_partner = d['partnerCode'].iloc[0]  # constant across d by construction (corridor filter)
    lookup_rows = pd.DataFrame({
        'reporterCode': agg['reporterCode'],
        'partnerCode': fixed_partner,
        'refYear': agg['refYear'],
        'dist': agg['dist'],
    })
    ef = MinMaxScaler().fit_transform(edge_features_full(lookup_rows, bilat_file))

    N = len(agg)
    idx = np.arange(N)
    eg = dgl.graph((idx, idx), num_nodes=N)  # self-loops only -- see docstring
    eg.ndata['feat'] = torch.tensor(Xe, dtype=torch.float32)
    eg.edata['ef'] = torch.tensor(ef, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(eg, eg.ndata['feat'], eg.edata['ef'])
    yp = np.expm1(st.inverse_transform(out.view(-1, 1).cpu().numpy()).flatten())
    yt = np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp


# ===================== job registry (mirrors train_benchmark.py main()) =====================

def build_job_list():
    jobs = []
    jobs.append(dict(name='GAT_first_none', kind='GAT', use_gdelt=False,
                      scoring='none', gdelt_file=None, fusion='concat', agg_mode='first'))
    for scoring, gfile in [('keepall', KEEPALL), ('topk', TOPK)]:
        for fusion in ['concat', 'blend', 'attention']:
            jobs.append(dict(name=f'GAT_first_{scoring}_{fusion}', kind='GAT', use_gdelt=True,
                              scoring=scoring, gdelt_file=gfile, fusion=fusion, agg_mode='first'))
    for combo, node_file, pair_file in SCORER_COMBOS:
        for fusion in ['concat', 'blend', 'attention']:
            jobs.append(dict(name=f'EdgeGAT_full_{fusion}_{combo}', kind='EdgeGAT_full',
                              scoring=combo, gdelt_file=node_file, bilat_file=pair_file,
                              fusion=fusion, agg_mode='first'))
    # RF baselines -- present in models/trained_models/ as RF_first_{none,keepall,topk}.pkl
    jobs.append(dict(name='RF_first_none', kind='RF', use_gdelt=False,
                      scoring='none', gdelt_file=None, fusion='none', agg_mode='first'))
    for scoring, gfile in [('keepall', KEEPALL), ('topk', TOPK)]:
        jobs.append(dict(name=f'RF_first_{scoring}', kind='RF', use_gdelt=True,
                          scoring=scoring, gdelt_file=gfile, fusion='none', agg_mode='first'))
    return jobs


def eval_rf(rf, sf, st, df_eval, feat_cols, use_gdelt, gdelt_file, agg_mode):
    d = df_eval.copy()
    for c in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        d[c] = d[c].fillna(d[c].mean())
    agg = build_agg(d, use_gdelt, gdelt_file, agg_mode).dropna(subset=feat_cols + ['primaryValue'])
    if len(agg) == 0:
        return None, None
    Xe = sf.transform(agg[feat_cols])
    ye = st.transform(agg[['y_log']])
    yp = np.expm1(st.inverse_transform(rf.predict(Xe).reshape(-1, 1)).flatten())
    yt = np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp


def prep_gat_scalers(train_data, use_gdelt, gdelt_file, agg_mode):
    feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
    agg = build_agg(train_data, use_gdelt, gdelt_file, agg_mode)
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
    sf, st = MinMaxScaler(), MinMaxScaler()
    sf.fit(agg[feat_cols])
    st.fit(agg[['y_log']])
    return feat_cols, cmap, sf, st


def prep_edge_full_scalers(train_data, gdelt_file, agg_mode):
    agg = build_agg(train_data, True, gdelt_file, agg_mode)
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
    sf, st = MinMaxScaler(), MinMaxScaler()
    sf.fit(agg[FULL_NODE_COLS])
    st.fit(agg[['y_log']])
    return cmap, sf, st


def filter_pair(df, reporter_code, partner_code):
    return df[(df['reporterCode'] == reporter_code) & (df['partnerCode'] == partner_code)].copy()


def get_pair_subset(test_df, pair_name, spec):
    sub = filter_pair(test_df, spec['reporterCode'], spec['partnerCode'])
    if pair_name == 'China_to_Taiwan' and len(sub) == 0:
        log(f"no rows for partnerCode=490 (Taiwan) — retrying with fallback code {TAIWAN_FALLBACK_CODE}", "WARN")
        sub = filter_pair(test_df, spec['reporterCode'], TAIWAN_FALLBACK_CODE)
    return sub


def main():
    global KEEPALL, TOPK, BILAT_KEEP, BILAT_TOPK, SCORER_COMBOS

    split_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'cache', 'split_cache')
    models_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('models', 'trained_models')
    interim_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join('data', 'interim')

    KEEPALL = os.path.join(interim_dir, KEEPALL_NAME)
    TOPK = os.path.join(interim_dir, TOPK_NAME)
    BILAT_KEEP = os.path.join(interim_dir, BILAT_KEEP_NAME)
    BILAT_TOPK = os.path.join(interim_dir, BILAT_TOPK_NAME)
    SCORER_COMBOS = [
        ('ka_ka', KEEPALL, BILAT_KEEP),
        ('tk_tk', TOPK,    BILAT_TOPK),
        ('ka_tk', KEEPALL, BILAT_TOPK),
        ('tk_ka', TOPK,    BILAT_KEEP),
    ]

    ftr = os.path.join(split_dir, 'train.parquet')
    fte = os.path.join(split_dir, 'test2023.parquet')
    if not (os.path.exists(ftr) and os.path.exists(fte)):
        raise FileNotFoundError(
            f"Expected {ftr} and {fte} from the cached split (built by "
            f"train_benchmark.py's load_or_split). Run training first, or point "
            f"this script at the right split_cache directory."
        )

    log(f"loading cached split from {split_dir}/")
    train_data = pd.read_parquet(ftr)
    test_data = pd.read_parquet(fte)
    log(f"train={len(train_data):,}  test2023={len(test_data):,}")

    if os.path.exists(RESULTS_FILE):
        log(f"{RESULTS_FILE} already exists — new rows will be appended", "WARN")

    # pre-slice the two bilateral corridors once
    pair_subsets = {name: get_pair_subset(test_data, name, spec)
                    for name, spec in COUNTRY_PAIRS.items()}
    for name, sub in pair_subsets.items():
        log(f"{name}: {len(sub):,} raw test rows matched "
            f"(years present: {sorted(sub['refYear'].dropna().unique().tolist())})")

    jobs = build_job_list()
    log(f"{len(jobs)} model configs registered; checking {models_dir}/ for saved weights")

    # cache scalers/cmap per unique (use_gdelt, gdelt_file, agg_mode) config so we
    # don't rebuild agg(train_data) redundantly for every fusion variant
    gat_scaler_cache = {}
    edge_scaler_cache = {}
    rf_scaler_cache = {}

    for job in jobs:
        name = job['name']
        ext = '.pkl' if job['kind'] == 'RF' else '.pt'
        pt_path = os.path.join(models_dir, f'{name}{ext}')
        if not os.path.exists(pt_path):
            log(f"skip {name}: weights not found at {pt_path}", "WARN")
            for pair_name in COUNTRY_PAIRS:
                log_result(None, name, job['kind'], job.get('use_gdelt'), job['scoring'],
                           job['fusion'], job['agg_mode'], pair_name, None, 0,
                           notes='weights not found')
            continue

        try:
            if job['kind'] == 'GAT':
                use_gdelt, gdelt_file, agg_mode = job['use_gdelt'], job['gdelt_file'], job['agg_mode']
                key = (use_gdelt, gdelt_file, agg_mode)
                if key not in gat_scaler_cache:
                    gat_scaler_cache[key] = prep_gat_scalers(train_data, use_gdelt, gdelt_file, agg_mode)
                feat_cols, cmap, sf, st = gat_scaler_cache[key]
                nt = len(BASE_COLS)
                fusion = job['fusion']
                if fusion == 'blend':
                    model = BlendGAT(nt, len(GDELT_COLS))
                elif fusion == 'attention':
                    model = AttnGAT(nt, len(GDELT_COLS))
                else:
                    model = GATRegressionModel(len(feat_cols))
                model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
                model.eval()

                for pair_name in COUNTRY_PAIRS:
                    sub = pair_subsets[pair_name]
                    yt, yp = (_eval_graph(model, sub, cmap, sf, st, feat_cols, use_gdelt,
                                          gdelt_file, agg_mode, fusion, nt)
                              if len(sub) > 0 else (None, None))
                    if yt is None or len(yt) == 0:
                        log_result(None, name, job['kind'], use_gdelt, job['scoring'], fusion,
                                   agg_mode, pair_name, None, 0, notes='no matching rows after filtering')
                        log(f"{name} | {pair_name}: no evaluable rows", "WARN")
                    else:
                        met = compute_metrics(yt, yp)
                        log_result(None, name, job['kind'], use_gdelt, job['scoring'], fusion,
                                   agg_mode, pair_name, met, len(yt))
                        log(f"{name} | {pair_name}: n={len(yt)} r2={met['r2']:.3f} "
                            f"log_r2={met['log_r2']:.3f} rho={met['spearman']:.3f}")

            elif job['kind'] == 'EdgeGAT_full':
                gdelt_file, bilat_file, agg_mode = job['gdelt_file'], job['bilat_file'], job['agg_mode']
                key = (gdelt_file, agg_mode)
                if key not in edge_scaler_cache:
                    edge_scaler_cache[key] = prep_edge_full_scalers(train_data, gdelt_file, agg_mode)
                cmap, sf, st = edge_scaler_cache[key]
                n_trade = len(FULL_NODE_COLS) - len(GDELT_COLS)
                n_gdelt = len(GDELT_COLS)
                ei = len(FULL_EDGE_COLS)
                fusion = job['fusion']
                model = FusedEdgeModel('EdgeGAT', n_trade, n_gdelt, ei, fusion=fusion)
                model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
                model.eval()

                for pair_name in COUNTRY_PAIRS:
                    sub = pair_subsets[pair_name]
                    yt, yp = (_eval_edge_full_bilateral(model, sub, sf, st, gdelt_file, bilat_file, agg_mode)
                              if len(sub) > 0 else (None, None))
                    if yt is None or len(yt) == 0:
                        log_result(None, name, job['kind'], True, job['scoring'], fusion,
                                   agg_mode, pair_name, None, 0, notes='no matching rows after filtering')
                        log(f"{name} | {pair_name}: no evaluable rows", "WARN")
                    else:
                        met = compute_metrics(yt, yp)
                        log_result(None, name, job['kind'], True, job['scoring'], fusion,
                                   agg_mode, pair_name, met, len(yt))
                        log(f"{name} | {pair_name}: n={len(yt)} r2={met['r2']:.3f} "
                            f"log_r2={met['log_r2']:.3f} rho={met['spearman']:.3f}")

            elif job['kind'] == 'RF':
                use_gdelt, gdelt_file, agg_mode = job['use_gdelt'], job['gdelt_file'], job['agg_mode']
                feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
                key = (use_gdelt, gdelt_file, agg_mode)
                if key not in rf_scaler_cache:
                    tr_agg = build_agg(train_data, use_gdelt, gdelt_file, agg_mode).dropna(
                        subset=feat_cols + ['primaryValue'])
                    sf, st = MinMaxScaler(), MinMaxScaler()
                    sf.fit(tr_agg[feat_cols])
                    st.fit(tr_agg[['y_log']])
                    rf_scaler_cache[key] = (sf, st)
                sf, st = rf_scaler_cache[key]
                with open(pt_path, 'rb') as f:
                    rf = pickle.load(f)

                for pair_name in COUNTRY_PAIRS:
                    sub = pair_subsets[pair_name]
                    yt, yp = (eval_rf(rf, sf, st, sub, feat_cols, use_gdelt, gdelt_file, agg_mode)
                              if len(sub) > 0 else (None, None))
                    if yt is None or len(yt) == 0:
                        log_result(None, name, job['kind'], use_gdelt, job['scoring'], job['fusion'],
                                   agg_mode, pair_name, None, 0, notes='no matching rows after filtering')
                        log(f"{name} | {pair_name}: no evaluable rows", "WARN")
                    else:
                        met = compute_metrics(yt, yp)
                        log_result(None, name, job['kind'], use_gdelt, job['scoring'], job['fusion'],
                                   agg_mode, pair_name, met, len(yt))
                        log(f"{name} | {pair_name}: n={len(yt)} r2={met['r2']:.3f} "
                            f"log_r2={met['log_r2']:.3f} rho={met['spearman']:.3f}")

        except Exception as e:
            log(f"FAILED on {name}: {e}", "ERROR")
            for pair_name in COUNTRY_PAIRS:
                log_result(None, name, job['kind'], job.get('use_gdelt'), job['scoring'],
                           job['fusion'], job['agg_mode'], pair_name, None, 0,
                           notes=f'error: {e}')

        gc.collect()

    log(f"DONE — results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()