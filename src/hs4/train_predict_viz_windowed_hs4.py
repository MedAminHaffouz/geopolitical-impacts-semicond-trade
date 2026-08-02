#!/usr/bin/env python3
"""
train_predict_viz_windowed_hs4.py
====================================
One consolidated script: expanding-window walk-forward training, prediction,
AND visualization -- no separate scripts to run in sequence this time.

WHAT CHANGED FROM train_predict_edge_hs4.py / train_windowed_edge_hs4.py:

  1. cmdCode is DROPPED as a raw scaled numeric edge feature. A raw numeric
     8541-vs-8542 column, run through MinMaxScaler, implies 8542 is "more"
     than 8541 -- a false ordinal relationship between two unrelated product
     categories. Replaced with a FIXED one-hot encoding: one binary column
     per distinct heading in CHIP_HEADINGS (currently 2: hs_8541, hs_8542).
     "Fixed" matters -- the columns are built from the CHIP_HEADINGS constant,
     not from whatever headings happen to appear in a given data slice (e.g.
     via pd.get_dummies), so train and every eval call always produce the
     same column set in the same order. A dynamic one-hot would silently
     misalign if a fold's eval slice happened to be missing one heading --
     exactly the kind of bug this project has hit repeatedly.

  2. Walk-forward loop, train, predict, AND plot all happen in this one
     script/run -- train_end=2017 -> test 2018, train_end=2018 -> test 2019,
     ... through train_end=2022 -> test 2023. Scope is 2017-2023 only for
     now (not the sparse 2015/2016/2024/2025 files) -- per "start from 2017
     and move forward for now."

  3. Each fold's models get fold-specific names (HS4W2_<model>_upto<year>)
     so no fold overwrites another's saved weights.

Usage:
    python train_predict_viz_windowed_hs4.py [all_products_ready.parquet]

Not runnable in this sandbox -- no real data/DGL environment here. Run
locally; paste back the console log, windowed2_results.csv,
windowed2_predictions.csv, and/or any plots in windowed2_plots/.
"""
import os
import pickle
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")


def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


CHIP_HEADINGS = [8541, 8542]
ONEHOT_COLS = [f'hs_{h}' for h in CHIP_HEADINGS]  # fixed, in this exact order, every time
GDELT_COLS = ['events_total', 'score_mean', 'score_max', 'score_vol', 'goldstein_wmean', 'tone_mean', 'active_months']
BILAT_COLS = ['pair_events', 'pair_score_mean', 'pair_score_max', 'pair_gold_mean']
KEEPALL = ['data/interim/gdelt_features_by_country_year.parquet',
           'data/interim/gdelt_features_by_country_year_2024_2025.parquet']
TOPK = ['data/interim/gdelt_features_topk.parquet',
        'data/interim/gdelt_features_topk_2024_2025.parquet']
BILAT_KEEP = ['data/interim/gdelt_bilateral_by_pair_year.parquet',
              'data/interim/gdelt_bilateral_by_pair_year_2024_2025.parquet']
BILAT_TOPK = ['data/interim/gdelt_bilateral_topk.parquet',
              'data/interim/gdelt_bilateral_topk_2024_2025.parquet']
# NOTE, CORRECTED: the previous version of this fix pointed at *_2017_2024.parquet,
# assuming that was the complete/authoritative multi-year file. It wasn't -- the
# diagnostic proved it only actually contains rows for 2023/2024/2025 (540 total,
# exactly 147+197+196), NOT 2017-2022 despite the filename. That silently zeroed out
# GDELT for 2017-2021 (same masking bug as before, just wider), and then caused a
# genuine scale-shock explosion (R2 in the negative millions) the moment real,
# large-magnitude GDELT values first appeared at eval time in the 2022+ folds.
#
# The bare, unsuffixed file (gdelt_features_by_country_year.parquet etc.) is the one
# actually confirmed good for 2017-2023 -- the VERY FIRST diagnostic run in this chat
# showed it with sensible, gradually-declining event counts across all 7 years
# (2017: 4838.6 ... 2023: 3644.1). That file was never the problem; only the earlier
# assumption that it also covered 2024/2025 was wrong. Combining bare (2017-2023) +
# the dedicated _2024_2025 pull (confirmed good via the extraction notebook's own
# output) is the correct fix -- not the _2017_2024 file, which appears incomplete.


def _load_gdelt_multi(paths, key_cols):
    """Loads and concatenates multiple GDELT range files, preferring the LATER path in
    the list on any (key_cols) overlap -- i.e. the dedicated per-range pull wins over
    whatever the broader multi-year file says for the same year, since the dedicated
    pull is the more deliberately-built source for that specific range."""
    parts = []
    for p in paths:
        if os.path.exists(p):
            d = pd.read_parquet(p)
            yr_col = 'year' if 'year' in d.columns else 'refYear'
            yrs = sorted(d[yr_col].dropna().unique().tolist()) if yr_col in d.columns else '?'
            log(f"    loaded {p}: {len(d)} rows, years={yrs}")
            parts.append(d)
        else:
            log(f"    WARNING: GDELT file not found, skipping: {p}")
    if not parts:
        raise FileNotFoundError(f"none of {paths} were found")
    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=key_cols, keep='last')
    if len(combined) < before:
        log(f"    _load_gdelt_multi: {before} -> {len(combined)} rows after preferring "
            f"later-listed source on {key_cols} overlap")
    return combined
NODE_MISSING_COLS = ['gdpcap', 'pop']
MODEL_DIR = 'models/trained_models_hs4'
RESULTS_FILE = 'results/evaluation/windowed2_results.csv'
PREDICTIONS_FILE = 'results/evaluation/windowed2_predictions.csv'
PLOT_DIR = 'results/windowed2_plots'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
DEVICE = torch.device('cpu')  # this DGL build is not CUDA-enabled
log(f"device = {DEVICE} (forced CPU)")

CORRIDORS = {'US_to_China': dict(reporter=842, partner=156),
             'China_to_Taiwan': dict(reporter=156, partner=490)}  # still unverified -- see chat


# ===== metrics =====

def _spearman(a, b):
    if len(a) < 3:
        return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')


def compute_metrics(yt, yp):
    ytl = np.log1p(np.clip(yt, 0, None))
    ypl = np.log1p(np.clip(yp, 0, None))
    return dict(mae=mean_absolute_error(yt, yp), rmse=mean_squared_error(yt, yp) ** 0.5,
                r2=r2_score(yt, yp), log_r2=(r2_score(ytl, ypl) if len(yt) > 2 else np.nan),
                spearman=_spearman(yt, yp))


def report(name, yt, yp):
    met = compute_metrics(yt, yp)
    log(f"  {name}: n={len(yt):,}  R2={met['r2']:.3f}  logR2={met['log_r2']:.3f}  "
        f"rho={met['spearman']:.3f}  RMSE={met['rmse']:,.0f}  MAE={met['mae']:,.0f}")
    return met


# ===== loading, chips-only filter, safe HS4 collapse =====

def diagnose_source_homogeneity(main_path, sparse_path):
    """DIAGNOSTIC: checks whether the pre-2023 file and the 2024/2025 file actually
    report gdpcap/pop/dist/primaryValue in the SAME raw units/scale BEFORE
    load_chip_rows() applies its normalization (/1e6, /1e3). load_chip_rows
    assumes both sources use the same units -- if the 2024/2025 pull used a
    different scale (e.g. gdpcap already in thousands vs raw dollars, or a
    different distance unit), that assumption breaks silently and produces
    exactly the kind of systematic bias we're chasing, with no error or
    warning anywhere else in the pipeline.
    """
    cols = ['refYear', 'reporterCode', 'partnerCode', 'cmdCode',
            'gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist', 'primaryValue']
    log("  RAW (pre-normalization) schema/scale comparison:")
    for path, label in [(main_path, 'main (pre-2023)'), (sparse_path, '2024/2025')]:
        if not os.path.exists(path):
            log(f"    {label}: {path} not found -- skipping")
            continue
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception as e:
            log(f"    {label}: could not read requested columns ({e}) -- reading all columns instead")
            df = pd.read_parquet(path)
        log(f"    {label} ({path}):")
        for c in cols:
            if c not in df.columns:
                log(f"      {c}: COLUMN MISSING")
                continue
            s = pd.to_numeric(df[c], errors='coerce')
            log(f"      {c}: dtype={df[c].dtype}  min={s.min():.4g}  max={s.max():.4g}  "
                f"mean={s.mean():.4g}  n_nan={s.isna().sum()}")


def load_chip_rows(path):
    cols = ['refYear', 'reporterCode', 'partnerCode', 'cmdCode',
            'gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist', 'primaryValue']
    df = pd.read_parquet(path, columns=cols)
    for c in ['gdpcap_o', 'gdpcap_d', 'dist', 'pop_o', 'pop_d', 'primaryValue',
              'refYear', 'cmdCode', 'reporterCode', 'partnerCode']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['gdpcap_o'] /= 1e6
    df['gdpcap_d'] /= 1e6
    df['dist'] /= 1e3
    df = df[df['cmdCode'].notna()].copy()
    df['cmdCode'] = df['cmdCode'].astype(int)
    if df['cmdCode'].max() >= 100000:
        df['cmdCode'] = df['cmdCode'] // 100
    df = df[df['cmdCode'].isin(CHIP_HEADINGS)].copy()
    return df


# ===== node table: (year, country) keyed =====

def build_node_table(df, imputer=None, fit_imputer=False):
    a = df[['refYear', 'reporterCode', 'gdpcap_o', 'pop_o']].rename(
        columns={'reporterCode': 'country', 'gdpcap_o': 'gdpcap', 'pop_o': 'pop'})
    b = df[['refYear', 'partnerCode', 'gdpcap_d', 'pop_d']].rename(
        columns={'partnerCode': 'country', 'gdpcap_d': 'gdpcap', 'pop_d': 'pop'})
    nodes = pd.concat([a, b], ignore_index=True)
    nodes = nodes.dropna(subset=['country']).copy()
    nodes['country'] = nodes['country'].astype(int)
    nodes['refYear'] = nodes['refYear'].astype(int)
    nodes = nodes.sort_values('gdpcap', na_position='last').drop_duplicates(['refYear', 'country'], keep='first')
    nodes = nodes.sort_values(['refYear', 'country']).reset_index(drop=True)

    if fit_imputer:
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20, n_jobs=2, random_state=0),
                                    max_iter=5, random_state=0)
        fit_sample = nodes[NODE_MISSING_COLS].sample(min(50_000, len(nodes)), random_state=0)
        imputer.fit(fit_sample)
    nodes[NODE_MISSING_COLS] = imputer.transform(nodes[NODE_MISSING_COLS])
    return nodes, imputer


def add_gdelt_node(nodes, gdelt_file):
    cy = _load_gdelt_multi(gdelt_file, key_cols=['reporterCode', 'year'])
    out = nodes.copy()
    out['_c'] = out['country'].astype(str)
    out['_y'] = out['refYear'].astype(str)
    cy['_c'] = cy['reporterCode'].astype('Int64').astype(str)
    cy['_y'] = cy['year'].astype('Int64').astype(str)
    out = out.merge(cy[['_c', '_y'] + GDELT_COLS], on=['_c', '_y'], how='left')
    out[GDELT_COLS] = out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_c', '_y'])


# ===== edge table: cmdCode -> FIXED one-hot, dist NaN-filled, real bilateral rows =====

def build_edge_table(df):
    e = (df.groupby(['refYear', 'reporterCode', 'partnerCode', 'cmdCode'])
           .agg(primaryValue=('primaryValue', 'sum'), dist=('dist', 'first')).reset_index())
    if e['dist'].isna().any():
        e['dist'] = e['dist'].fillna(e['dist'].median())
    # FIXED one-hot -- built from CHIP_HEADINGS, not from whatever's present in
    # this slice, so train and every eval call get identical columns/order.
    for h, col in zip(CHIP_HEADINGS, ONEHOT_COLS):
        e[col] = (e['cmdCode'] == h).astype(float)
    e['y_log'] = np.log1p(e['primaryValue'].clip(lower=0))
    return e


def add_gdelt_edge(edges, bilat_file):
    b = _load_gdelt_multi(bilat_file, key_cols=['reporterCode', 'partnerCode', 'year'])
    k = edges[['reporterCode', 'partnerCode', 'refYear']].copy()
    for c in ['reporterCode', 'partnerCode', 'refYear']:
        k[c] = k[c].astype('Int64')
    b['reporterCode'] = b['reporterCode'].astype('Int64')
    b['partnerCode'] = b['partnerCode'].astype('Int64')
    b['year'] = b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode', 'partnerCode', 'refYear'],
                right_on=['reporterCode', 'partnerCode', 'year'], how='left')
    out = edges.copy()
    out[BILAT_COLS] = m[BILAT_COLS].fillna(0).to_numpy()
    return out


# ===== graph builder =====

def build_graph(df, use_gdelt, gdelt_file, encoder_kind, bilat_file=None,
                 node_imputer=None, fit_node_imputer=False, node_scaler=None, edge_scaler=None,
                 target_scaler=None, fit_scalers=False):
    nodes, node_imputer = build_node_table(df, node_imputer, fit_node_imputer)
    if use_gdelt:
        nodes = add_gdelt_node(nodes, gdelt_file)
    node_feat_cols = ['gdpcap', 'pop'] + (GDELT_COLS if use_gdelt else [])

    edges = build_edge_table(df)
    if encoder_kind == 'edgegat':
        edges = add_gdelt_edge(edges, bilat_file)
    # NOTE: 'cmdCode' itself is gone from edge_feat_cols -- replaced by ONEHOT_COLS
    edge_feat_cols = ['dist'] + ONEHOT_COLS + (BILAT_COLS if encoder_kind == 'edgegat' else [])

    node_key = list(zip(nodes['refYear'].astype(int), nodes['country'].astype(int)))
    nmap = {k: i for i, k in enumerate(node_key)}
    edges['s'] = list(zip(edges['refYear'].astype(int), edges['reporterCode'].astype(int)))
    edges['d'] = list(zip(edges['refYear'].astype(int), edges['partnerCode'].astype(int)))
    edges = edges[edges['s'].isin(nmap) & edges['d'].isin(nmap)].copy()
    if len(edges) == 0:
        return None
    edges['sidx'] = edges['s'].map(nmap)
    edges['didx'] = edges['d'].map(nmap)

    if fit_scalers:
        node_scaler = MinMaxScaler().fit(nodes[node_feat_cols])
        edge_scaler = MinMaxScaler().fit(edges[edge_feat_cols])
        target_scaler = MinMaxScaler().fit(edges[['y_log']])

    Xn = node_scaler.transform(nodes[node_feat_cols])
    Xe = edge_scaler.transform(edges[edge_feat_cols])
    clip_stats = None
    if not fit_scalers:
        # DIAGNOSTIC: measure how much of the eval input is genuinely out-of-range
        # BEFORE clipping -- this quantifies candidate cause #3 (out-of-distribution
        # features getting flattened to the training range's boundary).
        n_total = Xn.size + Xe.size
        n_out = int(((Xn < 0) | (Xn > 1)).sum()) + int(((Xe < 0) | (Xe > 1)).sum())
        clip_frac = n_out / n_total if n_total else 0.0
        clip_stats = dict(n_total=n_total, n_out=n_out, frac=clip_frac)
        Xn = np.clip(Xn, 0.0, 1.0)
        Xe = np.clip(Xe, 0.0, 1.0)
    ye = target_scaler.transform(edges[['y_log']])[:, 0]

    g = dgl.graph((edges['sidx'].to_numpy(), edges['didx'].to_numpy()), num_nodes=len(nmap))
    g.ndata['feat'] = torch.tensor(Xn, dtype=torch.float32)
    g.edata['ef'] = torch.tensor(Xe, dtype=torch.float32)
    g.edata['y'] = torch.tensor(ye, dtype=torch.float32)

    return dict(graph=g, node_imputer=node_imputer, node_scaler=node_scaler,
                clip_stats=clip_stats,
                edge_scaler=edge_scaler, target_scaler=target_scaler,
                node_feat_cols=node_feat_cols, edge_feat_cols=edge_feat_cols, edges_df=edges)


# ===== model =====

class EdgeScoreModel(nn.Module):
    def __init__(s, encoder_kind, n_trade, n_gdelt, n_edge, fusion='none', proj=16, h=32, heads=4):
        super().__init__()
        s.encoder_kind = encoder_kind
        s.fusion = fusion
        s.n_trade = n_trade
        if fusion in ('none', 'concat'):
            inf = n_trade + (n_gdelt if fusion == 'concat' else 0)
        else:
            s.tp = nn.Linear(n_trade, proj)
            s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend':
                s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention':
                s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        if encoder_kind == 'gat':
            s.c1 = GATConv(inf, h, heads, allow_zero_in_degree=True)
            s.c2 = GATConv(h * heads, h, heads, allow_zero_in_degree=True)
        else:
            s.c1 = EdgeGATConv(inf, n_edge, h, heads, allow_zero_in_degree=True)
            s.c2 = EdgeGATConv(h * heads, n_edge, h, heads, allow_zero_in_degree=True)
        s.head = nn.Sequential(nn.Linear(h * 2 + n_edge, 32), nn.ReLU(), nn.Linear(32, 1))

    def _fuse(s, x):
        if s.fusion in ('none', 'concat'):
            return x
        xt, xg = x[:, :s.n_trade], x[:, s.n_trade:]
        t, d = torch.relu(s.tp(xt)), torch.relu(s.gp(xg))
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha)
            return a * d + (1 - a) * t
        st = torch.stack([t, d], dim=1)
        return s.attn(st, st, st)[0].mean(1)

    def forward(s, g, x, ef):
        node_in = s._fuse(x)
        if s.encoder_kind == 'gat':
            h1 = torch.relu(s.c1(g, node_in).flatten(1))
            h2 = s.c2(g, h1).mean(1)
        else:
            h1 = torch.relu(s.c1(g, node_in, ef).flatten(1))
            h2 = s.c2(g, h1, ef).mean(1)
        with g.local_scope():
            g.ndata['h'] = h2
            g.edata['ef2'] = ef
            g.apply_edges(lambda e: {'pred': s.head(
                torch.cat([e.src['h'], e.dst['h'], e.data['ef2']], dim=1)).squeeze(-1)})
            return g.edata['pred']


# ===== train / eval =====

def train_variant(name, encoder_kind, fusion, use_gdelt, gdelt_file, bilat_file, train_df, epochs=150):
    bundle = build_graph(train_df, use_gdelt, gdelt_file, encoder_kind, bilat_file,
                          fit_node_imputer=True, fit_scalers=True)
    g = bundle['graph']
    if torch.isnan(g.ndata['feat']).any() or torch.isnan(g.edata['ef']).any() or torch.isnan(g.edata['y']).any():
        raise ValueError(f"{name}: NaN in graph tensors after scaling -- check feature sources")
    n_trade = 2
    n_gdelt = len(GDELT_COLS) if use_gdelt else 0
    n_edge = len(bundle['edge_feat_cols'])
    model = EdgeScoreModel(encoder_kind, n_trade, n_gdelt, n_edge, fusion=fusion).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        pred = model(g, g.ndata['feat'], g.edata['ef'])
        loss = crit(pred, g.edata['y'])
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  {name}: final train loss={loss.item():.5f}  (n_edges={g.num_edges()}, n_nodes={g.num_nodes()})")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{name}.pt'))
    return model, bundle


def eval_variant(model, bundle_train, df_eval, use_gdelt, gdelt_file, encoder_kind, bilat_file):
    b = build_graph(df_eval, use_gdelt, gdelt_file, encoder_kind, bilat_file,
                     node_imputer=bundle_train['node_imputer'], fit_node_imputer=False,
                     node_scaler=bundle_train['node_scaler'], edge_scaler=bundle_train['edge_scaler'],
                     target_scaler=bundle_train['target_scaler'], fit_scalers=False)
    if b is None:
        return None, None, None, None
    g = b['graph']
    model.eval()
    with torch.no_grad():
        pred = model(g, g.ndata['feat'], g.edata['ef']).numpy()
    yp_log = bundle_train['target_scaler'].inverse_transform(pred.reshape(-1, 1)).flatten()
    yp = np.expm1(np.clip(yp_log, -50, 50))
    yt = np.expm1(bundle_train['target_scaler'].inverse_transform(g.edata['y'].numpy().reshape(-1, 1)).flatten())
    return yt, yp, b['edges_df'], b['clip_stats']


VARIANTS = [
    dict(name='HS4W2_GAT_none', encoder_kind='gat', fusion='none', use_gdelt=False, gdelt_file=None, bilat_file=None),
    dict(name='HS4W2_GAT_topk_concat', encoder_kind='gat', fusion='concat', use_gdelt=True, gdelt_file=TOPK, bilat_file=None),
    dict(name='HS4W2_EdgeGAT_attention_ka_tk', encoder_kind='edgegat', fusion='attention', use_gdelt=True, gdelt_file=KEEPALL, bilat_file=BILAT_TOPK),
    dict(name='HS4W2_EdgeGAT_blend_tk_ka', encoder_kind='edgegat', fusion='blend', use_gdelt=True, gdelt_file=TOPK, bilat_file=BILAT_KEEP),
]


def log_result(train_end, test_year, model_name, n, met):
    import csv
    row = dict(train_end=train_end, test_year=test_year, model=model_name, n=n,
               mae=met['mae'], rmse=met['rmse'], r2=met['r2'], log_r2=met['log_r2'], spearman=met['spearman'])
    new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


def log_predictions(train_end, test_year, model_name, edges_df, yt, yp):
    import csv
    edges_df = edges_df.reset_index(drop=True).copy()
    edges_df['actual'] = yt
    edges_df['predicted'] = yp
    rows = []
    for corridor, cfg in CORRIDORS.items():
        m = (edges_df['reporterCode'] == cfg['reporter']) & (edges_df['partnerCode'] == cfg['partner'])
        for _, row in edges_df[m].iterrows():
            rows.append(dict(train_end=train_end, test_year=test_year, model=model_name,
                              corridor=corridor, hs_code=int(row['cmdCode']),
                              actual=row['actual'], predicted=row['predicted']))
    if not rows:
        return 0
    new = not os.path.exists(PREDICTIONS_FILE)
    with open(PREDICTIONS_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


# ===== viz (runs automatically at the end -- no separate script needed) =====

MODEL_LABELS = {
    'HS4W2_GAT_none': 'GAT (no GDELT)',
    'HS4W2_GAT_topk_concat': 'GAT + GDELT (concat)',
    'HS4W2_EdgeGAT_attention_ka_tk': 'EdgeGAT + GDELT (attention)',
    'HS4W2_EdgeGAT_blend_tk_ka': 'EdgeGAT + GDELT (blend)',
}


def make_plots():
    plt.rcParams.update({
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff',
        'axes.edgecolor': '#b0b0b0', 'axes.grid': True, 'grid.color': '#d9d9d9', 'font.size': 11,
    })
    df = pd.read_csv(PREDICTIONS_FILE)
    for model in df['model'].unique():
        for corridor in df['corridor'].unique():
            sub = df[(df['model'] == model) & (df['corridor'] == corridor)]
            if sub.empty:
                continue
            agg = sub.groupby('test_year')[['actual', 'predicted']].sum().reset_index().sort_values('test_year')
            agg['actual_log'] = np.log10(agg['actual'].clip(lower=1))
            agg['predicted_log'] = np.log10(agg['predicted'].clip(lower=1))
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(agg['test_year'], agg['actual_log'], marker='s', markersize=7, lw=2, color='#1F4E78', label='actual')
            ax.plot(agg['test_year'], agg['predicted_log'], marker='D', markersize=6, lw=2, color='#ED7D31', label='predicted')
            ax.set_ylim(0, 12)
            ax.set_ylabel('log-scale trade value')
            ax.set_xlabel('year')
            ax.set_xticks(agg['test_year'].astype(int))
            label = MODEL_LABELS.get(model, model)
            ax.set_title(f'{corridor.replace("_", "-")} trade value \u2014 {label} (windowed, HS one-hot)', fontsize=11)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fname = os.path.join(PLOT_DIR, f'{model}_{corridor}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log(f"  saved {fname}")


# ===== main: train -> predict -> viz, all in one run =====

def diagnose_gdelt_coverage():
    """DIAGNOSTIC: compares GDELT event coverage per year across both node-level
    file sets, to check candidate cause #2 -- whether 2024/2025 have meaningfully
    thinner GDELT coverage than 2017-2023, which would systematically weaken
    the GDELT-fusion models (attention, blend) specifically on those years."""
    for paths, label in [(KEEPALL, 'keepall'), (TOPK, 'topk')]:
        try:
            gd = _load_gdelt_multi(paths, key_cols=['reporterCode', 'year'])
        except FileNotFoundError as e:
            log(f"  GDELT diagnostic: {e} -- skipping {label}")
            continue
        year_col = 'year' if 'year' in gd.columns else 'refYear'
        summary = gd.groupby(year_col).agg(
            n_country_years=('reporterCode', 'count'),
            mean_events_total=('events_total', 'mean') if 'events_total' in gd.columns else ('reporterCode', 'count'),
        ).reset_index()
        log(f"  GDELT coverage ({label}):")
        for _, row in summary.iterrows():
            log(f"    year={int(row[year_col])}  n_country_years={int(row['n_country_years'])}  "
                f"mean_events_total={row['mean_events_total']:.1f}")


def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    sparse_2024_2025 = 'data/processed/all_products_2024_2025_ready_gravity.parquet'

    log("running source homogeneity diagnostic (raw units/schema, before any normalization) ...")
    diagnose_source_homogeneity(data_file, sparse_2024_2025)

    log(f"loading + chips-only filtering: {data_file}")
    df = load_chip_rows(data_file)
    log(f"  edge feature columns this run: {['dist'] + ONEHOT_COLS} (+ bilateral GDELT for EdgeGAT variants)")

    log("running GDELT coverage diagnostic (candidate cause #2) ...")
    diagnose_gdelt_coverage()

    if os.path.exists(sparse_2024_2025):
        sparse = load_chip_rows(sparse_2024_2025)
        df = pd.concat([df, sparse], ignore_index=True)
        log(f"  folded in 2024/2025: +{len(sparse):,} rows -- fold range now extends to reach them")
    else:
        log(f"  WARNING: {sparse_2024_2025} not found -- 2024/2025 folds will be skipped "
            f"(update the path above if yours is named differently)")

    years_present = sorted(df['refYear'].dropna().unique().astype(int))
    train_ends = [y for y in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] if y + 1 in years_present]

    for train_end in train_ends:
        test_year = train_end + 1
        train_df = df[df['refYear'].between(2017, train_end)].copy()
        test_df = df[df['refYear'] == test_year].copy()
        log(f"=== fold: train 2017-{train_end} ({len(train_df):,} rows) -> test {test_year} "
            f"({len(test_df):,} rows) ===")
        if len(test_df) == 0:
            log(f"  no rows for test_year={test_year} -- skipping")
            continue
        for v in VARIANTS:
            fold_name = f"{v['name']}_upto{train_end}"
            model, bundle = train_variant(fold_name, v['encoder_kind'], v['fusion'],
                                           v['use_gdelt'], v['gdelt_file'], v['bilat_file'], train_df)
            yt, yp, edges_df, clip_stats = eval_variant(model, bundle, test_df, v['use_gdelt'], v['gdelt_file'],
                                                          v['encoder_kind'], v['bilat_file'])
            if yt is None:
                log(f"  {fold_name}: no evaluable edges for {test_year}")
                continue
            met = compute_metrics(yt, yp)
            log_result(train_end, test_year, v['name'], len(yt), met)
            n_pred = log_predictions(train_end, test_year, v['name'], edges_df, yt, yp)
            clip_note = (f"  clip: {clip_stats['n_out']}/{clip_stats['n_total']} values "
                         f"({clip_stats['frac']*100:.1f}%) out-of-range before clipping"
                         if clip_stats else "")
            log(f"  {fold_name}: n={len(yt):,}  R2={met['r2']:.3f}  logR2={met['log_r2']:.3f}  "
                f"rho={met['spearman']:.3f}  ({n_pred} corridor rows saved){clip_note}")

    log("=== training+prediction done -- building plots now ===")
    if os.path.exists(PREDICTIONS_FILE):
        make_plots()
    else:
        log("  no predictions file produced -- nothing to plot")

    log(f"=== ALL DONE -- {RESULTS_FILE}, {PREDICTIONS_FILE}, plots in {PLOT_DIR}/ ===")


if __name__ == "__main__":
    main()