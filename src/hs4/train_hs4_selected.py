#!/usr/bin/env python3
"""
train_hs4_selected.py
======================
Trains exactly the 4 models requested for the HS4-collapsed re-run:
  - HS4_GAT_none                  (GAT, no GDELT)
  - HS4_GAT_topk_concat           (GAT + GDELT, topk scorer, concat fusion)
  - HS4_EdgeGAT_attention_ka_tk   (EdgeGAT, keepall node scorer + topk bilateral scorer, attention fusion)
  - HS4_EdgeGAT_blend_tk_ka       (EdgeGAT, topk node scorer + keepall bilateral scorer, blend fusion)

WHAT'S DIFFERENT FROM train_benchmark.py, AND WHY:

1. cmdCode is collapsed from 6-digit to 4-digit HEADING level for ALL products
   (not just chips) via integer division `cmdCode // 100`, done once in
   load_and_split() before anything else. This is the "collapse the 4-digit
   same products into the same row" instruction, applied across the whole
   training set since you said train on all products, not chips-only.

2. build_agg() ALWAYS uses AGG_SPEC = {'primaryValue': 'sum', 'dist': 'first',
   'gdpcap_d': 'first', 'gdpcap_o': 'first', 'pop_o': 'first', 'pop_d': 'first'}.
   This dict already existed in train_benchmark.py but was never actually wired
   into any of the 4 models you asked for -- they all called build_agg with
   agg_mode='first', which routes through a DIFFERENT branch that uses 'mean'
   for primaryValue, not 'sum'. That's the exact bug that would have silently
   undercounted trade value once multiple 6-digit sub-products start colliding
   onto the same HS4 key. There's no 'mean' branch left in this file at all --
   removed on purpose so it can't be selected by accident.

3. The RF-based IterativeImputer used for missing gdpcap/pop/dist is now FIT
   ONCE and SAVED (trained_models/HS4_imputer.pkl), instead of being fit and
   immediately discarded like in train_benchmark.py. predict_hs4_corridors.py
   loads this exact same fitted imputer and reuses it on the sparse
   2015/2016/2024/2025 files -- this is what actually fixes "gdpcap of Taiwan"
   for the US corridor: those files get properly RF-imputed with the SAME
   imputer the model was trained against, not a crude per-call column-mean
   fallback (which is what train_benchmark.py's eval functions did before).

4. Model names are prefixed HS4_ deliberately, so they can't collide with (and
   get silently skipped by any done()-style check against) old 6-digit models
   already sitting in trained_models/ from earlier runs.

5. Split cache goes to split_cache_hs4/, NOT split_cache/ -- the HS4-collapsed
   data is not interchangeable with the old 6-digit split, so it gets its own
   cache directory rather than silently reusing (or corrupting) the old one.

Usage:
    python train_hs4_selected.py [all_products_ready.parquet]

Requires the same environment train_benchmark.py runs in (conda env
mathematical_implementations, DGL 2.4). Not runnable in this sandbox -- no
GPU, no DGL, no actual data here. Run it locally and send back the console
log + trained_models/HS4_*.pt files (or any traceback) for the next step.
"""
import os, sys, gc, pickle
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required import for IterativeImputer)
from sklearn.impute import IterativeImputer
import warnings; warnings.filterwarnings("ignore")


def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


GDELT_COLS = ['events_total', 'score_mean', 'score_max', 'score_vol', 'goldstein_wmean', 'tone_mean', 'active_months']
BASE_COLS = ['refYear', 'cmdCode', 'dist', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d']
BILAT_COLS = ['pair_events', 'pair_score_mean', 'pair_score_max', 'pair_gold_mean']
KEEPALL = 'data/interim/gdelt_features_by_country_year.parquet'
TOPK = 'data/interim/gdelt_features_topk.parquet'
BILAT_KEEP = 'data/interim/gdelt_bilateral_by_pair_year.parquet'
BILAT_TOPK = 'data/interim/gdelt_bilateral_topk.parquet'
AGG_SPEC = {'primaryValue': 'sum', 'dist': 'first', 'gdpcap_d': 'first',
            'gdpcap_o': 'first', 'pop_o': 'first', 'pop_d': 'first'}
FULL_NODE_COLS = ['refYear', 'cmdCode', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d'] + GDELT_COLS
MISSING_COLS = ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']
MODEL_DIR = 'models/trained_models_hs4'
SPLIT_DIR = 'data/cache/split_cache_hs4'
os.makedirs(MODEL_DIR, exist_ok=True)
# NOTE: torch.cuda.is_available() alone is NOT a safe check here -- this conda env's
# DGL build was not compiled with CUDA support, even though PyTorch itself can see the
# GPU. Calling g.to('cuda') throws dgl._ffi.base.DGLError: "Device API cuda is not
# enabled" the instant it's tried. Forcing CPU, matching how heavy edge models already
# run on CPU in this project's established environment.
DEVICE = torch.device('cpu')
log(f"device = {DEVICE}  (forced CPU -- this DGL build is not CUDA-enabled; "
    f"see comment above if you later install a CUDA-enabled DGL and want to change this)")


# ===== metrics =====

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
    return dict(mae=mean_absolute_error(yt, yp), rmse=mean_squared_error(yt, yp) ** 0.5,
                r2=r2_score(yt, yp), log_r2=(r2_score(ytl, ypl) if len(yt) > 2 else np.nan),
                spearman=_spearman(yt, yp), smape=_smape(yt, yp))


def report(name, yt, yp):
    met = compute_metrics(yt, yp)
    log(f"  {name}: n={len(yt):,}  R2={met['r2']:.3f}  logR2={met['log_r2']:.3f}  "
        f"rho={met['spearman']:.3f}  RMSE={met['rmse']:,.0f}  MAE={met['mae']:,.0f}")
    return met


# ===== features / aggregation =====

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


def build_agg(df, use_gdelt, gdelt_file):
    """Always AGG_SPEC (sum for primaryValue). This is the aggregation the whole
    HS4-collapse run depends on -- see module docstring point 2. No 'mean' branch
    exists in this file, on purpose."""
    agg = df.groupby(['refYear', 'reporterCode', 'cmdCode']).agg(
        **{k: (k, op) for k, op in AGG_SPEC.items()}).reset_index()
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
    return m[BILAT_COLS + ['dist']].to_numpy(dtype='float32')


# ===== missing-data: fit once, SAVE, reuse everywhere (train, test2023, and later 2015/2016/2024/2025) =====

def fit_missing_imputer(train, cols):
    imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20, n_jobs=2, random_state=0),
                            max_iter=5, random_state=0)
    fit_sample = train[cols].sample(min(500_000, len(train)), random_state=0)
    imp.fit(fit_sample)
    return imp


# ===== HS4 collapse + split (own cache dir, separate from the old 6-digit split_cache/) =====

def load_and_split(path):
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

    # --- THE HS4 COLLAPSE --- 854110 -> 8541, 854293 -> 8542, etc. Applied to
    # every product in the file, not just chips, per "train on all products."
    # Assumes cmdCode is a clean 6-digit numeric code site-wide -- same
    # assumption load_and_split already made everywhere else in this pipeline.
    df['cmdCode'] = (df['cmdCode'].astype(int) // 100)

    # NOTE: deliberately NOT deduplicating on (year, reporter, partner, cmdCode)
    # here like the old load_and_split did. Multiple original 6-digit rows are
    # SUPPOSED to collide onto the same HS4 key now -- build_agg's groupby+sum
    # is what consolidates them correctly. Deduplicating here first would
    # silently drop real trade value before it ever gets summed.
    tr_full = df[df['refYear'].isin([2017, 2018, 2019, 2020, 2021, 2022])].copy()
    te = df[df['refYear'] == 2023].copy()

    imp = fit_missing_imputer(tr_full, MISSING_COLS)
    tr_full[MISSING_COLS] = imp.transform(tr_full[MISSING_COLS])
    te[MISSING_COLS] = imp.transform(te[MISSING_COLS])
    imp_path = os.path.join(MODEL_DIR, 'HS4_imputer.pkl')
    with open(imp_path, 'wb') as f:
        pickle.dump(imp, f)
    log(f"  saved fitted imputer -> {imp_path}  (predict_hs4_corridors.py reuses this exact object)")

    train_data, val_data = [], []
    for _, grp in tr_full.groupby('reporterCode'):
        if len(grp) < 5:
            train_data.append(grp)
            continue
        a, b = train_test_split(grp, test_size=0.2, random_state=42)
        train_data.append(a)
        val_data.append(b)
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    log(f"  train {len(train_data):,} | val {len(val_data):,} | test2023 {len(te):,}  "
        f"(still pre-HS4-aggregation rows here -- build_agg does the actual sum)")
    return train_data, val_data, te


def load_or_split(path):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    ftr, fva, fte = (os.path.join(SPLIT_DIR, f) for f in ["train.parquet", "val.parquet", "test2023.parquet"])
    if all(os.path.exists(f) for f in (ftr, fva, fte)):
        log("HS4 split cache found -> loading (skip re-split)")
        return pd.read_parquet(ftr), pd.read_parquet(fva), pd.read_parquet(fte)
    log(f"no HS4 split cache -> building from {path}")
    tr, va, te = load_and_split(path)
    tr.to_parquet(ftr, index=False)
    va.to_parquet(fva, index=False)
    te.to_parquet(fte, index=False)
    return tr, va, te


# ===== model classes (identical architectures to train_benchmark.py) =====

class GATRegressionModel(nn.Module):
    def __init__(s, inf, h=32, heads=4):
        super().__init__()
        s.c1 = GATConv(inf, h, heads)
        s.c2 = GATConv(h * heads, 1, heads)

    def forward(s, g, x):
        return s.c2(g, torch.relu(s.c1(g, x).flatten(1))).mean(1)


class EdgeGAT(nn.Module):
    def __init__(s, inf, ef, h=32, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h * heads, ef, 1, heads, allow_zero_in_degree=True)

    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1).squeeze(-1)


class FusedEdgeModel(nn.Module):
    def __init__(s, n_trade, n_gdelt, n_edge, fusion='concat', proj=16, h=32):
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
        s.backbone = EdgeGAT(inf, n_edge, h)

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


# ===== eval: uses the SAVED imputer, not a naive fillna(mean) =====

def _prep_eval_frame(d, imp):
    d = d.copy()
    d[MISSING_COLS] = imp.transform(d[MISSING_COLS])
    return d


def eval_graph(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, imp, fusion, nt,
                bs=10000, device=torch.device('cpu')):
    d = _prep_eval_frame(df_eval, imp)
    d['nID'] = d['reporterCode'].map(cmap)
    d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    if len(d) == 0:
        return None, None
    agg = build_agg(d, use_gdelt, gdelt_file)
    emap = {c: i for i, c in enumerate(agg['reporterCode'])}
    Xe = sf.transform(agg[feat_cols])
    ye = st.transform(agg[['y_log']])
    d['eN'] = d['reporterCode'].map(emap)
    d['eP'] = d['partnerCode'].map(emap)
    d = d.dropna(subset=['eN', 'eP'])
    d['eN'] = d['eN'].astype(int)
    d['eP'] = d['eP'].astype(int)
    eg = dgl.graph((d['eN'].to_numpy(), d['eP'].to_numpy()))
    eg.ndata['feat'] = torch.tensor(Xe, dtype=torch.float32)
    eg = dgl.add_self_loop(eg)
    eg = eg.to(device)
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


def eval_edge_full(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, imp,
                    device=torch.device('cpu'), bs=20000):
    d = _prep_eval_frame(df_eval, imp)
    d['nID'] = d['reporterCode'].map(cmap)
    d['pID'] = d['partnerCode'].map(cmap)
    d = d.dropna(subset=['nID', 'pID']).copy()
    if len(d) == 0:
        return None, None
    agg = build_agg(d, True, gdelt_file)
    emap = {c: i for i, c in enumerate(agg['reporterCode'])}
    Xe = sf.transform(agg[FULL_NODE_COLS])
    ye = st.transform(agg[['y_log']])
    d['eN'] = d['reporterCode'].map(emap)
    d['eP'] = d['partnerCode'].map(emap)
    d = d.dropna(subset=['eN', 'eP'])
    d['eN'] = d['eN'].astype(int)
    d['eP'] = d['eP'].astype(int)
    ef = MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg = dgl.graph((d['eN'].to_numpy(), d['eP'].to_numpy()))
    eg.ndata['feat'] = torch.tensor(Xe, dtype=torch.float32)
    eg.edata['ef'] = torch.tensor(ef, dtype=torch.float32)
    eg = dgl.add_self_loop(eg, fill_data=0.)
    eg = eg.to(device)
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


# ===== the 4 training runners =====

def train_gat_none(train_data, data_2023, imp, epochs=100, device=torch.device('cpu')):
    feat_cols = BASE_COLS
    agg = build_agg(train_data, False, None)
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
    td = train_data.copy()
    td['nodeID'] = td['reporterCode'].map(cmap)
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(agg[feat_cols])
    ytr = st.fit_transform(agg[['y_log']])
    g = dgl.graph((td['nodeID'].to_numpy(), td['partnerCode'].map(cmap).to_numpy()))
    g.ndata['feat'] = torch.tensor(Xtr, dtype=torch.float32)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    model = GATRegressionModel(len(feat_cols)).to(device)
    ytr_t = torch.tensor(ytr[:, 0], dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    N = g.num_nodes()
    bs = 10000
    nb = N // bs + (N % bs > 0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn = list(range(i * bs, min((i + 1) * bs, N)))
            bg = g.subgraph(torch.tensor(bn))
            ft = bg.ndata['feat']
            loss = crit(model(bg, ft).view(-1, 1), ytr_t[bn].view(-1, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    yt, yp = eval_graph(model, data_2023, cmap, sf, st, feat_cols, False, None, imp, 'none', len(BASE_COLS), device=device)
    if yt is not None:
        report('HS4_GAT_none', yt, yp)
    gc.collect()
    return model, sf, st, cmap


def train_gat_topk_concat(train_data, data_2023, imp, epochs=100, device=torch.device('cpu')):
    feat_cols = BASE_COLS + GDELT_COLS
    agg = build_agg(train_data, True, TOPK)
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
    td = train_data.copy()
    td['nodeID'] = td['reporterCode'].map(cmap)
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(agg[feat_cols])
    ytr = st.fit_transform(agg[['y_log']])
    g = dgl.graph((td['nodeID'].to_numpy(), td['partnerCode'].map(cmap).to_numpy()))
    g.ndata['feat'] = torch.tensor(Xtr, dtype=torch.float32)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    model = GATRegressionModel(len(feat_cols)).to(device)  # concat fusion = plain GAT over the concatenated vector
    ytr_t = torch.tensor(ytr[:, 0], dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    N = g.num_nodes()
    bs = 10000
    nb = N // bs + (N % bs > 0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn = list(range(i * bs, min((i + 1) * bs, N)))
            bg = g.subgraph(torch.tensor(bn))
            ft = bg.ndata['feat']
            loss = crit(model(bg, ft).view(-1, 1), ytr_t[bn].view(-1, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    yt, yp = eval_graph(model, data_2023, cmap, sf, st, feat_cols, True, TOPK, imp, 'concat', len(BASE_COLS), device=device)
    if yt is not None:
        report('HS4_GAT_topk_concat', yt, yp)
    gc.collect()
    return model, sf, st, cmap


def train_edgegat(name, node_file, pair_file, fusion, train_data, data_2023, imp,
                   epochs=100, device=torch.device('cpu'), bs=20000):
    agg = build_agg(train_data, True, node_file)
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
    td = train_data.copy()
    td['nID'] = td['reporterCode'].map(cmap)
    td['pID'] = td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID', 'pID']).copy()
    td['nID'] = td['nID'].astype(int)
    td['pID'] = td['pID'].astype(int)
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(agg[FULL_NODE_COLS])
    ytr = st.fit_transform(agg[['y_log']])
    ef = MinMaxScaler().fit_transform(edge_features_full(td, pair_file))
    g = dgl.graph((td['nID'].to_numpy(), td['pID'].to_numpy()))
    g.ndata['feat'] = torch.tensor(Xtr, dtype=torch.float32)
    g.ndata['y'] = torch.tensor(ytr[:, 0], dtype=torch.float32)
    g.edata['ef'] = torch.tensor(ef, dtype=torch.float32)
    g = dgl.add_self_loop(g, fill_data=0.)
    g = g.to(device)
    n_trade = len(FULL_NODE_COLS) - len(GDELT_COLS)
    n_gdelt = len(GDELT_COLS)
    ei = len(BILAT_COLS) + 1
    model = FusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    N = g.num_nodes()
    nb = N // bs + (N % bs > 0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn = list(range(i * bs, min((i + 1) * bs, N)))
            bg = g.subgraph(torch.tensor(bn))
            out = model(bg, bg.ndata['feat'], bg.edata['ef'])
            loss = crit(out.view(-1, 1), bg.ndata['y'].view(-1, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    yt, yp = eval_edge_full(model, data_2023, cmap, sf, st, node_file, pair_file, imp, device=device, bs=bs)
    if yt is not None:
        report(name, yt, yp)
    gc.collect()
    return model, sf, st, cmap


# ===== save =====

def save_bundle(name, model, sf, st, cmap):
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{name}.pt'))
    with open(os.path.join(MODEL_DIR, f'{name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'sf': sf, 'st': st, 'cmap': cmap}, f)
    log(f"  saved -> {MODEL_DIR}/{name}.pt (+ _scalers.pkl)")


def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"loading + HS4-collapsing: {data_file}")
    train_data, val_data, data_2023 = load_or_split(data_file)
    with open(os.path.join(MODEL_DIR, 'HS4_imputer.pkl'), 'rb') as f:
        imp = pickle.load(f)

    log("=== HS4_GAT_none ===")
    m, sf, st, cmap = train_gat_none(train_data, data_2023, imp, device=DEVICE)
    save_bundle('HS4_GAT_none', m, sf, st, cmap)

    log("=== HS4_GAT_topk_concat ===")
    m, sf, st, cmap = train_gat_topk_concat(train_data, data_2023, imp, device=DEVICE)
    save_bundle('HS4_GAT_topk_concat', m, sf, st, cmap)

    log("=== HS4_EdgeGAT_attention_ka_tk (node scorer=keepall, bilateral scorer=topk) ===")
    m, sf, st, cmap = train_edgegat('HS4_EdgeGAT_attention_ka_tk', KEEPALL, BILAT_TOPK, 'attention',
                                     train_data, data_2023, imp, device=DEVICE)
    save_bundle('HS4_EdgeGAT_attention_ka_tk', m, sf, st, cmap)

    log("=== HS4_EdgeGAT_blend_tk_ka (node scorer=topk, bilateral scorer=keepall) ===")
    m, sf, st, cmap = train_edgegat('HS4_EdgeGAT_blend_tk_ka', TOPK, BILAT_KEEP, 'blend',
                                     train_data, data_2023, imp, device=DEVICE)
    save_bundle('HS4_EdgeGAT_blend_tk_ka', m, sf, st, cmap)

    log("=== ALL 4 MODELS DONE -- send back this console log + trained_models/HS4_*.pt for the next step ===")


if __name__ == "__main__":
    main()