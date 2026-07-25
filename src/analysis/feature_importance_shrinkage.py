#!/usr/bin/env python3
"""
feature_importance_shrinkage.py
=================================
Permutation importance for the SAVED pooling+shrinkage models (from
run_shrinkage_head.py), not the old dedicated chips/electronics models.
No retraining -- loads frozen weights from trained_models_shrinkage_head/
and perturbs the INPUT, same method as your earlier feature-importance run.

Two things this adds beyond the old per-category script:
  1. Runs on the actual best-performing pooling+shrinkage models, not the
     older, now-superseded dedicated architecture.
  2. A NEW permutable target that didn't exist before: `heading_idx` itself.
     Shuffling which product-head applies to which row (heading AND its
     parent chapter2, shuffled together as a pair so the combination stays
     internally consistent) measures how load-bearing the per-product head
     mechanism actually is -- a direct answer to "does knowing which
     product this is matter", not just "which trade feature matters".

Output is in the same shape as your existing feature_importance_*.csv
(feature, importance_log_r2, importance_spearman, model, dataset), so
analyze_feature_importance.py works on this file unmodified. `heading_idx`
shows up as its own row in that ranking, directly comparable to every
other feature.

First checks trained_models_shrinkage_head/ and tells you what's actually
there before doing anything -- per your question, this answers "are the
models saved" as the very first thing it does.

Usage:
    python feature_importance_shrinkage.py [all_products_ready.parquet]
"""
import os, sys, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)

# ==========================================================================
# config -- copied from run_shrinkage_head.py, must match exactly or saved
# state_dicts won't load (shapes have to line up)
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
FULL_NODE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS = BILAT_COLS + ['dist']
MISSING = 'rf'
KEEPALL='gdelt_features_by_country_year.parquet'; TOPK='gdelt_features_topk.parquet'
BILAT_KEEP='gdelt_bilateral_by_pair_year.parquet'; BILAT_TOPK='gdelt_bilateral_topk.parquet'
SCORER_COMBOS = {'ka_ka':(KEEPALL,BILAT_KEEP), 'ka_tk':(KEEPALL,BILAT_TOPK), 'tk_ka':(TOPK,BILAT_KEEP), 'tk_tk':(TOPK,BILAT_TOPK)}
CONDITIONS_BY_SCORING = {'none': (False, None), 'keepall': (True, KEEPALL), 'topk': (True, TOPK)}

NUM_HEADINGS = 10000
MODEL_DIR = 'trained_models_shrinkage_head'
SHRINKAGE_K = 50

# --- which saved models to run this on. Leave empty to run over EVERY
# saved model found in MODEL_DIR/ -- the recommended default now that you
# want both per-model and global numbers. Fill this in with specific names
# if you ever want to restrict to a subset again (e.g. while iterating). ---
TARGETS = []

def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)

def hs_chapter_2digit(heading):
    h = np.asarray(heading, dtype=np.int64)
    return (h // 100).astype(np.int64)

class UnparseableModelName(ValueError):
    """Raised for files in MODEL_DIR that don't match the expected naming
    convention -- e.g. orphaned files from an earlier version of
    run_shrinkage_head.py, saved before the fusion dimension was added."""
    pass

def parse_shrinkage_name(name):
    """Maps a saved model's filename back to (kind, fusion, scoring/combo)."""
    n = name.replace('_shrinkage', '')
    if n == 'GAT_first_none':
        return ('GAT', 'concat', 'none')
    if n.startswith('GAT_first_'):
        rest = n[len('GAT_first_'):]
        parts = rest.rsplit('_', 1)
        if len(parts) != 2:
            raise UnparseableModelName(
                f"'{name}' has no fusion suffix -- likely an orphaned file from an older "
                f"version of run_shrinkage_head.py (before fusion was added to GAT_TARGETS). "
                f"Delete it from {MODEL_DIR}/ if it's not one of the current 19 targets.")
        scoring, fusion = parts
        return ('GAT', fusion, scoring)
    if n.startswith('EdgeGAT_full_'):
        rest = n[len('EdgeGAT_full_'):]
        fusion, combo = rest.split('_', 1)
        return ('EdgeGAT_full', fusion, combo)
    raise UnparseableModelName(f"'{name}' doesn't match any known naming pattern -- skipping.")

# ==========================================================================
# metrics
# ==========================================================================
def _smape(yt, yp):
    d=(np.abs(yt)+np.abs(yp))/2; m=d>0
    return np.mean(np.abs(yt[m]-yp[m])/d[m])*100 if m.sum() else np.nan

def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_metrics(yt, yp):
    ytl=np.log1p(np.clip(yt,0,None)); ypl=np.log1p(np.clip(yp,0,None))
    return dict(mae=mean_absolute_error(yt,yp), rmse=mean_squared_error(yt,yp)**0.5,
                r2=r2_score(yt,yp), log_r2=(r2_score(ytl,ypl) if len(yt)>2 else np.nan),
                spearman=_spearman(yt,yp), smape=_smape(yt,yp))

# ==========================================================================
# feature engineering -- identical to run_shrinkage_head.py
# ==========================================================================
AGG_SPEC = {'primaryValue':'sum','dist':'first','gdpcap_d':'first','gdpcap_o':'first','pop_o':'first','pop_d':'first'}

def add_gdelt(agg, gdelt_file):
    cy=pd.read_parquet(gdelt_file)
    out=agg.copy()
    out['_r']=out['reporterCode'].astype('Int64').astype(str); out['_y']=out['refYear'].astype('Int64').astype(str)
    cy['_r']=cy['reporterCode'].astype('Int64').astype(str);  cy['_y']=cy['year'].astype('Int64').astype(str)
    out=out.merge(cy[['_r','_y']+GDELT_COLS], on=['_r','_y'], how='left')
    out[GDELT_COLS]=out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_r','_y'])

def build_agg(df, use_gdelt, gdelt_file, agg_mode='first'):
    if agg_mode == 'spec':
        agg = df.groupby(['refYear','reporterCode','cmdCode']).agg(
            **{k:(k,op) for k,op in AGG_SPEC.items()}).reset_index()
    else:
        grav = 'first' if agg_mode=='first' else 'sum'
        agg = (df.groupby(['refYear','reporterCode','cmdCode'])
                 .agg(primaryValue=('primaryValue','mean'), dist=('dist','first'),
                      gdpcap_d=('gdpcap_d',grav), gdpcap_o=('gdpcap_o',grav),
                      pop_o=('pop_o',grav), pop_d=('pop_d',grav)).reset_index())
    agg['y_log'] = np.log1p(agg['primaryValue'].clip(lower=0))
    agg['chapter'] = hs_chapter(agg['cmdCode'])
    agg['chapter2'] = hs_chapter_2digit(agg['chapter'])
    if use_gdelt:
        agg = add_gdelt(agg, gdelt_file)
    return agg

def edge_features_full(df_rows, bilat_file):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode','partnerCode','refYear','dist']].copy()
    for c in ['reporterCode','partnerCode','refYear']: k[c]=k[c].astype('Int64')
    b['reporterCode']=b['reporterCode'].astype('Int64'); b['partnerCode']=b['partnerCode'].astype('Int64'); b['year']=b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode','partnerCode','refYear'],
                right_on=['reporterCode','partnerCode','year'], how='left')
    m[BILAT_COLS]=m[BILAT_COLS].fillna(0); m['dist']=m['dist'].fillna(m['dist'].median())
    return m[FULL_EDGE_COLS].to_numpy(dtype='float32')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def handle_missing(train, test, strategy='rf'):
    cols=['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']
    if strategy=='rf':
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20, n_jobs=2, random_state=0),
                               max_iter=5, random_state=0)
        fit_sample = train[cols].sample(min(500_000, len(train)), random_state=0)
        imp.fit(fit_sample)
        tr, te = train.copy(), test.copy()
        tr[cols] = imp.transform(train[cols]); te[cols] = imp.transform(test[cols])
        return tr, te
    fill = train[cols].median()
    return train.fillna(fill).copy(), test.fillna(fill).copy()

def load_and_split(path):
    cols = ['refYear','reporterCode','partnerCode','cmdCode',
            'gdpcap_o','pop_o','gdpcap_d','pop_d','dist','primaryValue']
    df = pd.read_parquet(path, columns=cols)
    for c in ['gdpcap_o','gdpcap_d','dist','pop_o','pop_d','primaryValue','refYear','cmdCode','reporterCode','partnerCode']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['gdpcap_o']/=1e6; df['gdpcap_d']/=1e6; df['dist']/=1e3
    df = df[df['cmdCode'].notna()].copy(); df['cmdCode']=df['cmdCode'].astype(int)
    d = df.drop_duplicates(['refYear','reporterCode','partnerCode','cmdCode']).reset_index(drop=True)
    tr_full = d[d['refYear'].isin([2017,2018,2019,2020,2021,2022])].copy()
    te      = d[d['refYear']==2023].copy()
    tr_full, te = handle_missing(tr_full, te, strategy=MISSING)
    train_data, val_data = [], []
    for _, grp in tr_full.groupby('reporterCode'):
        if len(grp) < 5:
            train_data.append(grp); continue
        a,b = train_test_split(grp, test_size=0.2, random_state=42)
        train_data.append(a); val_data.append(b)
    train_data = pd.concat(train_data); val_data = pd.concat(val_data)
    log(f'  train {len(train_data):,} | val {len(val_data):,} | test2023 {len(te):,}')
    return train_data, val_data, te

def load_or_split(path):
    tag = os.path.splitext(os.path.basename(path))[0]
    split_dir = os.path.join('split_cache', tag)
    os.makedirs(split_dir, exist_ok=True)
    ftr, fva, fte = (os.path.join(split_dir, f) for f in ["train.parquet","val.parquet","test2023.parquet"])
    if all(os.path.exists(f) for f in (ftr, fva, fte)):
        log(f"  split cache found for '{tag}' -> loading")
        return pd.read_parquet(ftr), pd.read_parquet(fva), pd.read_parquet(fte)
    log(f"  no split cache for '{tag}' -> building from {path}")
    tr, va, te = load_and_split(path)
    tr.to_parquet(ftr, index=False); va.to_parquet(fva, index=False); te.to_parquet(fte, index=False)
    return tr, va, te

# ==========================================================================
# models -- identical to run_shrinkage_head.py (must match for state_dict to load)
# ==========================================================================
class PartialPoolingHead(nn.Module):
    def __init__(s, h2, num_headings=NUM_HEADINGS, num_chapters=100, k=SHRINKAGE_K):
        super().__init__()
        s.head_w = nn.Embedding(num_headings, h2); s.head_b = nn.Embedding(num_headings, 1)
        s.chap_w = nn.Embedding(num_chapters, h2); s.chap_b = nn.Embedding(num_chapters, 1)
        s.k = k
        s.register_buffer('shrink_alpha', torch.zeros(num_headings))
    def set_shrinkage(s, heading_counts):
        alpha = torch.zeros(s.head_w.num_embeddings)
        for h, n in heading_counts.items():
            if 0 <= h < len(alpha): alpha[h] = n / (n + s.k)
        s.shrink_alpha.copy_(alpha)
    def forward(s, hid, heading_idx, chapter2_idx):
        w_own  = s.head_w(heading_idx); b_own  = s.head_b(heading_idx).squeeze(-1)
        w_chap = s.chap_w(chapter2_idx); b_chap = s.chap_b(chapter2_idx).squeeze(-1)
        a = s.shrink_alpha[heading_idx].unsqueeze(-1)
        w = a * w_own + (1 - a) * w_chap
        b = a.squeeze(-1) * b_own + (1 - a.squeeze(-1)) * b_chap
        return (hid * w).sum(-1) + b

class PerProductGAT(nn.Module):
    def __init__(s, inf, num_chapters=NUM_HEADINGS, h=32, h2=16, heads=4):
        super().__init__()
        s.c1 = GATConv(inf, h, heads)
        s.c2 = GATConv(h*heads, h2, heads)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, heading_idx, chapter2_idx):
        hid = torch.relu(s.c1(g, x).flatten(1))
        hid = s.c2(g, hid).mean(1)
        return s.head(hid, heading_idx, chapter2_idx)

class PerProductFusedGAT(nn.Module):
    def __init__(s, n_trade, n_gdelt, fusion, num_chapters=NUM_HEADINGS, proj=16, h=32, h2=16, heads=4):
        super().__init__()
        assert fusion in ('blend', 'attention')
        s.fusion = fusion; s.n_trade = n_trade
        s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
        if fusion == 'blend':
            s.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.c1 = GATConv(proj, h, heads)
        s.c2 = GATConv(h*heads, h2, heads)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, heading_idx, chapter2_idx):
        xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
        t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha); f = a*d + (1-a)*t
        else:
            st = torch.stack([t, d], dim=1); f = s.attn(st, st, st)[0].mean(1)
        hid = torch.relu(s.c1(g, f).flatten(1))
        hid = s.c2(g, hid).mean(1)
        return s.head(hid, heading_idx, chapter2_idx)

def build_gat_model(len_feat_cols, n_trade, n_gdelt, use_gdelt, fusion):
    if not use_gdelt or fusion == 'concat':
        return PerProductGAT(len_feat_cols)
    return PerProductFusedGAT(n_trade, n_gdelt, fusion=fusion)

class PerProductEdgeGATBackbone(nn.Module):
    def __init__(s, inf, ef, h=32, h2=16, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h*heads, ef, h2, heads, allow_zero_in_degree=True)
    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1)

class PerProductFusedEdgeModel(nn.Module):
    def __init__(s, n_trade, n_gdelt, n_edge, fusion, num_chapters=NUM_HEADINGS, proj=16, h=32, h2=16):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        if fusion == 'concat':
            inf = n_trade + n_gdelt
        else:
            s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend':
                s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention':
                s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        s.backbone = PerProductEdgeGATBackbone(inf, n_edge, h, h2)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, ef, heading_idx, chapter2_idx):
        if s.fusion == 'concat':
            node = x
        else:
            xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
            t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
            if s.fusion == 'blend':
                a = torch.sigmoid(s.alpha); node = a*d + (1-a)*t
            else:
                st = torch.stack([t, d], dim=1); node = s.attn(st, st, st)[0].mean(1)
        hid = s.backbone(g, node, ef)
        return s.head(hid, heading_idx, chapter2_idx)

# ==========================================================================
# eval WITH permutation hooks -- perm_col shuffles one feature column,
# perm_heading shuffles (heading, chapter2) jointly as a pair (so the
# combination stays internally consistent -- no impossible heading/chapter
# mismatches), testing the per-product-head MECHANISM itself, not a feature.
# ==========================================================================
def _eval_graph_perm(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                      agg_mode='first', bs=10000, perm_col=None, perm_cols=None, perm_heading=False, rng=None):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d,use_gdelt,gdelt_file,agg_mode)
    emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    if perm_col is not None:
        Xe = Xe.copy(); Xe[:, perm_col] = rng.permutation(Xe[:, perm_col])
    if perm_cols is not None:
        Xe = Xe.copy()
        perm_idx = rng.permutation(len(Xe))
        Xe[:, perm_cols] = Xe[perm_idx][:, perm_cols]   # ONE shared row-shuffle applied to all listed columns together
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    if perm_heading:
        perm_idx = rng.permutation(len(heading))
        heading = heading[perm_idx]; chap2 = chap2[perm_idx]
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    eg.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    eg=dgl.add_self_loop(eg)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.ndata['heading'], bg.ndata['chap2']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp

def _eval_edge_full_perm(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first', bs=20000,
                          perm_col=None, perm_cols=None, ef_perm_col=None, perm_heading=False, rng=None):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    if perm_col is not None:
        Xe = Xe.copy(); Xe[:, perm_col] = rng.permutation(Xe[:, perm_col])
    if perm_cols is not None:
        Xe = Xe.copy()
        perm_idx = rng.permutation(len(Xe))
        Xe[:, perm_cols] = Xe[perm_idx][:, perm_cols]
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    if ef_perm_col is not None:
        ef = ef.copy(); ef[:, ef_perm_col] = rng.permutation(ef[:, ef_perm_col])
    if perm_heading:
        perm_idx = rng.permutation(len(heading))
        heading = heading[perm_idx]; chap2 = chap2[perm_idx]
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    eg.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef'], bg.ndata['heading'], bg.ndata['chap2']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp

# ==========================================================================
# permutation-importance drivers
# ==========================================================================
def perm_importance_gat(model, feat_cols, data_2023, cmap, sf, st, use_gdelt, gdelt_file,
                         agg_mode='first', n_repeats=3, seed=0):
    yt, yp = _eval_graph_perm(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode)
    base = compute_metrics(yt, yp)
    rng = np.random.default_rng(seed)
    rows = []
    for j, col in enumerate(feat_cols):
        d_r2, d_rho = [], []
        for _ in range(n_repeats):
            yt2, yp2 = _eval_graph_perm(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                                         agg_mode, perm_col=j, rng=rng)
            m = compute_metrics(yt2, yp2)
            d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
        rows.append({'feature': col, 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    # the mechanism test: shuffle which product-head applies
    d_r2, d_rho = [], []
    for _ in range(n_repeats):
        yt2, yp2 = _eval_graph_perm(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                                     agg_mode, perm_heading=True, rng=rng)
        m = compute_metrics(yt2, yp2)
        d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
    rows.append({'feature': 'heading_idx', 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    # NEW: grouped test -- shuffle the whole GDELT block together, preserving within-row
    # correlation, so a correlated sibling column can't let the model "cheat" the way it can
    # when each GDELT column is tested individually
    if use_gdelt:
        gdelt_idx = [feat_cols.index(c) for c in GDELT_COLS]
        d_r2, d_rho = [], []
        for _ in range(n_repeats):
            yt2, yp2 = _eval_graph_perm(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                                         agg_mode, perm_cols=gdelt_idx, rng=rng)
            m = compute_metrics(yt2, yp2)
            d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
        rows.append({'feature': 'gdelt_block', 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    return base, pd.DataFrame(rows).sort_values('importance_log_r2', ascending=False)


def perm_importance_edgegat_full(model, data_2023, cmap, sf, st, gdelt_file, bilat_file,
                                  agg_mode='first', n_repeats=3, seed=0):
    yt, yp = _eval_edge_full_perm(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode)
    base = compute_metrics(yt, yp)
    rng = np.random.default_rng(seed)
    rows = []
    all_cols = [(j, c, 'node') for j, c in enumerate(FULL_NODE_COLS)] + \
               [(j, c, 'edge') for j, c in enumerate(FULL_EDGE_COLS)]
    for j, col, side in all_cols:
        d_r2, d_rho = [], []
        for _ in range(n_repeats):
            kw = {'perm_col': j, 'rng': rng} if side == 'node' else {'ef_perm_col': j, 'rng': rng}
            yt2, yp2 = _eval_edge_full_perm(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode, **kw)
            m = compute_metrics(yt2, yp2)
            d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
        rows.append({'feature': f'{side}:{col}', 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    d_r2, d_rho = [], []
    for _ in range(n_repeats):
        yt2, yp2 = _eval_edge_full_perm(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode,
                                         perm_heading=True, rng=rng)
        m = compute_metrics(yt2, yp2)
        d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
    rows.append({'feature': 'heading_idx', 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    # NEW: same grouped GDELT test as above -- these features all live in FULL_NODE_COLS
    # for EdgeGAT_full too, so the index lookup is against that list instead of feat_cols
    gdelt_idx = [FULL_NODE_COLS.index(c) for c in GDELT_COLS]
    d_r2, d_rho = [], []
    for _ in range(n_repeats):
        yt2, yp2 = _eval_edge_full_perm(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode,
                                         perm_cols=gdelt_idx, rng=rng)
        m = compute_metrics(yt2, yp2)
        d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
    rows.append({'feature': 'node:gdelt_block', 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    return base, pd.DataFrame(rows).sort_values('importance_log_r2', ascending=False)


# ==========================================================================
# main
# ==========================================================================
def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'

    log(f"checking {MODEL_DIR}/ for saved models...")
    if not os.path.isdir(MODEL_DIR):
        log(f"  [!] {MODEL_DIR}/ does not exist -- nothing is saved. Run run_shrinkage_head.py first.")
        return
    on_disk = {f[:-3] for f in os.listdir(MODEL_DIR) if f.endswith('.pt')}
    log(f"  found {len(on_disk)} saved shrinkage models:")
    for name in sorted(on_disk):
        log(f"    - {name}")

    available_targets = sorted(on_disk) if not TARGETS else [t for t in TARGETS if t in on_disk]
    missing = [] if not TARGETS else [t for t in TARGETS if t not in on_disk]
    if missing:
        log(f"  [!] requested but NOT found (edit TARGETS at the top of this script if these are wrong): {missing}")
    if not available_targets:
        log("  nothing to run -- stopping.")
        return
    log(f"  proceeding with {len(available_targets)} model(s): {available_targets}")

    out_csv = 'feature_importance_shrinkage.csv'
    all_rows = []
    already_done = set()
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        already_done = set(prior['model'].unique())
        all_rows.append(prior)
        log(f"  found existing {out_csv} with {len(already_done)} model(s) already done -- resuming, skipping those.")

    train_data, val_data, data_2023 = load_or_split(data_file)

    for name in available_targets:
        if name in already_done:
            log(f"skip {name} (already in {out_csv} from a previous run)")
            continue

        try:
            kind, fusion, scoring = parse_shrinkage_name(name)
        except UnparseableModelName as e:
            log(f"  [!] skipping {name}: {e}")
            continue

        log(f"\n--- {name} (kind={kind}, fusion={fusion}, scoring/combo={scoring}) ---")

        try:
            if kind == 'GAT':
                use_gdelt, gfile = CONDITIONS_BY_SCORING[scoring]
                feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
                model = build_gat_model(len(feat_cols), len(BASE_COLS), len(GDELT_COLS), use_gdelt, fusion)
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu'))
                agg = build_agg(train_data, use_gdelt, gfile, 'first')
                cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
                sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
                base, imp = perm_importance_gat(model, feat_cols, data_2023, cmap, sf, st, use_gdelt, gfile)
            else:
                gfile, bfile = SCORER_COMBOS[scoring]
                n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
                model = PerProductFusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu'))
                agg = build_agg(train_data, True, gfile, 'first')
                cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
                sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
                base, imp = perm_importance_edgegat_full(model, data_2023, cmap, sf, st, gfile, bfile)
        except Exception as e:
            log(f"  [!] {name} FAILED ({type(e).__name__}: {e}) -- skipping, other models unaffected.")
            continue

        log(f"  base: log_r2={base['log_r2']:.3f}  spearman={base['spearman']:.3f}")
        print(imp.to_string(index=False))
        heading_row = imp[imp.feature == 'heading_idx']
        if len(heading_row):
            log(f"  >>> per-product head mechanism (heading_idx) importance: "
                f"log_r2 drop={heading_row.iloc[0]['importance_log_r2']:.4f}")
        imp['model'] = name
        imp['dataset'] = 'all_products_shrinkage'
        all_rows.append(imp)

        # checkpoint after EVERY model -- a crash or Ctrl+C partway through no longer loses prior work
        pd.concat(all_rows, ignore_index=True).to_csv(out_csv, index=False)
        gc.collect()

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_csv, index=False)
    log(f"\nsaved -> {out_csv}  (per-model rows; compatible with analyze_feature_importance.py, run it on this file directly)")

    # ---- global aggregation across every model that ran ----
    log(f"\n=== GLOBAL: aggregated across {result['model'].nunique()} model(s) ===")
    r = result.copy()
    r['feature_clean'] = r['feature'].apply(lambda f: f.split(':', 1)[-1] if ':' in f else f)
    # refYear is constant in the 2023-only eval set -> permutation importance can't measure it,
    # same caveat as your earlier analyze_feature_importance.py -- exclude from the summary verdicts
    r_for_summary = r[r['feature_clean'] != 'refYear'].copy()

    summary = r_for_summary.groupby('feature_clean').agg(
        n_runs=('importance_log_r2', 'size'),
        mean_log_r2=('importance_log_r2', 'mean'),
        std_log_r2=('importance_log_r2', 'std'),
        min_log_r2=('importance_log_r2', 'min'),
        max_log_r2=('importance_log_r2', 'max'),
        mean_spearman=('importance_spearman', 'mean'),
        frac_negative=('importance_log_r2', lambda s: (s < 0).mean()),
    ).round(4).sort_values('mean_log_r2', ascending=False)

    pd.set_option('display.width', 160)
    print(summary.to_string())

    summary_csv = 'feature_importance_shrinkage_summary.csv'
    summary.to_csv(summary_csv)
    log(f"\nsaved -> {summary_csv}  (one row per feature, aggregated across all {result['model'].nunique()} models)")

    heading_row = summary.loc['heading_idx'] if 'heading_idx' in summary.index else None
    if heading_row is not None:
        beat_by = (summary['mean_log_r2'] > heading_row['mean_log_r2']).sum()
        log(f"\n>>> heading_idx (per-product head mechanism): mean importance = {heading_row['mean_log_r2']:.4f}, "
            f"n={int(heading_row['n_runs'])} models. Only {beat_by} feature(s) rank above it.")

    # flag any model where EVERY feature importance came back exactly zero -- the degenerate/collapsed-
    # training signature you saw with GAT_first_topk_attention_shrinkage
    for name, sub in result.groupby('model'):
        non_heading = sub[sub.feature != 'heading_idx']
        if len(non_heading) and (non_heading['importance_log_r2'] == 0).all():
            log(f"  [!] {name}: ALL non-heading features show exactly 0 importance -- "
                f"likely a collapsed/degenerate training run, not a real 'no features matter' finding. "
                f"Consider excluding this model before computing any aggregate statistic, or retrain it.")


if __name__ == "__main__":
    main()