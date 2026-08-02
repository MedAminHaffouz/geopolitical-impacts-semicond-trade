#!/usr/bin/env python3
"""
retrain_track_metrics.py
============================
Retrains ONE shrinkage-head model from scratch (GAT_first_* or EdgeGAT_full_*)
and records the full metrics set after every epoch, to diagnose whether the
raw-scale MSE/MAE/R2 blowup seen in recompute_full_metrics.py is structural
(expm1 amplification of small log-space error, present from early training)
or something that develops as training progresses.

IMPORTANT: full-batch GAT/EdgeGAT training on an embedding-table-heavy
shrinkage head converges slowly -- one gradient step per epoch needs many
epochs before per-heading embeddings move meaningfully off their random init.
An undertrained head produces unconstrained log-space predictions, which can
overflow expm1 to literal inf rather than just "a large outlier". To keep the
trace informative even during that early phase, metrics are computed on the
finite-prediction subset, with a separate `overflow_frac` column tracking
what fraction of the test set is currently overflowing.

Outputs (checkpointed after every epoch, safe to Ctrl+C):
    results/grids/results_reseed_epoch_trace_<model>.csv

Usage:
    python src/retraining/retrain_track_metrics.py [all_products_ready.parquet] \
        [--epochs 300] [--lr 1e-3] [--clip_grad 5.0] [--model NAME]

NAME options (GAT): GAT_first_topk_blend_shrinkage, GAT_first_topk_attention_shrinkage,
GAT_first_topk_concat_shrinkage, GAT_first_keepall_blend_shrinkage,
GAT_first_keepall_attention_shrinkage, GAT_first_keepall_concat_shrinkage,
GAT_first_none_shrinkage

NAME options (EdgeGAT_full): EdgeGAT_full_{attention,blend,concat}_{ka_ka,ka_tk,tk_ka,tk_tk}_shrinkage
"""
import os, sys, gc, argparse
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
# config -- identical to recompute_full_metrics.py
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
FULL_NODE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS = BILAT_COLS + ['dist']
MISSING = 'rf'
GDELT_DIR = 'data/interim'
KEEPALL=os.path.join(GDELT_DIR, 'gdelt_features_by_country_year.parquet'); TOPK=os.path.join(GDELT_DIR, 'gdelt_features_topk.parquet')
BILAT_KEEP=os.path.join(GDELT_DIR, 'gdelt_bilateral_by_pair_year.parquet'); BILAT_TOPK=os.path.join(GDELT_DIR, 'gdelt_bilateral_topk.parquet')
SCORER_COMBOS = {'ka_ka':(KEEPALL,BILAT_KEEP), 'ka_tk':(KEEPALL,BILAT_TOPK), 'tk_ka':(TOPK,BILAT_KEEP), 'tk_tk':(TOPK,BILAT_TOPK)}
CONDITIONS_BY_SCORING = {'none': (False, None), 'keepall': (True, KEEPALL), 'topk': (True, TOPK)}
NUM_HEADINGS = 10000
SHRINKAGE_K = 50

def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)
def hs_chapter_2digit(heading):
    return (np.asarray(heading, dtype=np.int64) // 100).astype(np.int64)

def parse_shrinkage_name(name):
    n = name.replace('_shrinkage', '')
    if n == 'GAT_first_none':
        return ('GAT', 'concat', 'none')
    if n.startswith('GAT_first_'):
        rest = n[len('GAT_first_'):]
        scoring, fusion = rest.rsplit('_', 1)
        return ('GAT', fusion, scoring)
    if n.startswith('EdgeGAT_full_'):
        rest = n[len('EdgeGAT_full_'):]
        fusion, combo = rest.split('_', 1)
        return ('EdgeGAT_full', fusion, combo)
    raise ValueError(f"'{name}' doesn't match any known naming pattern.")

# ==========================================================================
# metrics -- overflow-aware
# ==========================================================================
def _smape(yt, yp):
    d=(np.abs(yt)+np.abs(yp))/2; m=d>0
    return np.mean(np.abs(yt[m]-yp[m])/d[m])*100 if m.sum() else np.nan

def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_full_metrics(yt, yp, yt_log, yp_log):
    """Log-space metrics (mse/rmse/mae/r2/spearman on yt_log vs yp_log) are
    computed directly from the model's log1p-domain output -- they never call
    expm1, so they can't blow up from exponentiation the way raw-scale metrics
    can. These are the metrics to actually trust/report as primary.

    Raw-scale metrics still go through expm1 and are computed on the
    finite-prediction subset only, with overflow_frac tracking what fraction
    of the test set blew up. Keep these as secondary/diagnostic context."""
    # --- log-space (primary, robust) ---
    log_finite = np.isfinite(yp_log)
    n_total = len(yp_log); n_log_finite = int(log_finite.sum())
    ytl, ypl = yt_log[log_finite], yp_log[log_finite]
    log_mse = mean_squared_error(ytl, ypl) if n_log_finite >= 3 else np.nan
    log_metrics = dict(
        log_mse=log_mse, log_rmse=(log_mse**0.5 if n_log_finite >= 3 else np.nan),
        log_mae=(mean_absolute_error(ytl, ypl) if n_log_finite >= 3 else np.nan),
        log_r2=(r2_score(ytl, ypl) if n_log_finite > 2 else np.nan),
        log_spearman=_spearman(ytl, ypl) if n_log_finite >= 3 else np.nan,
        log_n_finite=n_log_finite, log_n_total=n_total,
    )

    # --- raw-scale (secondary, overflow-prone) ---
    finite = np.isfinite(yp)
    n_finite = int(finite.sum())
    overflow_frac = 1.0 - n_finite / n_total if n_total else np.nan
    if n_finite < 3:
        raw_metrics = dict(mse=np.nan, rmse=np.nan, mae=np.nan, smape=np.nan, r2=np.nan,
                    spearman=np.nan, top1_sqerr_share=np.nan, top5_sqerr_share=np.nan,
                    overflow_frac=overflow_frac, n_finite=n_finite, n_total=n_total,
                    max_abs_pred_finite=np.nan)
    else:
        ytf, ypf = yt[finite], yp[finite]
        sq = (ytf - ypf) ** 2
        total_sq = sq.sum()
        top1_share = sq.max() / total_sq if total_sq > 0 else np.nan
        top5_share = np.sort(sq)[-5:].sum() / total_sq if total_sq > 0 else np.nan
        mse = mean_squared_error(ytf, ypf)
        raw_metrics = dict(
            mse=mse, rmse=mse**0.5, mae=mean_absolute_error(ytf, ypf), smape=_smape(ytf, ypf),
            r2=r2_score(ytf, ypf), spearman=_spearman(ytf, ypf),
            top1_sqerr_share=top1_share, top5_sqerr_share=top5_share,
            overflow_frac=overflow_frac, n_finite=n_finite, n_total=n_total,
            max_abs_pred_finite=np.abs(ypf).max(),
        )
    return {**log_metrics, **raw_metrics}

# ==========================================================================
# data pipeline -- identical to recompute_full_metrics.py
# ==========================================================================
def add_gdelt(agg, gdelt_file):
    cy=pd.read_parquet(gdelt_file)
    out=agg.copy()
    out['_r']=out['reporterCode'].astype('Int64').astype(str); out['_y']=out['refYear'].astype('Int64').astype(str)
    cy['_r']=cy['reporterCode'].astype('Int64').astype(str);  cy['_y']=cy['year'].astype('Int64').astype(str)
    out=out.merge(cy[['_r','_y']+GDELT_COLS], on=['_r','_y'], how='left')
    out[GDELT_COLS]=out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_r','_y'])

def build_agg(df, use_gdelt, gdelt_file, agg_mode='first'):
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
    return train_data, val_data, te

def load_or_split(path):
    tag = os.path.splitext(os.path.basename(path))[0]
    split_dir = os.path.join('data/cache/split_cache', tag)
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
# models -- identical to recompute_full_metrics.py
# ==========================================================================
class PartialPoolingHead(nn.Module):
    def __init__(s, h2, num_headings=NUM_HEADINGS, num_chapters=100, k=SHRINKAGE_K):
        super().__init__()
        s.head_w = nn.Embedding(num_headings, h2); s.head_b = nn.Embedding(num_headings, 1)
        s.chap_w = nn.Embedding(num_chapters, h2); s.chap_b = nn.Embedding(num_chapters, 1)
        nn.init.normal_(s.head_w.weight, std=0.05); nn.init.zeros_(s.head_b.weight)
        nn.init.normal_(s.chap_w.weight, std=0.05); nn.init.zeros_(s.chap_b.weight)
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

class PerProductFusedGAT(nn.Module):
    def __init__(s, n_trade, n_gdelt, fusion, num_chapters=NUM_HEADINGS, proj=16, h=32, h2=16, heads=4):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
        if fusion == 'blend': s.alpha = nn.Parameter(torch.tensor(0.5))
        else: s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.c1 = GATConv(proj, h, heads); s.c2 = GATConv(h*heads, h2, heads)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, heading_idx, chapter2_idx):
        xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
        t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha); f = a*d + (1-a)*t
        else:
            st = torch.stack([t, d], dim=1); f = s.attn(st, st, st)[0].mean(1)
        hid = torch.relu(s.c1(g, f).flatten(1)); hid = s.c2(g, hid).mean(1)
        return s.head(hid, heading_idx, chapter2_idx)

class PerProductGAT(nn.Module):
    def __init__(s, inf, num_chapters=NUM_HEADINGS, h=32, h2=16, heads=4):
        super().__init__()
        s.c1 = GATConv(inf, h, heads); s.c2 = GATConv(h*heads, h2, heads)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, heading_idx, chapter2_idx):
        hid = torch.relu(s.c1(g, x).flatten(1)); hid = s.c2(g, hid).mean(1)
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
            if fusion == 'blend': s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention': s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
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
# graph construction (shared between train and eval passes)
# ==========================================================================
def make_graph_gat(df_rows, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode='first'):
    d = df_rows.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    agg=build_agg(d,use_gdelt,gdelt_file,agg_mode)
    emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    eg.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    eg.ndata['y']=torch.tensor(ye,dtype=torch.float32)
    eg=dgl.add_self_loop(eg)
    heading_counts = agg['chapter'].value_counts().to_dict()
    return eg, heading_counts

def make_graph_edgegat(df_rows, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first'):
    d = df_rows.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    eg.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    eg.ndata['y']=torch.tensor(ye,dtype=torch.float32)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    heading_counts = agg['chapter'].value_counts().to_dict()
    return eg, heading_counts

def forward_pass(model, kind, g):
    if kind == 'GAT':
        return model(g, g.ndata['feat'], g.ndata['heading'], g.ndata['chap2'])
    return model(g, g.ndata['feat'], g.edata['ef'], g.ndata['heading'], g.ndata['chap2'])

def eval_on_graph(model, kind, eg, st, bs=10000):
    model.eval(); preds=[]; N=eg.num_nodes()
    with torch.no_grad():
        for i in range(0,N,bs):
            bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
            preds.append(forward_pass(model, kind, bg).unsqueeze(1))
    scaled_pred = torch.cat(preds,0).view(-1,1).cpu().numpy()
    yp_log = st.inverse_transform(scaled_pred).flatten().astype(np.float64)
    yt_log = st.inverse_transform(eg.ndata['y'].cpu().numpy()).flatten().astype(np.float64)
    with np.errstate(over='ignore'):
        yp = np.expm1(yp_log)
    yt = np.expm1(yt_log)
    return yt, yp, yt_log, yp_log

# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_file', nargs='?', default='all_products_ready.parquet')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--clip_grad', type=float, default=5.0)
    ap.add_argument('--model', default='GAT_first_topk_blend_shrinkage')
    ap.add_argument('--eval_every', type=int, default=1, help='eval on test set every N epochs (default: every epoch)')
    args = ap.parse_args()

    data_file = args.data_file
    if not os.path.exists(data_file):
        candidate = os.path.join('data/processed', data_file)
        if os.path.exists(candidate):
            data_file = candidate
        else:
            log(f"  [!] Could not find '{data_file}'. Run from repo root, or pass a full path.")
            return

    kind, fusion, scoring = parse_shrinkage_name(args.model)
    log(f"tracking model: {args.model}  (kind={kind}, fusion={fusion}, scoring={scoring})")
    train_data, val_data, data_2023 = load_or_split(data_file)

    if kind == 'GAT':
        use_gdelt, gfile = CONDITIONS_BY_SCORING[scoring]
        feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
        n_trade, n_gdelt = len(BASE_COLS), len(GDELT_COLS)
        agg = build_agg(train_data, use_gdelt, gfile, 'first')
        cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
        sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
        log("building train graph ...")
        train_g, heading_counts = make_graph_gat(train_data, cmap, sf, st, feat_cols, use_gdelt, gfile, 'first')
        log("building 2023 test graph ...")
        test_g, _ = make_graph_gat(data_2023, cmap, sf, st, feat_cols, use_gdelt, gfile, 'first')
        model = build_gat_model(len(feat_cols), n_trade, n_gdelt, use_gdelt, fusion)
    else:
        gfile, bfile = SCORER_COMBOS[scoring]
        n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
        agg = build_agg(train_data, True, gfile, 'first')
        cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
        sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
        log("building train graph ...")
        train_g, heading_counts = make_graph_edgegat(train_data, cmap, sf, st, gfile, bfile, 'first')
        log("building 2023 test graph ...")
        test_g, _ = make_graph_edgegat(data_2023, cmap, sf, st, gfile, bfile, 'first')
        model = PerProductFusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)

    model.head.set_shrinkage(heading_counts)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # mini-batch subgraph training -- matches run_shrinkage_head.py's convention
    # (bs=10000 for GAT, bs=20000 for EdgeGAT) instead of one full-graph backward
    # pass per epoch. This is the difference that keeps memory bounded: each
    # step only materializes activations/grads for a bs-node subgraph, not the
    # whole training graph plus every embedding row at once.
    train_bs = 10000 if kind == 'GAT' else 20000
    N_train = train_g.num_nodes()
    n_batches = N_train // train_bs + (N_train % train_bs > 0)
    y_train = train_g.ndata['y'].squeeze(-1)

    os.makedirs('results/grids', exist_ok=True)
    out_csv = f'results/grids/results_reseed_epoch_trace_{args.model}.csv'
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(N_train)
        for i in range(n_batches):
            bn = perm[i*train_bs:min((i+1)*train_bs, N_train)]
            bg = train_g.subgraph(bn)
            opt.zero_grad()
            pred = forward_pass(model, kind, bg)
            loss = loss_fn(pred, y_train[bn])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            epoch_loss += loss.item() * len(bn)
        epoch_loss /= N_train

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            yt, yp, yt_log, yp_log = eval_on_graph(model, kind, test_g, st)
            m = compute_full_metrics(yt, yp, yt_log, yp_log)
            m['epoch'] = epoch; m['train_loss'] = epoch_loss
            rows.append(m)
            log(f"  epoch {epoch:4d}  train_loss={epoch_loss:.5f}  "
                f"log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}  log_spearman={m['log_spearman']:.3f}  |  "
                f"raw_r2={m['r2']:.3g}  raw_mse={m['mse']:.3g}  overflow_frac={m['overflow_frac']:.3f}  "
                f"top1_sqerr_share={m['top1_sqerr_share']:.3f}")
            pd.DataFrame(rows).set_index('epoch').to_csv(out_csv)
        gc.collect()

    log(f"done -> {out_csv}")
    print("\n=== Epoch-by-epoch trace (key columns) ===")
    cols = ['train_loss','log_mse','log_r2','log_spearman','r2','mse','overflow_frac','top1_sqerr_share']
    print(pd.DataFrame(rows).set_index('epoch')[cols].round(4).to_string())

if __name__ == "__main__":
    main()
