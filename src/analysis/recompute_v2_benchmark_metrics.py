#!/usr/bin/env python3
"""
recompute_v2_benchmark_metrics.py
=====================================
Adds log_mse/log_rmse/log_mae/log_r2 (computed directly from log1p-space,
never routed through expm1) to train_benchmark.py's saved models -- the
"old approach" / dedicated per-category benchmark grid (results_v2_*.csv),
which only ever logged mae/rmse/r2/log_r2/spearman/smape (raw-scale MAE/RMSE
and log-space R2, but never log-space MAE/RMSE). No retraining: loads the
already-saved .pt weights from models/trained_models/ (per README's own
"## Models" section -- train_benchmark.py itself writes to a bare
trained_models/, which is the repo-root-relative path bug fixed here).

19 configs per data file, matching run_shrinkage_head.py's own grid:
    GAT_first_none
    GAT_first_{keepall,topk}_{concat,blend,attention}   (6)
    EdgeGAT_full_{concat,blend,attention}_{ka_ka,ka_tk,tk_ka,tk_tk}   (12)

Usage:
    python src/analysis/recompute_v2_benchmark_metrics.py [data_file]
    (default data_file = chips.parquet, matching the "old approach" comparison;
     pass medical.parquet / vehicles.parquet / oil.parquet / electronics.parquet
     for the other per-category benchmark grids)
"""
import os, sys, gc, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl, dgl.function as fn
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
# config -- copied verbatim from train_benchmark.py
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
FULL_NODE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS = BILAT_COLS + ['dist']
MISSING = 'rf'
GDELT_DIR = 'data/interim'
KEEPALL=os.path.join(GDELT_DIR,'gdelt_features_by_country_year.parquet'); TOPK=os.path.join(GDELT_DIR,'gdelt_features_topk.parquet')
BILAT_KEEP=os.path.join(GDELT_DIR,'gdelt_bilateral_by_pair_year.parquet'); BILAT_TOPK=os.path.join(GDELT_DIR,'gdelt_bilateral_topk.parquet')
SCORER_COMBOS = [('ka_ka',KEEPALL,BILAT_KEEP), ('tk_tk',TOPK,BILAT_TOPK), ('ka_tk',KEEPALL,BILAT_TOPK), ('tk_ka',TOPK,BILAT_KEEP)]
MODEL_DIR = 'models/trained_models'   # train_benchmark.py itself writes bare 'trained_models/' -- fixed here to match README

# ==========================================================================
# metrics -- log-space computed directly (never through expm1)
# ==========================================================================
def _smape(yt, yp):
    d=(np.abs(yt)+np.abs(yp))/2; m=d>0
    return np.mean(np.abs(yt[m]-yp[m])/d[m])*100 if m.sum() else np.nan

def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_full_metrics(yt, yp, yt_log, yp_log):
    log_finite = np.isfinite(yp_log)
    n_total = len(yp_log); n_log_finite = int(log_finite.sum())
    ytl, ypl = yt_log[log_finite], yp_log[log_finite]
    log_mse = mean_squared_error(ytl, ypl) if n_log_finite >= 3 else np.nan

    finite = np.isfinite(yp)
    ytf, ypf = (yt[finite], yp[finite]) if finite.sum() >= 3 else (yt, yp)
    mse = mean_squared_error(ytf, ypf)

    return dict(
        mse=mse, rmse=mse**0.5, mae=mean_absolute_error(ytf, ypf), smape=_smape(ytf, ypf), r2=r2_score(ytf, ypf),
        log_mse=log_mse, log_rmse=(log_mse**0.5 if n_log_finite >= 3 else np.nan),
        log_mae=(mean_absolute_error(ytl, ypl) if n_log_finite >= 3 else np.nan),
        log_r2=(r2_score(ytl, ypl) if n_log_finite > 2 else np.nan),
        spearman=_spearman(ytf, ypf),
        overflow_frac=1.0 - finite.sum()/n_total if n_total else np.nan,
    )

# ==========================================================================
# data pipeline -- copied from train_benchmark.py, split cache path fixed
# to data/cache/split_cache (matching the rest of the repo's convention;
# train_benchmark.py itself uses a bare 'split_cache/')
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
# model classes -- copied verbatim from train_benchmark.py (needed for
# state_dict compatibility; architecture must match exactly)
# ==========================================================================
class GATRegressionModel(nn.Module):
    def __init__(s,inf,h=32,heads=4):
        super().__init__(); s.c1=GATConv(inf,h,heads); s.c2=GATConv(h*heads,1,heads)
    def forward(s,g,x): return s.c2(g, torch.relu(s.c1(g,x).flatten(1))).mean(1)

class BlendGAT(nn.Module):
    def __init__(s,nt,ng,proj=16,h=32,heads=4):
        super().__init__(); s.tp=nn.Linear(nt,proj); s.gp=nn.Linear(ng,proj)
        s.alpha=nn.Parameter(torch.tensor(0.5)); s.c1=GATConv(proj,h,heads); s.c2=GATConv(h*heads,1,heads)
    def forward(s,g,xt,xg):
        a=torch.sigmoid(s.alpha); f=a*torch.relu(s.gp(xg))+(1-a)*torch.relu(s.tp(xt))
        return s.c2(g, torch.relu(s.c1(g,f).flatten(1))).mean(1)

class AttnGAT(nn.Module):
    def __init__(s,nt,ng,proj=16,h=32,heads=4):
        super().__init__(); s.tp=nn.Linear(nt,proj); s.gp=nn.Linear(ng,proj)
        s.attn=nn.MultiheadAttention(proj,1,batch_first=True); s.c1=GATConv(proj,h,heads); s.c2=GATConv(h*heads,1,heads)
    def forward(s,g,xt,xg):
        st=torch.stack([torch.relu(s.tp(xt)),torch.relu(s.gp(xg))],dim=1)
        f=s.attn(st,st,st)[0].mean(1)
        return s.c2(g, torch.relu(s.c1(g,f).flatten(1))).mean(1)

class EdgeGAT(nn.Module):
    def __init__(s, inf, ef, h=32, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h*heads, ef, 1, heads, allow_zero_in_degree=True)
    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1).squeeze(-1)

class FusedEdgeModel(nn.Module):
    def __init__(s, kind, n_trade, n_gdelt, n_edge, fusion='concat', proj=16, h=32):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        if fusion == 'concat':
            inf = n_trade + n_gdelt
        else:
            s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend': s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention': s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        s.backbone = EdgeGAT(inf, n_edge, h)   # kind == 'EdgeGAT' is the only one recompute_full_metrics.py/this script cover
    def forward(s, g, x, ef):
        if s.fusion == 'concat':
            node = x
        else:
            xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
            t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
            if s.fusion == 'blend':
                a = torch.sigmoid(s.alpha); node = a*d + (1-a)*t
            else:
                st = torch.stack([t, d], dim=1); node = s.attn(st, st, st)[0].mean(1)
        return s.backbone(g, node, ef)

# ==========================================================================
# eval -- returns yt, yp (raw, expm1'd) AND yt_log, yp_log (direct, pre-expm1)
# ==========================================================================
def eval_gat(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode, fusion, nt, bs=10000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None,None,None
    agg=build_agg(d,use_gdelt,gdelt_file,agg_mode)
    emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32); eg=dgl.add_self_loop(eg)
    fused = fusion in ('blend','attention')
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn)); ft=bg.ndata['feat']
        with torch.no_grad():
            out = model(bg, ft[:, :nt], ft[:, nt:]) if fused else model(bg, ft)
            preds.append(out.unsqueeze(1))
    scaled_pred = torch.cat(preds,0).view(-1,1).cpu().numpy()
    yp_log = st.inverse_transform(scaled_pred).flatten().astype(np.float64)
    yt_log = st.inverse_transform(ye).flatten().astype(np.float64)
    with np.errstate(over='ignore'):
        yp = np.expm1(yp_log)
    yt = np.expm1(yt_log)
    return yt, yp, yt_log, yp_log

def eval_edgegat_full(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode, bs=20000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None,None,None
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef']).unsqueeze(1))
    scaled_pred = torch.cat(preds,0).view(-1,1).cpu().numpy()
    yp_log = st.inverse_transform(scaled_pred).flatten().astype(np.float64)
    yt_log = st.inverse_transform(ye).flatten().astype(np.float64)
    with np.errstate(over='ignore'):
        yp = np.expm1(yp_log)
    yt = np.expm1(yt_log)
    return yt, yp, yt_log, yp_log

def fit_scalers_gat(train_data, feat_cols, use_gdelt, gdelt_file, agg_mode='first'):
    agg = build_agg(train_data, use_gdelt, gdelt_file, agg_mode)
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    sf, st = MinMaxScaler(), MinMaxScaler()
    sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
    return cmap, sf, st

def fit_scalers_edge(train_data, gdelt_file, agg_mode='first'):
    agg = build_agg(train_data, True, gdelt_file, agg_mode)
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    sf, st = MinMaxScaler(), MinMaxScaler()
    sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
    return cmap, sf, st

# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_file', nargs='?', default='chips.parquet')
    args = ap.parse_args()

    data_file = args.data_file
    if not os.path.exists(data_file):
        candidate = os.path.join('data/processed', data_file)
        if os.path.exists(candidate):
            data_file = candidate
        else:
            log(f"  [!] Could not find '{data_file}'. Run from repo root, or pass a full path.")
            return

    tag = os.path.splitext(os.path.basename(data_file))[0]
    log(f"recomputing v2 benchmark metrics for '{tag}' ({data_file})")
    train_data, val_data, data_2023 = load_or_split(data_file)

    rows = []

    # --- GAT_first_none ---
    name = 'GAT_first_none'
    model_path = os.path.join(MODEL_DIR, f'{name}.pt')
    if os.path.exists(model_path):
        feat_cols = BASE_COLS
        cmap, sf, st = fit_scalers_gat(train_data, feat_cols, False, None)
        model = GATRegressionModel(len(feat_cols))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        yt, yp, ytl, ypl = eval_gat(model, data_2023, cmap, sf, st, feat_cols, False, None, 'first', 'concat', len(BASE_COLS))
        m = compute_full_metrics(yt, yp, ytl, ypl); m['model'] = name
        log(f"  {name}: log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}")
        rows.append(m)
    else:
        log(f"  [!] {model_path} not found -- skipping.")
    gc.collect()

    # --- GAT_first_{scoring}_{fusion} ---
    for scoring, gdelt_file in [('keepall', KEEPALL), ('topk', TOPK)]:
        for fusion in ['concat', 'blend', 'attention']:
            name = f'GAT_first_{scoring}_{fusion}'
            model_path = os.path.join(MODEL_DIR, f'{name}.pt')
            if not os.path.exists(model_path):
                log(f"  [!] {model_path} not found -- skipping."); continue
            feat_cols = BASE_COLS + GDELT_COLS
            nt = len(BASE_COLS)
            cmap, sf, st = fit_scalers_gat(train_data, feat_cols, True, gdelt_file)
            if fusion == 'blend': model = BlendGAT(nt, len(GDELT_COLS))
            elif fusion == 'attention': model = AttnGAT(nt, len(GDELT_COLS))
            else: model = GATRegressionModel(len(feat_cols))
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            yt, yp, ytl, ypl = eval_gat(model, data_2023, cmap, sf, st, feat_cols, True, gdelt_file, 'first', fusion, nt)
            m = compute_full_metrics(yt, yp, ytl, ypl); m['model'] = name
            log(f"  {name}: log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}")
            rows.append(m); gc.collect()

    # --- EdgeGAT_full_{fusion}_{combo} ---
    for combo, gdelt_file, bilat_file in SCORER_COMBOS:
        for fusion in ['concat', 'blend', 'attention']:
            name = f'EdgeGAT_full_{fusion}_{combo}'
            model_path = os.path.join(MODEL_DIR, f'{name}.pt')
            if not os.path.exists(model_path):
                log(f"  [!] {model_path} not found -- skipping."); continue
            n_trade = len(FULL_NODE_COLS) - len(GDELT_COLS); n_gdelt = len(GDELT_COLS); n_edge = len(FULL_EDGE_COLS)
            cmap, sf, st = fit_scalers_edge(train_data, gdelt_file)
            model = FusedEdgeModel('EdgeGAT', n_trade, n_gdelt, n_edge, fusion=fusion)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            yt, yp, ytl, ypl = eval_edgegat_full(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, 'first')
            m = compute_full_metrics(yt, yp, ytl, ypl); m['model'] = name
            log(f"  {name}: log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}")
            rows.append(m); gc.collect()

    if not rows:
        log("  [!] no models found -- nothing to report. Check MODEL_DIR / data_file above."); return

    out = pd.DataFrame(rows).set_index('model')
    out_path = f'results/grids/results_v2_{tag}_full_metrics.csv'
    os.makedirs('results/grids', exist_ok=True)
    out.to_csv(out_path)
    print(f"\nsaved -> {out_path}")
    print(out.round(4).to_string())

if __name__ == "__main__":
    main()
