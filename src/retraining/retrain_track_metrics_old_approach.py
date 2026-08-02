#!/usr/bin/env python3
"""
retrain_track_metrics_old_approach.py
=========================================
Retrains ONE GAT config and ONE EdgeGAT_full config from scratch, OLD approach
(train_benchmark.py's plain models -- no per-product shrinkage head), on each
of chips.parquet and all_products_ready.parquet, tracking full metrics after
every epoch. Companion to src/retraining/retrain_track_metrics.py, which does
the same thing for the NEW approach (pooling+shrinkage) -- that script already
exists and covers GAT_first_*/EdgeGAT_full_*_shrinkage models; this one covers
the pre-shrinkage baseline so the two are directly comparable epoch-by-epoch.

NOTE: shrinkage's premise is pooling weight across many HS headings, so running
the NEW-approach script against chips.parquet (a handful of headings) isn't a
meaningful configuration -- the new-approach side of this comparison should
stay scoped to all_products_ready.parquet. This script's chips.parquet run is
specifically an OLD-approach condition (matching train_benchmark.py's own use
of chips.parquet for the dedicated per-category grid).

Same mini-batch convention as every other retraining script in this repo:
bs=10000 (GAT) / bs=20000 (EdgeGAT), lr=0.01, log-space metrics computed
directly (never routed through expm1, so they can't blow up from overflow).

Outputs (checkpointed after every epoch, safe to Ctrl+C):
    results/grids/results_reseed_epoch_trace_OLD_<model>_<data_tag>.csv

Usage:
    python src/retraining/retrain_track_metrics_old_approach.py \\
        [--gat_model GAT_first_topk_blend] [--edge_model EdgeGAT_full_attention_ka_ka] \\
        [--epochs 100] [--data_files chips.parquet,all_products_ready.parquet]

Model name options mirror train_benchmark.py's naming:
    GAT: GAT_first_none, GAT_first_{keepall,topk}_{concat,blend,attention}
    EdgeGAT: EdgeGAT_full_{concat,blend,attention}_{ka_ka,ka_tk,tk_ka,tk_tk}
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
# config -- copied verbatim from train_benchmark.py / recompute_v2_benchmark_metrics.py
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
SCORER_COMBOS = {'ka_ka':(KEEPALL,BILAT_KEEP), 'ka_tk':(KEEPALL,BILAT_TOPK), 'tk_ka':(TOPK,BILAT_KEEP), 'tk_tk':(TOPK,BILAT_TOPK)}

def parse_old_name(name):
    if name == 'GAT_first_none':
        return ('GAT', 'none', 'none')
    if name.startswith('GAT_first_'):
        rest = name[len('GAT_first_'):]
        scoring, fusion = rest.rsplit('_', 1)
        return ('GAT', fusion, scoring)
    if name.startswith('EdgeGAT_full_'):
        rest = name[len('EdgeGAT_full_'):]
        fusion, combo = rest.split('_', 1)
        return ('EdgeGAT_full', fusion, combo)
    raise ValueError(f"'{name}' doesn't match GAT_first_* or EdgeGAT_full_* naming.")

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
    n_finite = int(finite.sum())
    ytf, ypf = (yt[finite], yp[finite]) if n_finite >= 3 else (yt, yp)
    mse = mean_squared_error(ytf, ypf)

    return dict(
        mse=mse, rmse=mse**0.5, mae=mean_absolute_error(ytf, ypf), smape=_smape(ytf, ypf), r2=r2_score(ytf, ypf),
        log_mse=log_mse, log_rmse=(log_mse**0.5 if n_log_finite >= 3 else np.nan),
        log_mae=(mean_absolute_error(ytl, ypl) if n_log_finite >= 3 else np.nan),
        log_r2=(r2_score(ytl, ypl) if n_log_finite > 2 else np.nan),
        spearman=_spearman(ytf, ypf),
        overflow_frac=1.0 - n_finite/n_total if n_total else np.nan,
        n_finite=n_finite, n_total=n_total,
    )

# ==========================================================================
# data pipeline
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
# model classes -- copied verbatim from train_benchmark.py (no shrinkage head:
# single scalar output per node, this IS the old approach)
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
    def __init__(s, n_trade, n_gdelt, n_edge, fusion='concat', proj=16, h=32):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        if fusion == 'concat':
            inf = n_trade + n_gdelt
        else:
            s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend': s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention': s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        s.backbone = EdgeGAT(inf, n_edge, h)
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
# graph construction + eval
# ==========================================================================
def make_graph_gat(df_rows, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode='first'):
    d = df_rows.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    agg=build_agg(d,use_gdelt,gdelt_file,agg_mode)
    emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['y']=torch.tensor(ye,dtype=torch.float32)
    eg=dgl.add_self_loop(eg)
    return eg

def make_graph_edgegat(df_rows, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first'):
    d = df_rows.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['y']=torch.tensor(ye,dtype=torch.float32)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    return eg

def forward_pass(model, kind, fusion, nt, g):
    if kind == 'GAT':
        ft = g.ndata['feat']
        if fusion in ('blend','attention'):
            return model(g, ft[:, :nt], ft[:, nt:])
        return model(g, ft)
    return model(g, g.ndata['feat'], g.edata['ef'])

def eval_on_graph(model, kind, fusion, nt, eg, st, bs=10000):
    model.eval(); preds=[]; N=eg.num_nodes()
    with torch.no_grad():
        for i in range(0,N,bs):
            bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
            preds.append(forward_pass(model, kind, fusion, nt, bg).unsqueeze(1))
    scaled_pred = torch.cat(preds,0).view(-1,1).cpu().numpy()
    yp_log = st.inverse_transform(scaled_pred).flatten().astype(np.float64)
    yt_log = st.inverse_transform(eg.ndata['y'].cpu().numpy()).flatten().astype(np.float64)
    with np.errstate(over='ignore'):
        yp = np.expm1(yp_log)
    yt = np.expm1(yt_log)
    return yt, yp, yt_log, yp_log

def build_model(kind, fusion, feat_cols, nt, n_gdelt, n_edge=None):
    if kind == 'GAT':
        if fusion == 'blend': return BlendGAT(nt, n_gdelt)
        if fusion == 'attention': return AttnGAT(nt, n_gdelt)
        return GATRegressionModel(len(feat_cols))
    return FusedEdgeModel(nt, n_gdelt, n_edge, fusion=fusion)

# ==========================================================================
def train_and_track(model_name, data_file, epochs, lr, clip_grad):
    kind, fusion, scoring = parse_old_name(model_name)
    data_tag = os.path.splitext(os.path.basename(data_file))[0]
    log(f"=== {model_name}  on  {data_file}  (kind={kind}, fusion={fusion}, scoring={scoring}) ===")
    train_data, val_data, data_2023 = load_or_split(data_file)

    if kind == 'GAT':
        use_gdelt = scoring != 'none'
        gdelt_file = {'keepall': KEEPALL, 'topk': TOPK, 'none': None}[scoring]
        feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
        nt, n_gdelt = len(BASE_COLS), len(GDELT_COLS)
        agg = build_agg(train_data, use_gdelt, gdelt_file, 'first')
        cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
        sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
        log("  building train graph ..."); train_g = make_graph_gat(train_data, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, 'first')
        log("  building 2023 test graph ..."); test_g = make_graph_gat(data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, 'first')
        model = build_model('GAT', fusion, feat_cols, nt, n_gdelt)
        train_bs = 10000
    else:
        gdelt_file, bilat_file = SCORER_COMBOS[scoring]
        nt = len(FULL_NODE_COLS) - len(GDELT_COLS); n_gdelt = len(GDELT_COLS); n_edge = len(FULL_EDGE_COLS)
        agg = build_agg(train_data, True, gdelt_file, 'first')
        cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
        sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
        log("  building train graph ..."); train_g = make_graph_edgegat(train_data, cmap, sf, st, gdelt_file, bilat_file, 'first')
        log("  building 2023 test graph ..."); test_g = make_graph_edgegat(data_2023, cmap, sf, st, gdelt_file, bilat_file, 'first')
        model = build_model('EdgeGAT_full', fusion, None, nt, n_gdelt, n_edge)
        train_bs = 20000

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N_train = train_g.num_nodes()
    n_batches = N_train // train_bs + (N_train % train_bs > 0)
    y_train = train_g.ndata['y'].squeeze(-1)

    os.makedirs('results/grids', exist_ok=True)
    out_csv = f'results/grids/results_reseed_epoch_trace_OLD_{model_name}_{data_tag}.csv'
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(N_train)
        for i in range(n_batches):
            bn = perm[i*train_bs:min((i+1)*train_bs, N_train)]
            bg = train_g.subgraph(bn)
            opt.zero_grad()
            pred = forward_pass(model, kind, fusion, nt, bg)
            loss = loss_fn(pred, y_train[bn])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            epoch_loss += loss.item() * len(bn)
        epoch_loss /= N_train

        yt, yp, ytl, ypl = eval_on_graph(model, kind, fusion, nt, test_g, st)
        m = compute_full_metrics(yt, yp, ytl, ypl)
        m['epoch'] = epoch; m['train_loss'] = epoch_loss
        rows.append(m)
        log(f"  epoch {epoch:4d}  train_loss={epoch_loss:.5f}  "
            f"log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}  spearman={m['spearman']:.3f}  "
            f"overflow_frac={m['overflow_frac']:.3f}")
        pd.DataFrame(rows).set_index('epoch').to_csv(out_csv)
        gc.collect()

    log(f"  done -> {out_csv}")
    return out_csv

# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gat_model', default='GAT_first_topk_blend')
    ap.add_argument('--edge_model', default='EdgeGAT_full_attention_ka_ka')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--clip_grad', type=float, default=5.0)
    ap.add_argument('--data_files', default='chips.parquet,all_products_ready.parquet',
                     help='comma-separated list of data files, run for both models each')
    args = ap.parse_args()

    data_files = [f.strip() for f in args.data_files.split(',')]
    resolved = []
    for f in data_files:
        if os.path.exists(f):
            resolved.append(f)
        elif os.path.exists(os.path.join('data/processed', f)):
            resolved.append(os.path.join('data/processed', f))
        else:
            log(f"  [!] Could not find '{f}'. Run from repo root, or pass a full path. Skipping."); continue

    produced = []
    for data_file in resolved:
        for model_name in [args.gat_model, args.edge_model]:
            produced.append(train_and_track(model_name, data_file, args.epochs, args.lr, args.clip_grad))

    print("\n=== all traces written ===")
    for p in produced:
        print(f"  {p}")

if __name__ == "__main__":
    main()
