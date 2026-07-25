#!/usr/bin/env python3
"""
Semiconductor trade-flow benchmark — terminal training script (reduced, RAM-light).
Trains all models, saves each to trained_models/, logs to results_v2.csv.
Split is cached to split_cache/ so it is computed only once.
No visualization — training + saving only.

Usage:   python train_benchmark.py [data_file]
         (default data_file = all_products_ready.parquet; pass electronics.parquet to iterate fast)
"""
import os, sys, csv, gc, pickle
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl, dgl.function as fn
from dgl.nn import GATConv, GraphConv, GATv2Conv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings("ignore")

# ---------- [INFO]-style logging ----------
def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)

# ---------- config ----------
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
RESULTS_FILE = 'results_v2.csv'
PREDS = {}
MISSING = 'rf'  # 'drop' | 'median' | 'rf'
KEEPALL='gdelt_features_by_country_year.parquet'; TOPK='gdelt_features_topk.parquet'
BILAT_KEEP='gdelt_bilateral_by_pair_year.parquet'; BILAT_TOPK='gdelt_bilateral_topk.parquet'
AGG_SPEC = {'primaryValue':'sum','dist':'first','gdpcap_d':'first','gdpcap_o':'first','pop_o':'first','pop_d':'first'}
CONDITIONS = [('none',False,None),('keepall',True,KEEPALL),('topk',True,TOPK)]
SCORER_COMBOS = [('ka_ka',KEEPALL,BILAT_KEEP),('tk_tk',TOPK,BILAT_TOPK),('ka_tk',KEEPALL,BILAT_TOPK),('tk_ka',TOPK,BILAT_KEEP)]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"device = {DEVICE}")

# ===== metrics + logging =====

def _smape(yt, yp):
    d=(np.abs(yt)+np.abs(yp))/2; m=d>0
    return np.mean(np.abs(yt[m]-yp[m])/d[m])*100 if m.sum() else np.nan

def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_metrics(yt, yp):
    ytl=np.log1p(np.clip(yt,0,None)); ypl=np.log1p(np.clip(yp,0,None))
    return dict(mae=mean_absolute_error(yt,yp),
                rmse=mean_squared_error(yt,yp)**0.5,
                r2=r2_score(yt,yp),
                log_r2=(r2_score(ytl,ypl) if len(yt)>2 else np.nan),
                spearman=_spearman(yt,yp),
                smape=_smape(yt,yp))

def log_result(model, fusion, scoring, gdelt, risk_level, agg_mode, missing, tranche, met, n, notes=''):
    row={'timestamp':datetime.now().isoformat(timespec='seconds'),
         'model':model,'fusion':fusion,'scoring':scoring,'gdelt':gdelt,
         'risk_level':risk_level,'agg_mode':agg_mode,'missing':missing,'tranche':tranche,   # <-- added 'missing'
         'mae':round(met['mae'],2),'rmse':round(met['rmse'],2),'r2':round(met['r2'],4),
         'log_r2':round(float(met['log_r2']),4),'spearman':round(float(met['spearman']),4),
         'smape':round(met['smape'],2),'n':int(n),'notes':notes}
    new=not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row.keys())
        if new: w.writeheader()
        w.writerow(row)
    return row

def report_and_log(model, use_gdelt, scoring, agg_mode, y_true, y_pred, fusion_override=None, risk_level='node'):
    fus = fusion_override if fusion_override is not None else ('concat' if use_gdelt else 'none')
    sc  = scoring if use_gdelt else 'none'
    PREDS[f'{model}|{fus}|{sc}|{agg_mode}|{risk_level}'] = (y_true, y_pred)
    print(f'\n=== {model} gdelt={use_gdelt} scoring={sc} agg={agg_mode} fusion={fus} risk={risk_level} ===')
    for label, thr in [('all',None),('1M',1e6),('10M',1e7),('100M',1e8)]:
        m = np.ones_like(y_true,bool) if thr is None else (y_true>=thr)
        if m.sum()<10: continue
        met=compute_metrics(y_true[m],y_pred[m])
        print(f"  {label:4s} n={int(m.sum()):>6,} R2={met['r2']:.3f} logR2={met['log_r2']:.3f} "
              f"rho={met['spearman']:.3f} RMSE={met['rmse']:,.0f}")
        log_result(model, fus, sc, use_gdelt, risk_level, agg_mode, MISSING, label, met, int(m.sum()))

# ===== features / aggregation =====

def add_gdelt(agg, gdelt_file):
    cy=pd.read_parquet(gdelt_file)
    out=agg.copy()
    out['_r']=out['reporterCode'].astype('Int64').astype(str); out['_y']=out['refYear'].astype('Int64').astype(str)
    cy['_r']=cy['reporterCode'].astype('Int64').astype(str);  cy['_y']=cy['year'].astype('Int64').astype(str)
    out=out.merge(cy[['_r','_y']+GDELT_COLS], on=['_r','_y'], how='left')
    out[GDELT_COLS]=out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_r','_y'])

def build_agg(df, use_gdelt, gdelt_file, agg_mode='sum'):
    if agg_mode == 'spec':                       # Step B: per-feature aggregation
        agg = df.groupby(['refYear','reporterCode','cmdCode']).agg(
            **{k:(k,op) for k,op in AGG_SPEC.items()}).reset_index()
    else:                                        # old behaviour kept for comparison
        grav = 'first' if agg_mode=='first' else 'sum'
        agg = (df.groupby(['refYear','reporterCode','cmdCode'])
                 .agg(primaryValue=('primaryValue','mean'), dist=('dist','first'),
                      gdpcap_d=('gdpcap_d',grav), gdpcap_o=('gdpcap_o',grav),
                      pop_o=('pop_o',grav), pop_d=('pop_d',grav)).reset_index())
    agg['y_log'] = np.log1p(agg['primaryValue'].clip(lower=0))   # Step A: log AFTER agg
    if use_gdelt:
        agg = add_gdelt(agg, gdelt_file)
    return agg

def edge_features(df_rows, bilat_file):
    '''Per-row (reporter,partner,year) bilateral risk vector, aligned to df_rows order.'''
    b=pd.read_parquet(bilat_file)
    k=df_rows[['reporterCode','partnerCode','refYear']].copy()
    k['reporterCode']=k['reporterCode'].astype('Int64'); k['partnerCode']=k['partnerCode'].astype('Int64'); k['refYear']=k['refYear'].astype('Int64')
    b['reporterCode']=b['reporterCode'].astype('Int64'); b['partnerCode']=b['partnerCode'].astype('Int64'); b['year']=b['year'].astype('Int64')
    m=k.merge(b, left_on=['reporterCode','partnerCode','refYear'], right_on=['reporterCode','partnerCode','year'], how='left')
    return m[BILAT_COLS].fillna(0).to_numpy(dtype='float32')

# ===== split (+ caching) =====

from sklearn.experimental import enable_iterative_imputer   # required import
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def handle_missing(train, test, strategy='median'):
    cols=['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']
    if strategy=='drop':
        return train.dropna(subset=cols).copy(), test.dropna(subset=cols).copy()
    if strategy=='median':
        fill = train[cols].median()                       # fit on train only
        return train.fillna(fill).copy(), test.fillna(fill).copy()
    if strategy=='rf':
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20, n_jobs=2, random_state=0),
                               max_iter=5, random_state=0)
        fit_sample = train[cols].sample(min(500_000, len(train)), random_state=0)   # learn on 500k
        imp.fit(fit_sample)                                                          # fit small
        tr, te = train.copy(), test.copy()
        tr[cols] = imp.transform(train[cols])                                        # apply to all
        te[cols] = imp.transform(test[cols])
        return tr, te
    raise ValueError(strategy)

def load_and_split(path='all_products_ready.parquet'):
    # column-subset load: only what the models use (keeps 26M rows in RAM)
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
    print(f'train {len(train_data):,} | val {len(val_data):,} | test2023 {len(te):,}')
    return train_data, val_data, te

train_data, val_data, data_2023 = load_and_split('all_products_ready.parquet')


# --- split caching: save train/val/test once, reload thereafter ---
SPLIT_DIR = "split_cache"
def load_or_split(path, missing="median"):
    import os
    os.makedirs(SPLIT_DIR, exist_ok=True)
    ftr, fva, fte = (os.path.join(SPLIT_DIR, f) for f in
                     ["train.parquet","val.parquet","test2023.parquet"])
    if all(os.path.exists(f) for f in (ftr, fva, fte)):
        log("split cache found -> loading (skipping re-split)")
        return (pd.read_parquet(ftr), pd.read_parquet(fva), pd.read_parquet(fte))
    log(f"no split cache -> building split from {path}")
    tr, va, te = load_and_split(path)
    tr.to_parquet(ftr, index=False); va.to_parquet(fva, index=False); te.to_parquet(fte, index=False)
    log(f"split saved to {SPLIT_DIR}/ (train={len(tr):,} val={len(va):,} test={len(te):,})")
    return tr, va, te


# ===== model classes =====

class GATRegressionModel(nn.Module):
    def __init__(s,inf,h=32,heads=4):
        super().__init__(); s.c1=GATConv(inf,h,heads); s.c2=GATConv(h*heads,1,heads)
    def forward(s,g,x): return s.c2(g, torch.relu(s.c1(g,x).flatten(1))).mean(1)

class GCNRegressionModel(nn.Module):
    def __init__(s,inf,h=32):
        super().__init__(); s.c1=GraphConv(inf,h,allow_zero_in_degree=True); s.c2=GraphConv(h,1,allow_zero_in_degree=True)
    def forward(s,g,x): return s.c2(g, torch.relu(s.c1(g,x))).squeeze(-1)

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

# --- edge-aware conv: neighbor message = f(neighbor_node, EDGE_risk) ---
class EdgeConv(nn.Module):
    def __init__(s,inf,ef,out):
        super().__init__(); s.msg=nn.Linear(inf+ef,out); s.slf=nn.Linear(inf,out)
    def forward(s,g,x,ef):
        with g.local_scope():
            g.ndata['h']=x; g.edata['e']=ef
            g.update_all(lambda e:{'m':s.msg(torch.cat([e.src['h'],e.data['e']],1))}, fn.mean('m','agg'))
            return torch.relu(s.slf(x)+g.ndata['agg'])

class EdgeRiskGNN(nn.Module):
    '''Bilateral risk on edges: risk enters the message passed along each trade link.'''
    def __init__(s,inf,ef,h=32):
        super().__init__(); s.l1=EdgeConv(inf,ef,h); s.l2=EdgeConv(h,ef,1)
    def forward(s,g,x,ef): return s.l2(g, s.l1(g,x,ef), ef).squeeze(-1)

from dgl.nn import GATv2Conv

class GATv2RegressionModel(nn.Module):
    def __init__(s,inf,h=32,heads=4):
        super().__init__(); s.c1=GATv2Conv(inf,h,heads,allow_zero_in_degree=True); s.c2=GATv2Conv(h*heads,1,heads,allow_zero_in_degree=True)
    def forward(s,g,x): return s.c2(g, torch.relu(s.c1(g,x).flatten(1))).mean(1)

class TabTransformer(nn.Module):
    def __init__(self, n_feats, d=32, heads=4, layers=2):
        super().__init__()
        self.d=d
        self.feat_emb = nn.Linear(1, d)                       # each scalar feature -> d-dim token
        self.pos = nn.Parameter(torch.randn(n_feats, d)*0.02) # per-feature positional embedding
        enc = nn.TransformerEncoderLayer(d, heads, d*2, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,1))
    def forward(self, x):                # x: (batch, n_feats)
        t = self.feat_emb(x.unsqueeze(-1)) + self.pos      # (batch, n_feats, d)
        t = self.tf(t)
        return self.head(t.mean(1)).squeeze(-1)            # pool tokens -> scalar

from dgl.nn import EdgeGATConv

class EdgeGAT(nn.Module):
    """Attention + edge features: bilateral risk influences the attention weights."""
    def __init__(s, inf, ef, h=32, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h*heads, ef, 1, heads, allow_zero_in_degree=True)
    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))   # (N, heads*h)
        return s.c2(g, x, efeat).mean(1).squeeze(-1)    # (N,)

# ===== EdgeGATv2 =====

import dgl.function as fn

class EdgeGATv2Conv(nn.Module):
    """GATv2 dynamic attention + edge features, single-head for clarity."""
    def __init__(s, in_feats, edge_feats, out_feats):
        super().__init__()
        s.fc_src  = nn.Linear(in_feats, out_feats, bias=False)
        s.fc_dst  = nn.Linear(in_feats, out_feats, bias=False)
        s.fc_edge = nn.Linear(edge_feats, out_feats, bias=False)
        s.attn    = nn.Linear(out_feats, 1, bias=False)   # applied AFTER leakyrelu = GATv2
        s.leaky   = nn.LeakyReLU(0.2)
        s.out_feats = out_feats
    def forward(s, g, x, efeat):
        with g.local_scope():
            g.srcdata['xs'] = s.fc_src(x)
            g.dstdata['xd'] = s.fc_dst(x)
            g.edata['xe']   = s.fc_edge(efeat)
            # combine src + dst + edge on each edge, THEN nonlinearity, THEN score (v2 order)
            g.apply_edges(lambda e: {'e': s.attn(s.leaky(e.src['xs'] + e.dst['xd'] + e.data['xe']))})
            g.edata['a'] = dgl.nn.functional.edge_softmax(g, g.edata['e'])
            # message = attention-weighted (neighbor + edge)
            g.apply_edges(lambda e: {'m': e.data['a'] * (e.src['xs'] + e.data['xe'])})
            g.update_all(fn.copy_e('m','m'), fn.sum('m','h'))
            return g.dstdata['h']

class EdgeGATv2(nn.Module):
    def __init__(s, inf, ef, h=32):
        super().__init__()
        s.c1 = EdgeGATv2Conv(inf, ef, h)
        s.c2 = EdgeGATv2Conv(h, ef, 1)
    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat))
        return s.c2(g, x, efeat).squeeze(-1)

# ===== runners =====

# ================= CELL 1 of runners — device helper + node models + RF + TabTF =================
import torch, gc
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("default DEVICE:", DEVICE)



class FusedEdgeModel(nn.Module):
    """Edge model with a switchable NODE-fusion front-end (concat / blend / attention).
    Node features arrive as [trade_cols | gdelt_cols] concatenated; for blend/attention we
    split at n_trade, project each stream, fuse, then feed the fused vector to an edge backbone
    (EdgeRiskGNN / EdgeGAT / EdgeGATv2). concat = feed the raw concatenated vector (unchanged)."""
    def __init__(s, kind, n_trade, n_gdelt, n_edge, fusion='concat', proj=16, h=32):
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
        if   kind == 'EdgeGAT':   s.backbone = EdgeGAT(inf, n_edge, h)
        elif kind == 'EdgeGATv2': s.backbone = EdgeGATv2(inf, n_edge, h)
        else:                     s.backbone = EdgeRiskGNN(inf, n_edge, h)
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

def with_fallback(fn, *args, **kwargs):
    """Try GPU; on CUDA OOM, clean up and retry on CPU."""
    if torch.cuda.is_available():
        try:
            return fn(*args, device=torch.device('cuda'), **kwargs)
        except torch.cuda.OutOfMemoryError:
            print("  GPU OOM -> falling back to CPU")
            gc.collect(); torch.cuda.empty_cache()
            return fn(*args, device=torch.device('cpu'), **kwargs)
    return fn(*args, device=torch.device('cpu'), **kwargs)

def _eval_graph(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                agg_mode='sum', fusion='concat', nt=None, bs=10000, device=torch.device('cpu')):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d,use_gdelt,gdelt_file,agg_mode)
    emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy())); eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32); eg=dgl.add_self_loop(eg)
    eg=eg.to(device)                                                              # <-- graph to device
    fused=fusion in ('blend','attention'); model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn)); ft=bg.ndata['feat']
        with torch.no_grad():
            out=model(bg,ft[:,:nt],ft[:,nt:]) if fused else model(bg,ft)
            preds.append(out.unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())   # <-- .cpu()
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt,yp

def run_graph(kind, train_data, data_2023, use_gdelt, scoring, gdelt_file,
              fusion='concat', epochs=100, agg_mode='sum', device=torch.device('cpu')):
    feat_cols=BASE_COLS+(GDELT_COLS if use_gdelt else [])
    agg=build_agg(train_data,use_gdelt,gdelt_file,agg_mode); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
    td=train_data.copy(); td['nodeID']=td['reporterCode'].map(cmap)
    sf,st=MinMaxScaler(),MinMaxScaler(); Xtr=sf.fit_transform(agg[feat_cols]); ytr=st.fit_transform(agg[['y_log']])
    g=dgl.graph((td['nodeID'].to_numpy(),td['partnerCode'].map(cmap).to_numpy())); g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32); g=dgl.add_self_loop(g)
    g=g.to(device)                                                               # <-- graph to device
    fused=fusion in ('blend','attention'); nt=len(BASE_COLS)
    if fusion=='blend': model=BlendGAT(nt,len(GDELT_COLS))
    elif fusion=='attention': model=AttnGAT(nt,len(GDELT_COLS))
    elif kind=='GAT': model=GATRegressionModel(len(feat_cols))
    elif kind=='GATv2': model=GATv2RegressionModel(len(feat_cols))
    else: model=GCNRegressionModel(len(feat_cols))
    model=model.to(device)                                                       # <-- model to device
    ytr_t=torch.tensor(ytr[:,0],dtype=torch.float32).to(device)                  # <-- target to device
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss(); N=g.num_nodes(); bs=10000; nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn=list(range(i*bs,min((i+1)*bs,N))); bg=g.subgraph(torch.tensor(bn)); ft=bg.ndata['feat']
            logit=model(bg,ft[:,:nt],ft[:,nt:]) if fused else model(bg,ft)
            loss=crit(logit.view(-1,1),ytr_t[bn].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    yt,yp=_eval_graph(model,data_2023,cmap,sf,st,feat_cols,use_gdelt,gdelt_file,agg_mode,fusion,nt,device=device)
    if yt is not None: report_and_log(kind,use_gdelt,scoring,agg_mode,yt,yp,fusion_override=(fusion if use_gdelt else 'none'))
    if fusion=='blend': print(f'   learned alpha = {torch.sigmoid(model.alpha).item():.3f}')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

def run_rf(train_data, data_2023, use_gdelt, scoring, gdelt_file, agg_mode='sum', device=None):
    # RF is sklearn = CPU only; device arg ignored (kept for uniform driver calls)
    feat_cols=BASE_COLS+(GDELT_COLS if use_gdelt else [])
    tr=build_agg(train_data,use_gdelt,gdelt_file,agg_mode).dropna(subset=feat_cols+['primaryValue'])
    te=build_agg(data_2023,use_gdelt,gdelt_file,agg_mode).dropna(subset=feat_cols+['primaryValue'])
    sf,st=MinMaxScaler(),MinMaxScaler(); Xtr=sf.fit_transform(tr[feat_cols]); Xte=sf.transform(te[feat_cols])
    ytr=st.fit_transform(tr[['y_log']]); yte=st.transform(te[['y_log']])
    rf=RandomForestRegressor(n_estimators=200,max_depth=25,n_jobs=2,random_state=0).fit(Xtr,ytr.ravel())
    yp=np.expm1(st.inverse_transform(rf.predict(Xte).reshape(-1,1)).flatten()); yt=np.expm1(st.inverse_transform(yte).flatten())
    report_and_log('RF',use_gdelt,scoring,agg_mode,yt,yp)
    gc.collect()
    return rf

def run_tabtransformer(train_data, data_2023, use_gdelt, scoring, gdelt_file, agg_mode='sum', epochs=100, device=torch.device('cpu')):
    feat_cols=BASE_COLS+(GDELT_COLS if use_gdelt else [])
    tr=build_agg(train_data,use_gdelt,gdelt_file,agg_mode).dropna(subset=feat_cols+['primaryValue'])
    te=build_agg(data_2023,use_gdelt,gdelt_file,agg_mode).dropna(subset=feat_cols+['primaryValue'])
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=torch.tensor(sf.fit_transform(tr[feat_cols]),dtype=torch.float32).to(device)
    Xte=torch.tensor(sf.transform(te[feat_cols]),dtype=torch.float32).to(device)
    ytr=torch.tensor(st.fit_transform(tr[['y_log']]),dtype=torch.float32).to(device)
    model=TabTransformer(len(feat_cols)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.MSELoss(); bs=2048; N=len(Xtr)
    for ep in range(epochs):
        model.train(); perm=torch.randperm(N,device=device)
        for i in range(0,N,bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            loss=crit(model(Xtr[idx]).view(-1,1), ytr[idx].view(-1,1)); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): pred=model(Xte).view(-1,1).cpu().numpy()
    yp=np.expm1(st.inverse_transform(pred).flatten())
    yt=np.expm1(st.inverse_transform(st.transform(te[['y_log']])).flatten())
    report_and_log('TabTF',use_gdelt,scoring,agg_mode,yt,yp)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

def run_edge_graph(train_data, data_2023, scoring, bilat_file, agg_mode='first', epochs=100, device=torch.device('cpu')):
    '''Bilateral EDGE model (thin): gravity node feats + bilateral risk on edges, no attention.'''
    feat_cols=BASE_COLS
    agg=build_agg(train_data,False,None,agg_mode); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
    td=train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td=td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler(); Xtr=sf.fit_transform(agg[feat_cols]); ytr=st.fit_transform(agg[['y_log']])
    ef=MinMaxScaler().fit_transform(edge_features(td, bilat_file))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy())); g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)                 # <-- target on nodes (rides subgraph)
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)
    g=g.to(device)
    model=EdgeRiskGNN(len(feat_cols),len(BILAT_COLS)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss()
    N=g.num_nodes(); bs=20000; nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):                                                 # <-- BATCH over nodes
            bn=list(range(i*bs,min((i+1)*bs,N)))
            bg=g.subgraph(torch.tensor(bn,device=device))
            out=model(bg, bg.ndata['feat'], bg.edata['ef'])
            loss=crit(out.view(-1,1), bg.ndata['y'].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    yt,yp=_eval_edge(model,data_2023,cmap,sf,st,feat_cols,bilat_file,agg_mode,device=device)
    if yt is not None: report_and_log('EdgeGNN',True,scoring,agg_mode,yt,yp,fusion_override='edge',risk_level='bilateral')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

def _eval_edge(model, df_eval, cmap, sf, st, feat_cols, bilat_file, agg_mode, device=torch.device('cpu')):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d,False,None,agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features(d, bilat_file))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy())); eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    eg=eg.to(device)
    model.eval(); preds=[]; N=eg.num_nodes(); bs=20000
    for i in range(0,N,bs):                                                 # <-- BATCH eval
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn,device=device))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt,yp

def run_edge_gat(train_data, data_2023, scoring, bilat_file, agg_mode='first', epochs=100, device=torch.device('cpu')):
    '''Bilateral EDGE model (thin) with attention: EdgeGAT.'''
    feat_cols=BASE_COLS
    agg=build_agg(train_data,False,None,agg_mode); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
    td=train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td=td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler(); Xtr=sf.fit_transform(agg[feat_cols]); ytr=st.fit_transform(agg[['y_log']])
    ef=MinMaxScaler().fit_transform(edge_features(td, bilat_file))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy())); g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)                 # <-- target on nodes
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)
    g=g.to(device)
    model=EdgeGAT(len(feat_cols),len(BILAT_COLS)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss()
    N=g.num_nodes(); bs=20000; nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):                                                 # <-- BATCH over nodes
            bn=list(range(i*bs,min((i+1)*bs,N)))
            bg=g.subgraph(torch.tensor(bn,device=device))
            out=model(bg, bg.ndata['feat'], bg.edata['ef'])
            loss=crit(out.view(-1,1), bg.ndata['y'].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    yt,yp=_eval_edge(model,data_2023,cmap,sf,st,feat_cols,bilat_file,agg_mode,device=device)
    if yt is not None: report_and_log('EdgeGAT',True,scoring,agg_mode,yt,yp,fusion_override='edge',risk_level='bilateral')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

# ===== full edge runners =====

# ============ CELL 3 of runners — FULL edge models (GPU-ready) ============
FULL_NODE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS = BILAT_COLS + ['dist']

def edge_features_full(df_rows, bilat_file):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode','partnerCode','refYear','dist']].copy()
    for c in ['reporterCode','partnerCode','refYear']: k[c]=k[c].astype('Int64')
    b['reporterCode']=b['reporterCode'].astype('Int64'); b['partnerCode']=b['partnerCode'].astype('Int64'); b['year']=b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode','partnerCode','refYear'],
                right_on=['reporterCode','partnerCode','year'], how='left')
    m[BILAT_COLS]=m[BILAT_COLS].fillna(0); m['dist']=m['dist'].fillna(m['dist'].median())
    return m[FULL_EDGE_COLS].to_numpy(dtype='float32')

def run_edge_full(kind, train_data, data_2023, combo, gdelt_file, bilat_file,
                  agg_mode='first', epochs=100, device=torch.device('cpu'), bs=20000, node_fusion='concat'):
    agg = build_agg(train_data, True, gdelt_file, agg_mode)
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=sf.fit_transform(agg[FULL_NODE_COLS]); ytr=st.fit_transform(agg[['y_log']])
    ef=MinMaxScaler().fit_transform(edge_features_full(td, bilat_file))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy())); g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)                 # <-- attach target to nodes
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)
    n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
    model=FusedEdgeModel(kind, n_trade, n_gdelt, ei, fusion=node_fusion).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss()
    N=g.num_nodes(); nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):                                                 # <-- BATCH over nodes
            bn=list(range(i*bs, min((i+1)*bs, N)))
            bg=g.subgraph(torch.tensor(bn))                   # subgraph carries its edata['ef']
            out=model(bg, bg.ndata['feat'], bg.edata['ef'])
            loss=crit(out.view(-1,1), bg.ndata['y'].view(-1,1))             # target rides along in ndata
            opt.zero_grad(); loss.backward(); opt.step()
    yt,yp=_eval_edge_full(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode, device=device, bs=bs)
    if yt is not None:
        report_and_log(kind+'_full', True, combo, agg_mode, yt, yp, fusion_override=node_fusion, risk_level='bilateral')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

def _eval_edge_full(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode,
                    device=torch.device('cpu'), bs=20000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy())); eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):                                                 # <-- BATCH eval too
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt,yp

# ===== save / skip =====

import os, pickle, torch, gc

os.makedirs('trained_models', exist_ok=True)

def save_model(model, name):
    """Save sklearn via pickle, torch models via state_dict."""
    if hasattr(model, 'state_dict'):                    # torch
        torch.save(model.state_dict(), f'trained_models/{name}.pt')
    else:                                               # sklearn (RF)
        with open(f'trained_models/{name}.pkl','wb') as f: pickle.dump(model, f)

def done(name):
    return os.path.exists(f'trained_models/{name}.pt') or os.path.exists(f'trained_models/{name}.pkl')

def run_once(name, fn, *args, **kwargs):
    """Skip if already trained; otherwise run, save, and free memory."""
    if done(name):
        print(f'⏭  skip {name} (already trained)')
        return
    print(f'▶  {name}')
    model = fn(*args, **kwargs)
    if model is not None: save_model(model, name)
    gc.collect()                                        # free memory between runs

CONDITIONS = [('none',False,None),('keepall',True,KEEPALL),('topk',True,TOPK)]

# --- Block 1 ---
for agg_mode in ['first','spec']:
    for scoring,use_gdelt,gfile in CONDITIONS:
        tag=f'{agg_mode}_{scoring}'
        run_once(f'RF_{tag}',    run_rf,            train_data,data_2023,use_gdelt,scoring,gfile,agg_mode)
        run_once(f'TabTF_{tag}', run_tabtransformer,train_data,data_2023,use_gdelt,scoring,gfile,agg_mode)
        run_once(f'GCN_{tag}',   run_graph,'GCN',   train_data,data_2023,use_gdelt,scoring,gfile,agg_mode=agg_mode)
        run_once(f'GAT_{tag}',   run_graph,'GAT',   train_data,data_2023,use_gdelt,scoring,gfile,agg_mode=agg_mode)
        run_once(f'GATv2_{tag}', run_graph,'GATv2', train_data,data_2023,use_gdelt,scoring,gfile,agg_mode=agg_mode)

# --- Block 2 ---
for fusion in ['blend','attention']:
    for scoring,gfile in [('keepall',KEEPALL),('topk',TOPK)]:
        run_once(f'GAT_{fusion}_{scoring}', run_graph,'GAT',train_data,data_2023,True,scoring,gfile,fusion=fusion,agg_mode='first')

# --- Block 3 ---
BILAT_FILE='gdelt_bilateral_by_pair_year.parquet'
run_once('EdgeGNN_thin', run_edge_graph, train_data,data_2023,'bilateral',BILAT_FILE,agg_mode='first')
run_once('EdgeGAT_thin', run_edge_gat,   train_data,data_2023,'bilateral',BILAT_FILE,agg_mode='first')
for combo,node_file,pair_file in SCORER_COMBOS:
    run_once(f'EdgeGNN_full_{combo}',   run_edge_full,'EdgeGNN',  train_data,data_2023,combo,node_file,pair_file)
    run_once(f'EdgeGAT_full_{combo}',   run_edge_full,'EdgeGAT',  train_data,data_2023,combo,node_file,pair_file)
    run_once(f'EdgeGATv2_full_{combo}', run_edge_full,'EdgeGATv2',train_data,data_2023,combo,node_file,pair_file)

print('\n=== ALL RUNS DONE ===')

# ========================= DRIVER =========================
def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else "all_products_ready.parquet"
    log(f"loading data: {data_file}")
    train_data, val_data, data_2023 = load_or_split(data_file, missing=MISSING)
    log(f"train={len(train_data):,}  val={len(val_data):,}  test2023={len(data_2023):,}")

    log("=== Block 1: GAT only (no-GDELT, concat/keepall, concat/topk) ===")
    for agg_mode in ['first']:
        for scoring,use_gdelt,gfile in CONDITIONS:
            tag=f'{agg_mode}_{scoring}'
            for m in ['GAT']:
                run_once(f'{m}_{tag}', lambda *a,**k: with_fallback(run_graph,*a,**k), m,train_data,data_2023,use_gdelt,scoring,gfile,agg_mode=agg_mode)

    log("=== Block 3: EdgeGAT_full only (attention: ka_ka/ka_tk, blend: tk_ka/tk_tk) ===")
    TARGET_EDGE = {'ka_ka': 'attention', 'ka_tk': 'attention', 'tk_ka': 'blend', 'tk_tk': 'blend'}
    for combo,node_file,pair_file in SCORER_COMBOS:
        if combo not in TARGET_EDGE:
            continue
        nf = TARGET_EDGE[combo]
        for kind in ['EdgeGAT']:
            run_once(f'{kind}_full_{nf}_{combo}', run_edge_full,
                    kind,train_data,data_2023,combo,node_file,pair_file,
                    device=torch.device('cpu'), node_fusion=nf)

    log("=== ALL RUNS DONE ===")

if __name__ == "__main__":
    main()