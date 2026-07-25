#!/usr/bin/env python3
"""
run_category_experiments.py
=============================
For each dataset in ['electronics.parquet', 'chips.parquet']:
  1. Train (or load if already trained) 6 target models:
       GAT_first_none, GAT_first_keepall, GAT_first_topk,
       EdgeGAT_full_attention_ka_ka, EdgeGAT_full_attention_ka_tk, EdgeGAT_full_blend_tk_ka
  2. For the blend/attention fusion models: read the learned trade-vs-GDELT
     weighting straight out of the state_dict (alpha for blend), plus a
     forward-pass hook for attention (state_dict alone doesn't expose
     attention *weights*, only the projection matrices — see inspect_fusion()).
  3. Permutation importance: for every input feature, shuffle its column on
     the 2023 test set, re-run the already-trained model, and record the
     drop in log-R^2 / Spearman. No retraining involved anywhere in step 3.

Models + importance tables are saved per-dataset so electronics and chips
runs never collide or overwrite each other.

Usage:
    python run_category_experiments.py
"""
import os, sys, csv, gc
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl, dgl.function as fn
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings("ignore")

# ==========================================================================
# logging
# ==========================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)

# ==========================================================================
# config — copied from train_benchmark.py
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
FULL_NODE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS = BILAT_COLS + ['dist']
MISSING = 'rf'
KEEPALL='gdelt_features_by_country_year.parquet'; TOPK='gdelt_features_topk.parquet'
BILAT_KEEP='gdelt_bilateral_by_pair_year.parquet'; BILAT_TOPK='gdelt_bilateral_topk.parquet'
CONDITIONS = [('none',False,None),('keepall',True,KEEPALL),('topk',True,TOPK)]
SCORER_COMBOS = {'ka_ka':(KEEPALL,BILAT_KEEP), 'ka_tk':(KEEPALL,BILAT_TOPK), 'tk_ka':(TOPK,BILAT_KEEP), 'tk_tk':(TOPK,BILAT_TOPK)}

# --- the 6 targets, per your spec ---
GAT_TARGETS  = [('first_none','none',False,None), ('first_keepall','keepall',True,KEEPALL), ('first_topk','topk',True,TOPK)]
EDGE_TARGETS = [('ka_ka','attention'), ('ka_tk','attention'), ('tk_ka','blend')]

DATASETS = ['electronics.parquet', 'chips.parquet']

DEVICE = torch.device('cpu')   # forced CPU throughout — see earlier DGL/CUDA mismatch discussion

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
# feature engineering — copied from train_benchmark.py
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
    if strategy=='drop':
        return train.dropna(subset=cols).copy(), test.dropna(subset=cols).copy()
    if strategy=='median':
        fill = train[cols].median()
        return train.fillna(fill).copy(), test.fillna(fill).copy()
    if strategy=='rf':
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20, n_jobs=2, random_state=0),
                               max_iter=5, random_state=0)
        fit_sample = train[cols].sample(min(500_000, len(train)), random_state=0)
        imp.fit(fit_sample)
        tr, te = train.copy(), test.copy()
        tr[cols] = imp.transform(train[cols]); te[cols] = imp.transform(test[cols])
        return tr, te
    raise ValueError(strategy)

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

# --- split caching, FIXED: keyed by dataset name so electronics/chips can't collide ---
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
# model classes — copied from train_benchmark.py (only what these 6 targets need)
# ==========================================================================
class GATRegressionModel(nn.Module):
    def __init__(s,inf,h=32,heads=4):
        super().__init__(); s.c1=GATConv(inf,h,heads); s.c2=GATConv(h*heads,1,heads)
    def forward(s,g,x): return s.c2(g, torch.relu(s.c1(g,x).flatten(1))).mean(1)

class EdgeGAT(nn.Module):
    def __init__(s, inf, ef, h=32, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h*heads, ef, 1, heads, allow_zero_in_degree=True)
    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1).squeeze(-1)

class FusedEdgeModel(nn.Module):
    """Edge model with switchable node-fusion (blend/attention). Only EdgeGAT backbone needed here."""
    def __init__(s, n_trade, n_gdelt, n_edge, fusion, proj=16, h=32):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
        if fusion == 'blend':
            s.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion == 'attention':
            s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.backbone = EdgeGAT(proj, n_edge, h)
    def forward(s, g, x, ef, return_attn=False):
        xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
        t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
        attn_w = None
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha); node = a*d + (1-a)*t
        else:
            st = torch.stack([t, d], dim=1)
            out, attn_w = s.attn(st, st, st)
            node = out.mean(1)
        pred = s.backbone(g, node, ef)
        return (pred, attn_w) if return_attn else pred

# ==========================================================================
# eval functions, WITH permutation-importance hooks (perm_col shuffles one
# column of the input before the forward pass — no retraining, no backward)
# ==========================================================================
def _eval_graph(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                agg_mode='first', bs=10000, perm_col=None, rng=None):
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
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy())); eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32); eg=dgl.add_self_loop(eg)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn)); ft=bg.ndata['feat']
        with torch.no_grad():
            preds.append(model(bg,ft).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt,yp

def run_graph(train_data, data_2023, use_gdelt, gdelt_file, agg_mode='first', epochs=100):
    feat_cols=BASE_COLS+(GDELT_COLS if use_gdelt else [])
    agg=build_agg(train_data,use_gdelt,gdelt_file,agg_mode); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
    td=train_data.copy(); td['nodeID']=td['reporterCode'].map(cmap)
    sf,st=MinMaxScaler(),MinMaxScaler(); Xtr=sf.fit_transform(agg[feat_cols]); ytr=st.fit_transform(agg[['y_log']])
    g=dgl.graph((td['nodeID'].to_numpy(),td['partnerCode'].map(cmap).to_numpy())); g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32); g=dgl.add_self_loop(g)
    model=GATRegressionModel(len(feat_cols))
    ytr_t=torch.tensor(ytr[:,0],dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss(); N=g.num_nodes(); bs=10000; nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn=list(range(i*bs,min((i+1)*bs,N))); bg=g.subgraph(torch.tensor(bn)); ft=bg.ndata['feat']
            loss=crit(model(bg,ft).view(-1,1),ytr_t[bn].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    gc.collect()
    return model, feat_cols, cmap, sf, st

def _eval_edge_full(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first',
                    bs=20000, perm_col=None, ef_perm_col=None, rng=None):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    if perm_col is not None:
        Xe = Xe.copy(); Xe[:, perm_col] = rng.permutation(Xe[:, perm_col])
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    if ef_perm_col is not None:
        ef = ef.copy(); ef[:, ef_perm_col] = rng.permutation(ef[:, ef_perm_col])
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy())); eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt,yp

def run_edge_full(train_data, data_2023, combo, gdelt_file, bilat_file, fusion, agg_mode='first', epochs=100, bs=20000):
    agg = build_agg(train_data, True, gdelt_file, agg_mode)
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=sf.fit_transform(agg[FULL_NODE_COLS]); ytr=st.fit_transform(agg[['y_log']])
    ef=MinMaxScaler().fit_transform(edge_features_full(td, bilat_file))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy())); g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)
    n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
    model=FusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss()
    N=g.num_nodes(); nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn=list(range(i*bs, min((i+1)*bs, N)))
            bg=g.subgraph(torch.tensor(bn))
            out=model(bg, bg.ndata['feat'], bg.edata['ef'])
            loss=crit(out.view(-1,1), bg.ndata['y'].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    gc.collect()
    return model, cmap, sf, st

# ==========================================================================
# save / load / skip — per-dataset directory so electronics & chips never collide
# ==========================================================================
def model_dir(tag):
    d = os.path.join('trained_models_experiments', tag)
    os.makedirs(d, exist_ok=True)
    return d

def done(tag, name):
    return os.path.exists(os.path.join(model_dir(tag), f'{name}.pt'))

def save(tag, name, model):
    torch.save(model.state_dict(), os.path.join(model_dir(tag), f'{name}.pt'))

# ==========================================================================
# state_dict inspection — the zero-cost trick
# ==========================================================================
def inspect_fusion(tag, name, fusion, model_obj=None):
    """Read the learned trade-vs-GDELT weighting straight out of the saved
    state_dict. For 'blend' this is the whole story (a single scalar).
    For 'attention', the state_dict only holds the *projection* matrices
    (in_proj_weight / out_proj) -- the actual attention *weights* (the
    softmax scores over [trade, gdelt]) only exist at forward-pass time,
    so we also do one forward pass on a real batch and hook them out."""
    path = os.path.join(model_dir(tag), f'{name}.pt')
    sd = torch.load(path, map_location='cpu')
    print(f'\n--- state_dict for {name} ({tag}) ---')
    for k, v in sd.items():
        print(f'  {k:24s} shape={tuple(v.shape)}')

    if fusion == 'blend':
        alpha = torch.sigmoid(sd['alpha']).item()
        print(f'  => learned alpha (weight on GDELT stream) = {alpha:.3f}')
        print(f'     ({alpha:.0%} GDELT / {1-alpha:.0%} trade, per the model itself)')
        return {'alpha': alpha}

    if fusion == 'attention':
        # what state_dict alone tells us: the projection matrix norms, as a
        # weak proxy for "how much raw signal each stream's projection keeps"
        tp_norm = sd['tp.weight'].norm().item()
        gp_norm = sd['gp.weight'].norm().item()
        print(f'  tp.weight norm (trade projection)  = {tp_norm:.3f}')
        print(f'  gp.weight norm (gdelt projection)   = {gp_norm:.3f}')
        print('  (these are NOT attention weights -- just projection matrix scale,')
        print('   a weak proxy at best. Real attention weights need a forward pass:)')
        if model_obj is not None:
            print('  => see the "forward-pass attention" print right after this,')
            print('     computed from a real batch of 2023 test data.')
        return {'tp_norm': tp_norm, 'gp_norm': gp_norm}

def forward_pass_attention(model, g, x, ef, n_trade):
    """Run one real forward pass and pull the actual attention weights out —
    this is the 'what else can help us' answer for attention fusion models."""
    model.eval()
    with torch.no_grad():
        _, attn_w = model(g, x, ef, return_attn=True)
    # attn_w shape: (batch, 2, 2) -- attention over the 2 tokens [trade, gdelt]
    mean_w = attn_w.mean(0)   # average over all nodes in this batch
    print(f'  forward-pass attention (averaged over {attn_w.shape[0]:,} nodes):')
    print(f'    trade -> [trade={mean_w[0,0]:.3f}, gdelt={mean_w[0,1]:.3f}]')
    print(f'    gdelt -> [trade={mean_w[1,0]:.3f}, gdelt={mean_w[1,1]:.3f}]')
    gdelt_share = mean_w[:, 1].mean().item()
    print(f'    => average share of attention going to the GDELT token: {gdelt_share:.1%}')
    return gdelt_share

# ==========================================================================
# permutation importance
# ==========================================================================
def perm_importance_gat(model, feat_cols, data_2023, cmap, sf, st, use_gdelt, gdelt_file,
                         agg_mode='first', n_repeats=3, seed=0):
    yt, yp = _eval_graph(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode)
    base = compute_metrics(yt, yp)
    rng = np.random.default_rng(seed)
    rows = []
    for j, col in enumerate(feat_cols):
        d_r2, d_rho = [], []
        for _ in range(n_repeats):
            yt2, yp2 = _eval_graph(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gdelt_file,
                                    agg_mode, perm_col=j, rng=rng)
            m = compute_metrics(yt2, yp2)
            d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
        rows.append({'feature': col, 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    return base, pd.DataFrame(rows).sort_values('importance_log_r2', ascending=False)

def perm_importance_edgegat_full(model, data_2023, cmap, sf, st, gdelt_file, bilat_file,
                                  agg_mode='first', n_repeats=3, seed=0):
    yt, yp = _eval_edge_full(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode)
    base = compute_metrics(yt, yp)
    rng = np.random.default_rng(seed)
    rows = []
    all_cols = [(j, c, 'node') for j, c in enumerate(FULL_NODE_COLS)] + \
               [(j, c, 'edge') for j, c in enumerate(FULL_EDGE_COLS)]
    for j, col, side in all_cols:
        d_r2, d_rho = [], []
        for _ in range(n_repeats):
            kw = {'perm_col': j, 'rng': rng} if side == 'node' else {'ef_perm_col': j, 'rng': rng}
            yt2, yp2 = _eval_edge_full(model, data_2023, cmap, sf, st, gdelt_file, bilat_file, agg_mode, **kw)
            m = compute_metrics(yt2, yp2)
            d_r2.append(base['log_r2'] - m['log_r2']); d_rho.append(base['spearman'] - m['spearman'])
        rows.append({'feature': f'{side}:{col}', 'importance_log_r2': np.mean(d_r2), 'importance_spearman': np.mean(d_rho)})
    return base, pd.DataFrame(rows).sort_values('importance_log_r2', ascending=False)

# ==========================================================================
# main experiment loop
# ==========================================================================
def run_experiment(dataset_file):
    tag = os.path.splitext(os.path.basename(dataset_file))[0]
    log(f"===================== DATASET: {dataset_file} (tag='{tag}') =====================")
    train_data, val_data, data_2023 = load_or_split(dataset_file)

    trained = {}   # name -> (model, extra artifacts needed for eval)
    importance_rows = []

    # ---- GAT targets ----
    for name_suffix, scoring, use_gdelt, gfile in GAT_TARGETS:
        name = f'GAT_{name_suffix}'
        if done(tag, name):
            log(f"  skip {name} (already trained on {tag})")
            state = torch.load(os.path.join(model_dir(tag), f'{name}.pt'), map_location='cpu')
            feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
            model = GATRegressionModel(len(feat_cols)); model.load_state_dict(state)
            agg = build_agg(train_data, use_gdelt, gfile, 'first'); cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
            sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
        else:
            log(f"  training {name} ...")
            model, feat_cols, cmap, sf, st = run_graph(train_data, data_2023, use_gdelt, gfile, agg_mode='first')
            save(tag, name, model)
        trained[name] = dict(model=model, feat_cols=feat_cols, cmap=cmap, sf=sf, st=st, use_gdelt=use_gdelt, gfile=gfile)

        log(f"  permutation importance for {name} ...")
        base, imp = perm_importance_gat(model, feat_cols, data_2023, cmap, sf, st, use_gdelt, gfile)
        log(f"    base: log_r2={base['log_r2']:.3f}  spearman={base['spearman']:.3f}")
        print(imp.to_string(index=False))
        imp['model'] = name; imp['dataset'] = tag
        importance_rows.append(imp)

    # ---- EdgeGAT_full targets ----
    for combo, fusion in EDGE_TARGETS:
        name = f'EdgeGAT_full_{fusion}_{combo}'
        gfile, bfile = SCORER_COMBOS[combo]
        if done(tag, name):
            log(f"  skip {name} (already trained on {tag})")
            n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
            model = FusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
            model.load_state_dict(torch.load(os.path.join(model_dir(tag), f'{name}.pt'), map_location='cpu'))
            agg = build_agg(train_data, True, gfile, 'first'); cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
            sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
        else:
            log(f"  training {name} ...")
            model, cmap, sf, st = run_edge_full(train_data, data_2023, combo, gfile, bfile, fusion, agg_mode='first')
            save(tag, name, model)

        # --- state_dict inspection (the zero-cost trick) ---
        info = inspect_fusion(tag, name, fusion, model_obj=model)

        # --- forward-pass attention, for attention models only ---
        if fusion == 'attention':
            d = data_2023.copy()
            for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
            d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
            d=d.dropna(subset=['nID','pID']).copy()
            agg=build_agg(d, True, gfile, 'first'); emap={c:i for i,c in enumerate(agg['reporterCode'])}
            Xe=sf.transform(agg[FULL_NODE_COLS])
            d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
            d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
            ef=MinMaxScaler().fit_transform(edge_features_full(d, bfile))
            eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy())); eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
            eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
            n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS)
            forward_pass_attention(model, eg, eg.ndata['feat'], eg.edata['ef'], n_trade)

        log(f"  permutation importance for {name} ...")
        base, imp = perm_importance_edgegat_full(model, data_2023, cmap, sf, st, gfile, bfile)
        log(f"    base: log_r2={base['log_r2']:.3f}  spearman={base['spearman']:.3f}")
        print(imp.to_string(index=False))
        imp['model'] = name; imp['dataset'] = tag
        importance_rows.append(imp)

    all_imp = pd.concat(importance_rows, ignore_index=True)
    out_csv = f'feature_importance_{tag}.csv'
    all_imp.to_csv(out_csv, index=False)
    log(f"  saved -> {out_csv}")
    return all_imp


def main():
    all_results = []
    for dataset_file in DATASETS:
        all_results.append(run_experiment(dataset_file))
    log("===================== ALL EXPERIMENTS DONE =====================")
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv('feature_importance_all.csv', index=False)
    log("saved -> feature_importance_all.csv (electronics + chips combined)")


if __name__ == "__main__":
    main()