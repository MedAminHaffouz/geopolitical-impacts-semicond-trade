#!/usr/bin/env python3
"""
run_per_product_head.py
=========================
Tests "does giving the model product-specific weights help" WITHOUT training
N separate networks. One shared backbone (GAT / EdgeGAT_full, same as your
main models) + a small per-HS-HEADING (4-digit) final layer, looked up by
product. Backbone weights are shared and trained on ALL products at once;
only the last layer differs per heading -- this is what actually gives you
"different weight matrix per product" without a full separate-network-per-
product explosion.

NOTE: 4-digit headings (~1,200 possible codes) are a much finer grouping
than 2-digit chapters (~97) -- expect more per-group data sparsity, since
each heading gets far fewer training rows than a chapter would. Headings
with very few train rows will have noisier, less-trained weight vectors;
there's no backoff/shrinkage to a coarser group implemented here.

Targets trained (reduced set):
    GAT_first_none, GAT_first_topk,
    EdgeGAT_full_attention_ka_ka, EdgeGAT_full_blend_tk_tk

Must be run on a MULTI-product dataset (all_products_ready.parquet) --
a single-category file has only a handful of HS headings, so the per-
heading head would barely differentiate anything.

Usage:
    python run_per_product_head.py [all_products_ready.parquet]
"""
import os, sys, gc
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings("ignore")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)

# ==========================================================================
# config
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

GAT_TARGETS  = [('first_none','none',False,None), ('first_topk','topk',True,TOPK)]
EDGE_TARGETS = [('ka_ka','attention'), ('tk_tk','blend')]

NUM_HEADINGS = 10000  # 4-digit headings run up to ~9999; leaves headroom for any code < 10000 treated as heading-level
MODEL_DIR = 'trained_models_perproduct_heading'
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================================
# HS heading extraction (4-digit) -- magnitude-based, robust to leading-zero-
# stripped codes. Any code < 10000 is already at-or-coarser-than heading
# resolution and is used as-is; 6-digit codes get divided down to heading.
# ==========================================================================
def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)

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
# feature engineering (same as before)
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
    n_chap = pd.Series(hs_chapter(d['cmdCode'])).nunique()
    log(f'  {n_chap} distinct HS chapters present in this dataset')
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
# models: shared backbone -> shared hidden vector -> PER-CHAPTER final layer
# ==========================================================================
class PerProductGAT(nn.Module):
    """Shared GAT backbone (2 layers) -> per-HS-chapter linear head.
    The head IS a full weight vector + bias per chapter (output dim=1, so a
    full linear layer per chapter is just an embedding lookup, no approximation)."""
    def __init__(s, inf, num_chapters=NUM_HEADINGS, h=32, h2=16, heads=4):
        super().__init__()
        s.c1 = GATConv(inf, h, heads)
        s.c2 = GATConv(h*heads, h2, heads)          # shared -> per-node hidden vector
        s.head_w = nn.Embedding(num_chapters, h2)
        s.head_b = nn.Embedding(num_chapters, 1)
        nn.init.normal_(s.head_w.weight, std=0.05)
        nn.init.zeros_(s.head_b.weight)
    def forward(s, g, x, chapter_idx):
        hid = torch.relu(s.c1(g, x).flatten(1))
        hid = s.c2(g, hid).mean(1)                  # (N, h2)
        w = s.head_w(chapter_idx)                    # (N, h2) -- THIS node's product-specific weights
        b = s.head_b(chapter_idx).squeeze(-1)         # (N,)
        return (hid * w).sum(-1) + b

class PerProductEdgeGATBackbone(nn.Module):
    """Same idea as EdgeGAT, but the final EdgeGATConv layer outputs a hidden
    vector (h2) instead of a scalar -- the per-chapter head does the h2->1 step."""
    def __init__(s, inf, ef, h=32, h2=16, heads=4):
        super().__init__()
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h*heads, ef, h2, heads, allow_zero_in_degree=True)
    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1)             # (N, h2)

class PerProductFusedEdgeModel(nn.Module):
    """FusedEdgeModel (blend/attention node-fusion) + per-HS-chapter head."""
    def __init__(s, n_trade, n_gdelt, n_edge, fusion, num_chapters=NUM_HEADINGS, proj=16, h=32, h2=16):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
        if fusion == 'blend':
            s.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion == 'attention':
            s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.backbone = PerProductEdgeGATBackbone(proj, n_edge, h, h2)
        s.head_w = nn.Embedding(num_chapters, h2)
        s.head_b = nn.Embedding(num_chapters, 1)
        nn.init.normal_(s.head_w.weight, std=0.05)
        nn.init.zeros_(s.head_b.weight)
    def forward(s, g, x, ef, chapter_idx):
        xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
        t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha); node = a*d + (1-a)*t
        else:
            st = torch.stack([t, d], dim=1); node = s.attn(st, st, st)[0].mean(1)
        hid = s.backbone(g, node, ef)                # (N, h2)
        w = s.head_w(chapter_idx); b = s.head_b(chapter_idx).squeeze(-1)
        return (hid * w).sum(-1) + b

# ==========================================================================
# eval, with per-chapter metric breakdown
# ==========================================================================
def _eval_graph_pp(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode='first', bs=10000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None,None
    agg=build_agg(d,use_gdelt,gdelt_file,agg_mode)
    emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[feat_cols]); ye=st.transform(agg[['y_log']])
    chap = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['chap']=torch.tensor(chap,dtype=torch.long)
    eg=dgl.add_self_loop(eg)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.ndata['chap']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp, chap[:len(yt)]

def run_graph_pp(train_data, data_2023, use_gdelt, gdelt_file, agg_mode='first', epochs=100):
    feat_cols=BASE_COLS+(GDELT_COLS if use_gdelt else [])
    agg=build_agg(train_data,use_gdelt,gdelt_file,agg_mode); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
    td=train_data.copy(); td['nodeID']=td['reporterCode'].map(cmap)
    sf,st=MinMaxScaler(),MinMaxScaler(); Xtr=sf.fit_transform(agg[feat_cols]); ytr=st.fit_transform(agg[['y_log']])
    chap = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    g=dgl.graph((td['nodeID'].to_numpy(),td['partnerCode'].map(cmap).to_numpy()))
    g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['chap']=torch.tensor(chap,dtype=torch.long)
    g=dgl.add_self_loop(g)
    model=PerProductGAT(len(feat_cols))
    ytr_t=torch.tensor(ytr[:,0],dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss(); N=g.num_nodes(); bs=10000; nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn=list(range(i*bs,min((i+1)*bs,N))); bg=g.subgraph(torch.tensor(bn))
            loss=crit(model(bg,bg.ndata['feat'],bg.ndata['chap']).view(-1,1), ytr_t[bn].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    gc.collect()
    return model, feat_cols, cmap, sf, st

def _eval_edge_full_pp(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first', bs=20000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None,None
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS]); ye=st.transform(agg[['y_log']])
    chap = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_full(d, bilat_file))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()))
    eg.ndata['feat']=torch.tensor(Xe,dtype=torch.float32)
    eg.ndata['chap']=torch.tensor(chap,dtype=torch.long)
    eg.edata['ef']=torch.tensor(ef,dtype=torch.float32); eg=dgl.add_self_loop(eg, fill_data=0.)
    model.eval(); preds=[]; N=eg.num_nodes()
    for i in range(0,N,bs):
        bn=list(range(i,min(i+bs,N))); bg=eg.subgraph(torch.tensor(bn))
        with torch.no_grad():
            preds.append(model(bg, bg.ndata['feat'], bg.edata['ef'], bg.ndata['chap']).unsqueeze(1))
    yp=np.expm1(st.inverse_transform(torch.cat(preds,0).view(-1,1).cpu().numpy()).flatten())
    yt=np.expm1(st.inverse_transform(ye).flatten())
    return yt, yp, chap[:len(yt)]

def run_edge_full_pp(train_data, data_2023, combo, gdelt_file, bilat_file, fusion, agg_mode='first', epochs=100, bs=20000):
    agg = build_agg(train_data, True, gdelt_file, agg_mode)
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=sf.fit_transform(agg[FULL_NODE_COLS]); ytr=st.fit_transform(agg[['y_log']])
    chap = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    ef=MinMaxScaler().fit_transform(edge_features_full(td, bilat_file))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy()))
    g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['chap']=torch.tensor(chap,dtype=torch.long)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)
    n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
    model=PerProductFusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss()
    N=g.num_nodes(); nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn=list(range(i*bs, min((i+1)*bs, N)))
            bg=g.subgraph(torch.tensor(bn))
            out=model(bg, bg.ndata['feat'], bg.edata['ef'], bg.ndata['chap'])
            loss=crit(out.view(-1,1), bg.ndata['y'].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()
    gc.collect()
    return model, cmap, sf, st

# ==========================================================================
# save/skip
# ==========================================================================
def done(name): return os.path.exists(os.path.join(MODEL_DIR, f'{name}.pt'))
def save(name, model): torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{name}.pt'))

def report(name, yt, yp, chap):
    overall = compute_metrics(yt, yp)
    log(f"  {name}: log_r2={overall['log_r2']:.3f}  spearman={overall['spearman']:.3f}  n={len(yt):,}")
    rows = [{'chapter': -1, 'scope': 'overall', **overall, 'n': len(yt)}]
    for c in sorted(np.unique(chap)):
        m = chap == c
        if m.sum() < 30:   # too few rows for a stable per-chapter metric, skip
            continue
        met = compute_metrics(yt[m], yp[m])
        rows.append({'chapter': int(c), 'scope': 'per_chapter', **met, 'n': int(m.sum())})
    df = pd.DataFrame(rows); df['model'] = name
    return df

# ==========================================================================
# main
# ==========================================================================
def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"loading {data_file} (per-product-head experiment)")
    train_data, val_data, data_2023 = load_or_split(data_file)

    all_rows = []

    for name_suffix, scoring, use_gdelt, gfile in GAT_TARGETS:
        name = f'GAT_{name_suffix}_perproduct'
        if done(name):
            log(f"skip {name} (already trained)")
            feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
            model = PerProductGAT(len(feat_cols)); model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu'))
            agg = build_agg(train_data, use_gdelt, gfile, 'first'); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
            sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
        else:
            log(f"training {name} ...")
            model, feat_cols, cmap, sf, st = run_graph_pp(train_data, data_2023, use_gdelt, gfile, agg_mode='first')
            save(name, model)
        yt, yp, chap = _eval_graph_pp(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gfile, 'first')
        if yt is not None:
            all_rows.append(report(name, yt, yp, chap))

    for combo, fusion in EDGE_TARGETS:
        name = f'EdgeGAT_full_{fusion}_{combo}_perproduct'
        gfile, bfile = SCORER_COMBOS[combo]
        if done(name):
            log(f"skip {name} (already trained)")
            n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
            model = PerProductFusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu'))
            agg = build_agg(train_data, True, gfile, 'first'); cmap={c:i for i,c in enumerate(agg['reporterCode'])}
            sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
        else:
            log(f"training {name} ...")
            model, cmap, sf, st = run_edge_full_pp(train_data, data_2023, combo, gfile, bfile, fusion, agg_mode='first')
            save(name, model)
        yt, yp, chap = _eval_edge_full_pp(model, data_2023, cmap, sf, st, gfile, bfile, 'first')
        if yt is not None:
            all_rows.append(report(name, yt, yp, chap))

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv('results_perproduct_head.csv', index=False)
    log("saved -> results_perproduct_head.csv (overall + per-chapter breakdown for all 7 models)")

    print('\n=== Overall log-R^2, per-product-head vs. what to compare it against ===')
    print(result[result.scope=='overall'][['model','log_r2','spearman','n']].to_string(index=False))
    print('\nCompare these numbers to your pooled all_products results_v2.csv (same models, no per-chapter head)')
    print('to see whether the per-chapter head actually helps.')


if __name__ == "__main__":
    main()
