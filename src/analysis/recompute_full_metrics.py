#!/usr/bin/env python3
"""
recompute_full_metrics.py
============================
Recovers MSE, MAE, RMSE, SMAPE for the 19 already-trained shrinkage models --
NO retraining needed. Loads each saved model from trained_models_shrinkage_head/
and runs ONE clean forward pass on the 2023 test set (reusing the exact same
model-loading and eval code as feature_importance_shrinkage.py's baseline
step), then computes the full metrics set from the raw predictions.

This works because compute_metrics() was simplified partway through the
project (mae/rmse/smape got dropped from later scripts) -- but the trained
weights were never affected by that, so nothing is actually lost.

Usage:
    python recompute_full_metrics.py [all_products_ready.parquet]
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
# config -- identical to feature_importance_shrinkage.py
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
MODEL_DIR = 'models/trained_models_shrinkage_head'   # updated for the reorganized repo layout
SHRINKAGE_K = 50

def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)
def hs_chapter_2digit(heading):
    return (np.asarray(heading, dtype=np.int64) // 100).astype(np.int64)

class UnparseableModelName(ValueError):
    pass

def parse_shrinkage_name(name):
    n = name.replace('_shrinkage', '')
    if n == 'GAT_first_none':
        return ('GAT', 'concat', 'none')
    if n.startswith('GAT_first_'):
        rest = n[len('GAT_first_'):]
        parts = rest.rsplit('_', 1)
        if len(parts) != 2:
            raise UnparseableModelName(f"'{name}' has no fusion suffix -- likely orphaned, skipping.")
        scoring, fusion = parts
        return ('GAT', fusion, scoring)
    if n.startswith('EdgeGAT_full_'):
        rest = n[len('EdgeGAT_full_'):]
        fusion, combo = rest.split('_', 1)
        return ('EdgeGAT_full', fusion, combo)
    raise UnparseableModelName(f"'{name}' doesn't match any known naming pattern.")

# ==========================================================================
# FULL metrics -- this is the whole point of this script
# ==========================================================================
def _smape(yt, yp):
    d=(np.abs(yt)+np.abs(yp))/2; m=d>0
    return np.mean(np.abs(yt[m]-yp[m])/d[m])*100 if m.sum() else np.nan

def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_full_metrics(yt, yp):
    ytl=np.log1p(np.clip(yt,0,None)); ypl=np.log1p(np.clip(yp,0,None))
    mse = mean_squared_error(yt, yp)
    log_mse = mean_squared_error(ytl, ypl)
    return dict(
        mse=mse, rmse=mse**0.5, mae=mean_absolute_error(yt, yp), smape=_smape(yt, yp),
        r2=r2_score(yt, yp),
        log_mse=log_mse, log_rmse=log_mse**0.5, log_mae=mean_absolute_error(ytl, ypl),
        log_r2=(r2_score(ytl, ypl) if len(yt) > 2 else np.nan),
        spearman=_spearman(yt, yp),
    )

# ==========================================================================
# data pipeline -- identical to feature_importance_shrinkage.py
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
    split_dir = os.path.join('data/cache/split_cache', tag)   # updated for the reorganized repo layout
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
# models -- identical to feature_importance_shrinkage.py
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
        s.c1 = GATConv(inf, h, heads); s.c2 = GATConv(h*heads, h2, heads)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, heading_idx, chapter2_idx):
        hid = torch.relu(s.c1(g, x).flatten(1)); hid = s.c2(g, hid).mean(1)
        return s.head(hid, heading_idx, chapter2_idx)

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
# clean eval (no permutation) -- reused logic from feature_importance_shrinkage.py's baseline step
# ==========================================================================
def eval_gat(model, df_eval, cmap, sf, st, feat_cols, use_gdelt, gdelt_file, agg_mode='first', bs=10000):
    d=df_eval.copy()
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

def eval_edgegat_full(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first', bs=10000):
    d=df_eval.copy()
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
def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    # updated for the reorganized repo layout: accept either a bare filename (resolved
    # against data/processed/, matching how you've been invoking every other script) or a
    # full/relative path typed explicitly -- never silently guess between two possible files
    if not os.path.exists(data_file):
        candidate = os.path.join('data/processed', data_file)
        if os.path.exists(candidate):
            data_file = candidate
        else:
            log(f"  [!] Could not find '{data_file}' as given, or as data/processed/{data_file}. "
                f"Run this from the repo root, or pass a full path.")
            return
    log(f"checking {MODEL_DIR}/ for saved models...")
    if not os.path.isdir(MODEL_DIR):
        log(f"  [!] {MODEL_DIR}/ not found."); return
    on_disk = sorted({f[:-3] for f in os.listdir(MODEL_DIR) if f.endswith('.pt')})
    log(f"  found {len(on_disk)} saved models")

    train_data, val_data, data_2023 = load_or_split(data_file)

    rows = []
    for name in on_disk:
        try:
            kind, fusion, scoring = parse_shrinkage_name(name)
        except UnparseableModelName as e:
            log(f"  [!] skipping {name}: {e}"); continue

        log(f"evaluating {name} ...")
        try:
            if kind == 'GAT':
                use_gdelt, gfile = CONDITIONS_BY_SCORING[scoring]
                feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
                model = build_gat_model(len(feat_cols), len(BASE_COLS), len(GDELT_COLS), use_gdelt, fusion)
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu'))
                agg = build_agg(train_data, use_gdelt, gfile, 'first')
                cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
                sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[feat_cols]); st.fit(agg[['y_log']])
                yt, yp = eval_gat(model, data_2023, cmap, sf, st, feat_cols, use_gdelt, gfile)
            else:
                gfile, bfile = SCORER_COMBOS[scoring]
                n_trade=len(FULL_NODE_COLS)-len(GDELT_COLS); n_gdelt=len(GDELT_COLS); ei=len(FULL_EDGE_COLS)
                model = PerProductFusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{name}.pt'), map_location='cpu'))
                agg = build_agg(train_data, True, gfile, 'first')
                cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
                sf, st = MinMaxScaler(), MinMaxScaler(); sf.fit(agg[FULL_NODE_COLS]); st.fit(agg[['y_log']])
                yt, yp = eval_edgegat_full(model, data_2023, cmap, sf, st, gfile, bfile)
        except Exception as e:
            log(f"  [!] {name} FAILED ({type(e).__name__}: {e}) -- skipping."); continue

        m = compute_full_metrics(yt, yp)
        log(f"  {name}: mse={m['mse']:.2f} mae={m['mae']:.2f} rmse={m['rmse']:.2f} "
            f"smape={m['smape']:.2f}%  log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}  spearman={m['spearman']:.3f}")
        m['model'] = name
        rows.append(m)
        os.makedirs('results/grids', exist_ok=True)  # updated for the reorganized repo layout
        pd.DataFrame(rows).to_csv('results/grids/results_shrinkage_full_metrics.csv', index=False)  # checkpoint every model
        gc.collect()

    if rows:
        result = pd.DataFrame(rows).set_index('model')
        cols = ['mse','mae','rmse','smape','r2','log_mse','log_rmse','log_mae','log_r2','spearman']
        print("\n=== Full metrics, all evaluated models ===")
        print(result[cols].round(3).to_string())
        print(f"\nsaved -> results/grids/results_shrinkage_full_metrics.csv")


if __name__ == "__main__":
    main()
