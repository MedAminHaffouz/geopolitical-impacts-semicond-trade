#!/usr/bin/env python3
"""
retrain_reseed_diagnostic.py
===============================
Retrains GAT_first_topk_attention_shrinkage under 2 new random seeds, to
test whether its collapse (base log_r2=0.082, every feature importance
exactly 0.0) was a one-off training instability or a structural problem
with attention+topk fusion.

Saves each reseeded model separately (never overwrites the original), and
prints a direct comparison table: original vs. seed A vs. seed B.

If either reseed comes back healthy (non-zero feature importances, log_r2
comparable to your other GAT configs ~0.3), that confirms it was a fluke.
If ALL seeds collapse the same way, that's a structural problem worth
investigating in the architecture itself, not just bad luck.

Usage:
    python retrain_reseed_diagnostic.py [all_products_ready.parquet]
"""
import os, sys, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl
from dgl.nn import GATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)

# ==========================================================================
# config -- identical to run_shrinkage_head.py
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
MISSING = 'rf'
TOPK = 'gdelt_features_topk.parquet'
NUM_HEADINGS = 10000
SHRINKAGE_K = 50
MODEL_DIR = 'trained_models_reseed_diagnostic'
os.makedirs(MODEL_DIR, exist_ok=True)

SEEDS_TO_TRY = [1, 2]   # original run used whatever torch's default state was -- these are 2 fresh seeds

def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)

def hs_chapter_2digit(heading):
    h = np.asarray(heading, dtype=np.int64)
    return (h // 100).astype(np.int64)

# ==========================================================================
# metrics
# ==========================================================================
def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_metrics(yt, yp):
    ytl=np.log1p(np.clip(yt,0,None)); ypl=np.log1p(np.clip(yp,0,None))
    return dict(r2=r2_score(yt,yp), log_r2=(r2_score(ytl,ypl) if len(yt)>2 else np.nan), spearman=_spearman(yt,yp))

# ==========================================================================
# data pipeline -- identical to run_shrinkage_head.py
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
# model -- identical to run_shrinkage_head.py's PerProductFusedGAT (attention)
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

class PerProductFusedGAT(nn.Module):
    def __init__(s, n_trade, n_gdelt, fusion, num_chapters=NUM_HEADINGS, proj=16, h=32, h2=16, heads=4):
        super().__init__()
        s.fusion = fusion; s.n_trade = n_trade
        s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
        s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.c1 = GATConv(proj, h, heads)
        s.c2 = GATConv(h*heads, h2, heads)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, heading_idx, chapter2_idx):
        xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
        t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
        st = torch.stack([t, d], dim=1); f = s.attn(st, st, st)[0].mean(1)
        hid = torch.relu(s.c1(g, f).flatten(1))
        hid = s.c2(g, hid).mean(1)
        return s.head(hid, heading_idx, chapter2_idx)

# ==========================================================================
# eval + train
# ==========================================================================
def _eval(model, df_eval, cmap, sf, st, feat_cols, gdelt_file, agg_mode='first', bs=10000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d,True,gdelt_file,agg_mode)
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

def train_one(train_data, data_2023, seed, epochs=100):
    torch.manual_seed(seed); np.random.seed(seed)
    feat_cols = BASE_COLS + GDELT_COLS
    agg = build_agg(train_data, True, TOPK, 'first'); cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nodeID'] = td['reporterCode'].map(cmap)
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(agg[feat_cols]); ytr = st.fit_transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    g = dgl.graph((td['nodeID'].to_numpy(), td['partnerCode'].map(cmap).to_numpy()))
    g.ndata['feat'] = torch.tensor(Xtr, dtype=torch.float32)
    g.ndata['heading'] = torch.tensor(heading, dtype=torch.long)
    g.ndata['chap2'] = torch.tensor(chap2, dtype=torch.long)
    g = dgl.add_self_loop(g)

    model = PerProductFusedGAT(len(BASE_COLS), len(GDELT_COLS), fusion='attention')
    heading_counts = pd.Series(heading).value_counts().to_dict()
    model.head.set_shrinkage(heading_counts)

    ytr_t = torch.tensor(ytr[:,0], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.01); crit = nn.MSELoss()
    N = g.num_nodes(); bs = 10000; nb = N//bs + (N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn = list(range(i*bs, min((i+1)*bs, N))); bg = g.subgraph(torch.tensor(bn))
            loss = crit(model(bg, bg.ndata['feat'], bg.ndata['heading'], bg.ndata['chap2']).view(-1,1),
                        ytr_t[bn].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()

    yt, yp = _eval(model, data_2023, cmap, sf, st, feat_cols, TOPK, 'first')
    metrics = compute_metrics(yt, yp)
    gc.collect()
    return model, metrics

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    train_data, val_data, data_2023 = load_or_split(data_file)

    results = {}
    ORIGINAL_METRICS = {'log_r2': 0.082, 'spearman': 0.291}   # from your run
    log(f"original run: log_r2={ORIGINAL_METRICS['log_r2']:.3f}  spearman={ORIGINAL_METRICS['spearman']:.3f}  "
        f"(all feature importances came back exactly 0.0 -- the signature being tested here)")

    for seed in SEEDS_TO_TRY:
        model_path = os.path.join(MODEL_DIR, f'GAT_first_topk_attention_seed{seed}.pt')
        if os.path.exists(model_path):
            log(f"seed {seed}: model already trained, skipping (delete {model_path} to force retrain)")
            continue
        log(f"training with seed={seed} ...")
        model, metrics = train_one(train_data, data_2023, seed)
        torch.save(model.state_dict(), model_path)
        results[seed] = metrics
        log(f"  seed {seed}: log_r2={metrics['log_r2']:.3f}  spearman={metrics['spearman']:.3f}")

    print("\n=== Comparison ===")
    print(f"{'run':<12} {'log_r2':>8} {'spearman':>10}")
    print(f"{'original':<12} {ORIGINAL_METRICS['log_r2']:>8.3f} {ORIGINAL_METRICS['spearman']:>10.3f}")
    for seed, m in results.items():
        print(f"{'seed '+str(seed):<12} {m['log_r2']:>8.3f} {m['spearman']:>10.3f}")

    healthy = [s for s, m in results.items() if m['log_r2'] > 0.2]
    if healthy:
        print(f"\n>>> Seed(s) {healthy} came back healthy (log_r2 > 0.2) -- confirms the original was a training fluke, not structural.")
    elif results:
        print(f"\n>>> ALL reseeds still collapsed -- this suggests a structural problem with attention+topk fusion, "
              f"not bad luck. Worth investigating the attention layer's init/gradient flow for this specific combo "
              f"before including this config's numbers in any headline claim.")


if __name__ == "__main__":
    main()
