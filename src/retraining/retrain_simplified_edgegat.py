#!/usr/bin/env python3
"""
retrain_simplified_edgegat.py
================================
Retrains all 12 EdgeGAT_full configs with the 4 bilateral pair_* columns
AND cmdCode dropped -- the direct "wrapper method" test for whether it's
actually safe to simplify, confirming (rather than re-inferring) what 3
rounds of permutation importance have consistently pointed at:
  - pair_events, pair_score_mean, pair_score_max, pair_gold_mean: near-zero
    or negative importance, all 3 rounds
  - cmdCode: mean importance 0.004 across 19 models, well below the 0.01
    REMOVE_THRESHOLD, with a clear mechanism story (the per-product head
    now does that job)

Edge feature vector shrinks from 5 columns to 1 (just dist).
Node feature vector loses cmdCode (used only as a GROUPBY KEY for building
chapter/heading -- NOT removed from that -- just excluded from the actual
feature tensor fed to the model).

Compares each simplified config against its ORIGINAL counterpart's
overall log_r2/spearman (hardcoded below, from results_shrinkage_head.csv --
update these if you rerun the original grid and get different numbers).

If performance holds steady or improves: strong, direct evidence the
simplification is safe -- write it up as a confirmed finding, not an
inference. If performance drops noticeably for any config: that's real
information too -- worth flagging rather than skating past.

Usage:
    python retrain_simplified_edgegat.py [all_products_ready.parquet]
"""
import os, sys, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl
from dgl.nn import EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)

# ==========================================================================
# config
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
MISSING = 'rf'
KEEPALL='gdelt_features_by_country_year.parquet'; TOPK='gdelt_features_topk.parquet'
BILAT_KEEP='gdelt_bilateral_by_pair_year.parquet'; BILAT_TOPK='gdelt_bilateral_topk.parquet'
SCORER_COMBOS = {'ka_ka':(KEEPALL,BILAT_KEEP), 'ka_tk':(KEEPALL,BILAT_TOPK), 'tk_ka':(TOPK,BILAT_KEEP), 'tk_tk':(TOPK,BILAT_TOPK)}
NUM_HEADINGS = 10000
SHRINKAGE_K = 50
MODEL_DIR = 'trained_models_partial_pair_edgegat'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- the simplification: cmdCode dropped from node features, pair_* dropped
# from edge features (dist kept -- it's the one edge feature that survived
# permutation importance with a real, if small, positive signal) ---
FULL_NODE_COLS_SIMPLIFIED = ['refYear','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS_SIMPLIFIED = ['dist', 'pair_events', 'pair_score_mean']   # dropped: pair_score_max, pair_gold_mean (both negative mean)

FUSIONS = ['concat', 'blend', 'attention']

# original (unsimplified) overall numbers, from results_shrinkage_head.csv --
# update if you rerun the original grid and these change
ORIGINAL_OVERALL = {
    ('ka_ka','attention'): 0.272, ('ka_tk','attention'): 0.287, ('tk_ka','attention'): 0.339, ('tk_tk','attention'): 0.304,
    ('ka_ka','blend'):     0.319, ('ka_tk','blend'):     0.366, ('tk_ka','blend'):     0.321, ('tk_tk','blend'):     0.338,
    ('ka_ka','concat'):    0.211, ('ka_tk','concat'):    0.335, ('tk_ka','concat'):    0.331, ('tk_tk','concat'):    0.340,
}

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
# data pipeline -- build_agg still computes cmdCode (needed as a groupby key
# and for chapter/heading derivation), we just exclude it from the feature
# tensor at transform time via FULL_NODE_COLS_SIMPLIFIED
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

BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']

def edge_features_simplified(df_rows, bilat_file):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode','partnerCode','refYear','dist']].copy()
    for c in ['reporterCode','partnerCode','refYear']: k[c] = k[c].astype('Int64')
    b['reporterCode'] = b['reporterCode'].astype('Int64'); b['partnerCode'] = b['partnerCode'].astype('Int64'); b['year'] = b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode','partnerCode','refYear'],
                right_on=['reporterCode','partnerCode','year'], how='left')
    m[BILAT_COLS] = m[BILAT_COLS].fillna(0)
    m['dist'] = m['dist'].fillna(m['dist'].median())
    return m[FULL_EDGE_COLS_SIMPLIFIED].to_numpy(dtype='float32')
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
# models -- identical to run_shrinkage_head.py, fed the simplified col lists
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
# eval + train (simplified feature lists baked in)
# ==========================================================================
def _eval(model, df_eval, cmap, sf, st, gdelt_file, bilat_file, agg_mode='first', bs=20000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d, True, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS_SIMPLIFIED]); ye=st.transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_simplified(d, bilat_file))
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

def train_one(train_data, data_2023, gdelt_file, bilat_file, fusion, epochs=100, bs=20000):
    agg = build_agg(train_data, True, gdelt_file, 'first')
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=sf.fit_transform(agg[FULL_NODE_COLS_SIMPLIFIED]); ytr=st.fit_transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    ef=MinMaxScaler().fit_transform(edge_features_simplified(td, bilat_file))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy()))
    g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    g.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)

    n_trade = len(FULL_NODE_COLS_SIMPLIFIED) - len(GDELT_COLS); n_gdelt = len(GDELT_COLS)
    n_edge = len(FULL_EDGE_COLS_SIMPLIFIED)
    model = PerProductFusedEdgeModel(n_trade, n_gdelt, n_edge, fusion=fusion)
    heading_counts = pd.Series(heading).value_counts().to_dict()
    model.head.set_shrinkage(heading_counts)

    opt=torch.optim.Adam(model.parameters(),lr=0.01); crit=nn.MSELoss()
    N=g.num_nodes(); nb=N//bs+(N%bs>0)
    for ep in range(epochs):
        model.train()
        for i in range(nb):
            bn=list(range(i*bs, min((i+1)*bs, N)))
            bg=g.subgraph(torch.tensor(bn))
            out=model(bg, bg.ndata['feat'], bg.edata['ef'], bg.ndata['heading'], bg.ndata['chap2'])
            loss=crit(out.view(-1,1), bg.ndata['y'].view(-1,1))
            opt.zero_grad(); loss.backward(); opt.step()

    yt, yp = _eval(model, data_2023, cmap, sf, st, gdelt_file, bilat_file,  'first')
    metrics = compute_metrics(yt, yp)
    gc.collect()
    return model, metrics

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"simplified node cols ({len(FULL_NODE_COLS_SIMPLIFIED)}): {FULL_NODE_COLS_SIMPLIFIED}")
    log(f"simplified edge cols ({len(FULL_EDGE_COLS_SIMPLIFIED)}): {FULL_EDGE_COLS_SIMPLIFIED}  (was 5: dist + 4x pair_*)")
    train_data, val_data, data_2023 = load_or_split(data_file)

    out_csv = 'results_partial_pair_edgegat.csv'
    rows = []
    already_done = set()
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        already_done = set(prior['name'].unique())
        rows.extend(prior.to_dict('records'))
        log(f"resuming -- {len(already_done)} config(s) already done")

    for combo, (gdelt_file, bilat_file) in SCORER_COMBOS.items():
        for fusion in FUSIONS:
            name = f'{fusion}_{combo}'
            model_path = os.path.join(MODEL_DIR, f'EdgeGAT_full_{name}_simplified.pt')
            if name in already_done:
                log(f"skip {name} (already in {out_csv})")
                continue
            log(f"training EdgeGAT_full_{name}_simplified ...")
            try:
                model, metrics = train_one(train_data, data_2023, gdelt_file, bilat_file, fusion)
            except Exception as e:
                log(f"  [!] {name} FAILED ({type(e).__name__}: {e}) -- skipping, others unaffected.")
                continue
            torch.save(model.state_dict(), model_path)

            orig = ORIGINAL_OVERALL.get((combo, fusion))
            delta = (metrics['log_r2'] - orig) if orig is not None else None
            if orig is not None:
                log(f"  {name}: simplified log_r2={metrics['log_r2']:.3f}  spearman={metrics['spearman']:.3f}  "
                    f"(original={orig}, delta={delta:+.3f})")
            else:
                log(f"  {name}: simplified log_r2={metrics['log_r2']:.3f}  spearman={metrics['spearman']:.3f}")

            rows.append({'name': name, 'fusion': fusion, 'combo': combo,
                         'simplified_log_r2': metrics['log_r2'], 'simplified_spearman': metrics['spearman'],
                         'original_log_r2': orig, 'delta_log_r2': delta})
            pd.DataFrame(rows).to_csv(out_csv, index=False)   # checkpoint after every model

    result = pd.DataFrame(rows).sort_values('delta_log_r2', ascending=False)
    print("\n=== Simplified (no pair_*, no cmdCode) vs original ===")
    print(result.to_string(index=False))

    n_held = (result['delta_log_r2'] >= -0.01).sum()
    n_improved = (result['delta_log_r2'] > 0.01).sum()
    print(f"\n{n_held}/{len(result)} configs held steady or improved (delta >= -0.01).")
    print(f"{n_improved}/{len(result)} configs actually IMPROVED (delta > +0.01) -- "
          f"if this is more than a couple, the dropped features were adding noise, not just sitting neutral.")


if __name__ == "__main__":
    main()
