#!/usr/bin/env python3
"""
retrain_gravity_edge.py
==========================
Third variant in the ongoing feature-placement experiment. Recap:
  - original:    gdpcap_d/pop_d as crude NODE features (reporter-only groupby
                 collapses across partners via 'first' -- an aggregation
                 artifact, not a real destination-specific value)
  - simplified:  gdpcap_d/pop_d dropped entirely (retrain_simplified_edgegat.py)
  - gravity-edge (THIS SCRIPT): gdpcap_d/pop_d removed from nodes, replaced
                 by two genuinely pairwise EDGE features computed correctly
                 per (reporter, partner, year) -- no aggregation artifact,
                 because dist/gdpcap/pop are all constant per pair-year
                 (not per-product), so grouping by the actual pair instead
                 of by reporter alone is exact, not an approximation.

New edge features:
    gdpcap_gap = gdpcap_o - gdpcap_d   (economic asymmetry of the pair)
    pop_gap    = pop_o    - pop_d      (market-size asymmetry of the pair)
Both computed directly from train_data/data_2023's own raw rows (they
already carry reporterCode, partnerCode, gdpcap_o, gdpcap_d, pop_o, pop_d
per row, before the reporter-only groupby throws the partner distinction
away) -- no new external file needed, no bilateral GDELT-style parquet to
build separately.

Node features lose gdpcap_d/pop_d (their artifact-riddled home) and cmdCode
(per your earlier conclusion), keeping gdpcap_o/pop_o (these ARE correct as
node features -- a country's own GDP doesn't depend on its trade partner)
plus all 7 GDELT columns.

Trains all 12 EdgeGAT_full configs, same pattern as retrain_simplified_edgegat.py.
Compares against BOTH earlier variants where available.

Usage:
    python retrain_gravity_edge.py [all_products_ready.parquet]
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
MODEL_DIR = 'trained_models_gravity_edge'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- the redesign ---
FULL_NODE_COLS_GRAVITY = ['refYear', 'gdpcap_o', 'pop_o'] + GDELT_COLS   # gdpcap_d/pop_d/cmdCode removed
FULL_EDGE_COLS_GRAVITY = ['dist', 'gdpcap_gap', 'pop_gap']              # 2 new pairwise features added to dist

FUSIONS = ['concat', 'blend', 'attention']

# baselines from your other two variants, for the 3-way comparison --
# update these from your latest results_shrinkage_head.csv / results_simplified_edgegat.csv
ORIGINAL_OVERALL = {
    'ka_ka_attention': 0.272, 'ka_tk_attention': 0.287, 'tk_ka_attention': 0.339, 'tk_tk_attention': 0.304,
    'ka_ka_blend':      0.319, 'ka_tk_blend':      0.366, 'tk_ka_blend':      0.321, 'tk_tk_blend':      0.338,
    'ka_ka_concat':      0.211, 'ka_tk_concat':      0.335, 'tk_ka_concat':      0.331, 'tk_tk_concat':      0.340,
}
SIMPLIFIED_RESULTS_CSV = 'results_simplified_edgegat.csv'   # loaded automatically below if present

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

def build_agg(df, gdelt_file, agg_mode='first'):
    grav = 'first' if agg_mode=='first' else 'sum'
    agg = (df.groupby(['refYear','reporterCode','cmdCode'])
             .agg(primaryValue=('primaryValue','mean'), dist=('dist','first'),
                  gdpcap_d=('gdpcap_d',grav), gdpcap_o=('gdpcap_o',grav),
                  pop_o=('pop_o',grav), pop_d=('pop_d',grav)).reset_index())
    agg['y_log'] = np.log1p(agg['primaryValue'].clip(lower=0))
    agg['chapter'] = hs_chapter(agg['cmdCode'])
    agg['chapter2'] = hs_chapter_2digit(agg['chapter'])
    return add_gdelt(agg, gdelt_file)

def edge_features_gravity(df_rows):
    """The core of this experiment: gdpcap_gap/pop_gap computed directly from
    df_rows' own per-(reporter,partner) columns -- these are EXACT per-pair
    values (dist/gdpcap/pop don't vary by product), not an approximation."""
    k = df_rows[['dist','gdpcap_o','gdpcap_d','pop_o','pop_d']].copy()
    k['dist'] = k['dist'].fillna(k['dist'].median())
    k['gdpcap_gap'] = (k['gdpcap_o'] - k['gdpcap_d']).fillna(0)
    k['pop_gap'] = (k['pop_o'] - k['pop_d']).fillna(0)
    return k[FULL_EDGE_COLS_GRAVITY].to_numpy(dtype='float32')

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
# models
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
# eval + train
# ==========================================================================
def _eval(model, df_eval, cmap, sf, st, gdelt_file, agg_mode='first', bs=20000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[FULL_NODE_COLS_GRAVITY]); ye=st.transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    ef=MinMaxScaler().fit_transform(edge_features_gravity(d))
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

def train_one(train_data, data_2023, gdelt_file, fusion, epochs=100, bs=20000):
    agg = build_agg(train_data, gdelt_file, 'first')
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=sf.fit_transform(agg[FULL_NODE_COLS_GRAVITY]); ytr=st.fit_transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    ef=MinMaxScaler().fit_transform(edge_features_gravity(td))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy()))
    g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    g.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)

    n_trade = len(FULL_NODE_COLS_GRAVITY) - len(GDELT_COLS); n_gdelt = len(GDELT_COLS)
    n_edge = len(FULL_EDGE_COLS_GRAVITY)
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

    yt, yp = _eval(model, data_2023, cmap, sf, st, gdelt_file, 'first')
    metrics = compute_metrics(yt, yp)
    gc.collect()
    return model, metrics

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"gravity-edge node cols ({len(FULL_NODE_COLS_GRAVITY)}): {FULL_NODE_COLS_GRAVITY}")
    log(f"gravity-edge edge cols ({len(FULL_EDGE_COLS_GRAVITY)}): {FULL_EDGE_COLS_GRAVITY}")

    simplified = None
    if os.path.exists(SIMPLIFIED_RESULTS_CSV):
        simplified = pd.read_csv(SIMPLIFIED_RESULTS_CSV).set_index('name')['simplified_log_r2'].to_dict()
        log(f"loaded {SIMPLIFIED_RESULTS_CSV} for 3-way comparison")
    else:
        log(f"[note] {SIMPLIFIED_RESULTS_CSV} not found -- comparison will only show original vs gravity-edge")

    train_data, val_data, data_2023 = load_or_split(data_file)

    out_csv = 'results_gravity_edge.csv'
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
            model_path = os.path.join(MODEL_DIR, f'EdgeGAT_full_{name}_gravity.pt')
            if name in already_done:
                log(f"skip {name} (already in {out_csv})")
                continue
            log(f"training EdgeGAT_full_{name}_gravity ...")
            try:
                model, metrics = train_one(train_data, data_2023, gdelt_file, fusion)
            except Exception as e:
                log(f"  [!] {name} FAILED ({type(e).__name__}: {e}) -- skipping, others unaffected.")
                continue
            torch.save(model.state_dict(), model_path)

            orig = ORIGINAL_OVERALL.get(name)
            simp = simplified.get(name) if simplified else None
            delta_vs_orig = (metrics['log_r2'] - orig) if orig is not None else None
            delta_vs_simp = (metrics['log_r2'] - simp) if simp is not None else None
            log(f"  {name}: gravity_log_r2={metrics['log_r2']:.3f}  spearman={metrics['spearman']:.3f}  "
                f"(vs original={orig}, delta={delta_vs_orig}; vs simplified={simp}, delta={delta_vs_simp})")

            rows.append({'name': name, 'fusion': fusion, 'combo': combo,
                         'gravity_log_r2': metrics['log_r2'], 'gravity_spearman': metrics['spearman'],
                         'original_log_r2': orig, 'delta_vs_original': delta_vs_orig,
                         'simplified_log_r2': simp, 'delta_vs_simplified': delta_vs_simp})
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    result = pd.DataFrame(rows).sort_values('delta_vs_original', ascending=False)
    print("\n=== Gravity-edge (gdpcap_gap/pop_gap on edges) vs. original vs. simplified ===")
    print(result.to_string(index=False))
    print(f"\nsaved -> {out_csv}")

    if 'delta_vs_simplified' in result.columns and result['delta_vs_simplified'].notna().any():
        beats_simplified = (result['delta_vs_simplified'] > 0.01).sum()
        print(f"\n{beats_simplified}/{len(result)} configs: gravity-edge beats simplified by >0.01 log-R² --")
        print("if this is most/all of them, the destination-side GDP/pop signal was REAL, just misplaced.")
        print("If gravity-edge and simplified land about the same, the original gdpcap_d/pop_d genuinely")
        print("weren't adding value, independent of the aggregation artifact.")


if __name__ == "__main__":
    main()
