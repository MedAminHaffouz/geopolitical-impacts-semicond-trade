#!/usr/bin/env python3
"""
recompute_edgegat_variants_metrics.py
=========================================
Adds log_mse/log_rmse/log_mae/log_r2 (computed directly from log1p-space,
never routed through expm1) to the four EdgeGAT_full experiment families that
retrain_gravity_edge.py / retrain_simplified_edgegat.py / retrain_pruned_edgegat.py
/ retrain_leave_one_out.py each only reported r2/log_r2/spearman for. Loads
the already-saved .pt weights for each -- no retraining.

All four share the same PerProductFusedEdgeModel/PartialPoolingHead
architecture (retrain_pruned_edgegat.py and retrain_leave_one_out.py use a
version of the class with no 'concat' branch, but since both of those
experiment families only ever use fusion in {attention, blend}, the resulting
parameter shapes are identical to the fuller class used here -- so one
shared class correctly loads state_dicts from all four).

NOTE ON PATHS: the original retrain_*.py scripts cache splits under
`split_cache/<tag>/` (repo-root-relative), not `data/cache/split_cache/`
like recompute_full_metrics.py and everything else in the repo. That's an
inconsistency in the original scripts, not something to replicate --
this script uses `data/cache/split_cache/`, matching the rest of the repo
and reusing the same cache recompute_full_metrics.py already built.

Usage:
    python recompute_edgegat_variants_metrics.py --mode gravity     [data_file]
    python recompute_edgegat_variants_metrics.py --mode simplified  [data_file]
    python recompute_edgegat_variants_metrics.py --mode pruned      [data_file] [--importance_csv PATH]
    python recompute_edgegat_variants_metrics.py --mode loo         [data_file]
"""
import os, sys, gc, argparse
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
# shared config
# ==========================================================================
GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
TRADE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d']
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
FULL_EDGE_COLS = BILAT_COLS + ['dist']
MISSING = 'rf'
GDELT_DIR = 'data/interim'
KEEPALL=os.path.join(GDELT_DIR,'gdelt_features_by_country_year.parquet'); TOPK=os.path.join(GDELT_DIR,'gdelt_features_topk.parquet')
BILAT_KEEP=os.path.join(GDELT_DIR,'gdelt_bilateral_by_pair_year.parquet'); BILAT_TOPK=os.path.join(GDELT_DIR,'gdelt_bilateral_topk.parquet')
SCORER_COMBOS = {'ka_ka':(KEEPALL,BILAT_KEEP), 'ka_tk':(KEEPALL,BILAT_TOPK), 'tk_ka':(TOPK,BILAT_KEEP), 'tk_tk':(TOPK,BILAT_TOPK)}
NUM_HEADINGS = 10000
SHRINKAGE_K = 50
FUSIONS = ['concat', 'blend', 'attention']

# gravity-edge specific
FULL_NODE_COLS_GRAVITY = ['refYear', 'gdpcap_o', 'pop_o'] + GDELT_COLS
FULL_EDGE_COLS_GRAVITY = ['dist', 'gdpcap_gap', 'pop_gap']

# simplified/partial-pair specific
FULL_NODE_COLS_SIMPLIFIED = ['refYear','gdpcap_d','gdpcap_o','pop_o','pop_d'] + GDELT_COLS
FULL_EDGE_COLS_SIMPLIFIED = ['dist', 'pair_events', 'pair_score_mean']

# pruned/loo specific: 4-config target set both experiment families share
PRUNED_LOO_TARGETS = {
    'EdgeGAT_full_attention_ka_ka_shrinkage': ('attention', 'ka_ka', KEEPALL, BILAT_KEEP),
    'EdgeGAT_full_attention_ka_tk_shrinkage': ('attention', 'ka_tk', KEEPALL, BILAT_TOPK),
    'EdgeGAT_full_blend_tk_tk_shrinkage':     ('blend',     'tk_tk', TOPK,    BILAT_TOPK),
    'EdgeGAT_full_blend_tk_ka_shrinkage':     ('blend',     'tk_ka', TOPK,    BILAT_KEEP),
}
PRUNED_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]
LOO_ALL_FEATURES = TRADE_COLS + GDELT_COLS + FULL_EDGE_COLS

def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)
def hs_chapter_2digit(heading):
    return (np.asarray(heading, dtype=np.int64) // 100).astype(np.int64)

def leave_one_out_columns(feature_to_remove):
    trade_kept = [c for c in TRADE_COLS if c != feature_to_remove]
    gdelt_kept = [c for c in GDELT_COLS if c != feature_to_remove]
    edge_kept  = [c for c in FULL_EDGE_COLS if c != feature_to_remove]
    return trade_kept, gdelt_kept, edge_kept

def resolve_pruned_columns(imp_df, model_name, threshold):
    sub = imp_df[(imp_df.model == model_name) & (imp_df.feature != 'heading_idx')].copy()
    sub['clean'] = sub['feature'].apply(lambda f: f.split(':', 1)[-1])
    def keep_side(cols, side_prefix, safety_label):
        side = sub[sub['feature'].str.startswith(side_prefix)]
        keep = [c for c in cols if c in side[side.importance_log_r2 >= threshold]['clean'].tolist()]
        if not keep:
            best = side.sort_values('importance_log_r2', ascending=False).iloc[0]
            keep = [best['clean']]
        return keep
    trade_kept = keep_side(TRADE_COLS, 'node:', 'trade')
    gdelt_kept = keep_side(GDELT_COLS, 'node:', 'GDELT')
    edge_kept  = keep_side(FULL_EDGE_COLS, 'edge:', 'edge')
    return trade_kept, gdelt_kept, edge_kept

# ==========================================================================
# metrics -- log-space computed directly (never through expm1), matching
# the fix applied to recompute_full_metrics.py / retrain_track_metrics.py
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
# data pipeline -- identical to the retrain scripts, except load_or_split
# points at data/cache/split_cache (see NOTE ON PATHS above)
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

def edge_features_bilateral(df_rows, bilat_file, edge_cols):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode','partnerCode','refYear','dist']].copy()
    for c in ['reporterCode','partnerCode','refYear']: k[c]=k[c].astype('Int64')
    b['reporterCode']=b['reporterCode'].astype('Int64'); b['partnerCode']=b['partnerCode'].astype('Int64'); b['year']=b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode','partnerCode','refYear'],
                right_on=['reporterCode','partnerCode','year'], how='left')
    m[BILAT_COLS]=m[BILAT_COLS].fillna(0); m['dist']=m['dist'].fillna(m['dist'].median())
    return m[edge_cols].to_numpy(dtype='float32')

def edge_features_gravity(df_rows):
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
# shared model classes (see module docstring re: the concat-branch note)
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
    """Full-featured version (handles concat too) -- see module docstring:
    for pruned/loo (only ever attention/blend), parameter shapes are
    identical to the narrower class those scripts define locally."""
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
# generic eval, parameterized by node/edge column lists and edge-feature mode
# ==========================================================================
def eval_model(model, df_eval, cmap, sf, st, node_cols, edge_cols, gdelt_file, bilat_file,
               edge_mode='bilateral', agg_mode='first', bs=20000):
    d=df_eval.copy()
    for c in ['gdpcap_o','pop_o','gdpcap_d','pop_d','dist']: d[c]=d[c].fillna(d[c].mean())
    d['nID']=d['reporterCode'].map(cmap); d['pID']=d['partnerCode'].map(cmap)
    d=d.dropna(subset=['nID','pID']).copy()
    if len(d)==0: return None,None
    agg=build_agg(d, gdelt_file, agg_mode); emap={c:i for i,c in enumerate(agg['reporterCode'])}
    Xe=sf.transform(agg[node_cols]); ye=st.transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    d['eN']=d['reporterCode'].map(emap); d['eP']=d['partnerCode'].map(emap)
    d=d.dropna(subset=['eN','eP']); d['eN']=d['eN'].astype(int); d['eP']=d['eP'].astype(int)
    if edge_mode == 'gravity':
        ef = MinMaxScaler().fit_transform(edge_features_gravity(d))
    else:
        ef = MinMaxScaler().fit_transform(edge_features_bilateral(d, bilat_file, edge_cols))
    eg=dgl.graph((d['eN'].to_numpy(),d['eP'].to_numpy()), num_nodes=len(agg))
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

def fit_scalers(train_data, node_cols, gdelt_file):
    """Refits sf/st on train data with the given node_cols -- these scalers
    were never saved to disk (only state_dict was), so eval must rebuild
    them exactly as the original training run did: same columns, same
    'first' agg mode, same MinMaxScaler defaults."""
    agg = build_agg(train_data, gdelt_file, 'first')
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    sf, st = MinMaxScaler(), MinMaxScaler()
    sf.fit(agg[node_cols]); st.fit(agg[['y_log']])
    heading_counts = agg['chapter'].value_counts().to_dict()
    return cmap, sf, st, heading_counts

def load_model(path, n_trade, n_gdelt, n_edge, fusion, heading_counts):
    model = PerProductFusedEdgeModel(n_trade, n_gdelt, n_edge, fusion=fusion)
    model.head.set_shrinkage(heading_counts)   # overwritten by load_state_dict below if present, harmless either way
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model

# ==========================================================================
# mode: gravity
# ==========================================================================
def run_gravity(data_file):
    MODEL_DIR = 'models/trained_models_gravity_edge'
    train_data, val_data, data_2023 = load_or_split(data_file)
    rows = []
    for combo, (gdelt_file, bilat_file) in SCORER_COMBOS.items():
        for fusion in FUSIONS:
            name = f'{fusion}_{combo}'
            model_path = os.path.join(MODEL_DIR, f'EdgeGAT_full_{name}_gravity.pt')
            if not os.path.exists(model_path):
                log(f"  [!] {model_path} not found -- skipping."); continue
            cmap, sf, st, heading_counts = fit_scalers(train_data, FULL_NODE_COLS_GRAVITY, gdelt_file)
            n_trade = len(FULL_NODE_COLS_GRAVITY) - len(GDELT_COLS); n_gdelt = len(GDELT_COLS)
            model = load_model(model_path, n_trade, n_gdelt, len(FULL_EDGE_COLS_GRAVITY), fusion, heading_counts)
            yt, yp = eval_model(model, data_2023, cmap, sf, st, FULL_NODE_COLS_GRAVITY, FULL_EDGE_COLS_GRAVITY,
                                 gdelt_file, bilat_file, edge_mode='gravity')
            m = compute_full_metrics(yt, yp); m['name'] = name
            log(f"  {name}: log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}  spearman={m['spearman']:.3f}")
            rows.append(m); gc.collect()
    if not rows:
        log("  [!] no models found -- nothing to report. Check MODEL_DIR above actually has .pt files."); return
    out = pd.DataFrame(rows).set_index('name')
    out_path = 'results/grids/results_gravity_edge_full_metrics.csv'
    out.to_csv(out_path)
    print(f"\nsaved -> {out_path}")
    print(out.round(4).to_string())

# ==========================================================================
# mode: simplified / partial-pair
# ==========================================================================
def run_simplified(data_file):
    MODEL_DIR = 'models/trained_models_partial_pair_edgegat'
    train_data, val_data, data_2023 = load_or_split(data_file)
    rows = []
    for combo, (gdelt_file, bilat_file) in SCORER_COMBOS.items():
        for fusion in FUSIONS:
            name = f'{fusion}_{combo}'
            model_path = os.path.join(MODEL_DIR, f'EdgeGAT_full_{name}_simplified.pt')
            if not os.path.exists(model_path):
                log(f"  [!] {model_path} not found -- skipping."); continue
            cmap, sf, st, heading_counts = fit_scalers(train_data, FULL_NODE_COLS_SIMPLIFIED, gdelt_file)
            n_trade = len(FULL_NODE_COLS_SIMPLIFIED) - len(GDELT_COLS); n_gdelt = len(GDELT_COLS)
            model = load_model(model_path, n_trade, n_gdelt, len(FULL_EDGE_COLS_SIMPLIFIED), fusion, heading_counts)
            yt, yp = eval_model(model, data_2023, cmap, sf, st, FULL_NODE_COLS_SIMPLIFIED, FULL_EDGE_COLS_SIMPLIFIED,
                                 gdelt_file, bilat_file, edge_mode='bilateral')
            m = compute_full_metrics(yt, yp); m['name'] = name
            log(f"  {name}: log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}  spearman={m['spearman']:.3f}")
            rows.append(m); gc.collect()
    if not rows:
        log("  [!] no models found -- nothing to report. Check MODEL_DIR above actually has .pt files."); return
    out = pd.DataFrame(rows).set_index('name')
    out_path = 'results/grids/results_partial_pair_edgegat_full_metrics.csv'
    out.to_csv(out_path)
    print(f"\nsaved -> {out_path}")
    print(out.round(4).to_string())

# ==========================================================================
# mode: pruned (threshold sweep)
# ==========================================================================
def run_pruned(data_file, importance_csv):
    MODEL_DIR = 'models/trained_models_pruned_edgegat'
    if not os.path.exists(importance_csv):
        log(f"  [!] {importance_csv} not found -- required to resolve which columns each pruned model kept."); return
    imp_df = pd.read_csv(importance_csv)
    train_data, val_data, data_2023 = load_or_split(data_file)
    rows = []
    for threshold in PRUNED_THRESHOLDS:
        th_tag = str(threshold).replace('.', 'p')
        for name, (fusion, combo, gdelt_file, bilat_file) in PRUNED_LOO_TARGETS.items():
            model_path = os.path.join(MODEL_DIR, f'{name}_pruned_th{th_tag}.pt')
            if not os.path.exists(model_path):
                log(f"  [!] {model_path} not found -- skipping."); continue
            trade_kept, gdelt_kept, edge_kept = resolve_pruned_columns(imp_df, name, threshold)
            node_cols = trade_kept + gdelt_kept
            cmap, sf, st, heading_counts = fit_scalers(train_data, node_cols, gdelt_file)
            model = load_model(model_path, len(trade_kept), len(gdelt_kept), len(edge_kept), fusion, heading_counts)
            yt, yp = eval_model(model, data_2023, cmap, sf, st, node_cols, edge_kept, gdelt_file, bilat_file,
                                 edge_mode='bilateral')
            m = compute_full_metrics(yt, yp)
            m['model'] = name.replace('_shrinkage',''); m['threshold'] = threshold
            log(f"  {m['model']} @ th={threshold}: log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}")
            rows.append(m); gc.collect()
    if not rows:
        log("  [!] no models found -- nothing to report. Check MODEL_DIR above actually has .pt files."); return
    out = pd.DataFrame(rows).set_index(['model','threshold'])
    out_path = 'results/grids/results_pruned_edgegat_full_metrics.csv'
    out.to_csv(out_path)
    print(f"\nsaved -> {out_path}")
    print(out.round(4).to_string())

# ==========================================================================
# mode: leave-one-out (72 configs)
# ==========================================================================
def run_loo(data_file):
    MODEL_DIR = 'models/trained_models_leave_one_out'
    train_data, val_data, data_2023 = load_or_split(data_file)
    rows = []
    for name, (fusion, combo, gdelt_file, bilat_file) in PRUNED_LOO_TARGETS.items():
        for feature in LOO_ALL_FEATURES:
            safe_feat = feature.replace(' ', '_')
            model_path = os.path.join(MODEL_DIR, f'{name}_minus_{safe_feat}.pt')
            if not os.path.exists(model_path):
                log(f"  [!] {model_path} not found -- skipping."); continue
            trade_kept, gdelt_kept, edge_kept = leave_one_out_columns(feature)
            node_cols = trade_kept + gdelt_kept
            cmap, sf, st, heading_counts = fit_scalers(train_data, node_cols, gdelt_file)
            model = load_model(model_path, len(trade_kept), len(gdelt_kept), len(edge_kept), fusion, heading_counts)
            yt, yp = eval_model(model, data_2023, cmap, sf, st, node_cols, edge_kept, gdelt_file, bilat_file,
                                 edge_mode='bilateral')
            m = compute_full_metrics(yt, yp)
            m['model'] = name.replace('_shrinkage',''); m['feature_removed'] = feature
            log(f"  {m['model']} minus '{feature}': log_mse={m['log_mse']:.4f}  log_r2={m['log_r2']:.3f}")
            rows.append(m); gc.collect()
    if not rows:
        log("  [!] no models found -- nothing to report. Check MODEL_DIR above actually has .pt files."); return
    out = pd.DataFrame(rows).set_index(['model','feature_removed'])
    out_path = 'results/grids/results_leave_one_out_full_metrics.csv'
    out.to_csv(out_path)
    print(f"\nsaved -> {out_path}")
    print(out.round(4).to_string())

# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_file', nargs='?', default='all_products_ready.parquet')
    ap.add_argument('--mode', required=True, choices=['gravity','simplified','pruned','loo'])
    ap.add_argument('--importance_csv', default='results/feature_importance/feature_importance_shrinkage.csv')
    args = ap.parse_args()

    data_file = args.data_file
    if not os.path.exists(data_file):
        candidate = os.path.join('data/processed', data_file)
        if os.path.exists(candidate):
            data_file = candidate
        else:
            log(f"  [!] Could not find '{data_file}'. Run from repo root, or pass a full path.")
            return

    os.makedirs('results/grids', exist_ok=True)

    if args.mode == 'gravity':
        run_gravity(data_file)
    elif args.mode == 'simplified':
        run_simplified(data_file)
    elif args.mode == 'pruned':
        run_pruned(data_file, args.importance_csv)
    elif args.mode == 'loo':
        run_loo(data_file)

if __name__ == "__main__":
    main()