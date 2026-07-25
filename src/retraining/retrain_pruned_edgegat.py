#!/usr/bin/env python3
"""
retrain_pruned_edgegat.py
============================
For 4 target configs -- EdgeGAT_full attention/ka_ka, attention/ka_tk,
blend/tk_tk, blend/tk_ka -- reads that SPECIFIC model's own permutation
importance results (feature_importance_shrinkage.csv) and drops whatever
features scored below THRESHOLD (negative or near-zero) for THAT model,
then retrains with the pruned feature set. Each of the 4 gets its own,
different pruned feature list -- they don't all drop the same columns.

IMPORTANT EXCEPTION, applied and logged per model: every one of these 4
configs' own importance results would drop ALL edge features (including
dist) if the threshold rule were applied literally -- but EdgeGATConv
structurally requires at least 1 edge feature to exist. So the single
least-bad edge feature (highest importance, even if below threshold) is
always force-kept. Same safety rule applies to node/trade/GDELT groups in
case any of those ever collapse to empty too, though that doesn't happen
for these 4 in practice.

Outputs a comparison table: original (full feature set) vs pruned, for
all 4 configs, log_r2 and spearman side by side.

Usage:
    python retrain_pruned_edgegat.py [all_products_ready.parquet] [feature_importance_shrinkage.csv]
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
TRADE_COLS = ['refYear','cmdCode','gdpcap_d','gdpcap_o','pop_o','pop_d']   # node cols minus GDELT, in original order
FULL_NODE_COLS = TRADE_COLS + GDELT_COLS
BILAT_COLS = ['pair_events','pair_score_mean','pair_score_max','pair_gold_mean']
FULL_EDGE_COLS = BILAT_COLS + ['dist']
MISSING = 'rf'
KEEPALL='gdelt_features_by_country_year.parquet'; TOPK='gdelt_features_topk.parquet'
BILAT_KEEP='gdelt_bilateral_by_pair_year.parquet'; BILAT_TOPK='gdelt_bilateral_topk.parquet'
NUM_HEADINGS = 10000
SHRINKAGE_K = 50
MODEL_DIR = 'trained_models_pruned_edgegat'
os.makedirs(MODEL_DIR, exist_ok=True)

THRESHOLDS = [0.005, 0.01, 0.02, 0.05]   # deduped from your request -- 0.005 was listed twice, running
                                          # the identical threshold twice would waste compute for no new info

# (model name in feature_importance_shrinkage.csv) -> (fusion, combo, gdelt_file, bilat_file)
TARGETS = {
    'EdgeGAT_full_attention_ka_ka_shrinkage': ('attention', 'ka_ka', KEEPALL, BILAT_KEEP),
    'EdgeGAT_full_attention_ka_tk_shrinkage': ('attention', 'ka_tk', KEEPALL, BILAT_TOPK),
    'EdgeGAT_full_blend_tk_tk_shrinkage':     ('blend',     'tk_tk', TOPK,    BILAT_TOPK),
    'EdgeGAT_full_blend_tk_ka_shrinkage':     ('blend',     'tk_ka', TOPK,    BILAT_KEEP),
}

# original (full feature set) numbers, from your run -- update if you rerun and these change
ORIGINAL_METRICS = {
    'EdgeGAT_full_attention_ka_ka_shrinkage': {'log_r2': 0.272, 'spearman': 0.602},
    'EdgeGAT_full_attention_ka_tk_shrinkage': {'log_r2': 0.287, 'spearman': 0.576},
    'EdgeGAT_full_blend_tk_tk_shrinkage':     {'log_r2': 0.338, 'spearman': 0.588},
    'EdgeGAT_full_blend_tk_ka_shrinkage':     {'log_r2': 0.321, 'spearman': 0.587},
}

def hs_chapter(cmdCode):
    c = np.asarray(cmdCode, dtype=np.int64)
    return np.where(c < 10000, c, c // 100).astype(np.int64)

def hs_chapter_2digit(heading):
    h = np.asarray(heading, dtype=np.int64)
    return (h // 100).astype(np.int64)

# ==========================================================================
# per-model pruned feature-set resolution
# ==========================================================================
def resolve_pruned_columns(imp_df, model_name, threshold):
    """Returns (trade_kept, gdelt_kept, edge_kept) -- each a list of column
    names, in their ORIGINAL order (important: the fusion split logic
    depends on trade-then-gdelt ordering being preserved)."""
    sub = imp_df[(imp_df.model == model_name) & (imp_df.feature != 'heading_idx')].copy()
    sub['clean'] = sub['feature'].apply(lambda f: f.split(':', 1)[-1])

    def keep_side(cols, side_prefix, safety_label):
        side = sub[sub['feature'].str.startswith(side_prefix)]
        keep = [c for c in cols if c in side[side.importance_log_r2 >= threshold]['clean'].tolist()]
        if not keep:
            best = side.sort_values('importance_log_r2', ascending=False).iloc[0]
            keep = [best['clean']]
            log(f"    [safety] {model_name} (threshold={threshold}): ALL {safety_label} features were "
                f"below threshold -- force-keeping the least-bad one ('{best['clean']}', "
                f"importance={best['importance_log_r2']:.5f}) since the architecture needs at least 1.")
        return keep

    trade_kept = keep_side(TRADE_COLS, 'node:', 'trade')
    gdelt_kept = keep_side(GDELT_COLS, 'node:', 'GDELT')
    edge_kept  = keep_side(FULL_EDGE_COLS, 'edge:', 'edge')
    return trade_kept, gdelt_kept, edge_kept

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

def edge_features_pruned(df_rows, bilat_file, edge_kept):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode','partnerCode','refYear','dist']].copy()
    for c in ['reporterCode','partnerCode','refYear']: k[c]=k[c].astype('Int64')
    b['reporterCode']=b['reporterCode'].astype('Int64'); b['partnerCode']=b['partnerCode'].astype('Int64'); b['year']=b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode','partnerCode','refYear'],
                right_on=['reporterCode','partnerCode','year'], how='left')
    m[BILAT_COLS]=m[BILAT_COLS].fillna(0); m['dist']=m['dist'].fillna(m['dist'].median())
    return m[edge_kept].to_numpy(dtype='float32')

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
        s.tp = nn.Linear(n_trade, proj); s.gp = nn.Linear(n_gdelt, proj)
        if fusion == 'blend':
            s.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion == 'attention':
            s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.backbone = PerProductEdgeGATBackbone(proj, n_edge, h, h2)
        s.head = PartialPoolingHead(h2, num_headings=num_chapters)
    def forward(s, g, x, ef, heading_idx, chapter2_idx):
        xt = x[:, :s.n_trade]; xg = x[:, s.n_trade:]
        t = torch.relu(s.tp(xt)); d = torch.relu(s.gp(xg))
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha); node = a*d + (1-a)*t
        else:
            st = torch.stack([t, d], dim=1); node = s.attn(st, st, st)[0].mean(1)
        hid = s.backbone(g, node, ef)
        return s.head(hid, heading_idx, chapter2_idx)

# ==========================================================================
# eval + train, parameterized by the per-model pruned column lists
# ==========================================================================
def _eval(model, df_eval, cmap, sf, st, node_cols, edge_cols, gdelt_file, bilat_file, agg_mode='first', bs=20000):
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
    ef=MinMaxScaler().fit_transform(edge_features_pruned(d, bilat_file, edge_cols))
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

def train_one(train_data, data_2023, gdelt_file, bilat_file, fusion, trade_kept, gdelt_kept, edge_kept,
              epochs=100, bs=20000):
    node_cols = trade_kept + gdelt_kept   # order matters -- trade first, then gdelt, matches n_trade split
    agg = build_agg(train_data, gdelt_file, 'first')
    cmap = {c:i for i,c in enumerate(agg['reporterCode'])}
    td = train_data.copy(); td['nID']=td['reporterCode'].map(cmap); td['pID']=td['partnerCode'].map(cmap)
    td = td.dropna(subset=['nID','pID']).copy(); td['nID']=td['nID'].astype(int); td['pID']=td['pID'].astype(int)
    sf,st=MinMaxScaler(),MinMaxScaler()
    Xtr=sf.fit_transform(agg[node_cols]); ytr=st.fit_transform(agg[['y_log']])
    heading = np.clip(agg['chapter'].to_numpy(), 0, NUM_HEADINGS-1)
    chap2 = np.clip(agg['chapter2'].to_numpy(), 0, 99)
    ef=MinMaxScaler().fit_transform(edge_features_pruned(td, bilat_file, edge_kept))
    g=dgl.graph((td['nID'].to_numpy(),td['pID'].to_numpy()))
    g.ndata['feat']=torch.tensor(Xtr,dtype=torch.float32)
    g.ndata['heading']=torch.tensor(heading,dtype=torch.long)
    g.ndata['chap2']=torch.tensor(chap2,dtype=torch.long)
    g.ndata['y']=torch.tensor(ytr[:,0],dtype=torch.float32)
    g.edata['ef']=torch.tensor(ef,dtype=torch.float32); g=dgl.add_self_loop(g, fill_data=0.)

    model = PerProductFusedEdgeModel(len(trade_kept), len(gdelt_kept), len(edge_kept), fusion=fusion)
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

    yt, yp = _eval(model, data_2023, cmap, sf, st, node_cols, edge_kept, gdelt_file, bilat_file, 'first')
    metrics = compute_metrics(yt, yp)
    gc.collect()
    return model, metrics

# ==========================================================================
# main
# ==========================================================================
def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    imp_path  = sys.argv[2] if len(sys.argv) > 2 else 'feature_importance_shrinkage.csv'

    imp_df = pd.read_csv(imp_path)
    log(f"loaded {imp_path} -- sweeping thresholds {THRESHOLDS} x {len(TARGETS)} configs "
        f"= {len(THRESHOLDS)*len(TARGETS)} total runs")

    out_csv = 'results_pruned_edgegat_sweep.csv'
    rows = []
    already_done = set()
    if os.path.exists(out_csv):
        prior = pd.read_csv(out_csv)
        already_done = set(zip(prior['threshold'], prior['model']))
        rows.extend(prior.to_dict('records'))
        log(f"resuming -- {len(already_done)} (threshold, model) pair(s) already done")

    train_data, val_data, data_2023 = load_or_split(data_file)

    for threshold in THRESHOLDS:
        log(f"\n{'='*70}\nTHRESHOLD = {threshold}\n{'='*70}")

        pruned_cols = {}
        for name in TARGETS:
            trade_kept, gdelt_kept, edge_kept = resolve_pruned_columns(imp_df, name, threshold)
            pruned_cols[name] = (trade_kept, gdelt_kept, edge_kept)
            log(f"  {name}:")
            log(f"    trade kept ({len(trade_kept)}/{len(TRADE_COLS)}): {trade_kept}")
            log(f"    GDELT kept ({len(gdelt_kept)}/{len(GDELT_COLS)}): {gdelt_kept}")
            log(f"    edge  kept ({len(edge_kept)}/{len(FULL_EDGE_COLS)}): {edge_kept}")

        for name, (fusion, combo, gdelt_file, bilat_file) in TARGETS.items():
            model_label = name.replace('_shrinkage', '')
            if (threshold, model_label) in already_done:
                log(f"skip {model_label} @ threshold={threshold} (already in {out_csv})")
                continue

            trade_kept, gdelt_kept, edge_kept = pruned_cols[name]
            th_tag = str(threshold).replace('.', 'p')
            model_path = os.path.join(MODEL_DIR, f'{name}_pruned_th{th_tag}.pt')

            log(f"training {name} @ threshold={threshold} ...")
            try:
                model, metrics = train_one(train_data, data_2023, gdelt_file, bilat_file, fusion,
                                            trade_kept, gdelt_kept, edge_kept)
            except Exception as e:
                log(f"  [!] {model_label} @ threshold={threshold} FAILED "
                    f"({type(e).__name__}: {e}) -- skipping, others unaffected.")
                continue
            torch.save(model.state_dict(), model_path)

            orig = ORIGINAL_METRICS[name]
            delta = metrics['log_r2'] - orig['log_r2']
            log(f"  {model_label} @ threshold={threshold}: pruned log_r2={metrics['log_r2']:.3f}  "
                f"spearman={metrics['spearman']:.3f}  (original log_r2={orig['log_r2']:.3f}, delta={delta:+.3f})")

            rows.append({
                'threshold': threshold,
                'model': model_label,
                'n_features_kept': len(trade_kept)+len(gdelt_kept)+len(edge_kept),
                'n_features_original': len(TRADE_COLS)+len(GDELT_COLS)+len(FULL_EDGE_COLS),
                'original_log_r2': orig['log_r2'], 'pruned_log_r2': metrics['log_r2'],
                'delta_log_r2': delta,
                'original_spearman': orig['spearman'], 'pruned_spearman': metrics['spearman'],
                'delta_spearman': metrics['spearman'] - orig['spearman'],
            })
            # checkpoint after every single (threshold, model) run -- a crash partway through a
            # sweep of this size shouldn't lose everything before it
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    if not rows:
        log("nothing ran -- nothing to summarize.")
        return

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv}")

    print("\n=== Full sweep: pruned vs original, every (threshold, model) pair ===")
    print(result.sort_values(['model', 'threshold']).to_string(index=False))

    print("\n=== Pivot: delta_log_r2 by model x threshold ===")
    pivot = result.pivot_table(index='model', columns='threshold', values='delta_log_r2')
    print(pivot.round(4).to_string())

    print("\n=== Pivot: n_features_kept by model x threshold ===")
    pivot_n = result.pivot_table(index='model', columns='threshold', values='n_features_kept')
    print(pivot_n.astype(int).to_string())

    print("\n=== Per-threshold summary ===")
    for th in sorted(result['threshold'].unique()):
        sub = result[result.threshold == th]
        n_held = (sub['delta_log_r2'] >= -0.01).sum()
        print(f"  threshold={th}: {n_held}/{len(sub)} configs held steady or improved, "
              f"mean delta={sub['delta_log_r2'].mean():+.4f}, "
              f"mean features kept={sub['n_features_kept'].mean():.1f}/{sub['n_features_original'].iloc[0]}")


if __name__ == "__main__":
    main()
