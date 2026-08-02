#!/usr/bin/env python3
"""
train_predict_edge_hs4.py
===========================
Edge-regression rewrite of the 4 requested models, replacing the node-regression
architecture that had two compounding bugs:

  BUG 1 (found first): build_agg grouped by (year, reporter, product) with NO
  partner in the key. Trade value got summed across every partner a reporter
  traded with; dist/gdpcap_d/pop_d got taken from whichever partner sorted
  first ('first'), arbitrarily. The model was trained to predict a country's
  TOTAL exports while being fed one random partner's distance/GDP as if it were
  the target relationship.

  BUG 2 (found second, worse): cmap = {c: i for i, c in enumerate(agg['reporterCode'])}
  is a dict comprehension -- for any reporter appearing more than once (which
  happens for EVERY reporter across 6 training years, and for every reporter
  trading both HS8541 and HS8542), only the LAST occurrence survives. Every
  country was collapsing onto ONE node holding whichever year+product's
  features happened to be processed last. This wasn't a partner problem, it
  was a year- and product-identity problem baked into how graphs were built.

WHAT THIS SCRIPT DOES INSTEAD:
  - Node = (year, country). Node features = that country's OWN profile that
    year (gdpcap, population, GDELT country-year signal). No product, no
    partner baked into node identity -- this can't collapse across years or
    products anymore, because the node KEY includes the year explicitly.
  - Edge = a real (year, reporter, partner, cmdCode) trade relationship.
    Edge features = dist, cmdCode (scaled), + bilateral GDELT pair signal for
    the EdgeGAT variants. Edge target = log1p(primaryValue), summed only
    across true duplicate rows for the exact same (year,reporter,partner,cmdCode)
    key -- never across partners.
  - Prediction = MLP head over [reporter_node_embedding, partner_node_embedding,
    edge_features], read off each edge via g.apply_edges -- not a value read
    off a single node.
  - SCOPE: chips only (HS 8541/8542), not all products. See chat message for
    why -- an all-products edge graph here would be millions of edges and not
    trainable full-batch on CPU in reasonable time.
  - Training is FULL-BATCH (whole graph every epoch, no subgraph batching).
    The (year,country) node graph is small (well under a thousand nodes even
    across 6 years x ~150 countries); the chips-only edge count is in the tens
    of thousands, not millions -- full-batch avoids the edge-subgraph/edge-
    feature-alignment bugs that mini-batching an edge-prediction model over
    subgraphs would otherwise risk introducing.
  - Trains, evaluates on 2023, AND evaluates on the sparse 2015/2016/2024/2025
    corridor files in ONE run, reusing the exact same graph-building function
    for all three -- eliminating the train/predict interface mismatches that
    caused several of the earlier bugs in this conversation.

Not runnable in this sandbox -- no real data, no DGL/GPU environment here.
Run locally; paste back the console log (or traceback) same as before.

Usage:
    python train_predict_edge_hs4.py [all_products_ready.parquet]
"""
import os, pickle, sys
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn
import dgl
from dgl.nn import GATConv, EdgeGATConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")


def log(msg, level="INFO"):
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


CHIP_HEADINGS = [8541, 8542]
GDELT_COLS = ['events_total', 'score_mean', 'score_max', 'score_vol', 'goldstein_wmean', 'tone_mean', 'active_months']
BILAT_COLS = ['pair_events', 'pair_score_mean', 'pair_score_max', 'pair_gold_mean']
KEEPALL = 'data/interim/gdelt_features_by_country_year.parquet'
TOPK = 'data/interim/gdelt_features_topk.parquet'
BILAT_KEEP = 'data/interim/gdelt_bilateral_by_pair_year.parquet'
BILAT_TOPK = 'data/interim/gdelt_bilateral_topk.parquet'
NODE_MISSING_COLS = ['gdpcap', 'pop']
MODEL_DIR = 'models/trained_models_hs4'
OUT_DIR = 'results/hs4_edge_outputs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cpu')  # this DGL build is not CUDA-enabled -- see earlier fix in train_hs4_selected.py
log(f"device = {DEVICE} (forced CPU)")


# ===== metrics =====

def _spearman(a, b):
    if len(a) < 3:
        return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')


def compute_metrics(yt, yp):
    ytl = np.log1p(np.clip(yt, 0, None))
    ypl = np.log1p(np.clip(yp, 0, None))
    return dict(mae=mean_absolute_error(yt, yp), rmse=mean_squared_error(yt, yp) ** 0.5,
                r2=r2_score(yt, yp), log_r2=(r2_score(ytl, ypl) if len(yt) > 2 else np.nan),
                spearman=_spearman(yt, yp))


def report(name, yt, yp):
    met = compute_metrics(yt, yp)
    log(f"  {name}: n={len(yt):,}  R2={met['r2']:.3f}  logR2={met['log_r2']:.3f}  "
        f"rho={met['spearman']:.3f}  RMSE={met['rmse']:,.0f}  MAE={met['mae']:,.0f}")
    return met


# ===== loading, HS4-safe collapse, chips-only filter =====

def load_chip_rows(path):
    cols = ['refYear', 'reporterCode', 'partnerCode', 'cmdCode',
            'gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist', 'primaryValue']
    df = pd.read_parquet(path, columns=cols)
    for c in ['gdpcap_o', 'gdpcap_d', 'dist', 'pop_o', 'pop_d', 'primaryValue',
              'refYear', 'cmdCode', 'reporterCode', 'partnerCode']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['gdpcap_o'] /= 1e6
    df['gdpcap_d'] /= 1e6
    df['dist'] /= 1e3
    df = df[df['cmdCode'].notna()].copy()
    df['cmdCode'] = df['cmdCode'].astype(int)
    # collapse to HS4 ONLY if codes are genuinely 6-digit -- same guard as the
    # earlier sparse-file bug fix, since source granularity varies by file.
    if df['cmdCode'].max() >= 100000:
        df['cmdCode'] = df['cmdCode'] // 100
    df = df[df['cmdCode'].isin(CHIP_HEADINGS)].copy()
    return df


# ===== node table: (year, country) keyed, self-described profile only =====

def build_node_table(df, imputer=None, fit_imputer=False):
    a = df[['refYear', 'reporterCode', 'gdpcap_o', 'pop_o']].rename(
        columns={'reporterCode': 'country', 'gdpcap_o': 'gdpcap', 'pop_o': 'pop'})
    b = df[['refYear', 'partnerCode', 'gdpcap_d', 'pop_d']].rename(
        columns={'partnerCode': 'country', 'gdpcap_d': 'gdpcap', 'pop_d': 'pop'})
    nodes = pd.concat([a, b], ignore_index=True)
    nodes = nodes.dropna(subset=['country']).copy()
    nodes['country'] = nodes['country'].astype(int)
    nodes['refYear'] = nodes['refYear'].astype(int)
    # first non-null profile per (year, country) -- a country may appear with a
    # real self-reported profile in one row and only as someone else's partner
    # (possibly NaN) in another; prefer the non-null one.
    nodes = nodes.sort_values('gdpcap', na_position='last').drop_duplicates(['refYear', 'country'], keep='first')
    nodes = nodes.sort_values(['refYear', 'country']).reset_index(drop=True)

    if fit_imputer:
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20, n_jobs=2, random_state=0),
                                    max_iter=5, random_state=0)
        fit_sample = nodes[NODE_MISSING_COLS].sample(min(50_000, len(nodes)), random_state=0)
        imputer.fit(fit_sample)
    nodes[NODE_MISSING_COLS] = imputer.transform(nodes[NODE_MISSING_COLS])
    return nodes, imputer


def add_gdelt_node(nodes, gdelt_file):
    cy = pd.read_parquet(gdelt_file)
    out = nodes.copy()
    out['_c'] = out['country'].astype(str)
    out['_y'] = out['refYear'].astype(str)
    cy['_c'] = cy['reporterCode'].astype('Int64').astype(str)
    cy['_y'] = cy['year'].astype('Int64').astype(str)
    out = out.merge(cy[['_c', '_y'] + GDELT_COLS], on=['_c', '_y'], how='left')
    out[GDELT_COLS] = out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_c', '_y'])


# ===== edge table: genuine (year, reporter, partner, cmdCode) rows =====

def build_edge_table(df):
    e = (df.groupby(['refYear', 'reporterCode', 'partnerCode', 'cmdCode'])
           .agg(primaryValue=('primaryValue', 'sum'), dist=('dist', 'first')).reset_index())
    # dist can be genuinely NaN for pairs missing a CEPII distance entry -- unfilled,
    # this NaN survives into the scaled edge feature and poisons the WHOLE full-batch
    # loss via shared weights, not just that one edge's prediction. Same fillna the
    # older scripts always did for edge features, just missing here until now.
    if e['dist'].isna().any():
        n_missing = int(e['dist'].isna().sum())
        e['dist'] = e['dist'].fillna(e['dist'].median())
        log(f"    filled {n_missing} missing dist values with median ({e['dist'].median():.1f})")
    e['y_log'] = np.log1p(e['primaryValue'].clip(lower=0))
    return e


def add_gdelt_edge(edges, bilat_file):
    b = pd.read_parquet(bilat_file)
    k = edges[['reporterCode', 'partnerCode', 'refYear']].copy()
    for c in ['reporterCode', 'partnerCode', 'refYear']:
        k[c] = k[c].astype('Int64')
    b['reporterCode'] = b['reporterCode'].astype('Int64')
    b['partnerCode'] = b['partnerCode'].astype('Int64')
    b['year'] = b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode', 'partnerCode', 'refYear'],
                right_on=['reporterCode', 'partnerCode', 'year'], how='left')
    out = edges.copy()
    out[BILAT_COLS] = m[BILAT_COLS].fillna(0).to_numpy()
    return out


# ===== the full graph builder -- SAME function used for train, test2023, AND sparse corridors =====

def build_graph(df, use_gdelt, gdelt_file, encoder_kind, bilat_file=None,
                 node_imputer=None, fit_node_imputer=False, node_scaler=None, edge_scaler=None,
                 target_scaler=None, fit_scalers=False):
    nodes, node_imputer = build_node_table(df, node_imputer, fit_node_imputer)
    if use_gdelt:
        nodes = add_gdelt_node(nodes, gdelt_file)
    node_feat_cols = ['gdpcap', 'pop'] + (GDELT_COLS if use_gdelt else [])

    edges = build_edge_table(df)
    if encoder_kind == 'edgegat':
        edges = add_gdelt_edge(edges, bilat_file)
    edge_feat_cols = ['dist', 'cmdCode'] + (BILAT_COLS if encoder_kind == 'edgegat' else [])
    # cmdCode as a numeric edge feature -- only 2 distinct values (8541/8542) but
    # it's the thing distinguishing the two products on an otherwise-identical edge
    edges['cmdCode'] = edges['cmdCode'].astype(float)

    node_key = list(zip(nodes['refYear'].astype(int), nodes['country'].astype(int)))
    nmap = {k: i for i, k in enumerate(node_key)}  # SAFE now -- nodes already deduped per (year,country) above
    edges['s'] = list(zip(edges['refYear'].astype(int), edges['reporterCode'].astype(int)))
    edges['d'] = list(zip(edges['refYear'].astype(int), edges['partnerCode'].astype(int)))
    edges = edges[edges['s'].isin(nmap) & edges['d'].isin(nmap)].copy()
    if len(edges) == 0:
        return None
    edges['sidx'] = edges['s'].map(nmap)
    edges['didx'] = edges['d'].map(nmap)

    if fit_scalers:
        node_scaler = MinMaxScaler().fit(nodes[node_feat_cols])
        edge_scaler = MinMaxScaler().fit(edges[edge_feat_cols])
        target_scaler = MinMaxScaler().fit(edges[['y_log']])

    Xn = node_scaler.transform(nodes[node_feat_cols])
    Xe = edge_scaler.transform(edges[edge_feat_cols])
    if not fit_scalers:
        # MinMaxScaler extrapolates unboundedly outside its fitted range by default.
        # On new (sparse-year) data with genuinely out-of-range GDELT/feature values,
        # that produces extreme scaled inputs the model was never trained to handle --
        # this is what blew up HS4Edge_GAT_topk_concat's predictions to ~1e28. Clipping
        # to the fitted [0,1] range on eval only (never during training) caps the input
        # side of the problem, on top of the output-side clip in eval_variant.
        Xn = np.clip(Xn, 0.0, 1.0)
        Xe = np.clip(Xe, 0.0, 1.0)
    ye = target_scaler.transform(edges[['y_log']])[:, 0]

    g = dgl.graph((edges['sidx'].to_numpy(), edges['didx'].to_numpy()), num_nodes=len(nmap))
    g.ndata['feat'] = torch.tensor(Xn, dtype=torch.float32)
    g.edata['ef'] = torch.tensor(Xe, dtype=torch.float32)
    g.edata['y'] = torch.tensor(ye, dtype=torch.float32)

    bundle = dict(graph=g, node_imputer=node_imputer, node_scaler=node_scaler,
                  edge_scaler=edge_scaler, target_scaler=target_scaler,
                  node_feat_cols=node_feat_cols, edge_feat_cols=edge_feat_cols,
                  edges_df=edges)
    return bundle


# ===== model: encoder + genuine edge-readout head =====

class EdgeScoreModel(nn.Module):
    def __init__(s, encoder_kind, n_trade, n_gdelt, n_edge, fusion='none', proj=16, h=32, heads=4):
        super().__init__()
        s.encoder_kind = encoder_kind
        s.fusion = fusion
        s.n_trade = n_trade
        if fusion in ('none', 'concat'):
            inf = n_trade + (n_gdelt if fusion == 'concat' else 0)
        else:
            s.tp = nn.Linear(n_trade, proj)
            s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend':
                s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention':
                s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        if encoder_kind == 'gat':
            s.c1 = GATConv(inf, h, heads, allow_zero_in_degree=True)
            s.c2 = GATConv(h * heads, h, heads, allow_zero_in_degree=True)
        else:
            s.c1 = EdgeGATConv(inf, n_edge, h, heads, allow_zero_in_degree=True)
            s.c2 = EdgeGATConv(h * heads, n_edge, h, heads, allow_zero_in_degree=True)
        s.head = nn.Sequential(nn.Linear(h * 2 + n_edge, 32), nn.ReLU(), nn.Linear(32, 1))

    def _fuse(s, x):
        if s.fusion in ('none', 'concat'):
            return x
        xt, xg = x[:, :s.n_trade], x[:, s.n_trade:]
        t, d = torch.relu(s.tp(xt)), torch.relu(s.gp(xg))
        if s.fusion == 'blend':
            a = torch.sigmoid(s.alpha)
            return a * d + (1 - a) * t
        st = torch.stack([t, d], dim=1)
        return s.attn(st, st, st)[0].mean(1)

    def forward(s, g, x, ef):
        node_in = s._fuse(x)
        if s.encoder_kind == 'gat':
            h1 = torch.relu(s.c1(g, node_in).flatten(1))
            h2 = s.c2(g, h1).mean(1)
        else:
            h1 = torch.relu(s.c1(g, node_in, ef).flatten(1))
            h2 = s.c2(g, h1, ef).mean(1)
        with g.local_scope():
            g.ndata['h'] = h2
            g.edata['ef2'] = ef
            g.apply_edges(lambda e: {'pred': s.head(
                torch.cat([e.src['h'], e.dst['h'], e.data['ef2']], dim=1)).squeeze(-1)})
            return g.edata['pred']


# ===== train one variant, full-batch =====

def train_variant(name, encoder_kind, fusion, use_gdelt, gdelt_file, bilat_file,
                   train_df, epochs=150):
    bundle = build_graph(train_df, use_gdelt, gdelt_file, encoder_kind, bilat_file,
                          fit_node_imputer=True, fit_scalers=True)
    g = bundle['graph']
    for tensor_name in ('feat',):
        if torch.isnan(g.ndata[tensor_name]).any():
            raise ValueError(f"{name}: NaN in node feature '{tensor_name}' after imputation+scaling -- "
                              f"check node_feat_cols for a column the imputer didn't cover")
    for tensor_name in ('ef', 'y'):
        if torch.isnan(g.edata[tensor_name]).any():
            raise ValueError(f"{name}: NaN in edge tensor '{tensor_name}' after scaling -- "
                              f"check edge_feat_cols for an unfilled NaN source")
    n_trade = 2  # gdpcap, pop
    n_gdelt = len(GDELT_COLS) if use_gdelt else 0
    n_edge = len(bundle['edge_feat_cols'])
    model = EdgeScoreModel(encoder_kind, n_trade, n_gdelt, n_edge, fusion=fusion).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        pred = model(g, g.ndata['feat'], g.edata['ef'])
        loss = crit(pred, g.edata['y'])
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  {name}: final train loss={loss.item():.5f}  (n_edges={g.num_edges()}, n_nodes={g.num_nodes()})")

    with open(os.path.join(MODEL_DIR, f'{name}_bundle.pkl'), 'wb') as f:
        pickle.dump({k: v for k, v in bundle.items() if k not in ('graph', 'edges_df')}, f)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{name}.pt'))
    return model, bundle


def eval_variant(model, bundle_train, df_eval, use_gdelt, gdelt_file, encoder_kind, bilat_file):
    """Evaluate using the SAME build_graph function, reusing the fitted imputer/scalers
    from training -- this is what keeps train and eval on identical footing."""
    b = build_graph(df_eval, use_gdelt, gdelt_file, encoder_kind, bilat_file,
                     node_imputer=bundle_train['node_imputer'], fit_node_imputer=False,
                     node_scaler=bundle_train['node_scaler'], edge_scaler=bundle_train['edge_scaler'],
                     target_scaler=bundle_train['target_scaler'], fit_scalers=False)
    if b is None:
        return None, None, None
    g = b['graph']
    model.eval()
    with torch.no_grad():
        pred = model(g, g.ndata['feat'], g.edata['ef']).numpy()
    yp_log = bundle_train['target_scaler'].inverse_transform(pred.reshape(-1, 1)).flatten()
    n_clipped = int(np.sum(np.abs(yp_log) > 50))
    if n_clipped:
        log(f"    WARNING: {n_clipped}/{len(yp_log)} predictions clipped at exp guard -- model output is "
            f"out-of-range for this eval slice (likely OOD features for a 'concat'-fusion model; see chat)")
    yp = np.expm1(np.clip(yp_log, -50, 50))  # SAME guard convention used everywhere else in this project
    yt = np.expm1(bundle_train['target_scaler'].inverse_transform(g.edata['y'].numpy().reshape(-1, 1)).flatten())
    return yt, yp, b['edges_df']


VARIANTS = [
    dict(name='HS4Edge_GAT_none', encoder_kind='gat', fusion='none', use_gdelt=False, gdelt_file=None, bilat_file=None),
    dict(name='HS4Edge_GAT_topk_concat', encoder_kind='gat', fusion='concat', use_gdelt=True, gdelt_file=TOPK, bilat_file=None),
    dict(name='HS4Edge_EdgeGAT_attention_ka_tk', encoder_kind='edgegat', fusion='attention', use_gdelt=True, gdelt_file=KEEPALL, bilat_file=BILAT_TOPK),
    dict(name='HS4Edge_EdgeGAT_blend_tk_ka', encoder_kind='edgegat', fusion='blend', use_gdelt=True, gdelt_file=TOPK, bilat_file=BILAT_KEEP),
]


def extract_corridor_rows(name, model, bundle, df_eval, v, CORRIDORS, rows_by_hs, label):
    yt, yp, edges_df = eval_variant(model, bundle, df_eval, v['use_gdelt'], v['gdelt_file'],
                                     v['encoder_kind'], v['bilat_file'])
    if yt is None:
        log(f"  {name} [{label}]: no edges matched this model's training node map")
        return
    edges_df = edges_df.reset_index(drop=True)
    edges_df['actual'] = yt
    edges_df['predicted'] = yp
    n_added = 0
    for corridor, cfg in CORRIDORS.items():
        m = (edges_df['reporterCode'] == cfg['reporter']) & (edges_df['partnerCode'] == cfg['partner'])
        sub = edges_df[m]
        for _, row in sub.iterrows():
            hs = int(row['cmdCode'])
            if hs in rows_by_hs:
                rows_by_hs[hs].append(dict(hs_code=hs, corridor=corridor, year=int(row['refYear']),
                                            model=name, actual=row['actual'], predicted=row['predicted']))
                n_added += 1
    log(f"  {name} [{label}]: added {n_added} corridor rows")


def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"loading + chips-only filtering: {data_file}")
    df = load_chip_rows(data_file)
    train_df = df[df['refYear'].isin([2017, 2018, 2019, 2020, 2021, 2022])].copy()
    test_df = df[df['refYear'] == 2023].copy()
    log(f"  train rows={len(train_df):,}  test2023 rows={len(test_df):,}")

    trained = {}
    for v in VARIANTS:
        log(f"=== {v['name']} ===")
        model, bundle = train_variant(v['name'], v['encoder_kind'], v['fusion'],
                                       v['use_gdelt'], v['gdelt_file'], v['bilat_file'], train_df)
        yt, yp, _ = eval_variant(model, bundle, test_df, v['use_gdelt'], v['gdelt_file'],
                                  v['encoder_kind'], v['bilat_file'])
        if yt is not None:
            report(v['name'] + ' [test2023]', yt, yp)
        trained[v['name']] = (model, bundle, v)

    # ---- sparse-year files, same graph-building path, no separate reimplementation ----
    sparse_paths = ['data/processed/all_products_2015_2016_ready_gravity.parquet',
                     'data/processed/all_products_2024_2025_ready_gravity.parquet']
    parts = [load_chip_rows(p) for p in sparse_paths if os.path.exists(p)]
    sparse_df = pd.concat(parts, ignore_index=True) if parts else None
    if sparse_df is None:
        log("  no sparse files found -- corridor chart will only cover 2017-2023")

    CORRIDORS = {'US_to_China': dict(reporter=842, partner=156),
                 'China_to_Taiwan': dict(reporter=156, partner=490)}  # still unverified -- see chat

    # ---- corridor extraction across ALL THREE sources: train (2017-2022), test (2023), sparse (2015/16/24/25) ----
    # This is the actual fix for "why is there nothing in the middle" -- train_df/test_df
    # were already loaded and trained on above, they just were never run through the
    # corridor-extraction step before now. Nothing here is fabricated -- these are the
    # real training-year predictions, now finally included in the same chart.
    rows_by_hs = {8541: [], 8542: []}
    for name, (model, bundle, v) in trained.items():
        extract_corridor_rows(name, model, bundle, train_df, v, CORRIDORS, rows_by_hs, '2017-2022 train')
        extract_corridor_rows(name, model, bundle, test_df, v, CORRIDORS, rows_by_hs, '2023 test')
        if sparse_df is not None:
            extract_corridor_rows(name, model, bundle, sparse_df, v, CORRIDORS, rows_by_hs, 'sparse 2015/16/24/25')

    for hs, rows in rows_by_hs.items():
        out = pd.DataFrame(rows, columns=['hs_code', 'corridor', 'year', 'model', 'actual', 'predicted'])
        csv_path = os.path.join(OUT_DIR, f'hs{hs}_edge_predictions.csv')
        out.to_csv(csv_path, index=False)
        log(f"saved {csv_path} ({len(out)} rows)")
        for corridor in CORRIDORS:
            sub = out[out['corridor'] == corridor]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5.5))
            all_years = sorted(out['year'].unique())  # full range now: 2017-2022 train + 2023 test + sparse years
            actual_full = sub.drop_duplicates('year').set_index('year').reindex(all_years)['actual']
            ax.plot(all_years, actual_full, color='#555555', marker='o', lw=2, label='actual')
            colors = {'HS4Edge_GAT_none': '#4c8fbd', 'HS4Edge_GAT_topk_concat': '#c47f3e',
                      'HS4Edge_EdgeGAT_attention_ka_tk': '#2ca02c', 'HS4Edge_EdgeGAT_blend_tk_ka': '#888888'}
            for m in colors:
                d = sub[sub['model'] == m].sort_values('year')
                if d.empty:
                    continue
                d_full = d.set_index('year').reindex(all_years)['predicted']
                ax.plot(all_years, d_full, marker='o', lw=1.8, color=colors[m], label=m.replace('HS4Edge_', ''))
            ax.set_yscale('log'); ax.set_xlabel('year'); ax.set_ylabel('trade value (USD, log scale)')
            ax.set_xticks(all_years); ax.set_xticklabels([str(y) for y in all_years])
            ax.set_title(f'HS{hs} \u2014 {corridor} \u2014 edge-regression models', loc='left', fontweight='bold')
            ax.legend(fontsize=8)
            fig.tight_layout()
            png_path = os.path.join(OUT_DIR, f'hs{hs}_{corridor}_edge.png')
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log(f"  saved {png_path}")

    log("=== DONE ===")


if __name__ == "__main__":
    main()