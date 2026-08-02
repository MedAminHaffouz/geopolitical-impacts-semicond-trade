#!/usr/bin/env python3
"""
train_windowed.py
------------------
Expanding-window (walk-forward) retraining, per your supervisor's suggestion:

    Fold 1: train 2017         -> test 2018-2025
    Fold 2: train 2017-2018    -> test 2019-2025
    Fold 3: train 2017-2019    -> test 2020-2025
    Fold 4: train 2017-2020    -> test 2021-2025
    Fold 5: train 2017-2021    -> test 2022-2025
    Fold 6: train 2017-2022    -> test 2023-2025  (closest to the original single-split setup)

Only 4 configs are retrained per fold (not the full 22-model sweep -- 6 folds x 22 configs
would be ~132 training runs, not realistic given the deadline):

    GAT_first_none              -- no-GDELT baseline
    GAT_first_keepall_attention -- the GDELT-enabled GAT winner (US_to_China corridor)
    EdgeGAT_full_blend_tk_tk    -- the EdgeGAT_full winner (China_to_Taiwan corridor)
    RF_first_keepall            -- RF baseline, for the entity-memorization collapse comparison

*** IMPORTANT -- READ BEFORE RUNNING ***
I do not have train_benchmark.py's actual training loop in front of me. The model classes,
build_agg(), and scaler logic below are copied from test_bilateral_pairs.py (already verified
against your real data). The TRAINING loop (optimizer, loss, epoch count) is reconstructed from
your own project notes (lr=0.01, bs=10000 GAT / bs=20000 EdgeGAT, normal_(std=0.05) embedding
init, zero-init biases) -- NOT verified against your actual script. Check EPOCHS, the optimizer,
and the loss function against train_benchmark.py before trusting these results, and adjust the
CONFIG section below if anything differs.

Evaluates each fold's models on the South_Korea_to_China corridor (410->156) -- the one
confirmed to have real data across 2015/2016/2024/2025 -- using the same chips-only filter
and cmdCode zero-pad fix as corridor_timeseries.py.

Usage:
    python train_windowed.py [split_cache_dir] [interim_dir] [processed_dir] [out_models_dir]
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

DEVICE = torch.device('cpu')

# ============================================================
# ---- CONFIG -- verify against train_benchmark.py before trusting results ----
EPOCHS = 60            # ASSUMPTION -- not verified against your actual script
LR = 0.01               # from your project notes
BS_GAT = 10000          # from your project notes
BS_EDGE = 20000         # from your project notes
EMBED_INIT_STD = 0.05   # from your project notes
# ============================================================

CHIP_HEADINGS = ['8541', '8542']
CORRIDOR = {'reporterCode': 410, 'partnerCode': 156}   # South Korea -> China

BASE_COLS = ['refYear', 'cmdCode', 'dist', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d']
GDELT_COLS = ['events_total', 'score_mean', 'score_max', 'score_vol',
              'goldstein_wmean', 'tone_mean', 'active_months']
FULL_NODE_COLS = ['refYear', 'cmdCode', 'gdpcap_d', 'gdpcap_o', 'pop_o', 'pop_d'] + GDELT_COLS
BILAT_COLS = ['pair_events', 'pair_score_mean', 'pair_score_max', 'pair_gold_mean']
FULL_EDGE_COLS = BILAT_COLS + ['dist']


def log(msg, level='INFO'):
    from datetime import datetime
    print(f'[{level}] {datetime.now().strftime("%H:%M:%S")}  {msg}', flush=True)


# ===================== model classes (copied from test_bilateral_pairs.py) =====================

class GATRegressionModel(nn.Module):
    def __init__(s, inf, h=32, heads=4):
        super().__init__()
        from dgl.nn import GATConv
        s.c1 = GATConv(inf, h, heads)
        s.c2 = GATConv(h * heads, 1, heads)

    def forward(s, g, x):
        # .mean(1) only collapses the attention-heads dim -- the trailing singleton
        # out_feats dim survives, giving shape [N,1] instead of [N] and silently
        # broadcasting against an [N] target into a wrong [N,N] loss. squeeze(-1) fixes it.
        return s.c2(g, torch.relu(s.c1(g, x).flatten(1))).mean(1).squeeze(-1)


class AttnGAT(nn.Module):
    def __init__(s, nt, ng, proj=16, h=32, heads=4):
        super().__init__()
        from dgl.nn import GATConv
        s.tp = nn.Linear(nt, proj)
        s.gp = nn.Linear(ng, proj)
        s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
        s.c1 = GATConv(proj, h, heads)
        s.c2 = GATConv(h * heads, 1, heads)

    def forward(s, g, xt, xg):
        st = torch.stack([torch.relu(s.tp(xt)), torch.relu(s.gp(xg))], dim=1)
        f = s.attn(st, st, st)[0].mean(1)
        return s.c2(g, torch.relu(s.c1(g, f).flatten(1))).mean(1).squeeze(-1)  # same fix as GATRegressionModel


class EdgeGAT(nn.Module):
    def __init__(s, inf, ef, h=32, heads=4):
        super().__init__()
        from dgl.nn import EdgeGATConv
        s.c1 = EdgeGATConv(inf, ef, h, heads, allow_zero_in_degree=True)
        s.c2 = EdgeGATConv(h * heads, ef, 1, heads, allow_zero_in_degree=True)

    def forward(s, g, x, efeat):
        x = torch.relu(s.c1(g, x, efeat).flatten(1))
        return s.c2(g, x, efeat).mean(1).squeeze(-1)


class FusedEdgeModel(nn.Module):
    def __init__(s, n_trade, n_gdelt, n_edge, fusion='blend', proj=16, h=32):
        super().__init__()
        s.fusion = fusion
        s.n_trade = n_trade
        if fusion == 'concat':
            inf = n_trade + n_gdelt
        else:
            s.tp = nn.Linear(n_trade, proj)
            s.gp = nn.Linear(n_gdelt, proj)
            if fusion == 'blend':
                s.alpha = nn.Parameter(torch.tensor(0.5))
            elif fusion == 'attention':
                s.attn = nn.MultiheadAttention(proj, 1, batch_first=True)
            inf = proj
        s.backbone = EdgeGAT(inf, n_edge, h)

    def forward(s, g, x, ef):
        if s.fusion == 'concat':
            node = x
        else:
            xt = x[:, :s.n_trade]
            xg = x[:, s.n_trade:]
            t = torch.relu(s.tp(xt))
            d = torch.relu(s.gp(xg))
            if s.fusion == 'blend':
                a = torch.sigmoid(s.alpha)
                node = a * d + (1 - a) * t
            else:
                st = torch.stack([t, d], dim=1)
                node = s.attn(st, st, st)[0].mean(1)
        return s.backbone(g, node, ef)


def _init_weights(model):
    """Per your project notes: normal_(std=0.05) embedding-style init, zero-init biases."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, std=EMBED_INIT_STD)
        else:
            nn.init.zeros_(p)


# ===================== data helpers (copied from test_bilateral_pairs.py) =====================

def add_gdelt(agg, gdelt_file):
    cy = pd.read_parquet(gdelt_file)
    out = agg.copy()
    out['_r'] = out['reporterCode'].astype('Int64').astype(str)
    out['_y'] = out['refYear'].astype('Int64').astype(str)
    cy['_r'] = cy['reporterCode'].astype('Int64').astype(str)
    cy['_y'] = cy['year'].astype('Int64').astype(str)
    out = out.merge(cy[['_r', '_y'] + GDELT_COLS], on=['_r', '_y'], how='left')
    out[GDELT_COLS] = out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_r', '_y'])


def build_agg(df, use_gdelt, gdelt_file, agg_mode='first'):
    grav = 'first' if agg_mode == 'first' else 'sum'
    agg = (df.groupby(['refYear', 'reporterCode', 'cmdCode'])
             # primaryValue: 'sum', not 'mean'. This is what actually "compacts" a
             # sparse-granularity year -- e.g. 2015/2024/2025 source rows that all padded
             # to the same '854100'/'854200' code get summed into one true heading-level
             # total per year/reporter, instead of averaged down to a single row's value.
             # For 2017-2023 (real 6-digit codes), each group is already one distinct
             # sub-product, so sum vs. mean rarely differs there -- this only bites where
             # compaction is actually happening.
             .agg(primaryValue=('primaryValue', 'sum'), dist=('dist', 'first'),
                  gdpcap_d=('gdpcap_d', grav), gdpcap_o=('gdpcap_o', grav),
                  pop_o=('pop_o', grav), pop_d=('pop_d', grav)).reset_index())
    agg['y_log'] = np.log1p(agg['primaryValue'].clip(lower=0))
    if use_gdelt:
        agg = add_gdelt(agg, gdelt_file)
    return agg


def edge_features_full(df_rows, bilat_file):
    b = pd.read_parquet(bilat_file)
    k = df_rows[['reporterCode', 'partnerCode', 'refYear', 'dist']].copy()
    for c in ['reporterCode', 'partnerCode', 'refYear']:
        k[c] = k[c].astype('Int64')
    b['reporterCode'] = b['reporterCode'].astype('Int64')
    b['partnerCode'] = b['partnerCode'].astype('Int64')
    b['year'] = b['year'].astype('Int64')
    m = k.merge(b, left_on=['reporterCode', 'partnerCode', 'refYear'],
                right_on=['reporterCode', 'partnerCode', 'year'], how='left')
    m[BILAT_COLS] = m[BILAT_COLS].fillna(0)
    m['dist'] = m['dist'].fillna(m['dist'].median())
    return m[FULL_EDGE_COLS].to_numpy(dtype='float32')


def prep_chip_frame(df):
    """Chips-only filter + cmdCode zero-pad, matching corridor_timeseries.py's approach.

    We are NOT retraining on a different granularity -- cmdCode is a raw numeric model
    feature, and the scaler/model expects 6-digit-scale values. So instead of truncating
    everything down to 4 digits (which would change the numeric scale the model was fit
    on), we compact the other direction: filter to the two chip headings regardless of
    the source's digit width, then right-pad any short code up to 6 digits with zeros
    ('8541' -> '854100'). This keeps every year on the same numeric scale the model
    trains/evaluates on. build_agg's groupby + sum (below) then does the actual
    compaction: years with only 4-digit source data collapse naturally to at most 2 rows
    per reporter (since they only ever had 2 distinct padded codes to begin with); years
    with real 6-digit data keep their full sub-heading resolution, unchanged.
    """
    df = df.copy()
    df['cmdCode'] = df['cmdCode'].astype(str)
    df = df[df['cmdCode'].str[:4].isin(CHIP_HEADINGS)]
    df['cmdCode'] = df['cmdCode'].str.ljust(6, '0')
    return df


def filter_pair(df, reporter_code, partner_code):
    r = pd.to_numeric(df['reporterCode'], errors='coerce')
    p = pd.to_numeric(df['partnerCode'], errors='coerce')
    out = df[(r == reporter_code) & (p == partner_code)].copy()
    if len(out) == 0:
        # DIAGNOSTIC: previously this just returned empty silently. If you're seeing
        # "no evaluable rows" for a specific year and not others, this block tells you
        # why -- dtype mismatch, wrong codes present, or genuinely absent data are three
        # very different problems and look identical without this.
        log(f'    filter_pair MISS: looking for reporter={reporter_code} partner={partner_code} | '
            f'reporterCode dtype={df["reporterCode"].dtype}, sample={df["reporterCode"].dropna().unique()[:5].tolist()} | '
            f'partnerCode dtype={df["partnerCode"].dtype}, sample={df["partnerCode"].dropna().unique()[:5].tolist()} | '
            f'n_rows_in={len(df)}', 'WARN')
    return out


# ===================== training (reconstructed from project notes -- VERIFY) =====================

def train_gat_attn(train_df, use_gdelt, gdelt_file, epochs=EPOCHS):
    """AttnGAT with GDELT fusion (matches GAT_first_keepall_attention's family)."""
    agg = build_agg(train_df, use_gdelt, gdelt_file)
    feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
    sf, st = MinMaxScaler(), MinMaxScaler()
    X = sf.fit_transform(agg[feat_cols])
    y = st.fit_transform(agg[['y_log']])
    cmap = {c: i for i, c in enumerate(agg['reporterCode'])}

    nt = len(BASE_COLS)
    model = AttnGAT(nt, len(GDELT_COLS)) if use_gdelt else GATRegressionModel(len(feat_cols))
    _init_weights(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    g = dgl.graph((np.arange(len(agg)), np.arange(len(agg))), num_nodes=len(agg))  # self-loops (no bilateral edges for plain GAT)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).squeeze(-1)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        if use_gdelt:
            out = model(g, Xt[:, :nt], Xt[:, nt:])
        else:
            out = model(g, Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            log(f'    epoch {epoch}/{epochs}  loss={loss.item():.4f}')

    model.eval()
    return model, sf, st, feat_cols, cmap


def train_edgegat_full(train_df, gdelt_file, bilat_file, fusion='blend', epochs=EPOCHS):
    """FusedEdgeModel (matches EdgeGAT_full_blend_tk_tk's family)."""
    agg = build_agg(train_df, True, gdelt_file)
    sf, st = MinMaxScaler(), MinMaxScaler()
    X = sf.fit_transform(agg[FULL_NODE_COLS])
    y = st.fit_transform(agg[['y_log']])

    n_trade = len(FULL_NODE_COLS) - len(GDELT_COLS)
    n_gdelt = len(GDELT_COLS)
    ei = len(FULL_EDGE_COLS)
    model = FusedEdgeModel(n_trade, n_gdelt, ei, fusion=fusion)
    _init_weights(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    idx = np.arange(len(agg))
    ef = MinMaxScaler().fit_transform(
        edge_features_full(train_df.groupby(['refYear', 'reporterCode', 'cmdCode']).first().reset_index(), bilat_file)
    )
    g = dgl.graph((idx, idx), num_nodes=len(agg))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).squeeze(-1)
    eft = torch.tensor(ef, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(g, Xt, eft)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            log(f'    epoch {epoch}/{epochs}  loss={loss.item():.4f}')

    model.eval()
    return model, sf, st


def train_rf(train_df, gdelt_file):
    agg = build_agg(train_df, True, gdelt_file)
    feat_cols = BASE_COLS + GDELT_COLS
    sf, st = MinMaxScaler(), MinMaxScaler()
    X = sf.fit_transform(agg[feat_cols])
    y = st.fit_transform(agg[['y_log']]).ravel()
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(X, y)
    return rf, sf, st, feat_cols


# ===================== eval on the corridor =====================

def eval_gat(model, use_gdelt, sf, st, feat_cols, test_df, gdelt_file):
    sub = prep_chip_frame(filter_pair(test_df, CORRIDOR['reporterCode'], CORRIDOR['partnerCode']))
    if len(sub) == 0:
        return None, None
    agg = build_agg(sub, use_gdelt, gdelt_file)
    if len(agg) == 0:
        return None, None
    X = sf.transform(agg[feat_cols])
    y = st.transform(agg[['y_log']])
    nt = len(BASE_COLS)
    Xt = torch.tensor(X, dtype=torch.float32)
    g = dgl.graph((np.arange(len(agg)), np.arange(len(agg))), num_nodes=len(agg))
    with torch.no_grad():
        out = model(g, Xt[:, :nt], Xt[:, nt:]) if use_gdelt else model(g, Xt)
    yp = np.expm1(np.clip(st.inverse_transform(out.view(-1, 1).numpy()).flatten(), -50, 50))  # clip guards against overflow if a model still produces an extreme output
    yt_ = np.expm1(st.inverse_transform(y).flatten())
    return yt_, yp


def eval_edgegat_full(model, sf, st, test_df, gdelt_file, bilat_file):
    sub = prep_chip_frame(filter_pair(test_df, CORRIDOR['reporterCode'], CORRIDOR['partnerCode']))
    if len(sub) == 0:
        return None, None
    agg = build_agg(sub, True, gdelt_file)
    if len(agg) == 0:
        return None, None
    X = sf.transform(agg[FULL_NODE_COLS])
    y = st.transform(agg[['y_log']])
    idx = np.arange(len(agg))
    ef = MinMaxScaler().fit_transform(edge_features_full(sub, bilat_file))
    g = dgl.graph((idx, idx), num_nodes=len(agg))
    Xt = torch.tensor(X, dtype=torch.float32)
    eft = torch.tensor(ef, dtype=torch.float32)
    with torch.no_grad():
        out = model(g, Xt, eft)
    yp = np.expm1(np.clip(st.inverse_transform(out.view(-1, 1).numpy()).flatten(), -50, 50))  # clip guards against overflow if a model still produces an extreme output
    yt_ = np.expm1(st.inverse_transform(y).flatten())
    return yt_, yp


def eval_rf(rf, sf, st, feat_cols, test_df, gdelt_file):
    sub = prep_chip_frame(filter_pair(test_df, CORRIDOR['reporterCode'], CORRIDOR['partnerCode']))
    if len(sub) == 0:
        return None, None
    agg = build_agg(sub, True, gdelt_file)
    if len(agg) == 0:
        return None, None
    X = sf.transform(agg[feat_cols])
    y = st.transform(agg[['y_log']])
    yp = np.expm1(np.clip(st.inverse_transform(rf.predict(X).reshape(-1, 1)).flatten(), -50, 50))
    yt_ = np.expm1(st.inverse_transform(y).flatten())
    return yt_, yp


def compute_metrics(yt, yp):
    ytl, ypl = np.log1p(np.clip(yt, 0, None)), np.log1p(np.clip(yp, 0, None))
    return dict(
        mae=mean_absolute_error(yt, yp), rmse=mean_squared_error(yt, yp) ** 0.5,
        r2=r2_score(yt, yp), log_r2=r2_score(ytl, ypl) if len(yt) > 2 else np.nan,
    )


# ===================== main: build folds, train, eval, log =====================

def load_year_sources(split_dir, processed_dir):
    """Returns {year: dataframe} for every year 2015-2025 this can find data for."""
    sources = {}
    train_full = pd.read_parquet(os.path.join(split_dir, 'train.parquet'))
    val_full = pd.read_parquet(os.path.join(split_dir, 'val.parquet'))
    test2023 = pd.read_parquet(os.path.join(split_dir, 'test2023.parquet'))
    # DIAGNOSTIC: test2023 is the only source loaded from a separate file rather than
    # sliced out of train/val by refYear -- if its schema drifted (dtype, column names,
    # or refYear itself not actually being 2023 throughout), this is where it'll show.
    log(f'test2023.parquet: {len(test2023)} rows, refYear values present='
        f'{sorted(pd.to_numeric(test2023["refYear"], errors="coerce").dropna().unique().tolist())}, '
        f'reporterCode dtype={test2023["reporterCode"].dtype}, partnerCode dtype={test2023["partnerCode"].dtype}, '
        f'cmdCode dtype={test2023["cmdCode"].dtype} sample={test2023["cmdCode"].astype(str).head(3).tolist()}')
    hist = pd.concat([train_full, val_full], ignore_index=True)
    for y in range(2017, 2023):
        sources[y] = hist[pd.to_numeric(hist['refYear'], errors='coerce') == y]
    sources[2023] = test2023

    extra = [([2015, 2016], 'all_products_2015_2016_ready_gravity.parquet'),
             ([2024, 2025], 'all_products_2024_2025_ready_gravity.parquet')]
    for years, fname in extra:
        fpath = os.path.join(processed_dir, fname)
        if not os.path.exists(fpath):
            log(f'{fname} not found, skipping years {years}', 'WARN')
            continue
        df_range = pd.read_parquet(fpath)
        for y in years:
            sources[y] = df_range[pd.to_numeric(df_range['refYear'], errors='coerce') == y]
    return sources


def main():
    split_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'cache', 'split_cache')
    interim_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('data', 'interim')
    processed_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join('data', 'processed')
    out_models_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join('models', 'trained_models_windowed')
    os.makedirs(out_models_dir, exist_ok=True)

    KEEPALL = os.path.join(interim_dir, 'gdelt_features_by_country_year.parquet')
    TOPK = os.path.join(interim_dir, 'gdelt_features_topk.parquet')
    BILAT_TOPK = os.path.join(interim_dir, 'gdelt_bilateral_topk.parquet')

    log('loading year sources 2015-2025 ...')
    year_sources = load_year_sources(split_dir, processed_dir)
    log(f'years available: {sorted(year_sources.keys())}')

    fold_train_ends = [2017, 2018, 2019, 2020, 2021, 2022]
    results = []

    for train_end in fold_train_ends:
        train_years = [y for y in range(2017, train_end + 1) if y in year_sources]
        test_years = [y for y in sorted(year_sources.keys()) if y > train_end]
        log(f'\n{"="*60}\nFOLD: train {train_years} -> test {test_years}\n{"="*60}')

        train_df = pd.concat([year_sources[y] for y in train_years], ignore_index=True)
        train_df = prep_chip_frame(train_df)  # train on chips only too, for consistency

        try:
            log('  training GAT_first_none ...')
            m_none, sf_none, st_none, fc_none, _ = train_gat_attn(train_df, use_gdelt=False, gdelt_file=None)

            log('  training GAT_first_keepall_attention ...')
            m_attn, sf_attn, st_attn, fc_attn, _ = train_gat_attn(train_df, use_gdelt=True, gdelt_file=KEEPALL)

            log('  training EdgeGAT_full_blend_tk_tk ...')
            m_edge, sf_edge, st_edge = train_edgegat_full(train_df, TOPK, BILAT_TOPK, fusion='blend')

            log('  training RF_first_keepall ...')
            m_rf, sf_rf, st_rf, fc_rf = train_rf(train_df, KEEPALL)
        except Exception as e:
            log(f'  training FAILED for this fold: {type(e).__name__}: {e}', 'ERROR')
            continue

        for test_year in test_years:
            test_df = year_sources[test_year]
            if len(test_df) == 0:
                continue

            for name, fn in [
                ('GAT_first_none', lambda: eval_gat(m_none, False, sf_none, st_none, fc_none, test_df, None)),
                ('GAT_first_keepall_attention', lambda: eval_gat(m_attn, True, sf_attn, st_attn, fc_attn, test_df, KEEPALL)),
                ('EdgeGAT_full_blend_tk_tk', lambda: eval_edgegat_full(m_edge, sf_edge, st_edge, test_df, TOPK, BILAT_TOPK)),
                ('RF_first_keepall', lambda: eval_rf(m_rf, sf_rf, st_rf, fc_rf, test_df, KEEPALL)),
            ]:
                yt_, yp = fn()
                if yt_ is None or len(yt_) == 0:
                    log(f'  {test_year} | {name}: no evaluable rows', 'WARN')
                    continue
                met = compute_metrics(yt_, yp)
                results.append(dict(train_end=train_end, test_year=test_year, model=name,
                                    n=len(yt_), **met))
                log(f'  {test_year} | {name}: n={len(yt_)} log_r2={met["log_r2"]:.3f}')

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv('windowed_results.csv', index=False)
        log(f'\nsaved windowed_results.csv ({len(df_out)} rows)')
    else:
        log('no results produced -- check the errors above', 'ERROR')


if __name__ == '__main__':
    main()