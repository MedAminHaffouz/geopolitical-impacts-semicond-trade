#!/usr/bin/env python3
"""
corridor_timeseries.py
------------------------
Actual-vs-predicted trade value, per year, for the two bilateral corridors, using the
single best already-trained model per corridor (picked from bilateral_test_metrics.csv's
results):

    US -> China   : GAT_first_keepall_attention
    China -> Taiwan : EdgeGAT_full_blend_tk_tk

No retraining, ever. Covers 2015-2025 now: 2017-2022 is what the model actually trained on
(expected to fit well -- not evidence of anything); every other year (2015, 2016, 2023, 2024,
2025) is genuine out-of-sample -- some BEFORE the training window, some after. The output
chart marks train vs. test explicitly so this can't be misread either way.

2015-2016 and 2024-2025 need their own Comtrade extraction + GDP/pop/distance gravity merge
(separate notebooks) and their own GDELT scoring pass (different interim files, suffixed by
range) -- this script expects those to already exist; it does not build them.

Reuses every helper (model classes, build_agg, eval functions, scalers, path resolution) from
test_bilateral_pairs.py rather than reimplementing them -- run this from the same directory
you already run that script from.

Usage:
    python corridor_timeseries.py [split_cache_dir] [trained_models_dir] [interim_dir] [processed_dir]
    (same defaults as test_bilateral_pairs.py: data/cache/split_cache,
     models/trained_models, data/interim; processed_dir defaults to data/processed)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import test_bilateral_pairs as tbp

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff',
    'axes.edgecolor': '#000000', 'axes.labelcolor': '#000000',
    'text.color': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000',
    'axes.grid': True, 'grid.color': '#000000', 'grid.alpha': 0.15, 'font.size': 10,
})
BLUE, ORANGE, GRAY = '#4c8fbd', '#c47f3e', '#888888'

# ---- the winning config per corridor, per bilateral_test_metrics.csv ----
BEST_MODEL = {
    'US_to_China': dict(name='GAT_first_keepall_attention', kind='GAT',
                         use_gdelt=True, scoring='keepall', fusion='attention', agg_mode='first'),
    'China_to_Taiwan': dict(name='EdgeGAT_full_blend_tk_tk', kind='EdgeGAT_full',
                             scoring='tk_tk', fusion='blend', agg_mode='first'),
    # NOTE: no dedicated 22-model bake-off has been run for this corridor (that would be a
    # test_bilateral_pairs.py pass) -- reusing US_to_China's winning config as a reasonable
    # starting point, not a confirmed best-for-this-corridor choice.
    'South_Korea_to_China': dict(name='GAT_first_keepall_attention', kind='GAT',
                                  use_gdelt=True, scoring='keepall', fusion='attention', agg_mode='first'),
}

# M49 codes: South Korea=410, China=156. Not in tbp.COUNTRY_PAIRS (only US_to_China and
# China_to_Taiwan live there) -- defined locally and merged in at runtime instead of
# editing test_bilateral_pairs.py, which other scripts also depend on.
EXTRA_PAIRS = {
    'South_Korea_to_China': {'reporterCode': 410, 'partnerCode': 156},
}

# ---- known events worth annotating (US export-control / CHIPS Act timeline) ----
EVENTS = [
    (2022.7, 'CHIPS Act (Aug) /\nexport controls (Oct) 2022'),
]

# ---- chips-only scope, matching the paper's actual subject ----
# NOTE: applied via cmdCode[:4] matching -- works regardless of whether a given source
# file's cmdCode is 6-digit (e.g. '854110') or already collapsed to 4-digit ('8541').
# That means this filter is correct-by-construction for SCOPE, but does NOT by itself
# fix a separate, real bug: if a year's source data only ever HAD 4-digit (or 2-digit)
# codes to begin with, cmdCode as a raw numeric model feature is still out of the
# distribution the model trained on. The granularity check right after filtering below
# exists specifically to catch that and refuse to silently trust degenerate results.
CHIP_HEADINGS = ['8541', '8542']

# restrict this run to just this corridor -- set to None to run every pair in
# tbp.COUNTRY_PAIRS instead (e.g. once China_to_Taiwan's own data issues are sorted)
#
# NOTE: switched from US_to_China -- confirmed via direct inspection of the 2015-2016/
# 2024-2025 files that the US doesn't appear as a reporter AT ALL in 2016 (not a filter
# issue, genuinely absent from that pull), so no US corridor can have full 2015-2025
# coverage until that's re-pulled. South_Korea_to_China has confirmed rows in every one
# of 2015/2016/2024/2025 in the actual uploaded files, and is a strong real corridor in
# its own right (Samsung/SK Hynix = two of the largest global memory chipmakers).
#
# For this run: back to the two report corridors (US_to_China, China_to_Taiwan). Both
# have known gaps outside 2017-2023 (US missing 2016 entirely; Taiwan/"Other Asia, nes"
# coverage in 2024-2025 not confirmed) -- plot_corridor_zerostart below now reindexes to
# the full year range and leaves genuine gaps as gaps rather than bridging over them.
ONLY_PAIRS = ['US_to_China', 'China_to_Taiwan']

# ---- extra year ranges beyond the original 2017-2023 split ----
# each entry: (years_covered, comtrade_filename_in_processed_dir, gdelt_filename_suffix)
# gdelt_filename_suffix='' means "use the base 2017-2023 gdelt files" (no suffix)
EXTRA_RANGES = [
    ([2015, 2016], 'all_products_2015_2016_ready_gravity.parquet', '_2015_2016'),
    ([2024, 2025], 'all_products_2024_2025_ready_gravity.parquet', '_2024_2025'),
]


def load_split(split_dir):
    train = pd.read_parquet(os.path.join(split_dir, 'train.parquet'))
    val = pd.read_parquet(os.path.join(split_dir, 'val.parquet'))
    test = pd.read_parquet(os.path.join(split_dir, 'test2023.parquet'))
    hist = pd.concat([train, val], ignore_index=True)  # all of 2017-2022, actuals + in-sample preds
    return train, hist, test


def build_year_sources(hist, test, processed_dir, interim_dir):
    """Returns a list of (year, split_label, df_source, gdelt_suffix) covering every year
    this script knows how to evaluate. gdelt_suffix picks which *_features_*/*_bilateral_*
    files to use for that year -- '' for the base 2017-2023 range, or a range-specific
    suffix for 2015-2016 / 2024-2025. Missing extra-range files are skipped with a warning
    rather than crashing the whole run."""
    sources = []
    for y in range(2017, 2023):
        sources.append((y, 'train', hist, ''))
    sources.append((2023, 'test', test, ''))

    for years_covered, fname, suffix in EXTRA_RANGES:
        fpath = os.path.join(processed_dir, fname)
        if not os.path.exists(fpath):
            tbp.log(f'{fname} not found in {processed_dir} -- skipping years {years_covered}', 'WARN')
            continue
        df_range = pd.read_parquet(fpath)
        for y in years_covered:
            sources.append((y, 'test', df_range, suffix))

    return sorted(sources, key=lambda s: s[0])


def gdelt_paths_for_suffix(interim_dir, suffix):
    keepall = os.path.join(interim_dir, f'gdelt_features_by_country_year{suffix}.parquet')
    topk = os.path.join(interim_dir, f'gdelt_features_topk{suffix}.parquet')
    bilat_keep = os.path.join(interim_dir, f'gdelt_bilateral_by_pair_year{suffix}.parquet')
    bilat_topk = os.path.join(interim_dir, f'gdelt_bilateral_topk{suffix}.parquet')
    return keepall, topk, bilat_keep, bilat_topk


def pick_gdelt_files(cfg, keepall, topk, bilat_keep, bilat_topk):
    """Mirrors the exact selection logic test_bilateral_pairs.py's main() already uses for
    the base range -- factored out here so it can be reapplied per year-range suffix."""
    if cfg['kind'] == 'GAT':
        gdelt_file = keepall if cfg['scoring'] == 'keepall' else topk if cfg['scoring'] == 'topk' else None
        return gdelt_file, None
    else:  # EdgeGAT_full
        combo_lookup = {'ka_ka': (keepall, bilat_keep), 'tk_tk': (topk, bilat_topk),
                        'ka_tk': (keepall, bilat_topk), 'tk_ka': (topk, bilat_keep)}
        return combo_lookup[cfg['scoring']]


def yearly_actual_predicted(model_spec, kind, year_sources, interim_dir, sf, st, cmap_or_none,
                             feat_cols, use_gdelt, agg_mode, fusion, nt, pair_spec, taiwan_fallback, cfg):
    """Runs the given corridor's frame through eval, one year at a time, using whichever
    GDELT file suffix applies to that year's range. Returns a DataFrame:
    year, actual, predicted, split ('train' or 'test')."""
    rows = []
    for year, split_label, source, suffix in year_sources:
        keepall, topk, bilat_keep, bilat_topk = gdelt_paths_for_suffix(interim_dir, suffix)
        gdelt_file, bilat_file = pick_gdelt_files(cfg, keepall, topk, bilat_keep, bilat_topk)

        if gdelt_file is not None and not os.path.exists(gdelt_file):
            print(f'  [WARN] {year}: gdelt file missing ({gdelt_file}), skipping')
            continue
        if bilat_file is not None and not os.path.exists(bilat_file):
            print(f'  [WARN] {year}: bilateral gdelt file missing ({bilat_file}), skipping')
            continue

        yr_df = source[pd.to_numeric(source['refYear'], errors='coerce') == year]
        sub = tbp.filter_pair(yr_df, pair_spec['reporterCode'], pair_spec['partnerCode'])
        if len(sub) == 0 and pair_spec['partnerCode'] == 490:
            sub = tbp.filter_pair(yr_df, pair_spec['reporterCode'], taiwan_fallback)
        if len(sub) == 0:
            print(f'  [WARN] {year}: no rows for this corridor, skipping')
            continue

        # ---- chips-only scope: keep only HS 8541/8542, regardless of source's digit width ----
        pre_filter_n = len(sub)
        sub = sub.copy()
        sub['cmdCode'] = sub['cmdCode'].astype(str)
        sub = sub[sub['cmdCode'].str[:4].isin(CHIP_HEADINGS)]
        if len(sub) == 0:
            print(f'  [WARN] {year}: 0 rows left after chip-heading filter '
                  f'(had {pre_filter_n} before), skipping')
            continue

        # ---- numeric-consistency fix: right-pad short codes to 6 digits ----
        # cmdCode is a raw numeric model FEATURE, scaled on training data's 6-digit codes
        # (e.g. '854110'). A heading-level code reported as plain '8541' is a ~100x SMALLER
        # number than any real 6-digit subheading under it purely from missing trailing
        # zeros -- not because the underlying product scope is wrong. Right-padding ('8541'
        # -> '854100') restores the correct numeric magnitude (heading-level aggregate,
        # zero-filled subheading) instead of silently feeding the model an out-of-range value.
        sub['cmdCode'] = sub['cmdCode'].str.ljust(6, '0')

        if kind == 'GAT':
            yt, yp = tbp._eval_graph(model_spec, sub, cmap_or_none, sf, st, feat_cols,
                                      use_gdelt, gdelt_file, agg_mode, fusion, nt)
        else:  # EdgeGAT_full
            yt, yp = tbp._eval_edge_full_bilateral(model_spec, sub, sf, st, gdelt_file, bilat_file, agg_mode)

        if yt is None or len(yt) == 0:
            print(f'  [WARN] {year}: eval produced 0 rows, skipping')
            continue

        rows.append({'year': year, 'actual': float(np.sum(yt)), 'predicted': float(np.sum(yp)),
                     'split': split_label, 'n_products': len(yt)})
    return pd.DataFrame(rows)


def _draw_train_test_series(ax, d, value_col, label_train, label_test):
    """Draws `value_col` vs. year, colored/styled by split -- generalizes to ANY split
    pattern (test years before training, after, or both, contiguous or not), unlike a
    single 'bridge from train's last point' segment which assumed test always comes
    chronologically after train. Segment color reflects whichever endpoint is 'test'."""
    d = d.sort_values('year').reset_index(drop=True)
    for i in range(len(d) - 1):
        is_test_seg = (d.loc[i, 'split'] == 'test') or (d.loc[i + 1, 'split'] == 'test')
        color = ORANGE if is_test_seg else BLUE
        style = '--' if is_test_seg else '-'
        ax.plot(d.loc[i:i+1, 'year'], d.loc[i:i+1, value_col], color=color, lw=2, linestyle=style, zorder=2)

    train_mask = d['split'] == 'train'
    test_mask = d['split'] == 'test'
    if train_mask.any():
        ax.plot(d.loc[train_mask, 'year'], d.loc[train_mask, value_col], 'o', color=BLUE,
                markersize=7, label=label_train, zorder=3)
    if test_mask.any():
        ax.plot(d.loc[test_mask, 'year'], d.loc[test_mask, value_col], 'D', color=ORANGE,
                markersize=9, label=label_test, zorder=3)

    for y in sorted(d.loc[test_mask, 'year'].unique()):
        ax.axvspan(y - 0.5, y + 0.5, color=ORANGE, alpha=0.06, zorder=0)


def _draw_events(ax, years_present):
    for xpos, label in EVENTS:
        if years_present.min() <= xpos <= years_present.max():
            ax.axvline(xpos, color='black', lw=0.8, linestyle=':')
            ax.text(xpos, 0.98, label, fontsize=7, ha='center', va='top',
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))


def plot_corridor(pair, df, model_name):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df['year'], df['actual'], color=GRAY, marker='o', lw=2, label='actual (real trade value)')
    _draw_train_test_series(ax, df, 'predicted', 'predicted (trained on this year)', 'predicted (held-out test year)')
    _draw_events(ax, df['year'])

    ax.set_xlabel('year')
    ax.set_ylabel('total trade value (USD)')
    ax.set_title(f'{pair} \u2014 actual vs. predicted, {model_name}\n'
                 f'(shaded = held-out test years -- some before training window, some after)',
                 loc='left', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(sorted(df['year'].unique()))
    fig.tight_layout()
    return fig


def plot_corridor_indexed(pair, df, model_name):
    """Same data, indexed to 100 at the first year present -- makes TREND/DIRECTION
    comparable even though predicted's absolute level is structurally off (see script
    docstring / the mean-across-partners target-definition note)."""
    d = df.sort_values('year').reset_index(drop=True)
    base_actual = d['actual'].iloc[0]
    base_pred = d['predicted'].iloc[0]
    d = d.assign(actual_idx=100 * d['actual'] / base_actual, predicted_idx=100 * d['predicted'] / base_pred)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(d['year'], d['actual_idx'], color=GRAY, marker='o', lw=2,
            label=f"actual (indexed, {d['year'].iloc[0]}=100)")
    _draw_train_test_series(ax, d.rename(columns={'predicted_idx': 'predicted'}), 'predicted',
                            'predicted (trained on this year)', 'predicted (held-out test year)')
    _draw_events(ax, d['year'])

    ax.axhline(100, color='black', lw=0.5, alpha=0.3)
    ax.set_xlabel('year')
    ax.set_ylabel(f"index ({d['year'].iloc[0]} = 100)")
    ax.set_title(f'{pair} \u2014 TREND comparison (indexed), {model_name}\n'
                 f'absolute levels differ structurally \u2014 this chart isolates direction/shape instead',
                 loc='left', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(sorted(df['year'].unique()))
    fig.tight_layout()
    return fig


def plot_corridor_logscale(pair, df, model_name):
    """Log y-axis -- the space every model here actually trained in (agg['y_log'] =
    log1p(primaryValue), see build_agg() in test_bilateral_pairs.py)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df['year'], df['actual'], color=GRAY, marker='o', lw=2, label='actual (real trade value)')
    _draw_train_test_series(ax, df, 'predicted', 'predicted (trained on this year)', 'predicted (held-out test year)')
    ax.set_yscale('log')
    _draw_events(ax, df['year'])

    ax.set_xlabel('year')
    ax.set_ylabel('total trade value (USD, log scale)')
    ax.set_title(f'{pair} \u2014 actual vs. predicted, LOG SCALE, {model_name}\n'
                 f'this is the space the model actually trained in (log1p target)',
                 loc='left', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(sorted(df['year'].unique()))
    fig.tight_layout()
    return fig


def plot_corridor_zerostart(pair, df, model_name):
    """Linear scale, y-axis forced to start at 0.

    Reindexes to every year between min and max present so that any year with no
    evaluable row (e.g. US missing as a reporter in 2016) shows as a genuine break in
    both lines -- matplotlib skips NaN by default, so this is a real gap, not an
    invented value. Also annotates each predicted point with the actual/predicted
    ratio for that year, since 0-start linear compresses the visual gap between the
    two lines far less than log scale does -- the annotation keeps the true scale of
    the miss legible even though the axis itself no longer shows it clearly.
    """
    full_years = pd.RangeIndex(int(df['year'].min()), int(df['year'].max()) + 1)
    d = df.set_index('year').reindex(full_years).reset_index().rename(columns={'index': 'year'})
    # split is only meaningful where a row actually exists; leave NaN split rows out
    # of _draw_train_test_series's masks naturally (they won't match 'train' or 'test')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(d['year'], d['actual'], color=GRAY, marker='o', lw=2, label='actual (real trade value)')
    _draw_train_test_series(ax, d, 'predicted', 'predicted (trained on this year)', 'predicted (held-out test year)')
    ax.set_ylim(bottom=0)
    _draw_events(ax, df['year'])

    for _, row in d.dropna(subset=['actual', 'predicted']).iterrows():
        if row['predicted'] > 0:
            ratio = row['actual'] / row['predicted']
            ax.annotate(f'{ratio:.0f}\u00d7 off', (row['year'], row['actual']),
                        textcoords='offset points', xytext=(0, 10), ha='center',
                        fontsize=7, color='#444444')

    missing_years = sorted(set(full_years) - set(df['year']))
    if missing_years:
        ax.text(0.01, -0.12, f'no evaluable data for: {", ".join(map(str, missing_years))}'
                              f' (shown as a gap, not interpolated)',
                transform=ax.transAxes, fontsize=7, color='#888888')

    ax.set_xlabel('year')
    ax.set_ylabel('total trade value (USD)')
    ax.set_title(f'{pair} \u2014 actual vs. predicted, linear scale (0-start), {model_name}\n'
                 f'note: 0-start linear widens the visual gap between the two lines vs. log scale '
                 f'-- annotations show the true ratio',
                 loc='left', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(list(full_years))
    fig.tight_layout()
    return fig


def plot_convergence_ratio(pair, df, model_name):
    """How many times bigger reality is than the prediction, per year, and whether that
    gap is closing over time. Red shading marks the TRAINING period (whichever years are
    actually flagged 'train' in this run's data, not a hardcoded 2017-2022)."""
    d = df.sort_values('year').copy()
    ratio = d['actual'] / d['predicted']

    fig, ax = plt.subplots(figsize=(11, 6))

    train_years = d[d['split'] == 'train']['year']
    if len(train_years):
        ax.axvspan(train_years.min() - 0.5, train_years.max() + 0.5,
                   color='#d62728', alpha=0.08, zorder=0,
                   label=f'training period ({int(train_years.min())}-{int(train_years.max())})')

    ax.plot(d['year'], ratio, marker='o', lw=2, color=BLUE, zorder=3)
    for x, y in zip(d['year'], ratio):
        ax.annotate(f'{y:.1f}\u00d7', (x, y), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8)

    ax.axhline(1.0, color='black', lw=1, linestyle='--', label='perfect match (ratio = 1)')
    ax.set_yscale('log')
    ax.set_xlabel('year')
    ax.set_ylabel('actual \u00f7 predicted  (lower = closer to reality)')
    ax.set_title(f'{pair} \u2014 convergence trend, {model_name}\n'
                 f'is the gap between prediction and reality narrowing over time?',
                 loc='left', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xticks(sorted(d['year'].unique()))
    fig.tight_layout()
    return fig


def main():
    split_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'cache', 'split_cache')
    models_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('models', 'trained_models')
    interim_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join('data', 'interim')
    processed_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join('data', 'processed')

    KEEPALL = os.path.join(interim_dir, tbp.KEEPALL_NAME)
    TOPK = os.path.join(interim_dir, tbp.TOPK_NAME)
    BILAT_KEEP = os.path.join(interim_dir, tbp.BILAT_KEEP_NAME)
    BILAT_TOPK = os.path.join(interim_dir, tbp.BILAT_TOPK_NAME)

    tbp.log(f'loading split from {split_dir}/')
    train, hist, test = load_split(split_dir)
    tbp.log(f'train+val (2017-2022)={len(hist):,}  test2023={len(test):,}')

    year_sources = build_year_sources(hist, test, processed_dir, interim_dir)
    tbp.log(f'evaluating years: {[y for y, *_ in year_sources]}')

    all_pairs = {**tbp.COUNTRY_PAIRS, **EXTRA_PAIRS}
    all_results = {}
    pairs_to_run = {p: all_pairs[p] for p in ONLY_PAIRS} if ONLY_PAIRS else all_pairs
    for pair, spec in pairs_to_run.items():
        cfg = BEST_MODEL[pair]
        tbp.log(f'--- {pair}: {cfg["name"]} ---')
        pt_path = os.path.join(models_dir, f'{cfg["name"]}.pt')
        if not os.path.exists(pt_path):
            tbp.log(f'weights not found at {pt_path}, skipping', 'WARN')
            continue

        if cfg['kind'] == 'GAT':
            use_gdelt, gdelt_file, agg_mode, fusion = cfg['use_gdelt'], (
                KEEPALL if cfg['scoring'] == 'keepall' else TOPK if cfg['scoring'] == 'topk' else None
            ), cfg['agg_mode'], cfg['fusion']
            feat_cols, cmap, sf, st = tbp.prep_gat_scalers(train, use_gdelt, gdelt_file, agg_mode)
            nt = len(tbp.BASE_COLS)
            if fusion == 'blend':
                model = tbp.BlendGAT(nt, len(tbp.GDELT_COLS))
            elif fusion == 'attention':
                model = tbp.AttnGAT(nt, len(tbp.GDELT_COLS))
            else:
                model = tbp.GATRegressionModel(len(feat_cols))
            model.load_state_dict(torch.load(pt_path, map_location=tbp.DEVICE))
            model.eval()

            yearly = yearly_actual_predicted(model, 'GAT', year_sources, interim_dir, sf, st, cmap,
                                              feat_cols, use_gdelt, agg_mode, fusion, nt,
                                              spec, tbp.TAIWAN_FALLBACK_CODE, cfg)

        else:  # EdgeGAT_full
            gdelt_file, bilat_file = pick_gdelt_files(cfg, KEEPALL, TOPK, BILAT_KEEP, BILAT_TOPK)
            agg_mode, fusion = cfg['agg_mode'], cfg['fusion']
            cmap, sf, st = tbp.prep_edge_full_scalers(train, gdelt_file, agg_mode)
            n_trade = len(tbp.FULL_NODE_COLS) - len(tbp.GDELT_COLS)
            n_gdelt = len(tbp.GDELT_COLS)
            ei = len(tbp.FULL_EDGE_COLS)
            model = tbp.FusedEdgeModel('EdgeGAT', n_trade, n_gdelt, ei, fusion=fusion)
            model.load_state_dict(torch.load(pt_path, map_location=tbp.DEVICE))
            model.eval()

            yearly = yearly_actual_predicted(model, 'EdgeGAT_full', year_sources, interim_dir, sf, st, None,
                                              None, True, agg_mode, fusion, None,
                                              spec, tbp.TAIWAN_FALLBACK_CODE, cfg)

        if len(yearly) == 0:
            tbp.log(f'{pair}: no evaluable years, skipping', 'WARN')
            continue

        yearly['country_pair'] = pair
        yearly['model'] = cfg['name']
        all_results[pair] = yearly
        print(yearly.to_string(index=False))

        fig = plot_corridor(pair, yearly, cfg['name'])
        png_path = f'corridor_timeseries_{pair}.png'
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        tbp.log(f'saved {png_path}')

        fig_idx = plot_corridor_indexed(pair, yearly, cfg['name'])
        png_idx_path = f'corridor_timeseries_{pair}_indexed.png'
        fig_idx.savefig(png_idx_path, dpi=150, bbox_inches='tight')
        tbp.log(f'saved {png_idx_path}')

        fig_log = plot_corridor_logscale(pair, yearly, cfg['name'])
        png_log_path = f'corridor_timeseries_{pair}_logscale.png'
        fig_log.savefig(png_log_path, dpi=150, bbox_inches='tight')
        tbp.log(f'saved {png_log_path}')

        fig_conv = plot_convergence_ratio(pair, yearly, cfg['name'])
        png_conv_path = f'corridor_timeseries_{pair}_convergence.png'
        fig_conv.savefig(png_conv_path, dpi=150, bbox_inches='tight')
        tbp.log(f'saved {png_conv_path}')

        fig_zero = plot_corridor_zerostart(pair, yearly, cfg['name'])
        png_zero_path = f'corridor_timeseries_{pair}_zerostart.png'
        fig_zero.savefig(png_zero_path, dpi=150, bbox_inches='tight')
        tbp.log(f'saved {png_zero_path}')

    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv('corridor_timeseries.csv', index=False)
        tbp.log('saved corridor_timeseries.csv')
    tbp.log('DONE')


if __name__ == '__main__':
    main()