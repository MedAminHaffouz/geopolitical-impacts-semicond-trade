#!/usr/bin/env python3
"""
train_windowed_edge_hs4.py
============================
Expanding-window walk-forward training on the edge-regression architecture
from train_predict_edge_hs4.py.

CHANGES FROM THE PREVIOUS VERSION (per "why didn't the csv have actual/predicted"):
  1. Now ALSO loads the sparse-year files (patched 2016 + original 2024/2025),
     concatenated onto the main 2017-2023 data -- so folds can walk forward far
     enough to actually test on 2024 and 2025, not stop at 2023.
  2. Saves REAL actual/predicted pairs per fold, per corridor, to a second file
     (windowed_edge_predictions.csv) -- not just the aggregate metrics. This is
     what a plotting script needs; the old version only ever saved r2/log_r2/etc.
  3. Points at data/processed/all_products_2015_2016_ready_gravity_patched.parquet
     by default (the output of patch_2016_us_china.py) instead of the
     unpatched file -- update SPARSE_2015_2016 below if you renamed it.

Fold structure (expanding window, starting 2017):
  train_end=2017 -> test 2018   ... train_end=2023 -> test 2024   train_end=2024 -> test 2025

Usage:
    python train_windowed_edge_hs4.py [all_products_ready.parquet]

Not runnable in this sandbox -- no real data/DGL environment here. Run
locally; paste back windowed_edge_results.csv, windowed_edge_predictions.csv,
and/or the console log.
"""
import os
import sys
import csv
import pandas as pd

from train_predict_edge_hs4 import (
    load_chip_rows, train_variant, eval_variant, compute_metrics, log,
    VARIANTS, MODEL_DIR,
)

RESULTS_FILE = 'results/evaluation/windowed_edge_results.csv'
PREDICTIONS_FILE = 'results/evaluation/windowed_edge_predictions.csv'  # NEW -- real actual/predicted pairs

# update this path if patch_2016_us_china.py wrote a differently-named file
SPARSE_2015_2016 = 'data/processed/all_products_2015_2016_ready_gravity_patched.parquet'
SPARSE_2024_2025 = 'data/processed/all_products_2024_2025_ready_gravity.parquet'

CORRIDORS = {'US_to_China': dict(reporter=842, partner=156),
             'China_to_Taiwan': dict(reporter=156, partner=490)}  # still unverified -- see chat


def log_result(train_end, test_year, model_name, n, met):
    row = dict(train_end=train_end, test_year=test_year, model=model_name, n=n,
               mae=met['mae'], rmse=met['rmse'], r2=met['r2'],
               log_r2=met['log_r2'], spearman=met['spearman'])
    new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def log_predictions(train_end, test_year, model_name, edges_df, yt, yp):
    """NEW: writes real actual/predicted pairs, filtered to the two corridors,
    so a plotting script has something to read. Writing ALL edges here (often
    5,000-6,500 per fold) would make the CSV huge and mostly irrelevant --
    the corridor filter keeps it to just the rows anyone will actually plot."""
    edges_df = edges_df.reset_index(drop=True).copy()
    edges_df['actual'] = yt
    edges_df['predicted'] = yp
    rows = []
    for corridor, cfg in CORRIDORS.items():
        m = (edges_df['reporterCode'] == cfg['reporter']) & (edges_df['partnerCode'] == cfg['partner'])
        for _, row in edges_df[m].iterrows():
            rows.append(dict(train_end=train_end, test_year=test_year, model=model_name,
                              corridor=corridor, hs_code=int(row['cmdCode']),
                              actual=row['actual'], predicted=row['predicted']))
    if not rows:
        return 0
    new = not os.path.exists(PREDICTIONS_FILE)
    with open(PREDICTIONS_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"loading + chips-only filtering: {data_file}")
    df = load_chip_rows(data_file)

    for path, label in [(SPARSE_2015_2016, '2015/2016 (patched)'), (SPARSE_2024_2025, '2024/2025')]:
        if os.path.exists(path):
            sparse = load_chip_rows(path)
            df = pd.concat([df, sparse], ignore_index=True)
            log(f"  folded in {label}: +{len(sparse):,} rows")
        else:
            log(f"  WARNING: {path} not found -- {label} will be missing from this run "
                f"(check SPARSE_2015_2016/SPARSE_2024_2025 paths at the top of this script)")

    years_present = sorted(df['refYear'].dropna().unique().astype(int))
    log(f"  years now available: {years_present}")

    train_ends = [y for y in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] if y + 1 in years_present]
    for train_end in train_ends:
        test_year = train_end + 1
        train_df = df[df['refYear'].between(2015, train_end)].copy()
        test_df = df[df['refYear'] == test_year].copy()
        log(f"=== fold: train up to {train_end} ({len(train_df):,} rows) -> test {test_year} "
            f"({len(test_df):,} rows) ===")
        if len(test_df) == 0:
            log(f"  no rows for test_year={test_year} -- skipping this fold")
            continue

        for v in VARIANTS:
            fold_name = f"{v['name'].replace('HS4Edge_', 'HS4EdgeW_')}_upto{train_end}"
            model, bundle = train_variant(fold_name, v['encoder_kind'], v['fusion'],
                                           v['use_gdelt'], v['gdelt_file'], v['bilat_file'], train_df)
            yt, yp, edges_df = eval_variant(model, bundle, test_df, v['use_gdelt'], v['gdelt_file'],
                                             v['encoder_kind'], v['bilat_file'])
            if yt is None:
                log(f"  {fold_name}: no evaluable edges for {test_year} -- skipping")
                continue
            met = compute_metrics(yt, yp)
            log_result(train_end, test_year, v['name'], len(yt), met)
            n_pred = log_predictions(train_end, test_year, v['name'], edges_df, yt, yp)
            log(f"  {fold_name}: n={len(yt):,}  R2={met['r2']:.3f}  logR2={met['log_r2']:.3f}  "
                f"rho={met['spearman']:.3f}  ({n_pred} corridor rows saved to {PREDICTIONS_FILE})")

    log(f"=== DONE -- metrics in {RESULTS_FILE}, actual/predicted pairs in {PREDICTIONS_FILE} ===")


if __name__ == "__main__":
    main()