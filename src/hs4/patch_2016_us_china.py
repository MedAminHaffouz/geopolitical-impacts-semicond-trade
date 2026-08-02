#!/usr/bin/env python3
"""
patch_2016_us_china.py
========================
Merges the real, downloaded 2016 US->China Comtrade values into
data/processed/all_products_2015_2016_ready_gravity.parquet, closing the
genuine data gap (US did not report to Comtrade in 2016 -- confirmed by the
isReported=false/isAggregate=true flags in the downloaded CSV itself).

WHAT THIS DOES:
  - Inserts two new rows: (2016, reporter=842 USA, partner=156 China,
    cmdCode=8541/8542) with the real primaryValue from your Comtrade download.
  - Leaves gravity columns (dist, gdpcap_o, gdpcap_d, pop_o, pop_d) as NaN for
    these two new rows -- NOT hand-filled here. The existing RF imputer in
    your training/prediction scripts already handles this consistently for
    every other row; hand-filling here would be a special case that breaks
    that consistency for no real benefit (gravity features don't change much
    year to year, so 2017's US-China dist/gdpcap would be an equally-good
    stand-in either way -- let the imputer that's already fit on the full
    training panel make that call, not this script).
  - Tags both new rows with a `data_note` column so anyone reading the merged
    file later knows these are Comtrade-ESTIMATED values, not US-reported --
    this note is dropped before training (the model doesn't need to see it),
    but it's preserved in a sibling audit file for the report.

Usage:
    python patch_2016_us_china.py [comtrade_csv] [sparse_parquet]
    (defaults: TradeData_8_1_2026_20_15_32.csv,
               data/processed/all_products_2015_2016_ready_gravity.parquet)
"""
import sys
import csv
import pandas as pd
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)


def parse_comtrade_csv(path):
    """Handles the header/row field-count mismatch this UN Comtrade export has --
    47 header names but 48 fields per data row (one unnamed trailing column).
    Reads the fields we need by position from the END of each row instead of
    trusting pandas' left-aligned header match, which silently misaligns here."""
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        header = r.__next__()
        for row in r:
            rows.append(dict(
                refYear=int(row[3]),
                reporterCode=int(row[6]),
                partnerCode=int(row[11]),
                cmdCode=row[20],
                primaryValue=float(row[-5]),
                isReported=row[-3] == 'true',
                isAggregate=row[-2] == 'true',
            ))
    return pd.DataFrame(rows)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/TradeData_8_1_2026_20_15_32.csv'
    sparse_path = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/all_products_2015_2016_ready_gravity.parquet'

    log(f"parsing downloaded Comtrade CSV: {csv_path}")
    new_rows = parse_comtrade_csv(csv_path)
    log(f"  parsed {len(new_rows)} rows:\n{new_rows.to_string(index=False)}")

    log(f"loading existing sparse file: {sparse_path}")
    sparse = pd.read_parquet(sparse_path)
    before = len(sparse)

    # Only actually patch the 2016 rows -- 2015 already matches (confirmed above),
    # re-inserting it would just create a near-duplicate for no benefit.
    to_insert = new_rows[new_rows['refYear'] == 2016].copy()
    key_cols = ['refYear', 'reporterCode', 'partnerCode', 'cmdCode']

    # guard against double-patching if this script gets run twice
    existing_keys = set(map(tuple, sparse[key_cols].astype(str).values.tolist())) if all(c in sparse.columns for c in key_cols) else set()
    to_insert['cmdCode'] = to_insert['cmdCode'].astype(str)
    already_present = to_insert.apply(lambda r: tuple(str(r[c]) for c in key_cols) in existing_keys, axis=1)
    if already_present.any():
        log(f"  {already_present.sum()} row(s) already present in sparse file -- skipping those, not duplicating")
        to_insert = to_insert[~already_present]

    if to_insert.empty:
        log("  nothing new to insert -- sparse file already has these rows")
        return

    # gravity columns left NaN on purpose -- see module docstring
    for col in ['gdpcap_o', 'pop_o', 'gdpcap_d', 'pop_d', 'dist']:
        if col in sparse.columns:
            to_insert[col] = pd.NA

    to_insert['data_note'] = 'Comtrade-estimated (isReported=false, isAggregate=true) -- US did not report 2016 directly'

    merged = pd.concat([sparse, to_insert.drop(columns=['isReported', 'isAggregate'])], ignore_index=True)
    log(f"  {before} rows -> {len(merged)} rows ({len(merged) - before} inserted)")

    out_path = sparse_path.replace('.parquet', '_patched.parquet')
    merged.to_parquet(out_path, index=False)
    log(f"saved patched file -> {out_path}")
    log("NOTE: this writes a NEW file (_patched.parquet), it does not overwrite your original. "
        "Point your training/prediction scripts at this file once you've checked it looks right, "
        "or rename it to replace the original if you're confident.")

    # small audit CSV for the report -- just the human-readable trail of what got patched and why
    audit = to_insert[key_cols + ['primaryValue', 'data_note']]
    audit_path = 'results/evaluation/hs4_2016_us_china_patch_audit.csv'
    audit.to_csv(audit_path, index=False)
    log(f"saved audit trail -> {audit_path}")


if __name__ == "__main__":
    main()
