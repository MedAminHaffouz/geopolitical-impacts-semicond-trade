#!/usr/bin/env python3
"""
Fast profiler for the Comtrade + CEPII export CSV  (v2).

Reads in CHUNKS and only the columns we actually need, so it's lighter
and faster, and it prints PROGRESS so you can see it's alive.
Works on .csv or .csv.gz.

Usage:
    python3 profile_comtrade.py "comtradeExports_updatedH5[240924]-wb.csv.gz"

If you see "Killed", lower CHUNK below.
"""

import sys
import os
from collections import Counter

import numpy as np
import pandas as pd

CHUNK = 200_000  # rows per chunk


def h(n):
    return f"{int(n):,}"


def main(path):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    print(f"\n{'='*64}")
    print(f"Profiling: {path}")
    print(f"Size on disk: {os.path.getsize(path)/1e6:,.1f} MB")
    print('='*64, flush=True)

    # ---------- peek at structure (all columns) ----------
    head = pd.read_csv(path, nrows=5, dtype=str)
    all_cols = list(head.columns)
    print(f"\nColumns ({len(all_cols)}):\n  " + ", ".join(all_cols), flush=True)

    # ---------- only read the columns we use ----------
    NEEDED = [
        "cmdCode", "flowCode", "reporterCode", "partnerCode",
        "refYear", "qtyUnitCode", "classificationCode", "isAggregate",
        "qty", "netWgt", "grossWgt", "CIFValue", "FOBValue", "primaryValue",
        "pop_o", "gdp_o", "gdpcap_o", "pop_d", "gdp_d", "gdpcap_d", "dist",
    ]
    cols = [c for c in NEEDED if c in all_cols]
    print(f"\nReading {len(cols)} of {len(all_cols)} columns for speed.", flush=True)

    # ---------- accumulators ----------
    total_rows = 0
    missing = Counter()

    id_like = [c for c in [
        "reporterCode", "partnerCode", "flowCode",
        "refYear", "classificationCode", "qtyUnitCode",
    ] if c in cols]
    uniques = {c: set() for c in id_like}

    flow_counter = Counter()
    agg_counter = Counter()

    numeric_cols = [c for c in [
        "qty", "netWgt", "grossWgt", "CIFValue", "FOBValue", "primaryValue",
        "pop_o", "gdp_o", "gdpcap_o", "pop_d", "gdp_d", "gdpcap_d", "dist",
    ] if c in cols]
    nstats = {c: {"count": 0, "sum": 0.0, "min": np.inf, "max": -np.inf}
              for c in numeric_cols}

    has_cmd = "cmdCode" in cols
    elec_prefixes = {
        "8517": "8517 telecom/phone apparatus",
        "8541": "8541 semiconductor devices",
        "8542": "8542 integrated circuits (chips)",
    }
    elec_counts = Counter()
    chapter85 = 0
    cmd_counter = Counter()

    # ---------- streaming pass with progress ----------
    print("\nStreaming pass (a dot = one chunk):", flush=True)
    reader = pd.read_csv(path, usecols=cols, chunksize=CHUNK,
                         dtype=str, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        total_rows += len(chunk)

        for c in cols:
            missing[c] += chunk[c].isna().sum()
        for c in id_like:
            uniques[c].update(chunk[c].dropna().unique().tolist())
        if "flowCode" in cols:
            flow_counter.update(chunk["flowCode"].dropna().tolist())
        if "isAggregate" in cols:
            agg_counter.update(chunk["isAggregate"].dropna().tolist())

        for c in numeric_cols:
            v = pd.to_numeric(chunk[c], errors="coerce").dropna()
            if len(v):
                s = nstats[c]
                s["count"] += len(v)
                s["sum"] += float(v.sum())
                s["min"] = min(s["min"], float(v.min()))
                s["max"] = max(s["max"], float(v.max()))

        if has_cmd:
            code = chunk["cmdCode"].dropna().astype(str).str.zfill(6)
            cmd_counter.update(code.tolist())
            chapter85 += int(code.str.startswith("85").sum())
            for p in elec_prefixes:
                elec_counts[p] += int(code.str.startswith(p).sum())

        # progress: a dot each chunk, a running count every 10 chunks
        if i % 10 == 0:
            print(f" [{h(total_rows)} rows]", flush=True)
        else:
            print(".", end="", flush=True)

    print()
    total_rows = max(total_rows, 1)

    # ---------- report ----------
    print(f"\n{'='*64}")
    print(f"TOTAL ROWS: {h(total_rows)}")
    print('='*64)

    print("\n--- Coverage (unique values in key ID columns) ---")
    for c in id_like:
        u = uniques[c]
        sample = ", ".join(sorted(map(str, u))[:8])
        more = " ..." if len(u) > 8 else ""
        print(f"  {c:20} {len(u):>6} unique   e.g. {sample}{more}")

    if "refYear" in uniques:
        yrs = sorted(int(y) for y in uniques["refYear"] if str(y).isdigit())
        if yrs:
            print(f"\n  Year span: {yrs[0]} - {yrs[-1]}")

    if flow_counter:
        print("\n  Flow breakdown (X=export, M=import):")
        for f, cnt in flow_counter.most_common():
            print(f"    {f}: {h(cnt)}  ({100*cnt/total_rows:.1f}%)")

    if agg_counter:
        print("\n  isAggregate breakdown:")
        for a, cnt in agg_counter.most_common():
            print(f"    {a}: {h(cnt)}  ({100*cnt/total_rows:.1f}%)")

    print("\n--- Missing values (key columns with any nulls) ---")
    any_missing = False
    for c in cols:
        if missing[c]:
            print(f"  {c:20} {h(missing[c]):>14}  ({100*missing[c]/total_rows:5.1f}%)")
            any_missing = True
    if not any_missing:
        print("  none")

    print("\n--- Numeric columns (min / mean / max / non-null count) ---")
    for c in numeric_cols:
        s = nstats[c]
        if s["count"]:
            mean = s["sum"] / s["count"]
            print(f"  {c:11} min={s['min']:>16,.2f}  mean={mean:>16,.2f}  "
                  f"max={s['max']:>18,.2f}  n={h(s['count'])}")
        else:
            print(f"  {c:11} (no numeric values parsed)")

    if has_cmd:
        print("\n--- Electronics check (is it chips-only?) ---")
        print(f"  Total rows:                        {h(total_rows)}")
        print(f"  Chapter 85 (electrical machinery): {h(chapter85)}  "
              f"({100*chapter85/total_rows:.1f}%)")
        for p, label in elec_prefixes.items():
            print(f"    {label:36} {h(elec_counts[p])}")
        target = sum(elec_counts.values())
        print(f"  Your 3 target headings combined:   {h(target)}  "
              f"({100*target/total_rows:.1f}%)")
        if target == 0:
            print("  -> NOT pre-filtered. Filtering cmdCode is step one.")
        elif target >= total_rows * 0.99:
            print("  -> Already restricted to your target headings.")
        else:
            print("  -> Mixed: targets + other products. Filter cmdCode before training.")

        print("\n  Top 12 products overall (cmdCode -> row count):")
        for c, cnt in cmd_counter.most_common(12):
            print(f"    {c}  {h(cnt)}")

    print("\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 profile_comtrade.py <file.csv|file.csv.gz>")
    main(sys.argv[1])