"""Shared Comtrade country reference.
   Usage:  from country_codes import cname, drop_blocs, CODE2ISO3, ISO3_2_M49
"""
import json, os, requests

REPORTERS_URL = "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"
LOCAL = "comtrade_codes.json"

def _load():
    if os.path.exists(LOCAL):
        return json.load(open(LOCAL))
    data = requests.get(REPORTERS_URL, timeout=30).json()["results"]
    json.dump(data, open(LOCAL, "w"))
    return data

_data = _load()
CODE2NAME  = {str(r["reporterCode"]): r["reporterDesc"] for r in _data}
CODE2ISO3  = {str(r["reporterCode"]): (r.get("reporterCodeIsoAlpha3") or "").strip() for r in _data}
GROUP_CODES = {str(r["reporterCode"]) for r in _data if r.get("isGroup")}
DROP_CODES  = GROUP_CODES - {"490"}                      # keep 490 = Taiwan proxy
ISO3_2_M49  = {v: k for k, v in CODE2ISO3.items() if v}  # bridge for the GDELT join

def cname(code):
    return CODE2NAME.get(str(code), f"[{code}]")

def drop_blocs(df, cols=("reporterCode", "partnerCode")):
    before = len(df)
    for c in cols:
        if c in df.columns:
            df = df[~df[c].astype(str).isin(DROP_CODES)].copy()
    print(f"drop_blocs: removed {before - len(df):,} bloc rows (ASEAN, EU, …)")
    return df