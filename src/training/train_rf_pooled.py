#!/usr/bin/env python3
"""
train_rf_pooled.py
=====================
Trains the 3 RF configs (none/keepall/topk) on all_products_ready.parquet --
the SAME dataset your shrinkage models used, which is why these numbers are
comparable in rf_feature_crosscheck.py. NOT chips.parquet -- see the
explanation in the previous message for why that would be an apples-to-
oranges comparison.

Standalone on purpose: train_benchmark.py has a module-level
`load_and_split('all_products_ready.parquet')` call that fires the instant
you import anything from that file -- importing it here would trigger that
expensive, unwanted load. This script copies just the pieces it needs
instead (build_agg, handle_missing, run_rf logic), identical to your
existing train_benchmark.py so the resulting RF models are equivalent.

Saves into trained_models/ using the same naming convention as your GAT
models already there (RF_first_none.pkl, RF_first_keepall.pkl,
RF_first_topk.pkl), so rf_feature_crosscheck.py picks them up with zero
changes needed on its end.

Usage:
    python train_rf_pooled.py [all_products_ready.parquet]
"""
import os, sys, pickle, gc
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}", flush=True)

GDELT_COLS = ['events_total','score_mean','score_max','score_vol','goldstein_wmean','tone_mean','active_months']
BASE_COLS  = ['refYear','cmdCode','dist','gdpcap_d','gdpcap_o','pop_o','pop_d']
MISSING = 'rf'
KEEPALL='gdelt_features_by_country_year.parquet'; TOPK='gdelt_features_topk.parquet'
CONDITIONS = [('none', False, None), ('keepall', True, KEEPALL), ('topk', True, TOPK)]
MODEL_DIR = 'trained_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def add_gdelt(agg, gdelt_file):
    cy=pd.read_parquet(gdelt_file)
    out=agg.copy()
    out['_r']=out['reporterCode'].astype('Int64').astype(str); out['_y']=out['refYear'].astype('Int64').astype(str)
    cy['_r']=cy['reporterCode'].astype('Int64').astype(str);  cy['_y']=cy['year'].astype('Int64').astype(str)
    out=out.merge(cy[['_r','_y']+GDELT_COLS], on=['_r','_y'], how='left')
    out[GDELT_COLS]=out[GDELT_COLS].fillna(0)
    return out.drop(columns=['_r','_y'])

def build_agg(df, use_gdelt, gdelt_file, agg_mode='first'):
    grav = 'first' if agg_mode=='first' else 'sum'
    agg = (df.groupby(['refYear','reporterCode','cmdCode'])
             .agg(primaryValue=('primaryValue','mean'), dist=('dist','first'),
                  gdpcap_d=('gdpcap_d',grav), gdpcap_o=('gdpcap_o',grav),
                  pop_o=('pop_o',grav), pop_d=('pop_d',grav)).reset_index())
    agg['y_log'] = np.log1p(agg['primaryValue'].clip(lower=0))
    if use_gdelt:
        agg = add_gdelt(agg, gdelt_file)
    return agg

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

def _smape(yt, yp):
    d=(np.abs(yt)+np.abs(yp))/2; m=d>0
    return np.mean(np.abs(yt[m]-yp[m])/d[m])*100 if m.sum() else np.nan

def _spearman(a,b):
    if len(a)<3: return np.nan
    return pd.Series(a).corr(pd.Series(b), method='spearman')

def compute_metrics(yt, yp):
    ytl=np.log1p(np.clip(yt,0,None)); ypl=np.log1p(np.clip(yp,0,None))
    return dict(mae=mean_absolute_error(yt,yp), rmse=mean_squared_error(yt,yp)**0.5,
                r2=r2_score(yt,yp), log_r2=(r2_score(ytl,ypl) if len(yt)>2 else np.nan),
                spearman=_spearman(yt,yp), smape=_smape(yt,yp))

def run_rf(train_data, data_2023, use_gdelt, gdelt_file, agg_mode='first'):
    feat_cols = BASE_COLS + (GDELT_COLS if use_gdelt else [])
    tr = build_agg(train_data, use_gdelt, gdelt_file, agg_mode).dropna(subset=feat_cols+['primaryValue'])
    te = build_agg(data_2023,  use_gdelt, gdelt_file, agg_mode).dropna(subset=feat_cols+['primaryValue'])
    sf, st = MinMaxScaler(), MinMaxScaler()
    Xtr = sf.fit_transform(tr[feat_cols]); Xte = sf.transform(te[feat_cols])
    ytr = st.fit_transform(tr[['y_log']]); yte = st.transform(te[['y_log']])
    rf = RandomForestRegressor(n_estimators=200, max_depth=25, n_jobs=2, random_state=0).fit(Xtr, ytr.ravel())
    yp = np.expm1(st.inverse_transform(rf.predict(Xte).reshape(-1,1)).flatten())
    yt = np.expm1(st.inverse_transform(yte).flatten())
    metrics = compute_metrics(yt, yp)
    gc.collect()
    return rf, metrics

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'all_products_ready.parquet'
    log(f"training RF on {data_file} (pooled, all products -- matches the shrinkage models' dataset)")
    train_data, val_data, data_2023 = load_or_split(data_file)

    for scoring, use_gdelt, gdelt_file in CONDITIONS:
        name = f'RF_first_{scoring}'
        model_path = os.path.join(MODEL_DIR, f'{name}.pkl')
        if os.path.exists(model_path):
            log(f"skip {name} (already trained -- delete {model_path} to force retrain)")
            continue
        log(f"training {name} ...")
        rf, metrics = run_rf(train_data, data_2023, use_gdelt, gdelt_file, agg_mode='first')
        with open(model_path, 'wb') as f:
            pickle.dump(rf, f)
        log(f"  {name}: log_r2={metrics['log_r2']:.3f}  spearman={metrics['spearman']:.3f}  "
            f"saved -> {model_path}")

    log("done -- rf_feature_crosscheck.py trained_models will now find these 3 automatically.")


if __name__ == "__main__":
    main()
