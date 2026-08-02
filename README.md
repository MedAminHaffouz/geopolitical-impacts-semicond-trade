# Semiconductor Trade-Flow Prediction — GDELT Geopolitical Risk Integration

Predicting semiconductor/electronics trade-flow value by combining UN Comtrade
trade data with GDELT geopolitical event data, using graph neural networks
(GAT / EdgeGAT) with a hierarchical per-product-head architecture.

**Core research question:** does integrating GDELT geopolitical risk add
predictive value to trade-flow prediction, and does giving the model
per-product specialization (without training N separate models) outperform
both a fully pooled model and dedicated per-category models?

## Repository structure

```
data/
    raw/            Untouched original downloads (not tracked in git)
    interim/        GDELT intermediate extraction outputs (not tracked)
    processed/      Ready-to-train parquet files, per product category (not tracked)
    external/       Third-party reference data (not tracked)
    cache/          split_cache/, gdelt_chunks*/ -- regenerate anytime, safe to delete

notebooks/
    00_init/              Initial setup
    01_data_extraction/   GDELT extraction, feature engineering
    02_benchmarking/      Initial model grid experiments
    03_analysis/          Cross-experiment comparison notebooks

src/
    training/       Main training drivers (train_benchmark.py, run_shrinkage_head.py, ...)
    retraining/     Targeted retrains -- pruned features, reseed diagnostics, leave-one-out
    analysis/       Feature importance, category comparisons, metric recomputation
    evaluation/     Post-hoc evaluation on trained models -- bilateral corridor checks,
                    out-of-sample windowed tests, ad-hoc diagnostics (no training here)
    visualization/  Dashboard generation
    utils/          Shared helpers (country codes, data prep)

results/
    grids/                Full benchmark grid outputs, one CSV per run/category
    feature_importance/   Permutation importance results
    comparisons/          Head-to-head comparison outputs
    evaluation/           Root-level metric CSVs pulled out of the repo root
                           (bilateral_test_metrics.csv, windowed_results.csv)
    corridor_timeseries/  Actual-vs-predicted plots + CSV for the two bilateral
                           corridor case studies (US->China, China->Taiwan)

models/         Trained model weights (not tracked in git -- see Models section)
figures/        Generated plots and dashboards
docs/           Reference files
scripts/        Environment setup + one-off migration scripts (scripts/migrations/)
archive/        Superseded result bundles kept for reference only
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to use this repo

### 1. Reproduce the headline model (shrinkage head, all products)

```bash
python src/training/run_shrinkage_head.py [all_products_ready.parquet]
```

Trains the 19-model grid with `PartialPoolingHead` (the architecture the paper
and internship report both report on). Writes weights to
`models/trained_models_shrinkage_head/` and metrics to
`results/grids/results_shrinkage_head.csv` / `results_shrinkage_full_metrics.csv`.

### 2. Recompute metrics without retraining

```bash
python src/analysis/recompute_full_metrics.py [all_products_ready.parquet]
```

Reloads existing checkpoints and recomputes `log_r2` / Spearman / secondary
raw-scale metrics. Use this instead of retraining whenever only the metrics
definition changed, not the model.

### 3. Feature importance (permutation + retraining cross-check)

```bash
python src/analysis/feature_importance_shrinkage.py [all_products_ready.parquet]
# -> results/feature_importance/feature_importance_shrinkage.csv

python src/retraining/retrain_leave_one_out.py [all_products_ready.parquet]
# -> results/grids/results_leave_one_out.csv (actual retrain-without-feature results)
```

Always cross-check the two: permutation importance and leave-one-out retraining
disagree on several GDELT features (see Key findings #4) — the permutation
number alone is not sufficient evidence a feature is safe to drop.

### 4. Targeted retraining experiments

```bash
python src/retraining/retrain_gravity_edge.py [all_products_ready.parquet]       # gravity feature -> edge placement
python src/retraining/retrain_pruned_edgegat.py [all_products_ready.parquet] [feature_importance_shrinkage.csv]
python src/retraining/retrain_simplified_edgegat.py [all_products_ready.parquet]  # partial-pair experiment
```

Each writes to its own `models/trained_models_<experiment>/` and
`results/grids/results_<experiment>.csv` — never overwrites the shrinkage-head
baseline.

### 5. Bilateral corridor / out-of-sample evaluation

```bash
cd src/evaluation
python test_bilateral_pairs.py [split_cache_dir] [trained_models_dir] [interim_dir]
python corridor_timeseries.py  [split_cache_dir] [trained_models_dir] [interim_dir] [processed_dir]
python train_windowed.py       [split_cache_dir] [interim_dir] [processed_dir] [out_models_dir]
```

Run from `src/evaluation/` (or repo root, adjusting the relative defaults) —
`corridor_timeseries.py` imports `test_bilateral_pairs.py` directly, so keep
them in the same directory. These use already-trained checkpoints; nothing
here retrains a model. Output lands in `results/corridor_timeseries/` and
`results/evaluation/`.

### 6. Cross-experiment comparison notebooks

Open `notebooks/03_analysis/compare_all_experiments_v3.ipynb` or
`feature_importance_analysis.ipynb` for the plots that feed the paper figures
(`figures/best_of/`, `figures/writeup/`).

## Guide to `results/` CSVs

| File | What it is |
|---|---|
| `results/grids/results_shrinkage_head.csv`, `results_shrinkage_full_metrics.csv` | Headline 19-model grid, shrinkage head, primary metrics |
| `results/grids/results_leave_one_out*.csv` | Actual retrain-without-feature results (ground truth for feature necessity) |
| `results/grids/results_gravity_edge*.csv` | Gravity-as-edge-feature experiment (failed universally, 0/12 configs) |
| `results/grids/results_partial_pair_edgegat*.csv` | Partial-pair structural experiment (5/12 held, 4/12 improved) |
| `results/grids/results_pruned_edgegat*.csv` | Feature-pruned EdgeGAT retrains, by importance threshold |
| `results/grids/results_diagnostics_*.csv` | Bootstrap CI / Wilcoxon / timing diagnostics on top result |
| `results/feature_importance/feature_importance_shrinkage.csv` | Permutation importance, per feature, shrinkage-head models |
| `results/feature_importance/rf_vs_permutation_importance.csv` | RF cross-check against GNN permutation importance |
| `results/comparisons/perproduct_vs_pooled.csv` | Per-product head vs. classical pooling head-to-head |
| `results/evaluation/bilateral_test_metrics.csv` | Per-corridor test metrics for the two case-study corridors |
| `results/evaluation/windowed_results.csv` | Rolling-window out-of-sample results (`train_windowed.py`) |
| `results/corridor_timeseries/corridor_timeseries.csv` + PNGs | Actual-vs-predicted trade value by year, US→China and China→Taiwan, 2015–2025 (train years vs. genuine out-of-sample marked explicitly) |

`results/archive/` holds superseded runs (old mixed-category results, pre-v2
electronics) — kept for traceability, not for citing in the paper or report.

## Guide to `models/`

Trained model weights are **not** tracked in git (too large). Directory name
tells you the experiment; filename encodes architecture + fusion mode +
feature-placement combo (`ka`/`tk` = keepall/topk on origin/destination side):

- `models/trained_models/` — original 19-model benchmark grid (no shrinkage head)
- `models/trained_models_shrinkage_head/` — **headline architecture**, hierarchical per-product head
- `models/trained_models_leave_one_out/` — one checkpoint per model × per feature removed (`_minus_<feature>` suffix)
- `models/trained_models_pruned_edgegat/` — feature-pruned, suffixed by importance threshold (`_th0p005` etc.)
- `models/trained_models_partial_pair_edgegat/` — partial-pair structural experiment (`_simplified`)
- `models/trained_models_gravity_edge/` — gravity-on-edge experiment (`_gravity`)
- `models/trained_models_perproduct_heading/`, `models/trained_models_experiments/` — earlier iterations, superseded by the shrinkage head

To load a checkpoint, match the filename against the model class + `build_agg`
config in `src/evaluation/test_bilateral_pairs.py` — it has the canonical
loading logic every other evaluation script reuses.

## Key findings (summary — see `notebooks/03_analysis/` for full detail)

1. GDELT geopolitical risk adds real predictive value: grouped block
   permutation of the GDELT feature set shows a 41.8% lift over the gravity-model
   baseline. Individual GDELT features look weak in isolation — the signal is
   in the block, not any single column.
2. A pooled model with a hierarchical per-product head (`src/training/run_shrinkage_head.py`,
   `PartialPoolingHead`, `alpha = n/(n+K)`, K=50) matches or beats both a fully
   classical pooled model and dedicated per-category models, without the cost
   of training N separate networks. `heading_idx` is the single most important
   feature globally (mean importance ~0.34).
3. Bilateral (pair-level) GDELT features (`pair_*`) are consistently near-dead;
   GDELT's value is concentrated in node-level country signals, not the
   bilateral pair.
4. **Permutation importance and leave-one-out retraining disagree**: several
   features flagged as safe to drop by permutation consistently hurt
   performance when actually removed and retrained. Treated as a genuine
   methodological finding, not a bug — always check `results_leave_one_out.csv`
   before trusting a permutation-importance number.
5. Random Forest's strong raw performance reflects entity memorization on
   `heading_idx`/country codes, not learned generalization from the GDELT/gravity
   features — see `rf_feature_crosscheck.py`.

## Data sources

- UN Comtrade (trade flows), annual, full product universe, HS 8541–8542 as evaluation slice
- GDELT 2.0 (geopolitical events), streamed from the 67.5M-row all-country events file
- CEPII gravity features (distance, GDP, population)