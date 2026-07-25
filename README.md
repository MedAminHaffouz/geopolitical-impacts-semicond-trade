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
    cache/          split_cache/ etc. -- regenerate anytime, safe to delete

notebooks/
    00_init/             Initial setup
    01_data_extraction/   GDELT extraction, feature engineering
    02_benchmarking/      Initial model grid experiments
    03_analysis/          Cross-experiment comparison notebooks

src/
    training/       Main training drivers (train_benchmark.py, run_shrinkage_head.py, ...)
    retraining/     Targeted retrains -- pruned features, reseed diagnostics
    analysis/       Feature importance, category comparisons
    visualization/  Dashboard generation
    utils/          Shared helpers (country codes, data prep)

results/
    grids/                Full benchmark grid outputs, one CSV per run/category
    feature_importance/   Permutation importance results
    comparisons/           Head-to-head comparison outputs

models/         Trained model weights (not tracked in git -- see Models section)
figures/        Generated plots and dashboards
docs/           Reference files
scripts/        Environment setup
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Models

Trained model weights are **not** tracked in git (too large). They're
organized locally under `models/`:
- `models/trained_models/` — main benchmark grid
- `models/trained_models_shrinkage_head/` — hierarchical per-product-head models (headline architecture)
- `models/trained_models_pruned_edgegat/` — feature-pruned EdgeGAT_full retrains
- `models/trained_models_perproduct_heading/`, `models/trained_models_experiments/` — earlier iterations

## Key findings (summary — see `notebooks/03_analysis/` for full detail)

1. GDELT geopolitical risk adds small, consistent predictive value across every product category tested.
2. A pooled model with a hierarchical per-product head (`src/training/run_shrinkage_head.py`)
   matches or beats both a fully classical pooled model and dedicated per-category models,
   without the cost of training N separate networks.
3. Bilateral (pair-level) GDELT features consistently show negligible importance;
   node-level GDELT features and gravity variables (`gdpcap_o`, `pop_o`) dominate.

## Data sources

- UN Comtrade (trade flows)
- GDELT 2.0 (geopolitical events)
