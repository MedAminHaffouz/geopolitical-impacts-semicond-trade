#!/usr/bin/env bash
# ==========================================================================
# reorganize_repo.sh
# ==========================================================================
# Reorganizes electronic_prods_pred/ into a clean, modular structure.
# Run this FROM INSIDE electronic_prods_pred/.
#
# Safe by design:
#   - Uses `git mv` if you're in a git repo (preserves history), falls back
#     to plain `mv` otherwise.
#   - Skips (with a warning) any file that isn't found, rather than failing.
#   - Skips (with a warning) if the destination already exists, rather than
#     overwriting anything.
#   - Idempotent: safe to run more than once.
#
# It does NOT touch a short list of ambiguous/likely-stale files -- those
# are printed in a "NEEDS YOUR REVIEW" section at the end instead of being
# moved automatically. See that section before deleting anything.
#
# Usage:
#   chmod +x reorganize_repo.sh
#   ./reorganize_repo.sh
# ==========================================================================
set -uo pipefail

IS_GIT=0
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  IS_GIT=1
  echo "Detected git repo -- using 'git mv' (preserves history)."
else
  echo "Not a git repo -- using plain 'mv'. (Consider 'git init' first if you want history preserved.)"
fi

MOVED=0
SKIPPED_MISSING=0
SKIPPED_EXISTS=0

move() {
  local src="$1" dst="$2"
  if [ ! -e "$src" ]; then
    echo "  [skip: not found]   $src"
    SKIPPED_MISSING=$((SKIPPED_MISSING+1))
    return
  fi
  if [ -e "$dst" ]; then
    echo "  [skip: dest exists] $src -> $dst"
    SKIPPED_EXISTS=$((SKIPPED_EXISTS+1))
    return
  fi
  mkdir -p "$(dirname "$dst")"
  if [ "$IS_GIT" = "1" ]; then
    git mv "$src" "$dst" 2>/dev/null || mv "$src" "$dst"
  else
    mv "$src" "$dst"
  fi
  echo "  [moved]             $src -> $dst"
  MOVED=$((MOVED+1))
}

echo ""
echo "=== Creating directory skeleton ==="
mkdir -p data/raw data/interim data/processed data/external data/cache
mkdir -p notebooks/01_data_extraction notebooks/02_benchmarking notebooks/03_analysis notebooks/04_writeup
mkdir -p src/training src/retraining src/analysis src/visualization src/utils
mkdir -p results/grids results/feature_importance results/comparisons
mkdir -p models figures docs scripts

echo ""
echo "=== data/raw ==="
move "comtradeExports_updatedH5[240924]-wb.csv.gz" "data/raw/comtradeExports_updatedH5[240924]-wb.csv.gz"

echo ""
echo "=== data/processed (ready-to-train parquet files) ==="
move "all_products.parquet"          "data/processed/all_products.parquet"
move "all_products_ready.parquet"    "data/processed/all_products_ready.parquet"
move "chips.parquet"                 "data/processed/chips.parquet"
move "chip_trade_2017_2023.parquet"  "data/processed/chip_trade_2017_2023.parquet"
move "medical.parquet"               "data/processed/medical.parquet"
move "vehicles.parquet"              "data/processed/vehicles.parquet"
move "oil_fuel.parquet"              "data/processed/oil_fuel.parquet"
move "electronics.parquet"           "data/processed/electronics.parquet"
move "test_events.parquet"           "data/processed/test_events.parquet"

echo ""
echo "=== data/interim (GDELT intermediate files) ==="
move "gdelt_bilateral_by_pair_year.parquet"    "data/interim/gdelt_bilateral_by_pair_year.parquet"
move "gdelt_bilateral_topk.parquet"            "data/interim/gdelt_bilateral_topk.parquet"
move "gdelt_chip_events_2017_2023.parquet"     "data/interim/gdelt_chip_events_2017_2023.parquet"
move "gdelt_events_2017_2023.parquet"          "data/interim/gdelt_events_2017_2023.parquet"
move "gdelt_features_by_country_year.parquet"  "data/interim/gdelt_features_by_country_year.parquet"
move "gdelt_features_topk.parquet"             "data/interim/gdelt_features_topk.parquet"
move "gdelt_semiconductor_events.parquet"      "data/interim/gdelt_semiconductor_events.parquet"
move "comtrade_codes.json"                     "data/interim/comtrade_codes.json"
move "country_codes.py"                        "src/utils/country_codes.py"
move "gdelt_chunks"        "data/cache/gdelt_chunks"
move "gdelt_chunks_full"   "data/cache/gdelt_chunks_full"
move "gdelt_chunks_global" "data/cache/gdelt_chunks_global"
move "gdelt_chunks_raw"    "data/cache/gdelt_chunks_raw"

echo ""
echo "=== data/external (3rd-party reference data) ==="
move "monthly_semiconductor_supply_chain_risk.csv" "data/external/monthly_semiconductor_supply_chain_risk.csv"
move "ukraine_russia_critical_events.parquet"       "data/external/ukraine_russia_critical_events.parquet"

echo ""
echo "=== data/cache (regenerable, gitignored) ==="
move "split_cache" "data/cache/split_cache"

echo ""
echo "=== notebooks/01_data_extraction ==="
move "GDELT_Extraction.ipynb"               "notebooks/01_data_extraction/GDELT_Extraction.ipynb"
move "GDELT_Extraction_all_countries.ipynb" "notebooks/01_data_extraction/GDELT_Extraction_all_countries.ipynb"
move "GDELT_Extraction_full.ipynb"          "notebooks/01_data_extraction/GDELT_Extraction_full.ipynb"
move "gdelt_parqs_reader.ipynb"             "notebooks/01_data_extraction/gdelt_parqs_reader.ipynb"
move "gdelt_features_clean.ipynb"           "notebooks/01_data_extraction/gdelt_features_clean.ipynb"
move "feature_engineering.ipynb"            "notebooks/01_data_extraction/feature_engineering.ipynb"

echo ""
echo "=== notebooks/02_benchmarking ==="
move "benchmark_all.ipynb"    "notebooks/02_benchmarking/benchmark_all.ipynb"
move "benchmark_drop.ipynb"   "notebooks/02_benchmarking/benchmark_drop.ipynb"
move "benchmark_median.ipynb" "notebooks/02_benchmarking/benchmark_median.ipynb"
move "benchmark_models.ipynb" "notebooks/02_benchmarking/benchmark_models.ipynb"
move "benchmark_rf.ipynb"     "notebooks/02_benchmarking/benchmark_rf.ipynb"
move "benchmark_v2.ipynb"     "notebooks/02_benchmarking/benchmark_v2.ipynb"
move "benchmark_v2.txt"       "docs/benchmark_v2.txt"

echo ""
echo "=== notebooks/03_analysis ==="
move "compare_all_experiments.ipynb"    "notebooks/03_analysis/compare_all_experiments.ipynb"
move "compare_all_experiments_v2.ipynb" "notebooks/03_analysis/compare_all_experiments_v2.ipynb"
move "results_viz.ipynb"                "notebooks/03_analysis/results_viz.ipynb"
move "results_viz_full.ipynb"           "notebooks/03_analysis/results_viz_full.ipynb"
move "viz.ipynb"                        "notebooks/03_analysis/viz.ipynb"

echo ""
echo "=== notebooks/04_writeup ==="
move "Final_GAT_export-V3_revision-2017-2022.ipynb" "notebooks/04_writeup/Final_GAT_export-V3_revision-2017-2022.ipynb"
move "recap_writeup.ipynb"                          "notebooks/04_writeup/recap_writeup.ipynb"

echo ""
echo "=== src/training ==="
move "train_benchmark.py"          "src/training/train_benchmark.py"
move "run_shrinkage_head.py"       "src/training/run_shrinkage_head.py"
move "run_per_product_head.py"     "src/training/run_per_product_head.py"
move "run_category_experiments.py" "src/training/run_category_experiments.py"

echo ""
echo "=== src/retraining ==="
move "retrain_pruned_edgegat.py"     "src/retraining/retrain_pruned_edgegat.py"
move "retrain_reseed_diagnostic.py"  "src/retraining/retrain_reseed_diagnostic.py"
move "retrain_simplified_edgegat.py" "src/retraining/retrain_simplified_edgegat.py"

echo ""
echo "=== src/analysis ==="
move "feature_importance_shrinkage.py"   "src/analysis/feature_importance_shrinkage.py"
move "analyze_feature_importance.py"     "src/analysis/analyze_feature_importance.py"
move "analyze_dist_placement.py"         "src/analysis/analyze_dist_placement.py"
move "compare_categories.py"             "src/analysis/compare_categories.py"
move "compare_perproduct_vs_pooled.py"   "src/analysis/compare_perproduct_vs_pooled.py"

echo ""
echo "=== src/visualization ==="
move "generate_dashboard.py" "src/visualization/generate_dashboard.py"

echo ""
echo "=== src/utils ==="
move "benchmark_prep.py"    "src/utils/benchmark_prep.py"
move "profile_comtrade.py"  "src/utils/profile_comtrade.py"
move "benchmark.py"         "src/utils/benchmark.py"

echo ""
echo "=== results/grids (benchmark grid outputs) ==="
move "results_v2_all.csv"             "results/grids/results_v2_all.csv"
move "results_v2_chips.csv"           "results/grids/results_v2_chips.csv"
move "results_v2_chips_reproduced.csv" "results/grids/results_v2_chips_reproduced.csv"
move "results_v2_electronics.csv"     "results/grids/results_v2_electronics.csv"
move "results_v2_electro_v2.csv"      "results/grids/results_v2_electro_v2.csv"
move "results_v2_medical.csv"         "results/grids/results_v2_medical.csv"
move "results_v2_oil.csv"             "results/grids/results_v2_oil.csv"
move "results_v2_vehicles.csv"        "results/grids/results_v2_vehicles.csv"
move "results_electronics.csv"        "results/grids/results_electronics.csv"
move "results_shrinkage_head.csv"     "results/grids/results_shrinkage_head.csv"
move "results_perproduct_head.csv"    "results/grids/results_perproduct_head.csv"
move "results_pruned_edgegat.csv"     "results/grids/results_pruned_edgegat.csv"

echo ""
echo "=== results/feature_importance ==="
move "feature_importance_all.csv"             "results/feature_importance/feature_importance_all.csv"
move "feature_importance_chips.csv"           "results/feature_importance/feature_importance_chips.csv"
move "feature_importance_electronics.csv"     "results/feature_importance/feature_importance_electronics.csv"
move "feature_importance_shrinkage.csv"       "results/feature_importance/feature_importance_shrinkage.csv"
move "feature_importance_shrinkage_summary.csv" "results/feature_importance/feature_importance_shrinkage_summary.csv"

echo ""
echo "=== results/comparisons ==="
move "perproduct_vs_pooled.csv" "results/comparisons/perproduct_vs_pooled.csv"

echo ""
echo "=== models/ (gitignored -- large binaries) ==="
move "model_gat_2017_2022_chips_gdelt.pth" "models/model_gat_2017_2022_chips_gdelt.pth"
move "model_gat_2017_2022_chips.pth"       "models/model_gat_2017_2022_chips.pth"
move "model_gat_2017_2022_electro.pth"     "models/model_gat_2017_2022_electro.pth"
move "model_gat_2017_2022.pth"             "models/model_gat_2017_2022.pth"
move "model_gcn_2017_2022_chips.pth"       "models/model_gcn_2017_2022_chips.pth"
move "trained_models"                      "models/trained_models"
move "trained_models_experiments"          "models/trained_models_experiments"
move "trained_models_perproduct_heading"   "models/trained_models_perproduct_heading"
move "trained_models_pruned_edgegat"       "models/trained_models_pruned_edgegat"
move "trained_models_shrinkage_head"       "models/trained_models_shrinkage_head"

echo ""
echo "=== figures/ ==="
move "plots"                          "figures/plots"
move "dashboard.png"                  "figures/dashboard.png"
move "gdelt_insights_dashboard.png"   "figures/gdelt_insights_dashboard.png"
move "training_loss_2017_2022.png"    "figures/training_loss_2017_2022.png"

echo ""
echo "=== docs/ ==="
move "ISA.2013.GDELT.pdf"    "docs/ISA.2013.GDELT.pdf"
move "embedding_merge.pdf"   "docs/embedding_merge.pdf"

echo ""
echo "=== scripts/ ==="
move "setup_venv.sh" "scripts/setup_venv.sh"

echo ""
echo "=========================================================================="
echo "DONE.  moved=$MOVED  skipped(missing)=$SKIPPED_MISSING  skipped(dest exists)=$SKIPPED_EXISTS"
echo "=========================================================================="
echo ""
echo "NOT MOVED -- needs your review, since these are ambiguous or look stale:"
echo "  - dashoard1.png              (typo of dashboard.png -- duplicate? delete or rename+move)"
echo "  - results_v2_old.csv         (superseded by results_v2_all.csv? confirm before deleting)"
echo "  - results_old_mixed.csv      (unclear provenance -- check before deleting)"
echo "  - results_v2_no_rf.csv       (unclear provenance -- check before deleting)"
echo "  - country_test.py            (looks like a scratch/test file, not part of the pipeline)"
echo "  - test_mod'.py               (stray filename with a literal quote in it -- likely a typo/accident)"
echo "  - gat_original                (unclear contents -- inspect before deciding where it goes)"
echo "  - venv_train/                 (a Python virtualenv -- should NOT live inside the repo at all;"
echo "                                  left in place, added to .gitignore, but consider moving it"
echo "                                  outside the project directory entirely)"
echo "  - __pycache__/                (left in place, added to .gitignore -- safe to delete anytime)"
echo ""
echo "Also worth a manual pass: you now have THREE electronics result files"
echo "(results_v2_electronics.csv, results_v2_electro_v2.csv, results_electronics.csv)"
echo "in results/grids/ -- confirm which is authoritative and archive/delete the rest"
echo "before this becomes a source of confusion in the writeup."
