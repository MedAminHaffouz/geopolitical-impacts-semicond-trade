#!/usr/bin/env bash
# ==========================================================================
# reorganize_repo_update.sh
# ==========================================================================
# Supplementary migration for everything created SINCE the original
# reorganize_repo.sh -- all the retraining scripts, feature-importance work,
# new notebooks, and the results/models they produced.
#
# Safe to run regardless of whether you ran the original script:
#   - git mv if in a git repo, plain mv otherwise
#   - skips (with a warning) anything not found, rather than failing
#   - skips (with a warning) if the destination already exists
#   - idempotent -- safe to run more than once
#
# Usage:
#   chmod +x reorganize_repo_update.sh
#   ./reorganize_repo_update.sh
# ==========================================================================
set -uo pipefail

IS_GIT=0
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  IS_GIT=1
  echo "Detected git repo -- using 'git mv' (preserves history)."
else
  echo "Not a git repo -- using plain 'mv'."
fi

MOVED=0; SKIPPED_MISSING=0; SKIPPED_EXISTS=0

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
echo "=== src/analysis (feature-importance + comparison scripts) ==="
move "feature_importance_shrinkage.py"   "src/analysis/feature_importance_shrinkage.py"
move "rf_feature_crosscheck.py"          "src/analysis/rf_feature_crosscheck.py"
move "compare_shrinkage_results.py"      "src/analysis/compare_shrinkage_results.py"
move "analyze_dist_placement.py"         "src/analysis/analyze_dist_placement.py"
move "analyze_feature_importance.py"     "src/analysis/analyze_feature_importance.py"

echo ""
echo "=== src/training ==="
move "train_rf_pooled.py" "src/training/train_rf_pooled.py"

echo ""
echo "=== src/retraining (every removal/reseed experiment) ==="
move "retrain_reseed_diagnostic.py"    "src/retraining/retrain_reseed_diagnostic.py"
move "retrain_simplified_edgegat.py"   "src/retraining/retrain_simplified_edgegat.py"
move "retrain_pruned_edgegat.py"       "src/retraining/retrain_pruned_edgegat.py"
move "retrain_leave_one_out.py"        "src/retraining/retrain_leave_one_out.py"
move "retrain_gravity_edge.py"         "src/retraining/retrain_gravity_edge.py"

echo ""
echo "=== src/utils (one-off comparison-CSV patch scripts) ==="
move "fix_gravity_edge_comparison.py" "src/utils/fix_gravity_edge_comparison.py"
move "fix_simplified_comparison.py"   "src/utils/fix_simplified_comparison.py"

echo ""
echo "=== notebooks/03_analysis ==="
move "compare_all_experiments.ipynb"    "notebooks/03_analysis/compare_all_experiments.ipynb"
move "compare_all_experiments_v2.ipynb" "notebooks/03_analysis/compare_all_experiments_v2.ipynb"
move "feature_importance_analysis.ipynb" "notebooks/03_analysis/feature_importance_analysis.ipynb"

echo ""
echo "=== results/grids (new benchmark grid outputs) ==="
move "results_shrinkage_head.csv"          "results/grids/results_shrinkage_head.csv"
move "results_pruned_edgegat.csv"          "results/grids/results_pruned_edgegat.csv"
move "results_pruned_edgegat_sweep.csv"    "results/grids/results_pruned_edgegat_sweep.csv"
move "results_leave_one_out.csv"           "results/grids/results_leave_one_out.csv"
move "results_simplified_edgegat.csv"      "results/grids/results_simplified_edgegat.csv"
move "results_partial_pair_edgegat.csv"    "results/grids/results_partial_pair_edgegat.csv"
move "results_gravity_edge.csv"            "results/grids/results_gravity_edge.csv"

echo ""
echo "=== results/feature_importance ==="
move "feature_importance_shrinkage.csv"         "results/feature_importance/feature_importance_shrinkage.csv"
move "feature_importance_shrinkage_summary.csv" "results/feature_importance/feature_importance_shrinkage_summary.csv"
move "rf_feature_importance.csv"                "results/feature_importance/rf_feature_importance.csv"
move "rf_vs_permutation_importance.csv"         "results/feature_importance/rf_vs_permutation_importance.csv"

echo ""
echo "=== results/comparisons ==="
move "shrinkage_comparison.csv" "results/comparisons/shrinkage_comparison.csv"

echo ""
echo "=== models/ (gitignored -- large binaries, new trained-model directories) ==="
move "trained_models_shrinkage_head"        "models/trained_models_shrinkage_head"
move "trained_models_reseed_diagnostic"     "models/trained_models_reseed_diagnostic"
move "trained_models_simplified_edgegat"    "models/trained_models_simplified_edgegat"
move "trained_models_partial_pair_edgegat"  "models/trained_models_partial_pair_edgegat"
move "trained_models_pruned_edgegat"        "models/trained_models_pruned_edgegat"
move "trained_models_gravity_edge"          "models/trained_models_gravity_edge"
move "trained_models_leave_one_out"         "models/trained_models_leave_one_out"

echo ""
echo "=========================================================================="
echo "DONE.  moved=$MOVED  skipped(missing)=$SKIPPED_MISSING  skipped(dest exists)=$SKIPPED_EXISTS"
echo "=========================================================================="
echo ""
echo "NOT MOVED -- needs your review:"
echo "  - results_v2_chips_reproduced.csv and any other results_v2_* files you generated"
echo "    since the last reorg pass -- these follow the SAME pattern as before"
echo "    (results/grids/) but weren't explicitly listed here since I don't have a"
echo "    complete current directory listing. Move them the same way if present:"
echo "      mv results_v2_*.csv results/grids/"
echo "  - Any stray split_cache/<dataset_name>/ subfolders created by the newer"
echo "    scripts (they cache by dataset name now, e.g. split_cache/chips/,"
echo "    split_cache/all_products_ready/) -- these already live under whatever"
echo "    directory you ran each script from; if that's the repo root, they're"
echo "    fine where they are (gitignored via data/cache/ already covering split_cache/)."
