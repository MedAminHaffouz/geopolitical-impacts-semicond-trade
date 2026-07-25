"""
analyze_dist_placement.py
============================
Splits `dist`'s permutation importance by where it lives architecturally:
  - node:dist  -- in GAT models, dist is a node-level gravity feature
  - edge:dist  -- in EdgeGAT_full models, dist is an edge-level feature
The global summary (feature_importance_shrinkage_summary.csv) collapses
these into one 'dist' row via the node:/edge: prefix-stripping step --
this resolves that ambiguity using the raw per-model file, which still
has the prefixes.

Usage:
    python analyze_dist_placement.py [feature_importance_shrinkage.csv]
"""
import sys
import pandas as pd

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else 'feature_importance_shrinkage.csv'


def main():
    df = pd.read_csv(IN_PATH)
    dist_rows = df[df['feature'].isin(['dist', 'node:dist', 'edge:dist'])].copy()

    if len(dist_rows) == 0:
        print(f"No dist rows found in {IN_PATH} -- check the feature names in this file.")
        return

    dist_rows['placement'] = dist_rows['feature'].apply(
        lambda f: 'edge (EdgeGAT_full)' if f.startswith('edge:') else 'node (GAT)')

    summary = dist_rows.groupby('placement').agg(
        n_models=('importance_log_r2', 'size'),
        mean_log_r2=('importance_log_r2', 'mean'),
        std_log_r2=('importance_log_r2', 'std'),
        min_log_r2=('importance_log_r2', 'min'),
        max_log_r2=('importance_log_r2', 'max'),
        mean_spearman=('importance_spearman', 'mean'),
    ).round(5)

    pd.set_option('display.width', 140)
    print("\n=== dist importance, split by architectural placement ===\n")
    print(summary.to_string())

    print("\n=== per-model detail ===")
    print(dist_rows[['model', 'placement', 'importance_log_r2', 'importance_spearman']]
          .sort_values(['placement', 'importance_log_r2'], ascending=[True, False])
          .to_string(index=False))

    if len(summary) == 2:
        node_mean = summary.loc['node (GAT)', 'mean_log_r2']
        edge_mean = summary.loc['edge (EdgeGAT_full)', 'mean_log_r2']
        ratio = node_mean / edge_mean if edge_mean != 0 else float('inf')
        print(f"\n>>> As a node feature (GAT): mean importance = {node_mean:.5f}")
        print(f">>> As an edge feature (EdgeGAT_full): mean importance = {edge_mean:.5f}")
        if abs(node_mean) > abs(edge_mean) * 2:
            print(">>> distance matters meaningfully more when placed on NODES than on edges in this data.")
        elif abs(edge_mean) > abs(node_mean) * 2:
            print(">>> distance matters meaningfully more when placed on EDGES than on nodes in this data.")
        else:
            print(">>> distance's importance is similar regardless of placement -- not a strong effect either way.")


if __name__ == '__main__':
    main()
