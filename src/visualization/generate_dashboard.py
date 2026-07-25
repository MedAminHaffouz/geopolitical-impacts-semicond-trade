"""
Static results dashboard — GDELT trade-flow benchmark
=======================================================
Reads `results_v2.csv` and renders a single-page PNG dashboard with the
essentials for a supervisor readout:

  1. Does GDELT help? (per-model log-R^2, no-GDELT vs GDELT)
  2. Where should GDELT risk live? (global node/edge/both bars +
     model-family comparison table: GNN / GAT / GATv2 x node/edge/both)
  3. Which fusion technique wins, per model that supports fusion?
  4. Keep-all vs top-k event selection, per node model.

Usage:
    python dashboard.py [path/to/results_v2.csv] [path/to/output.png]

Defaults to ./results_v2.csv -> ./dashboard.png
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IN_PATH = sys.argv[1] if len(sys.argv) > 1 else 'results_electronics.csv'
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else 'dashboard.png'
TRANCHE = 'all'
MISSING = 'rf'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#111111',
    'axes.facecolor': '#111111',
    'savefig.facecolor': '#111111',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'font.size': 10,
})

POS, NEG = '#2ca02c', '#d62728'
BLUE, ORANGE = '#4c8fbd', '#c47f3e'

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load(path):
    r = pd.read_csv(path)
    r = r.drop_duplicates(
        ['model', 'fusion', 'scoring', 'gdelt', 'risk_level', 'agg_mode', 'missing', 'tranche'],
        keep='last'
    )
    return r


def base(df, tranche=TRANCHE, missing=MISSING):
    d = df[df.tranche == tranche]
    if missing is not None and 'missing' in d:
        d = d[d.missing == missing]
    return d.copy()


# --------------------------------------------------------------------------
# Panel 1 — GDELT added value, per model (log-R2)
# --------------------------------------------------------------------------
def panel_gdelt_value(ax, r):
    d = base(r)
    no_g = d[~d.gdelt].groupby('model')['log_r2'].max()
    with_g = d[d.gdelt].groupby('model')['log_r2'].max()
    piv = pd.DataFrame({'no-GDELT': no_g, 'GDELT': with_g}).dropna(how='all')
    # order by GDELT score, best first
    piv = piv.reindex(piv['GDELT'].fillna(piv['no-GDELT']).sort_values(ascending=False).index)

    x = np.arange(len(piv))
    w = 0.38
    ax.bar(x - w / 2, piv['no-GDELT'], w, label='no-GDELT', color=BLUE)
    ax.bar(x + w / 2, piv['GDELT'], w, label='GDELT', color=ORANGE)
    ax.axhline(0, color='w', lw=.6)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=40, ha='right')
    ax.set_ylabel('log-R²')
    ax.set_title('1. Does GDELT help? — log-R² per model (no-GDELT vs GDELT)', loc='left', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend(loc='upper right', framealpha=0.3)

    for xi, (nv, gv) in enumerate(zip(piv['no-GDELT'], piv['GDELT'])):
        if pd.notna(nv):
            ax.text(xi - w / 2, nv + (0.02 if nv >= 0 else -0.05), f'{nv:.2f}', ha='center', fontsize=7)
        if pd.notna(gv):
            ax.text(xi + w / 2, gv + (0.02 if gv >= 0 else -0.05), f'{gv:.2f}', ha='center', fontsize=7)


# --------------------------------------------------------------------------
# Panel 2 — Risk placement: global bars
# --------------------------------------------------------------------------
def _placement(row):
    m = row['model']
    if m in ('GCN', 'GAT', 'GATv2') and row['gdelt']:
        return 'node'
    if m in ('EdgeGNN', 'EdgeGAT'):
        return 'edge'
    if m in ('EdgeGNN_full', 'EdgeGAT_full', 'EdgeGATv2_full'):
        return 'both'
    return None


def _placement_data(r):
    d = r[r.tranche == TRANCHE].copy()  # placement compares across models; don't filter missing
    d['placement'] = d.apply(_placement, axis=1)
    return d.dropna(subset=['placement'])


def panel_placement_global(ax_r2, ax_sp, r):
    d = _placement_data(r)
    order = ['node', 'edge', 'both']
    for ax, metric, label in [(ax_r2, 'log_r2', 'log-R²'), (ax_sp, 'spearman', 'Spearman ρ')]:
        g = d.groupby('placement')[metric].max().reindex(order)
        colors = [POS if v >= 0 else NEG for v in g.values]
        ax.barh(g.index, g.values, color=colors)
        for i, v in enumerate(g.values):
            ax.text(v, i, f' {v:.3f}', va='center', fontsize=9)
        ax.axvline(0, color='w', lw=.6)
        ax.set_xlabel(label)
        ax.set_title(f'best {label} by risk placement', loc='left', fontsize=10)
        ax.invert_yaxis()


# --------------------------------------------------------------------------
# Panel 2b — Risk placement comparison table (family x placement)
# --------------------------------------------------------------------------
FAMILIES = {
    'GNN':   {'node': 'GCN',   'edge': 'EdgeGNN', 'both': 'EdgeGNN_full'},
    'GAT':   {'node': 'GAT',   'edge': 'EdgeGAT', 'both': 'EdgeGAT_full'},
    'GATv2': {'node': 'GATv2', 'edge': None,      'both': 'EdgeGATv2_full'},
}
COLS = ['node', 'edge', 'both']


def panel_placement_table(ax, r, metric='log_r2'):
    d = _placement_data(r)
    best = d.groupby('model')[metric].max()

    mat = np.full((len(FAMILIES), len(COLS)), np.nan)
    labels = np.empty((len(FAMILIES), len(COLS)), dtype=object)

    for i, (fam, mapping) in enumerate(FAMILIES.items()):
        for j, col in enumerate(COLS):
            mdl = mapping[col]
            if mdl is None:
                labels[i, j] = f'{fam} ({col})\nnot built'
                continue
            val = best.get(mdl, np.nan)
            mat[i, j] = val
            labels[i, j] = f'{mdl}\n{val:.3f}' if pd.notna(val) else f'{mdl}\nn/a'

    norm = Normalize(vmin=np.nanmin(mat) - 0.02, vmax=np.nanmax(mat) + 0.02)
    cmap = plt.get_cmap('RdYlGn')

    ax.set_xlim(0, len(COLS))
    ax.set_ylim(0, len(FAMILIES))
    ax.set_xticks(np.arange(len(COLS)) + 0.5)
    ax.set_xticklabels([c.upper() for c in COLS], fontweight='bold')
    ax.set_yticks(np.arange(len(FAMILIES)) + 0.5)
    ax.set_yticklabels(list(FAMILIES.keys())[::-1], fontweight='bold')
    ax.set_title(f'2b. Risk placement comparison table ({metric})', loc='left', fontweight='bold')
    ax.tick_params(length=0)

    n_fam = len(FAMILIES)
    for i, fam in enumerate(FAMILIES):
        row_from_top = i
        row_from_bottom = n_fam - 1 - row_from_top
        for j, col in enumerate(COLS):
            val = mat[i, j]
            face = cmap(norm(val)) if pd.notna(val) else '#333333'
            ax.add_patch(plt.Rectangle((j, row_from_bottom), 1, 1, facecolor=face,
                                        edgecolor='#111111', linewidth=2))
            txt_color = 'black' if pd.notna(val) and norm(val) > 0.35 else 'white'
            ax.text(j + 0.5, row_from_bottom + 0.5, labels[i, j], ha='center', va='center',
                    fontsize=9, color=txt_color, linespacing=1.6)

    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------
# Panel 3/4 — Fusion x scoring detail table, for a single model
# --------------------------------------------------------------------------
_SCORING_ORDERS = [
    ['keepall', 'topk'],
    ['ka_ka', 'ka_tk', 'tk_ka', 'tk_tk'],
]
_FUSION_ORDER = ['concat', 'blend', 'attention']
_LABEL_MAP = {'ka_ka': 'node=KA\npair=KA', 'ka_tk': 'node=KA\npair=TK',
              'tk_ka': 'node=TK\npair=KA', 'tk_tk': 'node=TK\npair=TK'}


def panel_fusion_scoring_table(ax, r, model, title):
    d = base(r, missing=None)
    fus = d[d.fusion.isin(_FUSION_ORDER)]
    sub = fus[fus.model == model]
    piv = sub.pivot_table(index='fusion', columns='scoring', values='log_r2', aggfunc='max')

    rows = [f for f in _FUSION_ORDER if f in piv.index]
    piv = piv.reindex(rows)
    cols = list(piv.columns)
    for cand in _SCORING_ORDERS:
        if set(cols) <= set(cand):
            cols = [c for c in cand if c in piv.columns]
            break
    piv = piv[cols]

    mat = piv.values.astype(float)
    n_rows, n_cols = mat.shape

    if n_rows == 0 or n_cols == 0 or np.all(np.isnan(mat)):
        ax.text(0.5, 0.5, f'no fusion data for {model}', ha='center', va='center')
        ax.axis('off')
        return

    norm = Normalize(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    cmap = plt.get_cmap('RdYlGn')

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    display_cols = [_LABEL_MAP.get(c, c) for c in cols]
    ax.set_xticklabels(display_cols, fontweight='bold')
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(list(piv.index)[::-1], fontweight='bold')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.tick_params(length=0)

    for i in range(n_rows):
        row_from_bottom = n_rows - 1 - i
        for j in range(n_cols):
            val = mat[i, j]
            face = cmap(norm(val)) if pd.notna(val) else '#333333'
            ax.add_patch(plt.Rectangle((j, row_from_bottom), 1, 1, facecolor=face,
                                        edgecolor='#111111', linewidth=2))
            txt_color = 'black' if pd.notna(val) and norm(val) > 0.35 else 'white'
            label = f'{val:.3f}' if pd.notna(val) else 'n/a'
            ax.text(j + 0.5, row_from_bottom + 0.5, label, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=txt_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0, -0.12, 'KA = keep-all   ·   TK = top-k (k=10)',
            transform=ax.transAxes, fontsize=8, color='#888888', ha='left', va='top')


# --------------------------------------------------------------------------
# Headline KPI strip
# --------------------------------------------------------------------------
def panel_headline(ax, r):
    d = base(r)

    gat_no_gdelt = d[(d.model == 'GAT') & (~d.gdelt)]['log_r2'].max()
    gat_gdelt    = d[(d.model == 'GAT') & (d.gdelt)]['log_r2'].max()
    edgegat_full = d[d.model == 'EdgeGAT_full']['log_r2'].max()

    ax.axis('off')
    ax.set_title('Trade-Flow prediction results benchmark - Tranche : semiconductors products',
                 fontsize=18, fontweight='bold', loc='left', pad=14)

    kpis = [
        f'GAT (no-GDELT): {gat_no_gdelt:.3f} log-R²',
        f'GAT (GDELT): {gat_gdelt:.3f} log-R²',
        f'EdgeGAT_full: {edgegat_full:.3f} log-R²',
    ]
    ax.text(0, 0.35, '   |   '.join(kpis), fontsize=12, color='#cccccc', va='center')


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def main():
    r = load(IN_PATH)

    fig = plt.figure(figsize=(16, 17))
    gs = GridSpec(
        5, 2, figure=fig,
        height_ratios=[0.2, 2.2, 1.6, 2.2, 1.8],
        width_ratios=[2, 4],
        hspace=0.65, wspace=0.28,
        left=0.07, right=0.97, top=0.97, bottom=0.03,
    )

    ax_head = fig.add_subplot(gs[0, :])
    panel_headline(ax_head, r)

    ax_gdelt = fig.add_subplot(gs[1, :])
    panel_gdelt_value(ax_gdelt, r)

    ax_place_r2 = fig.add_subplot(gs[2, 0])
    ax_place_sp = fig.add_subplot(gs[2, 1])
    panel_placement_global(ax_place_r2, ax_place_sp, r)

    ax_table = fig.add_subplot(gs[3, :])
    panel_placement_table(ax_table, r, metric='log_r2')

    ax_fus_gat = fig.add_subplot(gs[4, 0])
    ax_fus_edgegat = fig.add_subplot(gs[4, 1])
    panel_fusion_scoring_table(ax_fus_gat, r, 'GAT', '3. Fusion × scoring — GAT (log-R²)')
    panel_fusion_scoring_table(ax_fus_edgegat, r, 'EdgeGAT_full', '4. Fusion × scoring — EdgeGAT_full (log-R²)')

    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'saved -> {OUT_PATH}')


if __name__ == '__main__':
    main()