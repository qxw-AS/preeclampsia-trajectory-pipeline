"""
Module B Step 4 - STRING protein-protein interaction network
(skill: string-protein-interaction-analysis-with-omicverse)

Network built on the 196 EOPE-vs-control DEGs (FDR<0.05, |log2FC|>0.5).
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import omicverse as ov

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"

print("=" * 78)
print("STEP 4  STRING PPI network of EOPE-associated DEGs")
print("=" * 78)

deg = pd.read_csv(rf"{RES}\moduleB_deg_significant.csv", index_col=0)
gene_list = list(dict.fromkeys(deg.index.tolist()))
print(f"DEGs supplied to STRING: {len(gene_list)}")
assert len(gene_list) >= 2

up = deg.index[deg['sig'] == 'up'].tolist()
dn = deg.index[deg['sig'] == 'down'].tolist()
gene_type_dict = {g: ('Up in EOPE' if g in up else 'Down in EOPE') for g in gene_list}
gene_color_dict = {g: ('#C44E52' if g in up else '#4C72B0') for g in gene_list}
for g in gene_list:
    assert g in gene_type_dict and g in gene_color_dict

# ── query STRING (taxonomy 9606 = human) ─────────────────────────────────
edges = ov.bulk.string_interaction(gene_list, 9606)
print(f"STRING edges retrieved: {0 if edges is None else len(edges)}")
edges.to_csv(rf"{RES}\moduleB_ppi_edges.csv", index=False)

# ── network metrics ──────────────────────────────────────────────────────
G = nx.Graph()
G.add_nodes_from(gene_list)
ecols = {c.lower(): c for c in edges.columns}
src = ecols.get('preferredname_a') or ecols.get('#node1')
tgt = ecols.get('preferredname_b') or ecols.get('node2')
sc = ecols.get('score') or ecols.get('combined_score')
for _, r in edges.iterrows():
    G.add_edge(r[src], r[tgt], weight=float(r[sc]))

deg_cent = dict(G.degree())
btw = nx.betweenness_centrality(G) if G.number_of_edges() else {n: 0 for n in G}
metrics = pd.DataFrame({'gene': list(G.nodes)})
metrics['degree'] = metrics['gene'].map(deg_cent)
metrics['betweenness'] = metrics['gene'].map(btw)
metrics['direction'] = metrics['gene'].map(gene_type_dict)
metrics['log2FC'] = metrics['gene'].map(deg['log2FC'])
metrics = metrics.sort_values(['degree', 'betweenness'], ascending=False)
metrics.to_csv(rf"{RES}\moduleB_ppi_hub_metrics.csv", index=False)

print("\ntop hub genes (degree):")
print(metrics.head(12)[['gene', 'degree', 'betweenness', 'log2FC']].to_string(index=False))
print(f"\nnetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"density={nx.density(G):.4f}")

# ── figure: publication layout (largest connected component of connected genes) ──
sub = G.subgraph([n for n in G.nodes if G.degree(n) > 0]).copy()
comp = max(nx.connected_components(sub), key=len)
H = sub.subgraph(comp).copy()
print(f"plotted component: {H.number_of_nodes()} nodes / {H.number_of_edges()} edges")

pos = nx.spring_layout(H, k=0.42, seed=7, iterations=600, weight='weight')
deg_h = dict(H.degree())
for n in H.nodes:
    gene_color_dict.setdefault(n, '#BDBDBD')
node_color = [gene_color_dict[n] for n in H.nodes]
node_size = [28 + 26 * deg_h[n] for n in H.nodes]

fig, ax = plt.subplots(figsize=(6.4, 5.6))
nx.draw_networkx_edges(H, pos, ax=ax, edge_color='#D5D8DC', width=0.5, alpha=.8)
nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_color, node_size=node_size,
                       linewidths=.4, edgecolors='white', alpha=.95)

top = sorted(deg_h, key=deg_h.get, reverse=True)[:10]
cx = np.mean([pos[n][0] for n in H.nodes]); cy = np.mean([pos[n][1] for n in H.nodes])
for g in top:
    x, y = pos[g]
    dx, dy = x - cx, y - cy
    norm = np.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    ax.annotate(g, (x, y), xytext=(x + 0.135 * dx, y + 0.135 * dy),
                fontsize=7, fontweight='bold', color='#1B2631',
                ha='left' if dx > 0.15 else ('right' if dx < -0.15 else 'center'),
                va='bottom' if dy > 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.85),
                arrowprops=dict(arrowstyle='-', lw=.5, color='#7F8C8D',
                                shrinkA=0, shrinkB=3))

ax.set_axis_off()
handles = [plt.Line2D([], [], marker='o', ls='', ms=7, mfc=c, mec='white', label=l)
           for c, l in [('#C44E52', f'Up in EOPE (n={len(up)})'),
                        ('#4C72B0', f'Down in EOPE (n={len(dn)})')]]
ax.legend(handles=handles, frameon=False, fontsize=8, loc='lower center',
          bbox_to_anchor=(.5, -.02), ncol=2)
ax.set_title(f'STRING PPI network of EOPE-associated DEGs\n'
             f'({H.number_of_nodes()} connected proteins, {H.number_of_edges()} interactions; '
             f'node size = degree)', fontsize=9, pad=8)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS7_ppi_network.{ext}", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

json.dump({'n_input': len(gene_list), 'n_nodes': G.number_of_nodes(),
           'n_edges': G.number_of_edges(), 'density': nx.density(G),
           'top_hubs': metrics.head(10)['gene'].tolist()},
          open(rf"{RES}\moduleB_ppi_summary.json", 'w'), indent=2)
print("\nSaved PPI edges, hub metrics and figS7_ppi_network")
