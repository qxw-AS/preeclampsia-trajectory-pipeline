"""
Module B Step 8 — Immune cell deconvolution + platelet-endothelial axis

Method: exploratory ssGSEA of curated immune/endothelial marker sets.
        Scores are enrichment signatures, not estimated cell fractions or causal deconvolution.

Outputs:
  - Immune cell type scores per sample
  - EOPE vs control comparison
  - Correlation with Hypoxia / Platelet activation ssGSEA scores
  - Fig 6: immune composition + immune-pathway correlation heatmap
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
DATA = rf"{OUT}\data\geo\gse75010\GSE75010_data"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'figure.dpi': 300})

# Curated immune cell marker gene sets (from LM22/xCell/MCP-counter literature)
IMMUNE_MARKERS = {
    'CD4_T_cells':       ['CD4','IL7R','CD3D','CD3E','CD3G','LTB','CD27','CD28','ICOS','CD40LG'],
    'CD8_T_cells':       ['CD8A','CD8B','CD3D','CD3E','GZMA','GZMB','PRF1','NKG7','CCL5','KLRD1'],
    'Treg':              ['FOXP3','IL2RA','CTLA4','TIGIT','IKZF2','TNFRSF18','CCR8','BATF','SATB1','CD4'],
    'NK_cells':          ['NKG7','GNLY','KLRD1','KLRF1','NCR1','NKG2D','KLRC1','KLRB1','PRF1','GZMB'],
    'B_cells':           ['CD19','MS4A1','CD79A','CD79B','BLK','BANK1','CD22','PAX5','TCL1A'],
    'Plasma_cells':      ['MZB1','JCHAIN','XBP1','SDC1','IRF4','PRDM1','TNFRSF17','SLAMF7','FCRL5'],
    'Monocytes':         ['CD14','LYZ','S100A8','S100A9','FCN1','VCAN','CSF1R','CD68','ITGAM','CST3'],
    'Macrophages_M1':    ['CD68','NOS2','IL1B','TNF','IL6','CCR2','CD86','CD80','HLA-DRA','FCGR1A'],
    'Macrophages_M2':    ['CD163','MSR1','MRC1','ARG1','IL10','TGFB1','VEGFA','CCL18','CLEC10A'],
    'Dendritic_cells':   ['ITGAX','HLA-DRA','CD1C','CLEC10A','FCER1A','CLEC9A','XCR1','BATF3','IRF8','CCR7'],
    'Neutrophils':       ['S100A8','S100A9','CSF3R','FCGR3B','CXCR1','CXCR2','DEFA1','ELANE','MPO'],
    'Mast_cells':        ['KIT','CPA3','TPSAB1','TPSB2','HDC','MS4A2','FCER1A','GATA2','IL1RL1','CD200R3'],
    'Platelets':         ['ITGA2B','ITGB3','PF4','PPBP','NFE2','TUBB1','GP9','GP1BA'],
    'Endothelial':       ['PECAM1','VWF','CDH5','KDR','FLT1','ENG','ESAM','CLDN5','SOX18','ERG'],
}

print("=" * 78)
print("STEP 8  Immune cell deconvolution (xCell-style ssGSEA)")
print("=" * 78)

# ── 1. Load data ──────────────────────────────────────────────────────────
expr = pd.read_csv(rf"{DATA}\GSE75010_gene_matrix.csv.gz", index_col=0)
train = pd.read_csv(rf"{DATA}\GSE75010_training_labels.csv")
samples = train['sample_id'].tolist()
y = (train.set_index('sample_id').loc[samples, 'group'] == 'EOPE').astype(int).values
print(f"samples: {len(samples)} ({y.sum()} EOPE / {len(y)-y.sum()} control)")

# ── 2. Run ssGSEA for each immune cell type ───────────────────────────────
import gseapy as gp

print(f"\nRunning ssGSEA for {len(IMMUNE_MARKERS)} immune cell types...")
# gseapy.ssgsea needs gene sets as a dict of {name: [genes]}
ss = gp.ssgsea(data=expr.loc[:, samples], gene_sets=IMMUNE_MARKERS,
               outdir=None, sample_norm_method='rank', min_size=3)
# Results: res2d with columns Name, Term, ES
scores = ss.res2d.pivot(index='Name', columns='Term', values='ES').reindex(samples)
if scores.isna().any().any():
    raise ValueError('Missing or misaligned ssGSEA sample scores')
print(f"ssGSEA scores shape: {scores.shape}")
print(scores.head())

scores.to_csv(rf"{RES}\moduleB_immune_scores.csv")
print(f"saved moduleB_immune_scores.csv")

# ── 3. EOPE vs control comparison ─────────────────────────────────────────
print("\n--- EOPE vs control immune cell comparison ---")
comp = pd.DataFrame(index=scores.columns, columns=['EOPE_mean','Ctrl_mean','diff','pvalue','FDR'])
for ct in scores.columns:
    eope_v = scores.loc[y == 1, ct]
    ctrl_v = scores.loc[y == 0, ct]
    t, p = stats.ttest_ind(eope_v, ctrl_v, equal_var=False)
    comp.loc[ct] = [eope_v.mean(), ctrl_v.mean(), eope_v.mean() - ctrl_v.mean(), p, np.nan]
# BH-FDR correction across the 14 prespecified marker-set comparisons.
pvals = comp['pvalue'].astype(float).values
comp['FDR'] = multipletests(pvals, method='fdr_bh')[1]
comp = comp.sort_values('pvalue')
print(comp.to_string())
comp.to_csv(rf"{RES}\moduleB_immune_comparison.csv")

# ── 4. Correlate with pathway ssGSEA scores ───────────────────────────────
print("\n--- Immune cell ↔ pathway score correlation ---")
pathway_scores = pd.read_csv(rf"{RES}\moduleB_ssgsea_pathway_scores.csv", index_col=0)
# Align samples
common_samples = [s for s in samples if s in pathway_scores.index]
scores_aligned = scores.loc[common_samples]
pathway_aligned = pathway_scores.loc[common_samples]

# Focus on key pathways
key_pathways = ['Hypoxia', 'Platelet activation', 'Inflammatory response',
                'Angiogenesis', 'Complement & coagulation cascades', 'Apoptosis']
key_pathways = [p for p in key_pathways if p in pathway_aligned.columns]
print(f"using pathways: {key_pathways}")

corr = pd.DataFrame(index=scores.columns, columns=key_pathways, dtype=float)
pvals_corr = corr.copy()
for ct in scores.columns:
    for pw in key_pathways:
        x, yv = scores_aligned[ct].values, pathway_aligned[pw].values
        r, p = stats.spearmanr(x, yv)
        corr.loc[ct, pw] = r
        pvals_corr.loc[ct, pw] = p

corr.to_csv(rf"{RES}\moduleB_immune_pathway_corr.csv")
fdr_corr = pd.DataFrame(
    multipletests(pvals_corr.astype(float).values.ravel(), method='fdr_bh')[1].reshape(corr.shape),
    index=corr.index, columns=corr.columns)
fdr_corr.to_csv(rf"{RES}\moduleB_immune_pathway_corr_fdr.csv")
print("\nImmune cell ↔ pathway Spearman correlation:")
print(corr.to_string())

# ── 5. Figure 6 ───────────────────────────────────────────────────────────
print("\nGenerating Fig 6...")
fig = plt.figure(figsize=(13, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)

# Panel A: immune cell composition (EOPE vs control boxplots)
axA = fig.add_subplot(gs[0, 0])
# Sort by significance (p-value)
sig_cells = comp.sort_values('pvalue').index.tolist()
sig_cells = [c for c in sig_cells if comp.loc[c, 'FDR'] < 0.05][:10]
if len(sig_cells) < 6:
    sig_cells = comp.sort_values('pvalue').index[:8].tolist()

positions = []
box_data = []
labels = []
for i, ct in enumerate(sig_cells):
    box_data.append(scores.loc[y == 0, ct].values)
    box_data.append(scores.loc[y == 1, ct].values)
    positions.append(i * 2.5)
    positions.append(i * 2.5 + 1)
    labels.append(ct.replace('_', ' '))

bp = axA.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True,
                 showfliers=False, medianprops={'color': 'black', 'lw': 1.2})
colors_box = ['#BDC3C7', '#E74C3C']
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_box[i % 2])
    patch.set_alpha(0.7)
axA.set_xticks([p + 0.5 for p in positions[::2]])
axA.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=7, rotation=45, ha='right')
axA.set_ylabel('ssGSEA enrichment score')
axA.set_title('Immune cell scores: Control vs EOPE', fontsize=10, fontweight='bold', pad=8)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_box[0], label='Control'),
                   Patch(facecolor=colors_box[1], label='EOPE')]
axA.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.06),
           ncol=2, fontsize=8, frameon=True, edgecolor='#CCCCCC')
# Significance brackets
for i, ct in enumerate(sig_cells):
    p = comp.loc[ct, 'FDR']
    if p < 0.05:
        star = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
        ymax = max(np.max(box_data[i*2]), np.max(box_data[i*2+1]))
        axA.text(positions[i*2] + 0.5, ymax + 0.1, star, ha='center', fontsize=10, color='#E74C3C')
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)
axA.text(-0.1, 1.02, 'A', transform=axA.transAxes, fontsize=16, fontweight='bold', va='top')

# Panel B: immune-pathway correlation heatmap
axB = fig.add_subplot(gs[0, 1])
# Show all immune cells x key pathways
corr_plot = corr.astype(float)
im = axB.imshow(corr_plot.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
axB.set_xticks(range(len(key_pathways)))
short_pw = [p.split('_', 1)[-1].replace('_', ' ')[:22] for p in key_pathways]
axB.set_xticklabels(short_pw, rotation=45, ha='right', fontsize=7)
axB.set_yticks(range(len(corr_plot.index)))
axB.set_yticklabels([c.replace('_', ' ') for c in corr_plot.index], fontsize=7.5)
# Annotate significant correlations
for i in range(corr_plot.shape[0]):
    for j in range(corr_plot.shape[1]):
        r = corr_plot.iloc[i, j]
        p = fdr_corr.iloc[i, j]
        if p < 0.05:
            axB.text(j, i, f"{r:.2f}", ha='center', va='center', fontsize=6,
                     color='white' if abs(r) > 0.4 else 'black', fontweight='bold')
        else:
            axB.text(j, i, f"{r:.2f}", ha='center', va='center', fontsize=5.5, color='gray')
cb = fig.colorbar(im, ax=axB, shrink=0.75)
cb.set_label("Spearman's ρ", fontsize=8)
axB.set_title('Immune cell ↔ pathway co-activity', fontsize=10, fontweight='bold', pad=8)
axB.text(-0.08, 1.02, 'B', transform=axB.transAxes, fontsize=16, fontweight='bold', va='top')

fig.suptitle('Placental immune microenvironment in early-onset preeclampsia',
             fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout()
for ext in ['png', 'pdf', 'tiff']:
    fig.savefig(rf"{FIG}\fig6_immune_microenvironment.{ext}", dpi=300, bbox_inches='tight',
                pad_inches=0.15, facecolor='white', edgecolor='none')
plt.close(fig)
print("  saved fig6_immune_microenvironment.*")

# ── 6. Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
n_sig = (comp['FDR'] < 0.05).sum()
print(f"Immune/endothelial marker sets: {len(IMMUNE_MARKERS)}")
print(f"Significant EOPE vs control (BH-FDR<0.05): {n_sig}")
print(f"\nTop significant marker sets:")
for ct in comp[comp['FDR'] < 0.1].index:
    r = comp.loc[ct]
    star = '***' if r['pvalue'] < 0.001 else ('**' if r['pvalue'] < 0.01 else ('*' if r['pvalue'] < 0.05 else ''))
    print(f"  {ct:20s} diff={r['diff']:+.3f}  p={r['pvalue']:.4f} {star}")

# Key correlations with Hypoxia and Platelet activation
print(f"\nKey correlations (Spearman ρ):")
for ct in corr.index:
    h = corr.loc[ct, 'Hypoxia'] if 'Hypoxia' in corr.columns else np.nan
    p_h = fdr_corr.loc[ct, 'Hypoxia'] if 'Hypoxia' in fdr_corr.columns else np.nan
    pa = corr.loc[ct, 'Platelet activation'] if 'Platelet activation' in corr.columns else np.nan
    p_pa = fdr_corr.loc[ct, 'Platelet activation'] if 'Platelet activation' in fdr_corr.columns else np.nan
    if (p_h < 0.05 if not np.isnan(p_h) else False) or (p_pa < 0.05 if not np.isnan(p_pa) else False):
        print(f"  {ct:20s}  Hypoxia ρ={h:+.3f}(p={p_h:.3f})  Platelet_act ρ={pa:+.3f}(p={p_pa:.3f})")

summary = {
    'n_immune_cell_types': len(IMMUNE_MARKERS),
    'n_significant_EOPE_vs_control_FDR05': int(n_sig),
    'top_significant_cells_FDR10': comp[comp['FDR'] < 0.1].index.tolist(),
    'interpretation': 'Marker-set ssGSEA enrichment scores; not cell-fraction estimates',
}
json.dump(summary, open(rf"{RES}\moduleB_immune_summary.json", 'w'), indent=2)
print("\nAll results saved.")
