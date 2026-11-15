"""
Module B Step 2b — WGCNA deepening + HIF-1α upstream regulator analysis

Direction 4: indianred module hub genes, GO/KEGG enrichment, module-trait correlation
Direction 1: TF enrichment (ChEA3/TRRUST), HIF-1α regulatory subnetwork

Uses existing module assignments; recomputes module eigengenes, kME, GS.
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
DATA = rf"{OUT}\data\geo\gse75010\GSE75010_data"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'figure.dpi': 300})

print("=" * 78)
print("STEP 2b  WGCNA deepening + HIF-1α upstream regulator analysis")
print("=" * 78)

# ── 1. Load data ──────────────────────────────────────────────────────────
expr = pd.read_csv(rf"{DATA}\GSE75010_gene_matrix.csv.gz", index_col=0)
meta = pd.read_csv(rf"{DATA}\GSE75010_sample_metadata.csv").set_index('sample_id')
train = pd.read_csv(rf"{DATA}\GSE75010_training_labels.csv")
samples = train['sample_id'].tolist()

mod_assign = pd.read_csv(rf"{RES}\moduleB_wgcna_module_assignment.csv", index_col=0)
deg = pd.read_csv(rf"{RES}\moduleB_deg_significant.csv")
print(f"loaded: expr {expr.shape}, module assignment {len(mod_assign)}, DEG {len(deg)}")

# ── 2. Subset expression to the 5000 network genes, compute MEs/kME/GS ───
common_genes = [g for g in mod_assign.index if g in expr.index]
expr_sub = expr.loc[common_genes, samples].T.astype(float)   # samples x genes
mod_colors = mod_assign.loc[common_genes, 'module']
print(f"network genes: {expr_sub.shape[1]}; samples: {expr_sub.shape[0]}")

lab = train.set_index('sample_id')
eope = (lab.loc[samples, 'group'] == 'EOPE').astype(int).values
traits = pd.DataFrame(index=samples)
traits['EOPE'] = eope
for col in ['ga (week)', 'maternal age', 'maximum systolic bp', 'maximum diastolic bp',
            'newborn weight z-score', 'placental weight z-score']:
    traits[col] = lab.loc[samples, col].astype(float).values
traits.columns = ['EOPE', 'GA_weeks', 'Maternal_age', 'Max_SBP', 'Max_DBP',
                  'Newborn_weight_z', 'Placental_weight_z']

# Module eigengenes = first PC of each module's genes
MEs = pd.DataFrame(index=samples)
for m in mod_colors.unique():
    genes = mod_colors[mod_colors == m].index.tolist()
    X = expr_sub[genes].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    pca = PCA(n_components=1, svd_solver='arpack')
    me = pca.fit_transform(X).ravel()
    # sign: align with mean expression to keep sign stable
    if np.corrcoef(me, X.mean(axis=1))[0, 1] < 0:
        me = -me
    MEs[f'ME{m}'] = me
print(f"computed module eigengenes for {MEs.shape[1]} modules")

# kME: correlation of each gene with its module eigengene
kme = pd.DataFrame(index=expr_sub.columns, columns=['kME'], dtype=float)
for g in expr_sub.columns:
    m = mod_colors.loc[g]
    kme.loc[g, 'kME'] = np.corrcoef(expr_sub[g].values, MEs[f'ME{m}'].values)[0, 1]

# GS: correlation of each gene with EOPE
gs = pd.Series(
    [np.corrcoef(expr_sub[g].values, eope)[0, 1] for g in expr_sub.columns],
    index=expr_sub.columns, name='GS'
)

gene_info = pd.DataFrame({'module': mod_colors, 'kME': kme['kME'], 'GS': gs})
gene_info.to_csv(rf"{RES}\moduleB_wgcna_gene_info.csv")
print(f"saved gene_info (kME + GS) for {len(gene_info)} genes")

# ── 3. Hub genes of indianred module ──────────────────────────────────────
indianred = gene_info[gene_info['module'] == 'indianred'].copy()
# Standard WGCNA hub definition: |kME| > 0.8 AND |GS| > 0.5
indianred['kME_abs'] = indianred['kME'].abs()
indianred['GS_abs'] = indianred['GS'].abs()
hub_mask = (indianred['kME_abs'] > 0.8) & (indianred['GS_abs'] > 0.5)
hubs = indianred[hub_mask].sort_values('GS_abs', ascending=False)
print(f"\nindianred module: {len(indianred)} genes; hub genes (kME>0.8 & |GS|>0.5): {len(hubs)}")
if len(hubs) > 0:
    print(hubs[['kME', 'GS']].head(15).to_string())

hubs.to_csv(rf"{RES}\moduleB_indianred_hub_genes.csv")
gene_info.to_csv(rf"{RES}\moduleB_wgcna_hub_genes.csv")  # overwrite empty file

# ── 4. Module-trait correlation (full clinical traits) ────────────────────
cor = pd.DataFrame(index=MEs.columns, columns=traits.columns, dtype=float)
pval = cor.copy()
for m in MEs.columns:
    for t in traits.columns:
        x, y = MEs[m].values, traits[t].values
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() > 5 and x[ok].std() > 0 and y[ok].std() > 0:
            r, p = stats.pearsonr(x[ok], y[ok])
            cor.loc[m, t], pval.loc[m, t] = r, p

cor.to_csv(rf"{RES}\moduleB_wgcna_module_trait_r.csv")
pval.to_csv(rf"{RES}\moduleB_wgcna_module_trait_p.csv")

# Highlight indianred clinical correlations
print("\nindianred module correlations with clinical traits:")
for t in traits.columns:
    r, p = cor.loc['MEindianred', t], pval.loc['MEindianred', t]
    star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"  {t:22s} r={r:+.3f}  p={p:.2e} {star}")

# ── 5. Enrichment analysis (indianred module + DEGs) ──────────────────────
import gseapy as gp

def run_enrichr(gene_list, gene_set, label):
    """Run Enrichr ORA; return top 20 results DataFrame."""
    try:
        enr = gp.enrichr(gene_list=gene_list, gene_sets=gene_set,
                         organism='human', outdir=None)
        res = enr.results.copy()
        res['Gene_set'] = gene_set
        res['-log10FDR'] = -np.log10(res['Adjusted P-value'].clip(lower=1e-300))
        res = res.sort_values('Adjusted P-value').head(20)
        print(f"  {label} [{gene_set}]: {len(res)} significant terms (top adjP={res['Adjusted P-value'].min():.2e})")
        return res
    except Exception as e:
        print(f"  {label} [{gene_set}] FAILED: {e}")
        return pd.DataFrame()

indianred_genes = indianred.index.tolist()
deg_genes = deg['gene_symbol'].tolist()

print("\nEnrichment analysis (indianred module, 675 genes):")
enrich_sets = {
    'GO_BP': 'GO_Biological_Process_2023',
    'GO_CC': 'GO_Cellular_Component_2023',
    'GO_MF': 'GO_Molecular_Function_2023',
    'KEGG': 'KEGG_2021_Human',
    'Reactome': 'Reactome_2022',
}
enrich_results = {}
for name, gs in enrich_sets.items():
    enrich_results[name] = run_enrichr(indianred_genes, gs, f'indianred-{name}')

# Save all enrichment results
for name, df in enrich_results.items():
    if len(df) > 0:
        df.to_csv(rf"{RES}\moduleB_indianred_enrich_{name}.csv", index=False)

# ── 6. TF enrichment (Direction 1) ────────────────────────────────────────
print("\nTF enrichment (ChEA3 + TRRUST) on DEGs + indianred genes:")
tf_sets = {
    'ChEA3': 'ChEA_2016',
    'TRRUST': 'TRRUST_Transcription_Factors_2019',
    'ENCODE_TF': 'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X',
}
# Use union of DEG up-genes + indianred hub genes (most EOPE-relevant)
degs_up = deg[deg['log2FC'] > 0]['gene_symbol'].tolist()
tf_input = list(set(degs_up + hubs.index.tolist()))
print(f"TF enrichment input genes: {len(tf_input)} (DEG-up ∪ indianred hubs)")

tf_results = {}
for name, gs in tf_sets.items():
    tf_results[name] = run_enrichr(tf_input, gs, f'TF-{name}')
    if len(tf_results[name]) > 0:
        tf_results[name].to_csv(rf"{RES}\moduleB_tf_enrich_{name}.csv", index=False)

# ── 7. HIF-1α target gene verification ────────────────────────────────────
print("\nHIF-1α target gene verification:")
# Collect HIF1A targets from ChEA3/TRRUST results
hif_targets = set()
for name, df in tf_results.items():
    if len(df) > 0:
        hif_rows = df[df['Term'].str.contains('HIF1A|HIF-1', case=False, na=False)]
        for _, row in hif_rows.iterrows():
            genes = str(row['Genes']).split(';')
            hif_targets.update(genes)
hif_targets = {g.strip() for g in hif_targets if g.strip()}
print(f"HIF1A target genes from TF enrichment: {len(hif_targets)}")

# Overlap with DEGs and indianred
hif_in_deg = [g for g in hif_targets if g in deg['gene_symbol'].values]
hif_in_indianred = [g for g in hif_targets if g in indianred.index]
print(f"HIF targets in DEGs: {len(hif_in_deg)}")
print(f"HIF targets in indianred module: {len(hif_in_indianred)}")
if hif_in_deg:
    print("  HIF targets in DEGs:", ', '.join(hif_in_deg[:15]))

# Hypergeometric test: is the overlap of HIF targets with DEGs significant?
from scipy.stats import hypergeom
n_total = 5000  # background genes
n_deg = len(deg)
n_hif = len(hif_targets)
n_overlap = len(hif_in_deg)
p_hyper = hypergeom.sf(n_overlap - 1, n_total, n_hif, n_deg)
print(f"Hypergeometric test: overlap={n_overlap}, expected={n_hif*n_deg/n_total:.1f}, p={p_hyper:.2e}")

# Save HIF target list
pd.DataFrame({'gene': sorted(hif_targets)}).to_csv(rf"{RES}\moduleB_hif1a_targets.csv", index=False)

# ── 8. Figures ────────────────────────────────────────────────────────────
# Fig 5A: TF enrichment top TFs
print("\nGenerating figures...")

# Combine TF results, aggregate by TF term
all_tf = pd.concat([df for df in tf_results.values() if len(df) > 0], ignore_index=True)
if len(all_tf) > 0:
    # Take best result per TF term (lowest adjP) across databases
    all_tf['TF'] = all_tf['Term'].str.split(' ').str[0]
    top_tf = (all_tf.groupby('TF')['Adjusted P-value'].min()
              .sort_values().head(15).reset_index())
    top_tf['-log10FDR'] = -np.log10(top_tf['Adjusted P-value'].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ['#E74C3C' if 'HIF' in tf else '#2C3E50' for tf in top_tf['TF']]
    ax.barh(range(len(top_tf)), top_tf['-log10FDR'], color=colors)
    ax.set_yticks(range(len(top_tf)))
    ax.set_yticklabels(top_tf['TF'])
    ax.invert_yaxis()
    ax.set_xlabel('$-\log_{10}$(FDR)')
    ax.set_title('Top enriched transcription factors (ChEA3/TRRUST/ENCODE)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Add adjP values
    for i, (p, val) in enumerate(zip(top_tf['Adjusted P-value'], top_tf['-log10FDR'])):
        ax.text(val + 0.2, i, f'{p:.1e}', va='center', fontsize=7)
    fig.tight_layout()
    for ext in ['png', 'pdf', 'tiff']:
        fig.savefig(rf"{FIG}\fig5A_tf_enrichment.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  fig5A_tf_enrichment saved")

# Fig 5B: HIF-1α regulatory subnetwork (top targets in DEGs + indianred)
# Plot as a simple network: HIF1A center connected to target genes
if len(hif_in_deg) > 0 or len(hif_in_indianred) > 0:
    import networkx as nx
    G = nx.Graph()
    G.add_node('HIF1A', type='TF')
    targets = list(set(hif_in_deg + hif_in_indianred))[:30]  # limit to 30 for clarity
    for t in targets:
        in_deg = t in deg['gene_symbol'].values
        in_ind = t in indianred.index
        G.add_node(t, type='target', in_deg=in_deg, in_ind=in_ind)
        G.add_edge('HIF1A', t)

    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    # Draw HIF1A
    nx.draw_networkx_nodes(G, pos, nodelist=['HIF1A'], node_color='#E74C3C',
                           node_size=800, ax=ax)
    # Draw targets (color by DEG/indianred)
    deg_nodes = [n for n in targets if G.nodes[n].get('in_deg') and not G.nodes[n].get('in_ind')]
    ind_nodes = [n for n in targets if G.nodes[n].get('in_ind') and not G.nodes[n].get('in_deg')]
    both_nodes = [n for n in targets if G.nodes[n].get('in_deg') and G.nodes[n].get('in_ind')]
    if both_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=both_nodes, node_color='#F39C12',
                               node_size=400, ax=ax, label='DEG ∩ indianred')
    if deg_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=deg_nodes, node_color='#3498DB',
                               node_size=350, ax=ax, label='DEG only')
    if ind_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=ind_nodes, node_color='#95A5A6',
                               node_size=300, ax=ax, label='indianred only')
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.legend(loc='upper left', fontsize=7, frameon=True)
    ax.set_title('HIF-1α regulatory subnetwork in EOPE placenta')
    ax.axis('off')
    fig.tight_layout()
    for ext in ['png', 'pdf', 'tiff']:
        fig.savefig(rf"{FIG}\fig5B_hif1a_network.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  fig5B_hif1a_network saved")

# Fig S11A: indianred hub genes kME vs GS scatter
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(indianred['kME'], indianred['GS'], c='#BDC3C7', s=10, alpha=0.5, label='module genes')
ax.scatter(hubs['kME'], hubs['GS'], c='#E74C3C', s=25, zorder=5, label=f'hub genes (n={len(hubs)})')
ax.axvline(0.8, ls='--', c='gray', lw=0.5)
ax.axvline(-0.8, ls='--', c='gray', lw=0.5)
ax.axhline(0.5, ls='--', c='gray', lw=0.5)
ax.axhline(-0.5, ls='--', c='gray', lw=0.5)
ax.set_xlabel('Module membership (kME)')
ax.set_ylabel('Gene significance (GS, corr. with EOPE)')
ax.set_title('indianred module: hub gene selection')
ax.legend(loc='best', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS11A_hub_scatter.{ext}", dpi=300, bbox_inches='tight')
plt.close(fig)
print("  figS11A_hub_scatter saved")

# Fig S11B: indianred GO-BP enrichment bubble
if len(enrich_results.get('GO_BP', pd.DataFrame())) > 0:
    bp = enrich_results['GO_BP'].head(15).copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = bp['Overlap'].apply(lambda x: int(x.split('/')[0]))
    ax.scatter(bp['-log10FDR'], range(len(bp)), s=sizes * 3, c=bp['-log10FDR'],
               cmap='Reds', edgecolors='#2C3E50', linewidths=0.5)
    ax.set_yticks(range(len(bp)))
    ax.set_yticklabels(bp['Term'].str[:50], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('$-\log_{10}$(FDR)')
    ax.set_title('indianred module: GO Biological Process enrichment')
    # Size legend
    for s in [10, 30, 50]:
        ax.scatter([], [], s=s * 3, c='white', edgecolors='#2C3E50', label=f'{s} genes')
    ax.legend(loc='lower right', fontsize=7, title='overlap', title_fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(rf"{FIG}\figS11B_gobp_bubble.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  figS11B_gobp_bubble saved")

# Fig S11C: module-trait correlation heatmap (indianred highlighted)
fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(cor))))
im = ax.imshow(cor.astype(float).values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(traits.shape[1])); ax.set_xticklabels(traits.columns, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(cor))); ax.set_yticklabels(cor.index, fontsize=7)
# Highlight indianred row
ind_idx = list(cor.index).index('MEindianred')
ax.add_patch(plt.Rectangle((-0.5, ind_idx - 0.5), traits.shape[1], 1, fill=False,
                           edgecolor='#E74C3C', lw=2))
for i in range(len(cor)):
    for j in range(traits.shape[1]):
        r = cor.iloc[i, j]
        if pd.notna(r):
            ax.text(j, i, f"{r:.2f}", ha='center', va='center', fontsize=6,
                    color='white' if abs(r) > 0.55 else 'black')
cb = fig.colorbar(im, ax=ax, shrink=.6); cb.set_label('Pearson r', fontsize=8)
ax.set_title('Module–trait relationships (indianred highlighted)')
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS11C_module_trait.{ext}", dpi=300, bbox_inches='tight')
plt.close(fig)
print("  figS11C_module_trait saved")

# ── 9. Summary ────────────────────────────────────────────────────────────
hif_in_deg_logfc = deg[deg['gene_symbol'].isin(hif_in_deg)][['gene_symbol', 'log2FC', 'qvalue']].sort_values('qvalue')
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"indianred module genes    : {len(indianred)}")
print(f"hub genes (kME>0.8 & |GS|>0.5): {len(hubs)}")
print(f"TF enrichment databases   : {len(tf_results)}")
top_tf_name = top_tf.iloc[0]['TF'] if len(top_tf) > 0 else 'N/A'
print(f"Top enriched TF           : {top_tf_name}")
print(f"HIF1A target genes found  : {len(hif_targets)}")
print(f"HIF targets in DEGs       : {len(hif_in_deg)} (hypergeom p={p_hyper:.2e})")
if len(hif_in_deg_logfc) > 0:
    print("HIF targets in DEGs (top by FDR):")
    print(hif_in_deg_logfc.head(10).to_string(index=False))

summary = {
    'indianred_n_genes': int(len(indianred)),
    'indianred_n_hubs': int(len(hubs)),
    'top_tf': str(top_tf_name),
    'hif1a_targets_total': int(len(hif_targets)),
    'hif1a_targets_in_deg': int(len(hif_in_deg)),
    'hif1a_targets_in_indianred': int(len(hif_in_indianred)),
    'hif1a_overlap_hypergeom_p': float(p_hyper),
    'top_tf_list': top_tf['TF'].head(10).tolist(),
}
json.dump(summary, open(rf"{RES}\moduleB_wgcna_tf_summary.json", 'w'), indent=2)
print("\nAll results saved to results/ and figures/")
