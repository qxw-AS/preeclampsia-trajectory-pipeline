"""
Module B Step 2 - WGCNA (skill: bulk-wgcna-analysis-with-omicverse)

Placental co-expression network in the 84-sample GA-matched GSE75010 cohort.
Top 5,000 most variable genes (MAD), signed-hybrid network, module-trait
correlation against EOPE status / gestational age / clinical traits.
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels import robust

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
DATA = rf"{OUT}\data\geo\gse75010\GSE75010_data"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'figure.dpi': 300})

print("=" * 78)
print("STEP 2  WGCNA of the placental transcriptome (GSE75010, n=84)")
print("=" * 78)

expr = pd.read_csv(rf"{DATA}\GSE75010_gene_matrix.csv.gz", index_col=0)
meta = pd.read_csv(rf"{DATA}\GSE75010_sample_metadata.csv").set_index('sample_id')
train = pd.read_csv(rf"{DATA}\GSE75010_training_labels.csv")
samples = train['sample_id'].tolist()

expr = expr[samples]
keep = (expr > 4).sum(axis=1) >= 0.5 * expr.shape[1]
expr = expr.loc[keep]
print(f"expressed genes: {expr.shape[0]:,}")

gene_mad = expr.apply(robust.mad, axis=1)
top_genes = gene_mad.sort_values(ascending=False).index[:5000]
dat = expr.loc[top_genes].T          # samples x genes
print(f"MAD-filtered genes for network: {dat.shape[1]:,}; samples: {dat.shape[0]}")

assert dat.shape[1] >= 1000, "too few genes for WGCNA"
assert dat.shape[0] >= 6, "too few samples for WGCNA"
assert dat.dtypes.apply(lambda d: d.kind in 'iuf').all()

# ── numeric traits ───────────────────────────────────────────────────────
lab = train.set_index('sample_id')
traits = pd.DataFrame(index=samples)
traits['EOPE'] = (lab.loc[samples, 'group'] == 'EOPE').astype(int)
traits['GA_weeks'] = lab.loc[samples, 'ga (week)'].astype(float)
traits['Maternal_age'] = lab.loc[samples, 'maternal age'].astype(float)
traits['Max_systolic_BP'] = lab.loc[samples, 'maximum systolic bp'].astype(float)
traits['Max_diastolic_BP'] = lab.loc[samples, 'maximum diastolic bp'].astype(float)
traits['Newborn_weight_z'] = lab.loc[samples, 'newborn weight z-score'].astype(float)
traits['Placental_weight_z'] = lab.loc[samples, 'placental weight z-score'].astype(float)
print(f"traits: {list(traits.columns)}")

# ── network ──────────────────────────────────────────────────────────────
from omicverse.bulk._wgcna import pyWGCNA

wgcna = pyWGCNA(
    name='GSE75010_EOPE',
    species='homo sapiens',
    geneExp=dat,
    TPMcutoff=0,               # data are log2 RMA, not TPM
    powers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    RsquaredCut=0.85,
    networkType='signed hybrid',
    TOMType='signed',
    minModuleSize=30,
    MEDissThres=0.25,
    save=False,
)
print("object created; running preprocess()")
wgcna.preprocess()

wgcna.calculate_soft_threshold()
print(f"\nselected soft-threshold power = {wgcna.power}")

wgcna.calculating_adjacency_matrix()
wgcna.calculating_TOM_similarity_matrix()
wgcna.calculate_geneTree()
wgcna.calculate_dynamicMods()
wgcna.calculate_gene_module()

mod_colors = pd.Series(wgcna.datExpr.var['moduleColors'].values, index=wgcna.datExpr.var_names)
mod_colors.name = 'module'
print("\nmodule sizes:")
print(mod_colors.value_counts().to_string())

# ── module eigengenes + trait correlation ────────────────────────────────
MEs = wgcna.MEs.copy()
MEs.index = [str(i) for i in MEs.index]
common = [s for s in samples if s in MEs.index]
MEs = MEs.loc[common]
tr = traits.loc[common]

cor = pd.DataFrame(index=MEs.columns, columns=tr.columns, dtype=float)
pval = cor.copy()
for m in MEs.columns:
    for t in tr.columns:
        x, y = MEs[m].astype(float), tr[t].astype(float)
        okm = x.notna() & y.notna()
        if okm.sum() > 5 and x[okm].std() > 0 and y[okm].std() > 0:
            r, p = stats.pearsonr(x[okm], y[okm])
            cor.loc[m, t], pval.loc[m, t] = r, p

cor.to_csv(rf"{RES}\moduleB_wgcna_module_trait_r.csv")
pval.to_csv(rf"{RES}\moduleB_wgcna_module_trait_p.csv")
mod_colors.to_csv(rf"{RES}\moduleB_wgcna_module_assignment.csv")

eope_cor = cor['EOPE'].astype(float).sort_values()
print("\nmodule-EOPE correlation:")
for m, r in eope_cor.items():
    print(f"  {m:20s} r={r:+.3f}  p={float(pval.loc[m,'EOPE']):.2e}  n={int((mod_colors==m).sum())}")

# ── hub genes (kME) ──────────────────────────────────────────────────────
kme = {}
for m in MEs.columns:
    genes = mod_colors[mod_colors == m].index
    if len(genes) < 3:
        continue
    sub = wgcna.datExpr.to_df()[genes]
    k = sub.apply(lambda col: stats.pearsonr(col, MEs[m].astype(float))[0])
    kme[m] = k.sort_values(ascending=False)
hub = pd.DataFrame({m: s.index[:50] for m, s in kme.items()}).to_csv(rf"{RES}\moduleB_wgcna_hub_genes.csv")

# ── figures ──────────────────────────────────────────────────────────────
# 1) scale-free fit
sft = wgcna.sft.copy()
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].plot(sft['Power'], -np.sign(sft['slope']) * sft['SFT.R.sq'], 'o-', color='#2C3E50', ms=4)
ax[0].axhline(0.85, color='#E74C3C', ls='--', lw=.8)
ax[0].set_xlabel('Soft-threshold power'); ax[0].set_ylabel('Scale-free topology $R^2$')
ax[0].set_title('Scale independence', fontsize=9)
ax[1].plot(sft['Power'], sft['mean(k)'], 'o-', color='#2C3E50', ms=4)
ax[1].set_xlabel('Soft-threshold power'); ax[1].set_ylabel('Mean connectivity')
ax[1].set_title('Mean connectivity', fontsize=9)
for a in ax:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS5_wgcna_soft_threshold.{ext}", dpi=300, bbox_inches='tight')
plt.close(fig)

# 2) module-trait heatmap
fig, ax = plt.subplots(figsize=(5.2, max(2.6, 0.32 * len(cor))))
im = ax.imshow(cor.astype(float).values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(tr.shape[1])); ax.set_xticklabels(tr.columns, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(cor))); ax.set_yticklabels(cor.index, fontsize=7)
for i in range(len(cor)):
    for j in range(tr.shape[1]):
        r = cor.iloc[i, j]
        if pd.notna(r):
            ax.text(j, i, f"{r:.2f}", ha='center', va='center', fontsize=6,
                    color='white' if abs(r) > 0.55 else 'black')
cb = fig.colorbar(im, ax=ax, shrink=.6); cb.set_label('Pearson r', fontsize=8)
ax.set_title('Module–trait relationships', fontsize=9)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS6_wgcna_module_trait.{ext}", dpi=300, bbox_inches='tight')
plt.close(fig)

# ── verification ─────────────────────────────────────────────────────────
print("\nLayer 4 verification")
print(f"  genes in network      : {dat.shape[1]:,}")
print(f"  samples               : {dat.shape[0]}")
print(f"  modules (excl. grey)  : {len([m for m in mod_colors.unique() if m != 'grey'])}")
print(f"  soft power            : {wgcna.power}")
print(f"  strongest EOPE module : {eope_cor.index[-1]} (r={eope_cor.iloc[-1]:+.3f})")

summary = {
    'n_samples': int(dat.shape[0]), 'n_genes_network': int(dat.shape[1]),
    'soft_power': int(wgcna.power),
    'modules': {m: int(c) for m, c in mod_colors.value_counts().items()},
    'best_eope_module': str(eope_cor.index[-1]),
    'best_eope_r': float(eope_cor.iloc[-1]),
    'best_eope_p': float(pval.loc[eope_cor.index[-1], 'EOPE']),
}
json.dump(summary, open(rf"{RES}\moduleB_wgcna_summary.json", 'w'), indent=2)
print("\nSaved WGCNA tables and figures figS5/figS6")
