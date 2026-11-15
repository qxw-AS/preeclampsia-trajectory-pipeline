"""
Module B Step 5 - External validation in GSE204835 + cross-platform harmonisation
(skill: bulk-rna-seq-batch-correction-with-combat)

GSE204835: FFPE placental QuantSeq 3'-mRNA-seq, hg38, Entrez gene IDs.
Groups: PE (n=16), Control (n=12), Chronic hypertension (n=13),
        Superimposed PE (n=7).

Analysis
--------
1. Entrez -> HGNC symbol mapping (mygene).
2. CPM + log2 normalisation; ssGSEA of the same seven pathway sets.
3. Direction-of-effect concordance with the GSE75010 discovery cohort.
4. ComBat harmonisation of the two platforms on shared genes (z-scored),
   visualised by PCA before/after correction (supplementary QC figure).
"""
import os, re, json, gzip, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import anndata as ad
import omicverse as ov

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"
GS = rf"{OUT}\genesets"

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'figure.dpi': 300})

print("=" * 78)
print("STEP 5  Validation cohort GSE204835 + ComBat harmonisation")
print("=" * 78)

# ── 1. series matrix annotation ──────────────────────────────────────────
lines = gzip.open(rf"{OUT}\data\geo\GSE204835_series_matrix.txt.gz", 'rt',
                  encoding='utf-8', errors='ignore').read().split('\n')
attrs = {}
for l in lines:
    if l.startswith('!Sample_'):
        f = [x.strip('"') for x in l.split('\t')]
        attrs.setdefault(f[0], []).append(f[1:])

gsm = attrs['!Sample_geo_accession'][0]
desc = attrs['!Sample_description'][0]
char_rows = attrs['!Sample_characteristics_ch1']
def char(name):
    for row in char_rows:
        if row[0].lower().startswith(name):
            return [v.split(':', 1)[1].strip() for v in row]
    return None

meta = pd.DataFrame({'gsm': gsm, 'col': desc,
                     'ga': char('gestational age'),
                     'disease': char('disease state')})
meta['disease'] = meta['disease'].str.replace('Supermposed', 'Superimposed', regex=False)
print(meta['disease'].value_counts().to_string())
meta.to_csv(rf"{RES}\moduleB_gse204835_metadata.csv", index=False)

cnt = pd.read_csv(rf"{OUT}\data\geo\GSE204835_counts.csv.gz", index_col=0)
print(f"counts matrix: {cnt.shape[0]:,} genes x {cnt.shape[1]} samples")
cnt = cnt[[c for c in meta['col'] if c in cnt.columns]]
meta['col'] = meta['col'].astype(str)
meta = meta.set_index('col').loc[[c for c in cnt.columns]].reset_index()

# ── 2. Entrez -> symbol ──────────────────────────────────────────────────
cache = rf"{RES}\moduleB_gse204835_entrez2sym.csv"
if os.path.exists(cache):
    mp = pd.read_csv(cache)
else:
    import mygene
    mg = mygene.MyGeneInfo()
    ids = [str(i).split('.')[0] for i in cnt.index]
    q = mg.querymany(ids, scopes='entrezgene', fields='symbol', species='human',
                     as_dataframe=True, verbose=False)
    mp = q.reset_index().rename(columns={'query': 'entrez', 'symbol': 'symbol'})
    mp = mp.dropna(subset=['symbol'])[['entrez', 'symbol']]
    mp.to_csv(cache, index=False)
print(f"mapped Entrez -> symbol: {len(mp):,}")

cnt.index = [str(i).split('.')[0] for i in cnt.index]
mp['entrez'] = mp['entrez'].astype(str).str.split('.').str[0]
mp = mp.drop_duplicates('entrez')
cnt = cnt.loc[cnt.index.isin(mp['entrez'])]
sym = cnt.index.map(mp.set_index('entrez')['symbol'])
df = cnt.copy(); df.index = sym
df = df.groupby(level=0).max()
print(f"symbol-level matrix: {df.shape}")

# ── 3. CPM-log2 + ssGSEA ─────────────────────────────────────────────────
cpm = df / df.sum(axis=0) * 1e6
cpm = cpm.loc[(cpm > 1).sum(axis=1) >= 0.3 * cpm.shape[1]]
logcpm = np.log2(cpm + 1)
print(f"expressed genes after CPM filter: {logcpm.shape[0]:,}")


def load_gmt(path):
    d = {}
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if len(f) > 2:
            d[f[0]] = sorted({g.strip().upper() for g in f[2:] if g.strip()})
    return d

kegg = load_gmt(rf"{GS}\KEGG_2021_Human.gmt")
hallmark = load_gmt(rf"{GS}\MSigDB_Hallmark_2020.gmt")
PATHWAYS = {
    'Platelet activation':                kegg['Platelet activation'],
    'Complement & coagulation cascades':  kegg['Complement and coagulation cascades'],
    'Angiogenesis':                       hallmark['Angiogenesis'],
    'Inflammatory response':              hallmark['Inflammatory Response'],
    'Hypoxia':                            hallmark['Hypoxia'],
    'PPAR signalling (lipid metabolism)': kegg['PPAR signaling pathway'],
    'Apoptosis':                          kegg['Apoptosis'],
}

import gseapy
ss = gseapy.ssgsea(data=logcpm, gene_sets=PATHWAYS, outdir=None,
                   sample_norm_method='rank', min_size=15, max_size=500,
                   permutation_num=0, no_plot=True, threads=4, seed=123)
scores = ss.res2d.pivot(index='Name', columns='Term', values='ES').loc[:, list(PATHWAYS)]
scores.index.name = 'col'
scores = scores.loc[cnt.columns]
scores.to_csv(rf"{RES}\moduleB_gse204835_ssgsea_scores.csv")

meta_i = meta.set_index('col').loc[scores.index]
# GA-balanced primary contrast: term PE vs term control (12 vs 12)
term = meta_i['ga'] == 'Term'
pe = meta_i.index[(meta_i['disease'] == 'Preeclampsia') & term]
ctl = meta_i.index[(meta_i['disease'] == 'Control') & term]
pe_all = meta_i.index[meta_i['disease'] == 'Preeclampsia']
ctl_all = meta_i.index[meta_i['disease'] == 'Control']
print(f"\nvalidation contrast (GA-matched, term): PE n={len(pe)} vs Control n={len(ctl)}")
print(f"sensitivity contrast (all GA)        : PE n={len(pe_all)} vs Control n={len(ctl_all)}")

disc = pd.read_csv(rf"{RES}\moduleB_ssgsea_group_stats.csv")
disc = disc.set_index('pathway')


def compare(a_idx, b_idx, label):
    rows = []
    for p in PATHWAYS:
        a, b = scores.loc[a_idx, p].astype(float), scores.loc[b_idx, p].astype(float)
        u, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
        rows.append({'pathway': p, 'contrast': label,
                     'mean_case': a.mean(), 'mean_control': b.mean(),
                     'delta_validation': a.mean() - b.mean(),
                     'rank_biserial': 2 * u / (len(a) * len(b)) - 1,
                     'p_value': pu,
                     'delta_discovery': disc.loc[p, 'delta']})
    out = pd.DataFrame(rows)
    out['fdr'] = multipletests(out['p_value'], method='fdr_bh')[1]
    out['direction_concordant'] = np.sign(out['delta_validation']) == np.sign(out['delta_discovery'])
    return out


val = compare(pe, ctl, 'term_PE_vs_term_control')
val_all = compare(pe_all, ctl_all, 'all_PE_vs_control')
pd.concat([val, val_all]).to_csv(rf"{RES}\moduleB_validation_concordance.csv", index=False)

print("\n-- pathway direction concordance (GA-matched term contrast) --")
print(val[['pathway', 'delta_discovery', 'delta_validation', 'p_value', 'fdr',
           'direction_concordant']].to_string(index=False))
print(f"\nconcordant pathways: {int(val['direction_concordant'].sum())}/{len(val)}")

# ── gene-level direction concordance for the 196 DEGs ───────────────────
deg = pd.read_csv(rf"{RES}\moduleB_deg_significant.csv", index_col=0)
gl = [g for g in deg.index if g in logcpm.index]
a = logcpm.loc[gl, pe].astype(float); b = logcpm.loc[gl, ctl].astype(float)
uu, pp = stats.mannwhitneyu(a.values, b.values, axis=1, alternative='two-sided')
gene_val = pd.DataFrame({'gene': gl,
                         'log2FC_discovery': deg.loc[gl, 'log2FC'].values,
                         'delta_validation': (a.mean(1) - b.mean(1)).values,
                         'p_validation': pp})
gene_val['same_direction'] = np.sign(gene_val['log2FC_discovery']) == np.sign(gene_val['delta_validation'])
gene_val['fdr_validation'] = multipletests(gene_val['p_validation'], method='fdr_bh')[1]
gene_val = gene_val.sort_values('p_validation')
gene_val.to_csv(rf"{RES}\moduleB_validation_gene_level.csv", index=False)
rho, rho_p = stats.spearmanr(gene_val['log2FC_discovery'], gene_val['delta_validation'])
print(f"\ngene-level: {len(gene_val)} DEGs testable in validation; "
      f"same direction {gene_val['same_direction'].mean()*100:.1f}%; "
      f"Spearman rho={rho:.3f} (p={rho_p:.2e}); "
      f"FDR<0.05 in validation: {(gene_val['fdr_validation']<0.05).sum()}")

# ── 4. ComBat harmonisation across platforms ─────────────────────────────
disc_expr = pd.read_csv(rf"{OUT}\data\geo\gse75010\GSE75010_data\GSE75010_gene_matrix.csv.gz",
                        index_col=0)
train = pd.read_csv(rf"{OUT}\data\geo\gse75010\GSE75010_data\GSE75010_training_labels.csv")
disc_expr = disc_expr[train['sample_id'].tolist()]

shared = sorted(set(disc_expr.index) & set(logcpm.index))
print(f"\nshared genes across platforms: {len(shared):,}")
X1 = disc_expr.loc[shared]
X2 = logcpm.loc[shared]
z1 = X1.sub(X1.mean(1), axis=0).div(X1.std(1).replace(0, np.nan), axis=0)
z2 = X2.sub(X2.mean(1), axis=0).div(X2.std(1).replace(0, np.nan), axis=0)
z1, z2 = z1.dropna(), z2.dropna()
shared = sorted(set(z1.index) & set(z2.index))
z1, z2 = z1.loc[shared], z2.loc[shared]

comb = pd.concat([z1, z2], axis=1).T.fillna(0)         # samples x genes
adata = ad.AnnData(X=comb.values,
                   obs=pd.DataFrame({'dataset': ['GSE75010'] * z1.shape[1] + ['GSE204835'] * z2.shape[1],
                                     'group': list(train.set_index('sample_id').loc[z1.columns, 'group']) +
                                              list(meta_i.loc[z2.columns, 'disease'])},
                                    index=comb.index),
                   var=pd.DataFrame(index=shared))
adata.obsm['X_raw'] = adata.X.copy()
ov.bulk.batch_correction(adata, batch_key='dataset', key_added='combat')
adata.obsm['X_combat'] = adata.layers['combat'].copy()
print("ComBat finished")

def pca2(X):
    from sklearn.decomposition import PCA
    p = PCA(n_components=2, random_state=0).fit_transform(np.asarray(X))
    v = PCA(n_components=2, random_state=0).fit(np.asarray(X)).explained_variance_ratio_
    return p, v

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
for ax, key, ttl in [(axes[0], 'X_raw', 'Before ComBat'), (axes[1], 'X_combat', 'After ComBat')]:
    p, v = pca2(adata.obsm[key])
    for ds, c in [('GSE75010', '#2C3E50'), ('GSE204835', '#E74C3C')]:
        m = (adata.obs['dataset'] == ds).values
        ax.scatter(p[m, 0], p[m, 1], s=22, c=c, alpha=.8, label=ds, linewidths=0)
    ax.set_xlabel(f'PC1 ({v[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({v[1]*100:.1f}%)')
    ax.set_title(ttl, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle('Cross-platform harmonisation of shared placental transcripts', fontsize=10, y=1.02)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS8_combat_pca.{ext}", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

summary = {
    'n_samples_validation': int(cnt.shape[1]),
    'groups': meta['disease'].value_counts().to_dict(),
    'n_shared_genes': len(shared),
    'concordant_pathways': int(val['direction_concordant'].sum()),
    'n_pathways': len(val),
    'gene_same_direction_pct': float(gene_val['same_direction'].mean() * 100),
    'gene_level_spearman_rho': float(rho),
    'gene_fdr05_validation': int((gene_val['fdr_validation'] < 0.05).sum()),
}
json.dump(summary, open(rf"{RES}\moduleB_validation_summary.json", 'w'), indent=2, default=str)
print("\nSaved validation tables and figS8_combat_pca")
