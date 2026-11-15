"""
Module B Step 3 - GSEA and ssGSEA (skill: gsea-enrichment-analysis)

- Pre-ranked GSEA (KEGG_2021_Human and MSigDB_Hallmark_2020) on the
  EOPE-vs-control log2 fold-change ranking of all expressed genes.
- ssGSEA per sample for seven a-priori pathways spanning the
  platelet / coagulation / angiogenic / inflammatory / hypoxia /
  lipid / apoptotic axes.
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
DATA = rf"{OUT}\data\geo\gse75010\GSE75010_data"
RES = rf"{OUT}\results"
GS = rf"{OUT}\genesets"
os.makedirs(RES, exist_ok=True)

print("=" * 78)
print("STEP 3  GSEA / ssGSEA (GSE75010, EOPE vs control)")
print("=" * 78)


# ── helpers ──────────────────────────────────────────────────────────────
def ov_gsea(rnk_df, gs_dict):
    import omicverse as ov
    res = ov.bulk.geneset_enrichment_GSEA(
        gene_rnk=rnk_df.set_index('gene')['score'].to_frame(),
        pathways_dict=gs_dict,
        permutation_num=1000, outdir=None, organism='Human',
        min_size=15, max_size=500, progress=False)
    df = res.res2d.reset_index()
    df = df.rename(columns={'Term': 'Term', 'term': 'Term', 'es': 'ES', 'nes': 'NES',
                            'pval': 'NOM p-val', 'fdr': 'FDR q-val',
                            'geneset_size': 'Size', 'matched_size': 'Matched',
                            'lead_genes': 'Lead_genes'})
    return df


def gseapy_ssgsea(expr_df, sets):
    import gseapy
    return gseapy.ssgsea(data=expr_df, gene_sets=sets, outdir=None,
                         sample_norm_method='rank', min_size=15, max_size=500,
                         permutation_num=0, no_plot=True, threads=4, seed=123)


def multipletests_bh(p):
    from statsmodels.stats.multitest import multipletests
    return multipletests(p, method='fdr_bh')[1]

expr = pd.read_csv(rf"{DATA}\GSE75010_gene_matrix.csv.gz", index_col=0)
train = pd.read_csv(rf"{DATA}\GSE75010_training_labels.csv")
deg = pd.read_csv(rf"{RES}\moduleB_deg_all.csv", index_col=0)

samples = train['sample_id'].tolist()
eope = train.loc[train['group'] == 'EOPE', 'sample_id'].tolist()
ctrl = train.loc[train['group'] == 'Control', 'sample_id'].tolist()
expr = expr[samples]
keep = (expr > 4).sum(axis=1) >= 0.5 * expr.shape[1]
expr = expr.loc[keep]

# ══════════════════════════════════════════════════════════════════════════
# 1. Pre-ranked GSEA
# ══════════════════════════════════════════════════════════════════════════
def load_gmt(path):
    d = {}
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if len(f) > 2:
            genes = [g.strip().upper() for g in f[2:] if g.strip()]
            d[f[0]] = sorted(set(genes))
    return d

kegg = load_gmt(rf"{GS}\KEGG_2021_Human.gmt")
hallmark = load_gmt(rf"{GS}\MSigDB_Hallmark_2020.gmt")
print(f"KEGG_2021_Human: {len(kegg)} sets | MSigDB_Hallmark_2020: {len(hallmark)} sets")

rnk = deg['log2FC'].dropna().sort_values(ascending=False)
rnk_df = pd.DataFrame({'gene': rnk.index, 'score': rnk.values})

gsea_out = {}
for name, gs in [('KEGG', kegg), ('Hallmark', hallmark)]:
    print(f"\n-- pre-ranked GSEA [{name}] --")
    res = ov_gsea(rnk_df, gs)
    gsea_out[name] = res
    res.to_csv(rf"{RES}\moduleB_gsea_{name.lower()}.csv", index=False)
    sig = res[res['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
    print(f"   sets with FDR<0.25: {len(sig)} (up {int((sig.NES>0).sum())}, down {int((sig.NES<0).sum())})")
    print(sig.head(6)[['Term', 'NES', 'FDR q-val']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 2. ssGSEA - seven a-priori pathways
# ══════════════════════════════════════════════════════════════════════════
PATHWAYS = {
    'Platelet activation':                 kegg['Platelet activation'],
    'Complement & coagulation cascades':   kegg['Complement and coagulation cascades'],
    'Angiogenesis':                        hallmark['Angiogenesis'],
    'Inflammatory response':               hallmark['Inflammatory Response'],
    'Hypoxia':                             hallmark['Hypoxia'],
    'PPAR signalling (lipid metabolism)':  kegg['PPAR signaling pathway'],
    'Apoptosis':                           kegg['Apoptosis'],
}
print("\n-- ssGSEA pathway sizes --")
for k, v in PATHWAYS.items():
    print(f"   {k:38s} {len(v):4d} genes")

ss = gseapy_ssgsea(expr, PATHWAYS)
scores = ss.res2d.pivot(index='Name', columns='Term', values='ES').loc[:, list(PATHWAYS)]
scores.index.name = 'sample_id'
scores = scores.loc[[s for s in samples if s in scores.index]]
scores.to_csv(rf"{RES}\moduleB_ssgsea_pathway_scores.csv")

# group statistics
rows = []
for p in PATHWAYS:
    a = scores.loc[eope, p].astype(float)
    b = scores.loc[ctrl, p].astype(float)
    u, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
    # rank-biserial effect size
    rb = 2 * u / (len(a) * len(b)) - 1
    rows.append({'pathway': p, 'n_eope': len(a), 'n_control': len(b),
                 'mean_eope': a.mean(), 'sd_eope': a.std(),
                 'mean_control': b.mean(), 'sd_control': b.std(),
                 'delta': a.mean() - b.mean(), 'rank_biserial': rb,
                 'p_value': pu})
stats_df = pd.DataFrame(rows)
stats_df['fdr'] = multipletests_bh(stats_df['p_value'])
stats_df.to_csv(rf"{RES}\moduleB_ssgsea_group_stats.csv", index=False)
print("\n-- ssGSEA group comparison (EOPE vs control) --")
print(stats_df[['pathway', 'delta', 'rank_biserial', 'p_value', 'fdr']].to_string(index=False))

summary = {
    'n_pathways': len(PATHWAYS),
    'pathway_sizes': {k: len(v) for k, v in PATHWAYS.items()},
    'n_significant_fdr05': int((stats_df['fdr'] < 0.05).sum()),
    'gsea_kegg_fdr25': int((gsea_out['KEGG']['FDR q-val'] < 0.25).sum()),
    'gsea_hallmark_fdr25': int((gsea_out['Hallmark']['FDR q-val'] < 0.25).sum()),
}
json.dump(summary, open(rf"{RES}\moduleB_gsea_summary.json", 'w'), indent=2)
print("\nSaved ssGSEA scores, group statistics and GSEA result tables")
