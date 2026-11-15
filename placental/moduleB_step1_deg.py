"""
Module B Step 1 - Differential expression (skill: bulk-rna-seq-differential-expression-with-omicverse)

Data : GSE75010 (GPL6244, Affymetrix Human Gene 1.0 ST), log2 RMA
Cohort: 84 GA-matched placenta samples (EOPE n=49 vs preterm control n=35, GA 25-33 wk)

Notes
-----
- Input is already RMA-normalised log2 intensity, therefore DESeq2 size-factor
  normalisation (dds.normalize()) is NOT applied; this is documented in Methods.
- Welch t-test (pyDEG method='ttest') is performed on log2 intensities.
- log2 fold change is recomputed as the difference of group means on the log2
  scale (standard for one-colour microarray) because pyDEG's ttest branch
  derives fold change from a pseudo-count ratio that is not valid for log2 data.
- Significance: FDR (Benjamini-Hochberg) < 0.05 and |log2FC| > 0.5.
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import omicverse as ov

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
DATA = rf"{OUT}\data\geo\gse75010\GSE75010_data"
RES = rf"{OUT}\results"
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'figure.dpi': 300})

# ── 1. Load ──────────────────────────────────────────────────────────────
print("=" * 78)
print("STEP 1  Differential expression: EOPE vs preterm control (GSE75010)")
print("=" * 78)

expr = pd.read_csv(rf"{DATA}\GSE75010_gene_matrix.csv.gz", index_col=0)
meta = pd.read_csv(rf"{DATA}\GSE75010_sample_metadata.csv")
train = pd.read_csv(rf"{DATA}\GSE75010_training_labels.csv")

print(f"gene-level matrix : {expr.shape[0]:,} genes x {expr.shape[1]} samples")
print(f"training cohort   : {train.shape[0]} samples "
      f"(EOPE {int((train.group=='EOPE').sum())} / Control {int((train.group=='Control').sum())})")

cols = [c for c in train['sample_id'] if c in expr.columns]
assert len(cols) == train.shape[0], "sample id mismatch between labels and matrix"
expr = expr[cols]

# ── 2. Filter low-expressed probes/genes ─────────────────────────────────
# log2 RMA background is ~3.5-4.5; require expression above background in >=50% samples
keep = (expr > 4).sum(axis=1) >= 0.5 * expr.shape[1]
expr_f = expr.loc[keep]
print(f"expressed genes   : {expr_f.shape[0]:,} (max > 4 in >=50% of samples)")

# ── 3. pyDEG -------------------------------------------------------------
dds = ov.bulk.pyDEG(expr_f)
dds.drop_duplicates_index()

eope = train.loc[train['group'] == 'EOPE', 'sample_id'].tolist()
ctrl = train.loc[train['group'] == 'Control', 'sample_id'].tolist()

all_cols = set(dds.data.columns)
for g in eope + ctrl:
    assert g in all_cols, f"Sample {g} not in matrix"
assert not set(eope) & set(ctrl), "groups overlap"

res = dds.deg_analysis(eope, ctrl, method='ttest', alpha=0.05,
                       multipletests_method='fdr_bh')

# ── 4. Recompute log2FC on log2 scale ────────────────────────────────────
m1 = dds.data[eope].mean(axis=1)
m2 = dds.data[ctrl].mean(axis=1)
res['mean_EOPE'] = m1
res['mean_Control'] = m2
res['log2FC'] = m1 - m2
res['FoldChange'] = 2 ** res['log2FC']
res['abs(log2FC)'] = res['log2FC'].abs()
res['BaseMean'] = (m1 + m2) / 2
res = res.loc[res['mean_EOPE'].notna() & res['mean_Control'].notna()]

FC_CUT, FDR_CUT = 0.5, 0.05
res['sig'] = 'normal'
res.loc[(res['qvalue'] < FDR_CUT) & (res['log2FC'] >= FC_CUT), 'sig'] = 'up'
res.loc[(res['qvalue'] < FDR_CUT) & (res['log2FC'] <= -FC_CUT), 'sig'] = 'down'
dds.result = res

n_up = int((res['sig'] == 'up').sum())
n_dn = int((res['sig'] == 'down').sum())
print(f"\nDEG (FDR<{FDR_CUT}, |log2FC|>{FC_CUT}): up {n_up}, down {n_dn}, total {n_up + n_dn}")

# ── 5. Save tables ───────────────────────────────────────────────────────
res_sorted = res.sort_values('qvalue')
res_sorted.to_csv(rf"{RES}\moduleB_deg_all.csv")
sig = res_sorted[res_sorted['sig'] != 'normal']
sig.to_csv(rf"{RES}\moduleB_deg_significant.csv")

rank = res_sorted[['log2FC', 'pvalue', 'qvalue']].copy()
rank['rank_metric'] = rank['log2FC'] * (-np.log10(rank['pvalue'].clip(lower=1e-300)))
rank.sort_values('log2FC').to_csv(rf"{RES}\moduleB_deg_ranked_for_gsea.csv")

print(f"top up   : {', '.join(sig[sig.sig=='up'].head(10).index)}")
print(f"top down : {', '.join(sig[sig.sig=='down'].head(10).index)}")

# ── 6. Volcano plot (preliminary; journal figure regenerated later) ───────
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ns = res[res['sig'] == 'normal']
ax.scatter(ns['log2FC'], -np.log10(ns['qvalue']), s=3, c='#BDBDBD', alpha=.6, linewidths=0)
ax.scatter(res.loc[res.sig == 'up', 'log2FC'], -np.log10(res.loc[res.sig == 'up', 'qvalue']),
           s=4, c='#C44E52', alpha=.85, linewidths=0, label=f'Up ({n_up})')
ax.scatter(res.loc[res.sig == 'down', 'log2FC'], -np.log10(res.loc[res.sig == 'down', 'qvalue']),
           s=4, c='#4C72B0', alpha=.85, linewidths=0, label=f'Down ({n_dn})')
ax.axhline(-np.log10(0.05), color='#999999', ls='--', lw=.7)
ax.axvline(FC_CUT, color='#999999', ls='--', lw=.7)
ax.axvline(-FC_CUT, color='#999999', ls='--', lw=.7)
lab = list(sig[sig.sig == 'up'].head(5).index) + list(sig[sig.sig == 'down'].head(5).index)
for g in lab:
    if g in res.index:
        ax.annotate(g, (res.loc[g, 'log2FC'], -np.log10(res.loc[g, 'qvalue'])),
                    fontsize=6, xytext=(2, 2), textcoords='offset points')
ax.set_xlabel('log$_2$ fold change (EOPE vs control)')
ax.set_ylabel('-log$_{10}$ FDR')
ax.legend(frameon=False, loc='upper left')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{OUT}\figures\figS4_volcano.{ext}", dpi=300, bbox_inches='tight')
plt.close(fig)

# ── 7. Layer 4 verification ──────────────────────────────────────────────
print("\nLayer 4 verification")
checks = [
    ('samples in DEG', 84, len(eope) + len(ctrl)),
    ('EOPE n', 49, len(eope)),
    ('control n', 35, len(ctrl)),
    ('DEG total', n_up + n_dn, int((res.sig != 'normal').sum())),
    ('expressed genes', expr_f.shape[0], res.shape[0]),
]
ok = True
for label, exp, act in checks:
    st = 'PASS' if exp == act else 'FAIL'
    ok &= (exp == act)
    print(f"  [{st}] {label:22s} expected={exp:<8} actual={act}")
print("  ALL CHECKS PASSED" if ok else "  CHECK FAILED")

summary = {
    'n_samples': int(len(cols)), 'n_eope': len(eope), 'n_control': len(ctrl),
    'genes_tested': int(res.shape[0]), 'n_up': n_up, 'n_down': n_dn,
    'fc_cutoff_log2': FC_CUT, 'fdr_cutoff': FDR_CUT,
}
json.dump(summary, open(rf"{RES}\moduleB_deg_summary.json", 'w'), indent=2)
print("\nSaved: moduleB_deg_all.csv, moduleB_deg_significant.csv, moduleB_deg_ranked_for_gsea.csv")
