"""
Module B Step 7 — Candidate gene prioritization (LASSO + SVM-RFE) + druggability

Input: 196 DEGs as features, EOPE/control labels (GSE75010, n=84)
Methods:
  1. LASSO logistic regression (L1 penalty, CV-selected C)
  2. SVM-RFE (linear SVC, CV-selected n_features)
  3. Intersection = candidate genes
  4. External validation in GSE204835 (direction + FDR)
  5. Druggability (DGIdb API, fallback: curated druggable families)
Output: Fig 7 (LASSO path + candidate heatmap + druggability table)
"""
import os, json, warnings, urllib.request, urllib.parse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
from matplotlib.lines import Line2D

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
DATA = rf"{OUT}\data\geo\gse75010\GSE75010_data"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'figure.dpi': 300})

COLORS = {'primary': '#2C3E50', 'accent': '#E74C3C', 'up': '#E74C3C', 'down': '#2C3E50',
          'neutral': '#BDC3C7'}

print("=" * 78)
print("STEP 7  Candidate gene prioritization (LASSO + SVM-RFE)")
print("=" * 78)

# ── 1. Load data ──────────────────────────────────────────────────────────
expr = pd.read_csv(rf"{DATA}\GSE75010_gene_matrix.csv.gz", index_col=0)
train = pd.read_csv(rf"{DATA}\GSE75010_training_labels.csv")
samples = train['sample_id'].tolist()
deg = pd.read_csv(rf"{RES}\moduleB_deg_significant.csv")
deg_genes = deg['gene_symbol'].tolist()
print(f"DEG features: {len(deg_genes)}; samples: {len(samples)}")

# Feature matrix: genes x samples -> samples x genes
X = expr.loc[[g for g in deg_genes if g in expr.index], samples].T.astype(float)
y = (train.set_index('sample_id').loc[samples, 'group'] == 'EOPE').astype(int).values
print(f"X: {X.shape}, y: {y.sum()} EOPE / {len(y)-y.sum()} control")

# ── 2. LASSO logistic regression ──────────────────────────────────────────
print("\n--- LASSO ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
C_grid = np.logspace(-3, 2, 25)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Repeated 5-fold CV (5 repeats) -> mean AUC and standard error for each C
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
cv_mean, cv_se, cv_nnz = [], [], []
for C in C_grid:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', LogisticRegression(penalty='l1', solver='liblinear',
                                                C=C, max_iter=5000, random_state=42))])
    scores = cross_val_score(pipe, X, y, cv=rcv, scoring='roc_auc')
    cv_mean.append(scores.mean())
    cv_se.append(scores.std(ddof=1) / np.sqrt(len(scores)))
    m_full = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                                max_iter=5000, random_state=42).fit(X_scaled, y)
    cv_nnz.append(int((m_full.coef_[0] != 0).sum()))
cv_mean = np.array(cv_mean)
cv_se = np.array(cv_se)
best_idx = int(np.argmax(cv_mean))
best_C = float(C_grid[best_idx])
best_cv_score = float(cv_mean[best_idx])
# 1-SE rule: strongest penalty (smallest C) with mean AUC within 1 SE of the best
se_threshold = best_cv_score - cv_se[best_idx]
idx_1se = int(np.where(cv_mean >= se_threshold)[0][0])
C_1se = float(C_grid[idx_1se])
print(f"min  C={best_C:.4f}, CV AUC={best_cv_score:.3f} (SE={cv_se[best_idx]:.3f})")
print(f"1-SE C={C_1se:.4f}, CV AUC={cv_mean[idx_1se]:.3f}")
print("\nCV curve near the optimum (within 2 SE of best):")
for i in range(len(C_grid)):
    if cv_mean[i] >= best_cv_score - 2 * cv_se[best_idx]:
        print(f"  C={C_grid[i]:8.4f}  AUC={cv_mean[i]:.3f}+/-{cv_se[i]:.3f}  non-zero genes={cv_nnz[i]}")
pd.DataFrame({'C': C_grid, 'mean_auc': cv_mean, 'se_auc': cv_se,
              'n_nonzero_genes': cv_nnz}).to_csv(rf"{RES}\moduleB_lasso_cv_curve.csv", index=False)

# Fit final LASSO at min-C
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=best_C,
                           max_iter=5000, random_state=42)
lasso.fit(X_scaled, y)
coef = pd.Series(lasso.coef_[0], index=X.columns)
lasso_selected = coef[coef != 0].sort_values(key=abs, ascending=False)
print(f"LASSO selected {len(lasso_selected)} genes (non-zero coef)")
print(lasso_selected.head(15).to_string())

# Model at the 1-SE C (more parsimonious) — reported for comparison
lasso_1se = LogisticRegression(penalty='l1', solver='liblinear', C=C_1se,
                               max_iter=5000, random_state=42)
lasso_1se.fit(X_scaled, y)
coef_1se = pd.Series(lasso_1se.coef_[0], index=X.columns)
sel_1se = coef_1se[coef_1se != 0].sort_values(key=abs, ascending=False)
print(f"1-SE model selected {len(sel_1se)} genes: {list(sel_1se.index)}")

# LASSO coefficient path across C values for visualization
coef_path = []
for C in np.logspace(-3, 2, 50):
    m = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=5000, random_state=42)
    m.fit(X_scaled, y)
    coef_path.append(m.coef_[0])
coef_path = np.array(coef_path)  # (n_C, n_genes)
lasso_c_grid = np.logspace(-3, 2, 50)

# ── 3. SVM-RFE ────────────────────────────────────────────────────────────
print("\n--- SVM-RFE ---")
# Find optimal number of features via CV
n_features_grid = [3, 5, 8, 10, 15, 20, 30, 50]
best_rfe_score = -1
best_n_feat = None
for n_feat in n_features_grid:
    try:
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('rfe', RFE(SVC(kernel='linear', C=1.0, random_state=42),
                                     n_features_to_select=n_feat, step=1))])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        if scores.mean() > best_rfe_score:
            best_rfe_score = scores.mean()
            best_n_feat = n_feat
    except Exception as e:
        print(f"  n={n_feat} failed: {e}")
print(f"Best n_features={best_n_feat}, CV AUC={best_rfe_score:.3f}")

# Fit final SVM-RFE
svc = SVC(kernel='linear', C=1.0, random_state=42)
rfe = RFE(svc, n_features_to_select=best_n_feat, step=1)
rfe.fit(X_scaled, y)
svm_selected = pd.Series(X.columns[rfe.support_], index=X.columns[rfe.support_])
print(f"SVM-RFE selected {len(svm_selected)} genes")

# SVM weights for ranking
svc_final = SVC(kernel='linear', C=1.0, random_state=42)
svc_final.fit(X_scaled[:, rfe.support_], y)
svm_weights = pd.Series(svc_final.coef_[0], index=X.columns[rfe.support_]).sort_values(key=abs, ascending=False)
print(svm_weights.head(15).to_string())

# ── 4. Intersection = candidate genes ─────────────────────────────────────
lasso_set = set(lasso_selected.index)
svm_set = set(svm_selected.index)
candidates = sorted(lasso_set & svm_set)
print(f"\n--- Candidate genes (LASSO ∩ SVM-RFE): {len(candidates)} ---")
print(candidates)

# If intersection too small, expand to union of top-10 from each
if len(candidates) < 5:
    lasso_top = set(lasso_selected.head(10).index)
    svm_top = set(svm_weights.head(10).index)
    candidates = sorted(lasso_top | svm_top)
    print(f"  (expanded to top-10 union: {len(candidates)})")

# ── 5. External validation in GSE204835 ───────────────────────────────────
print("\n--- External validation (GSE204835) ---")
val = pd.read_csv(rf"{RES}\moduleB_validation_gene_level.csv")
val_cand = val[val['gene'].isin(candidates)].copy()
n_concordant = val_cand['same_direction'].sum()
n_validated_fdr = (val_cand['fdr_validation'] < 0.05).sum()
print(f"Candidates testable in validation: {len(val_cand)}/{len(candidates)}")
print(f"Same direction: {n_concordant}/{len(val_cand)} ({100*n_concordant/max(len(val_cand),1):.1f}%)")
print(f"FDR<0.05 in validation: {n_validated_fdr}")
if len(val_cand) > 0:
    print(val_cand[['gene','log2FC_discovery','delta_validation','same_direction','fdr_validation']].to_string(index=False))

# ── 6. Druggability analysis ──────────────────────────────────────────────
print("\n--- Druggability (DGIdb) ---")
def dgidb_druggable(genes):
    """Query DGIdb API for drug-gene interactions."""
    results = {}
    try:
        url = "https://dgidb.org/api/v2/interactions.json?genes=" + urllib.parse.quote(','.join(genes))
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        for match in data.get('matchedTerms', []):
            g = match['searchTerm']
            interactions = match.get('interactions', [])
            drugs = list(set([i.get('drugName', '') for i in interactions if i.get('drugName')]))
            results[g] = {'n_drugs': len(drugs), 'drugs': drugs[:10], 'druggable': len(drugs) > 0}
        return results, 'dgidb'
    except Exception as e:
        print(f"  DGIdb API failed: {e}")
        return None, 'failed'

dgidb_res, source = dgidb_druggable(candidates)
if dgidb_res is None:
    # Fallback: curated druggable gene families
    druggable_families = {
        'kinase': ['KIT', 'EGFR', 'ERBB2', 'PDGFRB', 'FLT1', 'FLT4', 'JAK2', 'MAPK', 'PIK3', 'AKT', 'SRC'],
        'gpcr': ['ADR', 'CCR', 'CXCR', 'EDN', 'FPR', 'GALR', 'NPY', 'PTAFR'],
        'nuclear_receptor': ['ESR', 'PPAR', 'VDR', 'NR3', 'NR1', 'THR', 'RAR'],
        'ion_channel': ['SCN', 'KCN', 'CACN', 'CLCN', 'TRP'],
        'enzyme': ['HK2', 'LDHA', 'PFKP', 'PDK1', 'EGLN', 'HIF', 'CASP', 'MMP', 'PLAU'],
        'cytokine_receptor': ['IL', 'IFN', 'TNF', 'TGF', 'LEP', 'INH'],
    }
    dgidb_res = {}
    for g in candidates:
        families = []
        for fam, prefixes in druggable_families.items():
            if any(g.startswith(p) for p in prefixes):
                families.append(fam)
        dgidb_res[g] = {'n_drugs': 0, 'drugs': [], 'families': families,
                        'druggable': len(families) > 0}
    source = 'curated_families'
print(f"Druggability source: {source}")
druggable = [g for g, r in dgidb_res.items() if r.get('druggable', False)]
print(f"Druggable candidates: {len(druggable)}/{len(candidates)}")
for g in candidates:
    info = dgidb_res.get(g, {})
    flag = '✓' if info.get('druggable') else '✗'
    extra = info.get('drugs', info.get('families', []))
    print(f"  {flag} {g:12s} {'druggable' if info.get('druggable') else 'not druggable'}  {extra[:5] if extra else ''}")

# ── 7. Save results ───────────────────────────────────────────────────────
cand_df = pd.DataFrame({
    'gene': candidates,
    'lasso_coef': [coef.get(g, 0) for g in candidates],
    'in_lasso': [g in lasso_set for g in candidates],
    'in_svm_rfe': [g in svm_set for g in candidates],
    'druggable': [dgidb_res.get(g, {}).get('druggable', False) for g in candidates],
    'drugs': [';'.join(dgidb_res.get(g, {}).get('drugs', [])) for g in candidates],
})
# Add validation info
val_map = val_cand.set_index('gene')[['same_direction', 'fdr_validation', 'log2FC_discovery']].to_dict('index')
cand_df['validated_direction'] = cand_df['gene'].map(lambda g: val_map.get(g, {}).get('same_direction', None))
cand_df['validation_fdr'] = cand_df['gene'].map(lambda g: val_map.get(g, {}).get('fdr_validation', None))
cand_df['discovery_log2FC'] = cand_df['gene'].map(lambda g: val_map.get(g, {}).get('log2FC_discovery', None))
cand_df.to_csv(rf"{RES}\moduleB_candidate_genes.csv", index=False)
print(f"\nSaved {len(cand_df)} candidate genes to moduleB_candidate_genes.csv")

# Save LASSO and SVM selected genes
pd.DataFrame({'gene': lasso_selected.index, 'coef': lasso_selected.values}).to_csv(
    rf"{RES}\moduleB_lasso_selected.csv", index=False)
pd.DataFrame({'gene': svm_weights.index, 'weight': svm_weights.values}).to_csv(
    rf"{RES}\moduleB_svmrfe_selected.csv", index=False)

# ── 8. Figure 7 ───────────────────────────────────────────────────────────
print("\nGenerating Fig 7...")
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 0.9], wspace=0.35)

# Panel A: LASSO coefficient path
axA = fig.add_subplot(gs[0, 0])
# Plot only genes that become non-zero at some C
nonzero_genes = np.any(coef_path != 0, axis=0)
for i in range(coef_path.shape[1]):
    if nonzero_genes[i]:
        color = COLORS['accent'] if X.columns[i] in candidates else COLORS['neutral']
        lw = 1.5 if X.columns[i] in candidates else 0.5
        axA.plot(lasso_c_grid, coef_path[:, i], color=color, lw=lw, alpha=0.8 if X.columns[i] in candidates else 0.3)
axA.axvline(best_C, color=COLORS['primary'], ls='--', lw=1.0,
            label=f'min: C={best_C:.3f} ({len(lasso_selected)} genes)')
axA.axvline(C_1se, color='#27AE60', ls='--', lw=1.0,
            label=f'1-SE: C={C_1se:.3f} ({len(sel_1se)} genes)')
axA.text(best_C, 0.985, 'min', transform=axA.get_xaxis_transform(),
         ha='left', va='top', fontsize=7.5, fontweight='bold', color=COLORS['primary'])
axA.text(C_1se, 0.985, '1-SE', transform=axA.get_xaxis_transform(),
         ha='right', va='top', fontsize=7.5, fontweight='bold', color='#27AE60')
axA.set_xscale('log')
axA.set_xlabel('Regularization parameter C (1/λ)')
axA.set_ylabel('LASSO coefficient')
axA.set_title('LASSO coefficient path', fontsize=10, fontweight='bold', pad=8)
axA.legend(loc='lower left', fontsize=7, frameon=True, edgecolor='#CCCCCC')
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)
axA.text(-0.15, 1.02, 'A', transform=axA.transAxes, fontsize=16, fontweight='bold', va='top')

# Panel B: Candidate gene expression heatmap (EOPE vs control)
axB = fig.add_subplot(gs[0, 1])
# Z-score each gene across samples
X_cand = X[candidates].copy()
X_z = (X_cand - X_cand.mean()) / (X_cand.std() + 1e-9)
# Order genes by mean expression difference
gene_order = X_z.columns[np.argsort(X_z[y == 1].mean() - X_z[y == 0].mean())][::-1]
# Order samples: controls first, then EOPE
sample_order = np.argsort(y)
im = axB.imshow(X_z.values[sample_order][:, [list(X_z.columns).index(g) for g in gene_order]].T,
                cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
axB.set_yticks(range(len(gene_order)))
axB.set_yticklabels(gene_order, fontsize=7)
axB.set_xticks([])
axB.set_title('Candidate gene expression (z-scored)', fontsize=10, fontweight='bold', pad=8)
# Group separator
n_ctrl = int((y == 0).sum())
n_eope = int(len(y) - n_ctrl)
axB.axvline(n_ctrl - 0.5, color='white', lw=1.5)
# Colored range blocks flush against the heatmap bottom (x-axis), so each
# block visually aligns with the sample columns it covers
from matplotlib.patches import Rectangle
ctrl_color = '#BDC3C7'   # Control = gray
eope_color = '#E74C3C'   # EOPE = red
block_h = 0.06           # axes-coord height of the range block
block_y = -block_h       # top edge at 0 -> touches heatmap bottom
axB.add_patch(Rectangle((-0.5, block_y), n_ctrl, block_h,
                        facecolor=ctrl_color, edgecolor='none',
                        transform=axB.get_xaxis_transform(), clip_on=False))
axB.add_patch(Rectangle((n_ctrl - 0.5, block_y), n_eope, block_h,
                        facecolor=eope_color, edgecolor='none',
                        transform=axB.get_xaxis_transform(), clip_on=False))
# Group labels directly below the range blocks (no extra subtitle underneath)
axB.text(n_ctrl / 2 - 0.5, block_y - 0.015, 'Control',
         ha='center', va='top', fontsize=7.5, fontweight='bold',
         transform=axB.get_xaxis_transform())
axB.text(n_ctrl - 0.5 + n_eope / 2, block_y - 0.015, 'EOPE',
         ha='center', va='top', fontsize=7.5, fontweight='bold', color='#E74C3C',
         transform=axB.get_xaxis_transform())
cb = fig.colorbar(im, ax=axB, shrink=0.7)
cb.set_label('z-score', fontsize=8)
axB.text(-0.08, 1.02, 'B', transform=axB.transAxes, fontsize=16, fontweight='bold', va='top')

# Panel C: Druggability + validation forest
axC = fig.add_subplot(gs[0, 2])
# Show candidate genes with druggability and validation status
y_pos = range(len(candidates))
for i, g in enumerate(candidates):
    is_drug = dgidb_res.get(g, {}).get('druggable', False)
    is_val = val_map.get(g, {}).get('same_direction', False)
    color = COLORS['accent'] if is_drug else COLORS['neutral']
    axC.barh(i, 1, color=color, edgecolor='white', height=0.7)
    # Markers for validation
    if is_val:
        axC.plot(1.02, i, 'v', color='#27AE60', markersize=8)
axC.set_yticks(y_pos)
axC.set_yticklabels(candidates, fontsize=7.5)
axC.invert_yaxis()
axC.set_xlim(0, 1.2)
axC.set_xticks([])
axC.set_title('Druggability & validation', fontsize=10, fontweight='bold', pad=8)
# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['accent'], label='Druggable'),
    Patch(facecolor=COLORS['neutral'], label='Not druggable'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#27AE60', markersize=8, label='Validated (same dir.)'),
]
axC.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
           ncol=1, fontsize=6.5, frameon=True, edgecolor='#CCCCCC')
axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)
axC.spines['bottom'].set_visible(False)
axC.text(-0.1, 1.02, 'C', transform=axC.transAxes, fontsize=16, fontweight='bold', va='top')

fig.suptitle('Candidate gene prioritization by LASSO + SVM-RFE',
             fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout()
for ext in ['png', 'pdf', 'tiff']:
    fig.savefig(rf"{FIG}\fig7_candidate_genes.{ext}", dpi=300, bbox_inches='tight',
                pad_inches=0.15, facecolor='white', edgecolor='none')
plt.close(fig)
print("  saved fig7_candidate_genes.*")

# ── 9. Summary & verification ─────────────────────────────────────────────
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"LASSO selected  : {len(lasso_selected)} genes (best C={best_C:.3f}, CV AUC={best_cv_score:.3f})")
print(f"SVM-RFE selected: {len(svm_selected)} genes (best n={best_n_feat}, CV AUC={best_rfe_score:.3f})")
print(f"Candidates (∩)  : {len(candidates)} genes")
print(f"  Druggable     : {len(druggable)}")
print(f"  Validated dir : {n_concordant}/{len(val_cand)}")
print(f"  FDR<0.05 in val: {n_validated_fdr}")
print(f"Candidates: {', '.join(candidates)}")

summary = {
    'lasso_best_C': float(best_C), 'lasso_cv_auc': float(best_cv_score),
    'lasso_cv_se': float(cv_se[best_idx]),
    'lasso_1se_C': float(C_1se), 'lasso_1se_cv_auc': float(cv_mean[idx_1se]),
    'lasso_1se_genes': list(sel_1se.index),
    'lasso_n_selected': int(len(lasso_selected)),
    'svm_best_n_features': int(best_n_feat), 'svm_cv_auc': float(best_rfe_score),
    'svm_n_selected': int(len(svm_selected)),
    'n_candidates': int(len(candidates)),
    'candidates': candidates,
    'n_druggable': int(len(druggable)),
    'druggable_genes': druggable,
    'n_validated_direction': int(n_concordant),
    'n_validated_fdr05': int(n_validated_fdr),
    'druggability_source': source,
}
json.dump(summary, open(rf"{RES}\moduleB_candidate_summary.json", 'w'), indent=2)
print("\nAll results saved.")
