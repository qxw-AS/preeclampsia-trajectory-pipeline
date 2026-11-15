"""
Panel-revision analyses for the JTM manuscript.

1. Patient-level bootstrap stability (ARI) of the four-class GMM.
2. Silhouette permutation null.
3. Two-class and continuous platelet-slope outcome models.
4. Case-status-adjusted module-trait correlations for the indianred module.

Outputs are written to results/panel_revision_*.json/csv.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\1\Documents\Trae-CN\temp_20261113_01")
RES = ROOT / "results"
RNG = np.random.default_rng(20261115)

FEATURES = [
    "platelet_slope", "platelet_intercept", "platelet_first", "platelet_mean",
    "ast_slope", "ast_intercept", "ast_first", "ast_mean",
    "alt_slope", "alt_intercept", "alt_first", "alt_mean",
]

# ---------------------------------------------------------------- load clinical data
feat = pd.read_csv(ROOT / "moduleA_trajectory_features.csv")
assign = pd.read_csv(ROOT / "moduleA_cluster_assignments.csv")
adm = pd.read_csv(RES / "moduleA_robustness_cache_admissions.csv.gz")

feat["hadm_id"] = feat["hadm_id"].astype(int)
assign["hadm_id"] = assign["hadm_id"].astype(int)
adm["hadm_id"] = adm["hadm_id"].astype(int)
df = assign.merge(adm[["hadm_id", "subject_id"]], on="hadm_id", how="left")
if df["subject_id"].isna().any():
    raise RuntimeError("Missing subject_id after merge")
df["subject_id"] = df["subject_id"].astype(int)
df = df.dropna(subset=FEATURES + ["composite_outcome", "class_name"]).reset_index(drop=True)
print(f"Analysis set: {len(df)} admissions, {df['subject_id'].nunique()} patients")
print(df["class_name"].value_counts().to_dict())

X = df[FEATURES].to_numpy(float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
orig_labels = df["class_name"].astype(str).to_numpy()

# ---------------------------------------------------------------- refit full model (sanity check)
gmm4 = GaussianMixture(n_components=4, covariance_type="full", n_init=20,
                       max_iter=500, random_state=42)
gmm4.fit(Xs)
full_pred = gmm4.predict(Xs)
sanity_ari = adjusted_rand_score(orig_labels, full_pred)
orig_sil = silhouette_score(Xs, full_pred)
print(f"Full-fit ARI vs saved labels: {sanity_ari:.4f}; silhouette={orig_sil:.4f}")

# ---------------------------------------------------------------- bootstrap ARI
B = 500
by_subject = df.groupby("subject_id").indices
subjects = np.array(list(by_subject.keys()))
aris = []
valid = 0
for b in range(B):
    sampled = RNG.choice(subjects, size=len(subjects), replace=True)
    idx = np.concatenate([by_subject[s] for s in sampled])
    if len(np.unique(df.loc[idx, "subject_id"])) < 10:
        continue
    Xb = StandardScaler().fit_transform(X[idx])
    try:
        g = GaussianMixture(n_components=4, covariance_type="full", n_init=10,
                            max_iter=500, random_state=1000 + b)
        g.fit(Xb)
        pred = g.predict(Xs)
    except Exception:
        continue
    aris.append(adjusted_rand_score(orig_labels, pred))
    valid += 1
    if (b + 1) % 100 == 0:
        print(f"  bootstrap {b + 1}/{B}; valid={valid}; mean ARI={np.mean(aris):.3f}")

aris = np.array(aris)
boot_summary = {
    "requested": B,
    "valid": int(valid),
    "mean": float(np.mean(aris)),
    "sd": float(np.std(aris, ddof=1)),
    "ci_low": float(np.percentile(aris, 2.5)),
    "ci_high": float(np.percentile(aris, 97.5)),
    "prop_ge_0_60": float(np.mean(aris >= 0.60)),
    "prop_ge_0_40": float(np.mean(aris >= 0.40)),
    "min": float(np.min(aris)),
    "max": float(np.max(aris)),
}

# ---------------------------------------------------------------- silhouette permutation null
P = 200
null_sil = []
for p in range(P):
    Xp = np.empty_like(X)
    for j in range(X.shape[1]):
        Xp[:, j] = RNG.permutation(X[:, j])
    Xps = StandardScaler().fit_transform(Xp)
    try:
        gp = GaussianMixture(n_components=4, covariance_type="full", n_init=5,
                             max_iter=500, random_state=5000 + p)
        gp.fit(Xps)
        null_sil.append(silhouette_score(Xps, gp.predict(Xps)))
    except Exception:
        continue
    if (p + 1) % 50 == 0:
        print(f"  silhouette null {p + 1}/{P}")

null_sil = np.array(null_sil)
emp_p = (1 + np.sum(null_sil >= orig_sil)) / (1 + len(null_sil))
sil_summary = {
    "observed": float(orig_sil),
    "null_n": int(len(null_sil)),
    "null_mean": float(np.mean(null_sil)),
    "null_sd": float(np.std(null_sil, ddof=1)),
    "null_ci_low": float(np.percentile(null_sil, 2.5)),
    "null_ci_high": float(np.percentile(null_sil, 97.5)),
    "empirical_p": float(emp_p),
}

# ---------------------------------------------------------------- 2-class model
gmm2 = GaussianMixture(n_components=2, covariance_type="full", n_init=20,
                       max_iter=500, random_state=42)
gmm2.fit(Xs)
lab2 = gmm2.predict(Xs)
slope_means = [df.loc[lab2 == k, "platelet_slope"].mean() for k in range(2)]
platelet_like = int(np.argmin(slope_means))
df["two_class_platelet"] = (lab2 == platelet_like).astype(int)
print(f"2-class platelet-like n={df['two_class_platelet'].sum()}")

def fit_logit(y, Xmat, names):
    Xc = sm.add_constant(pd.DataFrame(Xmat, columns=names))
    m = sm.Logit(y, Xc.astype(float)).fit(disp=0)
    return m

y = df["composite_outcome"].astype(float).to_numpy()
base_cov = df[["anchor_age", "platelet_first", "ast_first", "alt_first"]].astype(float).to_numpy()

m2 = fit_logit(y, np.column_stack([df["two_class_platelet"], base_cov]),
               ["two_class_platelet", "age", "plt0", "ast0", "alt0"])
coef = m2.params["two_class_platelet"]
se = m2.bse["two_class_platelet"]
p2 = m2.predict(sm.add_constant(pd.DataFrame(
    np.column_stack([df["two_class_platelet"], base_cov]),
    columns=["two_class_platelet", "age", "plt0", "ast0", "alt0"])))
two_class = {
    "n_platelet_like": int(df["two_class_platelet"].sum()),
    "OR": float(np.exp(coef)),
    "CI_low": float(np.exp(coef - 1.96 * se)),
    "CI_high": float(np.exp(coef + 1.96 * se)),
    "p_value": float(m2.pvalues["two_class_platelet"]),
    "AUC": float(roc_auc_score(y, p2)),
}

# continuous platelet slope (per SD)
slope_z = (df["platelet_slope"] - df["platelet_slope"].mean()) / df["platelet_slope"].std(ddof=0)
mc = fit_logit(y, np.column_stack([slope_z, base_cov]),
               ["platelet_slope_z", "age", "plt0", "ast0", "alt0"])
coefc = mc.params["platelet_slope_z"]
sec = mc.bse["platelet_slope_z"]
pc = mc.predict(sm.add_constant(pd.DataFrame(
    np.column_stack([slope_z, base_cov]),
    columns=["platelet_slope_z", "age", "plt0", "ast0", "alt0"])))
continuous = {
    "OR_per_SD": float(np.exp(coefc)),
    "CI_low": float(np.exp(coefc - 1.96 * sec)),
    "CI_high": float(np.exp(coefc + 1.96 * sec)),
    "p_value": float(mc.pvalues["platelet_slope_z"]),
    "AUC": float(roc_auc_score(y, pc)),
    "slope_sd": float(df["platelet_slope"].std(ddof=0)),
}

# four-class adjusted model for a same-sample AUC comparison
class_order = ["Stable", "Low-grade", "Liver-dominant decompensation", "Platelet-dominant decline"]
df["class_num"] = pd.Categorical(df["class_name"], categories=class_order, ordered=True).codes
m4 = fit_logit(y, np.column_stack([pd.get_dummies(df["class_num"], drop_first=True).astype(float), base_cov]),
               ["class_1", "class_2", "class_3", "age", "plt0", "ast0", "alt0"])
p4 = m4.predict(sm.add_constant(pd.concat([
    pd.get_dummies(df["class_num"], drop_first=True).astype(float),
    pd.DataFrame(base_cov, columns=["age", "plt0", "ast0", "alt0"])], axis=1)))
four_class = {
    "AUC": float(roc_auc_score(y, p4)),
    "platelet_OR": float(np.exp(m4.params["class_3"])),
    "platelet_CI_low": float(np.exp(m4.params["class_3"] - 1.96 * m4.bse["class_3"])),
    "platelet_CI_high": float(np.exp(m4.params["class_3"] + 1.96 * m4.bse["class_3"])),
    "platelet_p": float(m4.pvalues["class_3"]),
}

# ---------------------------------------------------------------- module-trait adjusted correlations
gene = pd.read_csv(ROOT / "data/geo/gse75010/GSE75010_data/GSE75010_gene_matrix.csv.gz")
gene = gene.set_index("gene_symbol")
assign_mod = pd.read_csv(RES / "moduleB_wgcna_module_assignment.csv")
labels = pd.read_csv(ROOT / "data/geo/gse75010/GSE75010_data/GSE75010_training_labels.csv")
labels["sample_id"] = labels["sample_id"].astype(str)

indian = set(assign_mod.loc[assign_mod["module"].astype(str).str.lower() == "indianred", "gene_symbol"])
genes = [g for g in gene.index if g in indian and g in gene.columns]
# gene index is genes; columns are samples
sample_cols = [c for c in labels["sample_id"] if c in gene.columns]
if len(sample_cols) < 20:
    raise RuntimeError(f"Too few overlapping samples: {len(sample_cols)}")
mat = gene.loc[[g for g in gene.index if g in indian], sample_cols].astype(float)
mat = mat.dropna(axis=0, how="any")
print(f"Indianred eigengene: {mat.shape[0]} genes x {len(sample_cols)} samples")
Z = StandardScaler().fit_transform(mat.T)  # samples x genes
pca = PCA(n_components=1, random_state=42)
me = pca.fit_transform(Z)[:, 0]
labels = labels.set_index("sample_id").loc[sample_cols].reset_index()
labels["EOPE"] = (labels["group"].astype(str).str.upper() == "EOPE").astype(int)
if np.corrcoef(me, labels["EOPE"])[0, 1] < 0:
    me = -me

def partial_corr(x, yv, cov):
    rx = sm.OLS(x, sm.add_constant(cov)).fit().resid
    ry = sm.OLS(yv, sm.add_constant(cov)).fit().resid
    r, p = stats.pearsonr(rx, ry)
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    p_partial = 2 * stats.t.sf(abs(t), df=n - 3)
    return float(r), float(p_partial)

traits = {
    "EOPE": labels["EOPE"].astype(float).to_numpy(),
    "maximum systolic bp": labels["maximum systolic bp"].astype(float).to_numpy(),
    "maximum diastolic bp": labels["maximum diastolic bp"].astype(float).to_numpy(),
    "newborn weight z-score": labels["newborn weight z-score"].astype(float).to_numpy(),
    "placental weight z-score": labels["placental weight z-score"].astype(float).to_numpy(),
}
mod_rows = []
for name, tv in traits.items():
    mask = np.isfinite(me) & np.isfinite(tv)
    if name == "EOPE":
        r, p = stats.pearsonr(me[mask], tv[mask])
        mod_rows.append({"trait": name, "n": int(mask.sum()), "r_unadjusted": float(r),
                         "p_unadjusted": float(p), "r_adjusted_for_EOPE": float(r),
                         "p_adjusted_for_EOPE": float(p)})
    else:
        r, p = stats.pearsonr(me[mask], tv[mask])
        ra, pa = partial_corr(me[mask], tv[mask], labels.loc[mask, "EOPE"].astype(float).to_numpy())
        mod_rows.append({"trait": name, "n": int(mask.sum()), "r_unadjusted": float(r),
                         "p_unadjusted": float(p), "r_adjusted_for_EOPE": float(ra),
                         "p_adjusted_for_EOPE": float(pa)})
mod_df = pd.DataFrame(mod_rows)
mod_df.to_csv(RES / "panel_revision_module_trait_adjusted.csv", index=False)
print(mod_df.to_string(index=False))

# ---------------------------------------------------------------- save
out = {
    "clinical": {
        "n_admissions": int(len(df)),
        "n_patients": int(df["subject_id"].nunique()),
        "class_sizes": df["class_name"].value_counts().to_dict(),
        "full_fit_ari_vs_saved": float(sanity_ari),
        "observed_silhouette": float(orig_sil),
    },
    "bootstrap_ari": boot_summary,
    "silhouette_null": sil_summary,
    "two_class_model": two_class,
    "continuous_model": continuous,
    "four_class_model": four_class,
    "module_trait": mod_rows,
}
with open(RES / "panel_revision_analyses.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
with open(RES / "panel_revision_bootstrap_ari.csv", "w", encoding="utf-8") as f:
    pd.DataFrame({"ari": aris}).to_csv(f, index=False)
with open(RES / "panel_revision_silhouette_null.csv", "w", encoding="utf-8") as f:
    pd.DataFrame({"silhouette": null_sil}).to_csv(f, index=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
