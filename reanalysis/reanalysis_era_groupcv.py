# -*- coding: utf-8 -*-
"""Reviewer-requested reanalyses (C2, M3).

C2: admission-era (= ICD coding system) stratified and era-adjusted models.
M3: patient-grouped, fold-local-GMM cross-validation for incremental discrimination.
Outputs: results/reviewer_era_models.csv, results/reviewer_groupcv.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(r"C:\Users\1\Documents\Trae-CN\temp_20261113_01")
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
CLASSES = ["Stable", "Low-grade", "Liver-dominant decompensation", "Platelet-dominant decline"]
FEAT = ["platelet_slope", "platelet_intercept", "platelet_first", "platelet_mean",
        "ast_slope", "ast_intercept", "ast_first", "ast_mean",
        "alt_slope", "alt_intercept", "alt_first", "alt_mean"]

cohort = pd.read_csv(ROOT / "moduleA_landmark_cohort.csv")
assign = pd.read_csv(ROOT / "moduleA_cluster_assignments.csv")
df = assign.merge(cohort[["hadm_id", "subject_id", "admittime"]], on="hadm_id", how="inner")
df = df.dropna(subset=FEAT + ["class_name", "composite_outcome"]).copy()
df["admittime"] = pd.to_datetime(df["admittime"])

# admittime is subject-level date-shifted in MIMIC-IV and cannot recover calendar era,
# so the coding system itself (ICD-9 vs ICD-10) is used as the era stratifier.
diag = pd.read_csv(r"D:\Program Files\mimic-iv-3.1\hosp\diagnoses_icd.csv.gz",
                   usecols=["hadm_id", "icd_version"])
ver = diag.groupby("hadm_id")["icd_version"].agg(lambda s: s.value_counts().idxmax())
df["icd_version"] = df["hadm_id"].map(ver)
df["era"] = np.where(df["icd_version"] == 9, "ICD9", "ICD10")
print("n =", len(df), "| era counts:", df["era"].value_counts().to_dict())

# ---------------- C2: era stratified / adjusted ----------------
formula = ("composite_outcome ~ C(class_name, Treatment('Stable')) + anchor_age + "
           "platelet_first + ast_first + alt_first")
rows = []


def fit(sub, label, extra=""):
    m = smf.logit(formula + extra, data=sub).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": sub["subject_id"]})
    for term in m.params.index:
        if term.startswith("C(class_name"):
            rows.append({
                "model": label, "n": int(sub.shape[0]),
                "events": int(sub["composite_outcome"].sum()), "term": term,
                "OR": float(np.exp(m.params[term])),
                "CI_low": float(np.exp(m.conf_int().loc[term, 0])),
                "CI_high": float(np.exp(m.conf_int().loc[term, 1])),
                "P": float(m.pvalues[term]),
            })
    return m


fit(df, "overall")
fit(df, "era_adjusted", extra=" + C(era)")
for era, sub in df.groupby("era"):
    fit(sub, f"stratum_{era}")

out = pd.DataFrame(rows)
out.to_csv(RES / "reviewer_era_models.csv", index=False)
print(out.round(3).to_string(index=False))

# ---------------- M3: patient-grouped, fold-local GMM CV ----------------
# Global centroid of each named class, used to match fold-local clusters.
gmeans = df.groupby("class_name")[FEAT].mean().loc[CLASSES]
Xall = df[FEAT].to_numpy(float)
y = df["composite_outcome"].to_numpy(int)
groups = df["subject_id"].to_numpy()
cov = df[["anchor_age", "platelet_first", "ast_first", "alt_first"]].to_numpy(float)

repeats = []
for seed in range(1, 6):
    gkf = GroupKFold(n_splits=5)
    pred_base = np.zeros(len(df))
    pred_ext = np.zeros(len(df))
    for tr, te in gkf.split(Xall, y, groups):
        scaler = StandardScaler().fit(Xall[tr])
        gmm = GaussianMixture(n_components=4, covariance_type="full", n_init=10,
                              max_iter=500, random_state=seed).fit(scaler.transform(Xall[tr]))
        # match fold clusters to the named global classes by centroid distance
        new_cent = scaler.inverse_transform(gmm.means_)
        dist = np.linalg.norm(new_cent[:, None, :] - gmeans.to_numpy()[None, :, :], axis=2)
        r, c = linear_sum_assignment(dist)
        mapping = {int(ri): CLASSES[ci] for ri, ci in zip(r, c)}

        def design(idx):
            lab = pd.Series([mapping[int(k)] for k in gmm.predict(scaler.transform(Xall[idx]))],
                            index=idx)
            d = pd.get_dummies(lab).reindex(
                columns=[c for c in CLASSES if c != "Stable"], fill_value=0).astype(float)
            Xb = np.column_stack([np.ones(len(idx)), cov[idx]])
            return Xb, np.column_stack([Xb, d.to_numpy()])

        Xb_tr, Xe_tr = design(tr)
        Xb_te, Xe_te = design(te)
        for Xtr, Xte, store in ((Xb_tr, Xb_te, pred_base), (Xe_tr, Xe_te, pred_ext)):
            lr = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
            store[te] = lr.predict_proba(Xte)[:, 1]
    repeats.append({"repeat": seed,
                    "auc_base": roc_auc_score(y, pred_base),
                    "auc_ext": roc_auc_score(y, pred_ext)})
    repeats[-1]["delta"] = repeats[-1]["auc_ext"] - repeats[-1]["auc_base"]

rep = pd.DataFrame(repeats)
summary = {
    "n": int(len(df)), "patients": int(df["subject_id"].nunique()),
    "folds": "5-fold GroupKFold by subject_id, 5 repeats",
    "auc_base_mean": float(rep["auc_base"].mean()),
    "auc_ext_mean": float(rep["auc_ext"].mean()),
    "delta_mean": float(rep["delta"].mean()),
    "delta_ci95": [float(rep["delta"].quantile(0.025)), float(rep["delta"].quantile(0.975))],
    "per_repeat": repeats,
}
(RES / "reviewer_groupcv.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "per_repeat"}, indent=2))
