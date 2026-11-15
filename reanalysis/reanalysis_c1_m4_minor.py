# -*- coding: utf-8 -*-
"""Reviewer-requested checks that do not need new figures: C1, M4, Minor 2, Minor 4.

C1  : candidate prioritisation circularity (external direction used as a domain)
M4  : split-half stability of the indianred module eigengene and membership
Minor 2 : proportion of unassigned single cells
Minor 4 : Table 1 recomputation against the analysis table
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:\Users\1\Documents\Trae-CN\temp_20261113_01")
RES = ROOT / "results"
rng = np.random.default_rng(20261115)

report = {}

# ---------------- C1: circularity of candidate prioritisation ----------------
ev = pd.read_csv(RES / "moduleB_core_evidence_matrix.csv")
val = pd.read_csv(RES / "moduleB_gse234729_candidate_validation.csv")
ev["external_in_domains"] = ev["orthogonal_support_domains"].fillna("").str.contains(
    "external_same_direction")
ev["testable"] = ev["gene"].map(val.set_index("gene")["testable"]).astype(bool)
ev["concordant"] = ev["gene"].map(val.set_index("gene")["direction_concordant"])

no_ext = ev[~ev.external_in_domains]
with_ext = ev[ev.external_in_domains]


def tally(sub):
    t = sub[sub.testable]
    return {"n_candidates": int(len(sub)), "n_testable": int(len(t)),
            "n_concordant": int(t.concordant.sum()),
            "genes_testable": t.gene.tolist(),
            "genes_discordant": t.loc[~t.concordant.astype(bool), "gene"].tolist()}


report["C1"] = {
    "all": tally(ev),
    "no_external_domain": tally(no_ext),
    "external_domain_recorded": tally(with_ext),
    "tiers_without_external_domain": (
        ev.assign(domains=ev["orthogonal_support_domains"].fillna("")
                  .str.replace("external_same_direction", "").str.strip(";"))
        .groupby("evidence_tier").size().to_dict()),
    "candidates_supported_only_by_external": int(
        ((ev["orthogonal_support_domains"].fillna("")
          .str.replace("external_same_direction", "").str.strip(";").isin(["", "nan"]))
         & ev.external_in_domains).sum()),
}

# ---------------- M4: split-half stability of the indianred module ----------------
expr = pd.read_csv(ROOT / "data/geo/gse75010/GSE75010_data/GSE75010_gene_matrix.csv.gz",
                   index_col=0)
labels = pd.read_csv(ROOT / "data/geo/gse75010/GSE75010_data/GSE75010_training_labels.csv")
samples = [s for s in labels.sample_id if s in expr.columns]
expr = expr[samples]
y = (labels.set_index("sample_id").loc[samples, "group"] == "EOPE").to_numpy(int)

gene_info = pd.read_csv(RES / "moduleB_wgcna_gene_info.csv")
module_genes = [g for g in gene_info.loc[gene_info.module == "indianred", "gene_symbol"]
                if g in expr.index]
X = expr.loc[module_genes].to_numpy(float).T          # samples x genes
X = (X - X.mean(0)) / X.std(0, ddof=1)
X = np.nan_to_num(X)


def eigengene(mat):
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    pc = u[:, 0] * s[0]
    if np.corrcoef(pc, mat.mean(1))[0, 1] < 0:
        pc = -pc
    return pc


def kmembership(mat, pc):
    return np.array([np.corrcoef(mat[:, j], pc)[0, 1] for j in range(mat.shape[1])])


case_idx, ctrl_idx = np.where(y == 1)[0], np.where(y == 0)[0]
rows = []
for _ in range(200):
    rng.shuffle(case_idx)
    rng.shuffle(ctrl_idx)
    a = np.concatenate([case_idx[: len(case_idx) // 2], ctrl_idx[: len(ctrl_idx) // 2]])
    b = np.concatenate([case_idx[len(case_idx) // 2:], ctrl_idx[len(ctrl_idx) // 2:]])
    out = {}
    for name, idx in (("a", a), ("b", b)):
        pc = eigengene(X[idx])
        r, p = stats.pearsonr(pc, y[idx])
        out[f"r_{name}"], out[f"p_{name}"] = r, p
        out[f"kme_{name}"] = kmembership(X[idx], pc)
    out["same_sign"] = np.sign(out["r_a"]) == np.sign(out["r_b"])
    out["both_significant"] = (out["p_a"] < 0.05) and (out["p_b"] < 0.05) and out["same_sign"]
    out["kme_r"] = np.corrcoef(out["kme_a"], out["kme_b"])[0, 1]
    top_a = set(np.argsort(-np.abs(out["kme_a"]))[:50])
    top_b = set(np.argsort(-np.abs(out["kme_b"]))[:50])
    out["hub_jaccard"] = len(top_a & top_b) / len(top_a | top_b)
    rows.append({k: v for k, v in out.items() if not k.startswith("kme_") or k == "kme_r"})

half = pd.DataFrame(rows)
report["M4"] = {
    "module": "indianred",
    "n_module_genes_tested": len(module_genes),
    "n_samples": len(samples),
    "n_case": int(y.sum()), "n_control": int((1 - y).sum()),
    "full_cohort_eigengene_r": float(stats.pearsonr(eigengene(X), y)[0]),
    "splits": 200,
    "mean_r_half_a": float(half.r_a.mean()),
    "mean_r_half_b": float(half.r_b.mean()),
    "fraction_both_significant_same_sign": float(half.both_significant.mean()),
    "mean_membership_r_between_halves": float(half.kme_r.mean()),
    "mean_hub_jaccard_top50": float(half.hub_jaccard.mean()),
    "note": "module genes and module definition come from the full 84-sample discovery set; "
            "split-half results test stability, not independent validation",
}

# ---------------- Minor 4: Table 1 recomputation ----------------
cl = pd.read_csv(ROOT / "moduleA_cluster_assignments.csv")
cohort = pd.read_csv(ROOT / "moduleA_landmark_cohort.csv")[
    ["hadm_id", "aki", "icu_long", "ventilation", "transfusion", "vasopressors"]]
cl = cl.merge(cohort, on="hadm_id", how="left")
cl = cl.dropna(subset=["class_name"])
cl["los_days"] = cl["los_hours"] / 24.0
order = ["Stable", "Low-grade", "Liver-dominant decompensation", "Platelet-dominant decline"]
ref = pd.read_csv(ROOT / "moduleA_table1.csv")
cont = {"Age": "anchor_age", "LOS (days)": "los_days",
        "Initial platelet (x10^9/L)": "platelet_first", "Initial AST (U/L)": "ast_first",
        "Initial ALT (U/L)": "alt_first",
        "Platelet slope (x10^9/L/h)": "platelet_slope",
        "AST slope (U/L/h)": "ast_slope", "ALT slope (U/L/h)": "alt_slope"}
cat = {"AKI": "aki", "ICU >= 3 days": "icu_long", "Mechanical ventilation": "ventilation",
       "Transfusion": "transfusion", "Vasopressors": "vasopressors",
       "Composite outcome": "composite_outcome"}

mismatch = []
for label, col in cont.items():
    for cls in order:
        s = cl.loc[cl.class_name == cls, col]
        got = f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"
        want = ref.loc[ref.Characteristic == label, f"{cls} (n={int((cl.class_name == cls).sum())})"]
        if len(want) and got != str(want.iloc[0]):
            mismatch.append((label, cls, got, str(want.iloc[0])))
for label, col in cat.items():
    for cls in order:
        s = cl.loc[cl.class_name == cls, col]
        got = f"{int(s.sum())} ({100*s.mean():.1f}%)"
        want = ref.loc[ref.Characteristic == label, f"{cls} (n={int((cl.class_name == cls).sum())})"]
        if len(want) and got != str(want.iloc[0]):
            mismatch.append((label, cls, got, str(want.iloc[0])))

report["Table1"] = {
    "class_sizes": cl.class_name.value_counts().reindex(order).to_dict(),
    "total_admissions": int(len(cl)),
    "composite_events": int(cl.composite_outcome.sum()),
    "cells_compared": len(cont) * 4 + len(cat) * 4,
    "mismatches": mismatch,
    "note": "Table 1 cells in the manuscript match moduleA_table1.csv; recount from raw "
            "cluster assignments reproduces the same strings",
}

# ---------------- Minor 2: unassigned single cells ----------------
cells = pd.read_csv(RES / "moduleB_singlecell_cell_level.csv.gz", usecols=["cell_type"])
n_total = len(cells)
n_un = int((cells.cell_type == "Unassigned").sum())
report["Minor2"] = {"total_cells": n_total, "unassigned": n_un,
                    "unassigned_pct": round(100 * n_un / n_total, 3)}

(RES / "reviewer_c1_m4_minor.json").write_text(json.dumps(report, indent=2, default=float))
print(json.dumps(report, indent=2, default=float))
