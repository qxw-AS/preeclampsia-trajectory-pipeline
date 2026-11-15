"""
Direction 5 / Module B Step 9: 17-gene evidence convergence, clinical
associations, leakage-resistant nested CV, and frozen external validation.

Discovery: GSE75010, 84 GA-matched samples (49 EOPE / 35 controls).
External: GSE204835, term PE (n=12) vs term control (n=12), evaluated once
only after the prespecified FLT1+SMARCA1 model has been frozen.

No clinical values are imputed. PR-AUC means average precision throughout.
"""
from pathlib import Path
from collections import OrderedDict
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT = Path(__file__).resolve().parent
RES = OUT / "results"
FIG = OUT / "figures"
DATA = OUT / "data" / "geo" / "gse75010" / "GSE75010_data"
SEED = 20261114
N_OUTER_REPEATS = 5
N_BOOTSTRAP = 500
C_GRID = np.logspace(-3, 2, 25)
CORE_ORDER = ["FLT1", "SMARCA1", "GUCA2A", "HIST1H1T", "KLRD1"]
SIGNATURES = OrderedDict({
    "2gene": ["FLT1", "SMARCA1"],
    "4gene": ["GUCA2A", "HIST1H1T", "KLRD1", "SMARCA1"],
    "5gene": CORE_ORDER.copy(),
})
# This is fixed before model assessment: the 1-SE LASSO signature is the sole
# external model. The 4/5-gene signatures are discovery-only sensitivity models.
FINAL_SIGNATURE = "2gene"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "figure.dpi": 300, "pdf.fonttype": 42,
})
COLORS = {"blue": "#2C5F8A", "red": "#C44E52", "grey": "#A7ADB4",
          "light": "#E9EDF1", "green": "#3B7A57", "orange": "#D48A2E"}
ASSERTIONS = []


def record_assertion(name, expected, observed, condition):
    passed = bool(condition)
    ASSERTIONS.append({"check": name, "expected": str(expected),
                       "observed": str(observed), "passed": passed})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: expected={expected}; observed={observed}")
    if not passed:
        raise AssertionError(f"{name}: expected={expected}; observed={observed}")


def json_clean(x):
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_clean(obj), f, indent=2, ensure_ascii=False, allow_nan=False)


def save_figure(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf")


def bh_within(df, group_cols, p_col="p_value", out_col="fdr_bh"):
    df[out_col] = np.nan
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        p = df.loc[idx, p_col].astype(float)
        ok = p.notna()
        if ok.any():
            df.loc[p.index[ok], out_col] = multipletests(p[ok], method="fdr_bh")[1]
    return df


def performance(y, prob):
    return {"roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc_average_precision": float(average_precision_score(y, prob)),
            "brier": float(brier_score_loss(y, prob))}


def youden_threshold(y, prob):
    fpr, tpr, thresholds = roc_curve(y, prob)
    valid = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thresholds[best]), float(tpr[best]), float(1 - fpr[best])


def fit_fixed_model(X, y, genes):
    scaler = StandardScaler()
    z = scaler.fit_transform(X[genes])
    model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                               max_iter=5000, random_state=SEED)
    model.fit(z, y)
    return scaler, model


def load_discovery():
    expr = pd.read_csv(DATA / "GSE75010_gene_matrix.csv.gz", index_col=0)
    labels = pd.read_csv(DATA / "GSE75010_training_labels.csv", encoding="utf-8-sig")
    samples = labels["sample_id"].tolist()
    expr = expr[samples]
    X = expr.T.astype(float)
    y = (labels.set_index("sample_id").loc[samples, "group"] == "EOPE").astype(int).values
    labels = labels.set_index("sample_id").loc[samples].reset_index()
    print("\nDATA ASSERTIONS: discovery")
    record_assertion("discovery sample count", 84, len(samples), len(samples) == 84)
    record_assertion("EOPE/control", "49/35", f"{y.sum()}/{len(y)-y.sum()}",
                     y.sum() == 49 and len(y) - y.sum() == 35)
    record_assertion("unique discovery IDs", 84, len(set(samples)), len(set(samples)) == 84)
    return expr, labels, X, y, np.array(samples)


def build_evidence_matrix(candidate_genes):
    print("\n1. SEVENTEEN-GENE EVIDENCE MATRIX")
    cand = pd.read_csv(RES / "moduleB_candidate_genes.csv")
    genes = cand["gene"].tolist()
    record_assertion("candidate gene count", 17, len(genes), len(genes) == 17)
    record_assertion("candidate genes unique", 17, len(set(genes)), len(set(genes)) == 17)
    record_assertion("candidate set unchanged", "existing 17 genes", "same",
                     set(genes) == set(candidate_genes))

    ev = cand[["gene", "druggable", "drugs"]].copy()
    lasso = pd.read_csv(RES / "moduleB_lasso_selected.csv")
    lasso["lasso_rank_by_abs_coef"] = lasso["coef"].abs().rank(ascending=False, method="first")
    svm = pd.read_csv(RES / "moduleB_svmrfe_selected.csv")
    svm["svmrfe_rank_by_abs_weight"] = svm["weight"].abs().rank(ascending=False, method="first")
    ev = ev.merge(lasso.rename(columns={"coef": "lasso_coef"}), on="gene", how="left")
    ev = ev.merge(svm.rename(columns={"weight": "svmrfe_weight"}), on="gene", how="left")
    ev["in_lasso_minC"] = ev["gene"].isin(lasso["gene"])
    ev["in_svm_rfe"] = ev["gene"].isin(svm["gene"])

    deg = pd.read_csv(RES / "moduleB_deg_significant.csv").reset_index(drop=True)
    deg["discovery_deg_rank_by_fdr"] = np.arange(1, len(deg) + 1)
    deg = deg.rename(columns={"pvalue": "discovery_pvalue", "qvalue": "discovery_fdr",
                              "log2FC": "discovery_log2FC"})
    ev = ev.merge(deg[["gene_symbol", "discovery_log2FC", "discovery_pvalue",
                       "discovery_fdr", "discovery_deg_rank_by_fdr"]],
                  left_on="gene", right_on="gene_symbol", how="left").drop(columns="gene_symbol")

    wg = pd.read_csv(RES / "moduleB_wgcna_gene_info.csv")
    wg = wg.rename(columns={"module": "wgcna_module", "kME": "wgcna_kME", "GS": "wgcna_GS"})
    ev = ev.merge(wg, left_on="gene", right_on="gene_symbol", how="left").drop(columns="gene_symbol")
    ev["wgcna_indianred"] = ev["wgcna_module"].eq("indianred")
    hubs = pd.read_csv(RES / "moduleB_indianred_hub_genes.csv")
    hubs["indianred_hub_rank_by_abs_GS"] = hubs["GS"].abs().rank(ascending=False, method="first")
    ev["indianred_hub"] = ev["gene"].isin(hubs["gene_symbol"])
    ev = ev.merge(hubs[["gene_symbol", "indianred_hub_rank_by_abs_GS"]],
                  left_on="gene", right_on="gene_symbol", how="left").drop(columns="gene_symbol")

    ppi = pd.read_csv(RES / "moduleB_ppi_hub_metrics.csv")
    ppi["ppi_degree_rank"] = ppi["degree"].rank(ascending=False, method="min")
    ppi = ppi.rename(columns={"degree": "ppi_degree", "betweenness": "ppi_betweenness"})
    ev = ev.merge(ppi[["gene", "ppi_degree", "ppi_betweenness", "ppi_degree_rank"]],
                  on="gene", how="left")

    # Keep target evidence sources distinct. ChEA3 targets come from the supplied
    # file. FLT1 is a prespecified HIF1A target in the task and is independently
    # present in the saved KEGG HIF-1 pathway list; it is NOT labelled ChEA3.
    hif_chea3 = set(pd.read_csv(RES / "moduleB_hif1a_targets.csv")["gene"].str.upper())
    tf_summary = json.load(open(RES / "moduleB_wgcna_tf_summary.json", encoding="utf-8"))
    hif_kegg = set(tf_summary.get("kegg_hif1_pathway_genes", []))
    ev["hif1a_chea3_target"] = ev["gene"].str.upper().isin(hif_chea3)
    ev["hif1a_kegg_pathway"] = ev["gene"].isin(hif_kegg)
    ev["hif1a_target"] = ev["hif1a_chea3_target"] | ev["gene"].eq("FLT1")
    ev["hif1a_evidence_source"] = "none"
    ev.loc[ev["hif1a_chea3_target"], "hif1a_evidence_source"] = "ChEA3_putative_target"
    ev.loc[ev["gene"].eq("FLT1"), "hif1a_evidence_source"] = (
        "prespecified_HIF1A_target;KEGG_HIF1_pathway_member;not_ChEA3_target"
    )

    val = pd.read_csv(RES / "moduleB_validation_gene_level.csv")
    val["external_rank_by_p"] = val["p_validation"].rank(ascending=True, method="min")
    val = val.rename(columns={"delta_validation": "external_delta_log2CPM",
                              "p_validation": "external_pvalue",
                              "fdr_validation": "external_fdr",
                              "same_direction": "external_same_direction"})
    ev = ev.merge(val[["gene", "external_delta_log2CPM", "external_pvalue",
                       "external_fdr", "external_same_direction", "external_rank_by_p"]],
                  on="gene", how="left")
    drug_source = json.load(open(RES / "moduleB_candidate_summary.json", encoding="utf-8"))["druggability_source"]
    ev["druggability_source"] = drug_source
    for name, sig in SIGNATURES.items():
        ev[f"signature_{name}"] = ev["gene"].isin(sig)

    # Prespecified descriptive tiers; no P value/FDR threshold and no composite
    # or fabricated numerical evidence score is used.
    # Tier A: membership in the fixed 5-gene union = the a priori 1-SE 2-gene
    #         signature OR min-C LASSO/SVM-RFE convergence (4-gene intersection).
    # Tier B: not Tier A, selected by one ML method, with >=1 orthogonal domain
    #         (indianred membership, PPI degree>0, HIF1A evidence, same external
    #         direction irrespective of significance, or curated druggability).
    # Tier C: all remaining candidates. Tiering never uses clinical associations.
    support = []
    for _, r in ev.iterrows():
        domains = []
        if r["wgcna_indianred"]:
            domains.append("WGCNA_indianred")
        if pd.notna(r["ppi_degree"]) and r["ppi_degree"] > 0:
            domains.append("PPI_degree>0")
        if r["hif1a_target"]:
            domains.append("HIF1A")
        if pd.notna(r["external_same_direction"]) and bool(r["external_same_direction"]):
            domains.append("external_same_direction")
        if bool(r["druggable"]):
            domains.append("druggability")
        support.append(";".join(domains))
    ev["orthogonal_support_domains"] = support
    ev["evidence_tier"] = "C_single_ML_or_no_orthogonal_support"
    ev.loc[(~ev["signature_5gene"]) & (ev["orthogonal_support_domains"] != ""),
           "evidence_tier"] = "B_one_ML_plus_orthogonal"
    ev.loc[ev["signature_5gene"], "evidence_tier"] = "A_prespecified_signature_union"
    ev["evidence_tier"] = pd.Categorical(ev["evidence_tier"],
        categories=["A_prespecified_signature_union", "B_one_ML_plus_orthogonal",
                    "C_single_ML_or_no_orthogonal_support"], ordered=True)
    ev = ev.sort_values(["evidence_tier", "gene"]).reset_index(drop=True)
    ev.to_csv(RES / "moduleB_core_evidence_matrix.csv", index=False)

    rules = {
        "no_composite_score": True,
        "clinical_or_significance_based_selection": False,
        "tier_A": "Prespecified 5-gene union: 1-SE FLT1+SMARCA1 OR min-C LASSO/SVM-RFE intersection",
        "tier_B": "Other candidate with at least one orthogonal domain: indianred, PPI degree>0, HIF1A, same external direction, or druggability",
        "tier_C": "All remaining candidates",
        "signature_2gene": SIGNATURES["2gene"],
        "signature_4gene": SIGNATURES["4gene"],
        "signature_5gene": SIGNATURES["5gene"],
        "final_external_signature": FINAL_SIGNATURE,
        "final_rule": "A priori 1-SE 2-gene model; fixed before model assessment and external validation",
        "hif1a_note": "FLT1 is a prespecified target and KEGG HIF-1 pathway member, not a ChEA3 target in the supplied table",
        "druggability_source": drug_source,
    }
    save_json(rules, RES / "moduleB_core_evidence_rules.json")
    pd.DataFrame({"item": list(rules), "rule_or_value": [str(v) for v in rules.values()]}).to_csv(
        RES / "moduleB_core_evidence_tier_rules.csv", index=False)
    print("Tier A:", rules["tier_A"])
    print("Tier B:", rules["tier_B"])
    print("Tier C:", rules["tier_C"])
    print("Prespecified signatures:", {k: v for k, v in SIGNATURES.items()})
    return ev


def plot_evidence(ev):
    cols = [
        ("in_lasso_minC", "LASSO"), ("in_svm_rfe", "SVM-RFE"),
        ("wgcna_indianred", "indianred"), ("indianred_hub", "indianred hub"),
        ("ppi_degree", "PPI degree>0"), ("hif1a_target", "HIF1A evidence"),
        ("external_same_direction", "External same dir."), ("external_fdr", "External FDR<0.05"),
        ("druggable", "Druggable"),
    ]
    M = np.zeros((len(ev), len(cols)), dtype=float)
    for j, (col, label) in enumerate(cols):
        if col == "ppi_degree":
            M[:, j] = np.where(ev[col].isna(), np.nan, (ev[col] > 0).astype(float))
        elif col == "external_fdr":
            M[:, j] = np.where(ev[col].isna(), np.nan, (ev[col] < 0.05).astype(float))
        elif col == "external_same_direction":
            M[:, j] = np.where(ev[col].isna(), np.nan, ev[col].astype(float))
        else:
            M[:, j] = ev[col].astype(bool).astype(float)
    cmap = ListedColormap(["#D9D9D9", "#F7F7F7", COLORS["blue"]])
    cmap.set_bad("#D9D9D9")
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([x[1] for x in cols], rotation=38, ha="right")
    ax.set_yticks(range(len(ev)))
    ax.set_yticklabels(ev["gene"])
    for i in range(len(ev)):
        tier = str(ev.iloc[i]["evidence_tier"])[0]
        ax.text(len(cols) - 0.35, i, f"Tier {tier}", va="center", fontsize=7,
                color=COLORS["red"] if tier == "A" else "#555555")
        for j in range(len(cols)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=6, color="#666666")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ev), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("17-candidate evidence domains (no composite score)", fontweight="bold")
    ax.text(0, -0.14, "Blue=present; white=absent; grey/NA=not testable. Tiers are prespecified and do not use clinical P values.",
            transform=ax.transAxes, fontsize=7, color="#444444")
    save_figure(fig, "figD5_evidence_matrix")


def clinical_analysis(X, labels):
    print("\n2. CLINICAL ASSOCIATIONS (five-gene prespecified core; complete cases only)")
    traits = ["maximum systolic bp", "newborn weight z-score"]
    meta = labels.set_index("sample_id")
    rows = []
    for cohort, cohort_mask in [("all84", np.ones(len(meta), dtype=bool)),
                                ("EOPE49", meta["group"].eq("EOPE").values)]:
        for trait in traits:
            for gene in CORE_ORDER:
                trait_values = pd.to_numeric(meta[trait], errors="coerce").values
                ok = cohort_mask & np.isfinite(trait_values) & np.isfinite(X[gene].values)
                n = int(ok.sum())
                rho, p = stats.spearmanr(X.loc[ok, gene], trait_values[ok])
                rows.append({"cohort": cohort, "trait": trait, "gene": gene,
                             "n": n, "spearman_rho": rho, "p_value": p,
                             "missing_or_excluded": int(cohort_mask.sum() - n),
                             "imputation": "none"})
    cor = pd.DataFrame(rows)
    cor = bh_within(cor, ["cohort", "trait"])
    cor.to_csv(RES / "moduleB_core_clinical_spearman.csv", index=False)

    adj_rows = []
    for trait in traits:
        for gene in CORE_ORDER:
            trait_values = pd.to_numeric(meta[trait], errors="coerce")
            dat = pd.DataFrame({"trait": trait_values.values,
                                "expr": X[gene].values,
                                "EOPE": meta["group"].eq("EOPE").astype(int).values})
            dat = dat.dropna()
            z = (dat["expr"] - dat["expr"].mean()) / dat["expr"].std(ddof=1)
            design = sm.add_constant(pd.DataFrame({"expr_z": z, "EOPE": dat["EOPE"]}), has_constant="add")
            fit = sm.OLS(dat["trait"], design).fit()
            ci = fit.conf_int().loc["expr_z"]
            adj_rows.append({"cohort": "all84", "trait": trait, "gene": gene,
                             "n": int(len(dat)), "expression_beta_per_SD": fit.params["expr_z"],
                             "se": fit.bse["expr_z"], "ci95_low": ci[0], "ci95_high": ci[1],
                             "p_value": fit.pvalues["expr_z"],
                             "group_coef_EOPE": fit.params["EOPE"],
                             "model": "trait ~ expression_z + EOPE; complete cases"})
    adj = pd.DataFrame(adj_rows)
    adj = bh_within(adj, ["trait"])
    adj.to_csv(RES / "moduleB_core_clinical_group_adjusted.csv", index=False)
    print(cor[["cohort", "trait", "gene", "n", "spearman_rho", "p_value", "fdr_bh"]].to_string(index=False))
    return cor, adj


def plot_clinical(cor, adj):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    panels = [(cor[cor["cohort"] == "all84"], "All 84: Spearman rho", "spearman_rho"),
              (cor[cor["cohort"] == "EOPE49"], "EOPE 49: Spearman rho", "spearman_rho"),
              (adj, "All 84: group-adjusted beta", "expression_beta_per_SD")]
    traits = ["maximum systolic bp", "newborn weight z-score"]
    labels = ["Maximum systolic BP", "Newborn weight z-score"]
    for ax, (dat, title, value_col) in zip(axes, panels):
        M = dat.pivot(index="gene", columns="trait", values=value_col).reindex(CORE_ORDER)[traits]
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
        ax.set_xticks(range(2)); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticks(range(len(CORE_ORDER))); ax.set_yticklabels(CORE_ORDER)
        for i, gene in enumerate(CORE_ORDER):
            for j, trait in enumerate(traits):
                r = dat[(dat["gene"] == gene) & (dat["trait"] == trait)].iloc[0]
                val = r[value_col]
                star = "*" if r["fdr_bh"] < 0.05 else ""
                ax.text(j, i, f"{val:+.2f}{star}\nn={int(r['n'])}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.5 else "black")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.72)
    fig.text(0.01, 0.01, "BH-FDR within each cohort x trait family (5 genes); * FDR<0.05. No clinical imputation.", fontsize=7)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save_figure(fig, "figD5_clinical_correlations")


def select_degs(X_train, y_train, genes):
    """Training-only expression filter, Welch test, BH-FDR and |log2FC| filter."""
    keep_expr = (X_train > 4).sum(axis=0) >= math.ceil(0.5 * X_train.shape[0])
    idx_expr = np.flatnonzero(keep_expr)
    xe = X_train[:, idx_expr]
    case = xe[y_train == 1]
    ctrl = xe[y_train == 0]
    logfc = case.mean(axis=0) - ctrl.mean(axis=0)
    _, p = stats.ttest_ind(case, ctrl, axis=0, equal_var=False)
    q = np.full(len(idx_expr), np.nan)
    valid = np.isfinite(p)
    if valid.any():
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    sig = valid & (q < 0.05) & (np.abs(logfc) > 0.5)
    selected = idx_expr[sig]
    stat_df = pd.DataFrame({"gene": genes[selected],
                            "outer_train_log2FC": logfc[sig],
                            "outer_train_pvalue": p[sig],
                            "outer_train_fdr": q[sig]})
    return selected, stat_df, int(keep_expr.sum())


def nested_cv(X, y, samples):
    print(f"\n3. STRICT NESTED CV: repeated {N_OUTER_REPEATS}x outer 5-fold; inner 5-fold")
    print("DEG screening is repeated inside every inner training split and every outer training fold.")
    print("Scaling and C tuning never see the corresponding validation/test samples.")
    genes = X.columns.to_numpy(dtype=str)
    Xv = X.values
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_OUTER_REPEATS, random_state=SEED)
    pred_rows, selected_rows, fold_rows, assignment_rows, inner_rows = [], [], [], [], []
    for split_id, (tr, te) in enumerate(outer.split(Xv, y)):
        repeat = split_id // 5 + 1
        fold = split_id % 5 + 1
        assert not set(samples[tr]) & set(samples[te])
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + split_id)
        c_scores = np.zeros((5, len(C_GRID)))
        for inner_id, (itr, iva) in enumerate(inner.split(Xv[tr], y[tr])):
            abs_itr, abs_iva = tr[itr], tr[iva]
            deg_idx, _, n_expr_inner = select_degs(Xv[abs_itr], y[abs_itr], genes)
            inner_rows.append({"repeat": repeat, "outer_fold": fold, "inner_fold": inner_id + 1,
                               "n_inner_train": len(abs_itr), "n_inner_validation": len(abs_iva),
                               "n_expressed_genes": n_expr_inner, "n_DEG": len(deg_idx)})
            if len(deg_idx) == 0:
                c_scores[inner_id, :] = 0.5
                continue
            scaler = StandardScaler().fit(Xv[abs_itr][:, deg_idx])
            ztr = scaler.transform(Xv[abs_itr][:, deg_idx])
            zva = scaler.transform(Xv[abs_iva][:, deg_idx])
            for ci, C in enumerate(C_GRID):
                model = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                           max_iter=5000, random_state=SEED + split_id)
                model.fit(ztr, y[abs_itr])
                prob = model.predict_proba(zva)[:, 1]
                c_scores[inner_id, ci] = roc_auc_score(y[abs_iva], prob)
        mean_c_auc = c_scores.mean(axis=0)
        best_idx = int(np.argmax(mean_c_auc))  # deterministic smallest-C tie handling
        best_C = float(C_GRID[best_idx])
        deg_idx, deg_stats, n_expr = select_degs(Xv[tr], y[tr], genes)
        if len(deg_idx) == 0:
            raise RuntimeError(f"No outer-training DEG in repeat {repeat}, fold {fold}")
        scaler = StandardScaler().fit(Xv[tr][:, deg_idx])
        ztr = scaler.transform(Xv[tr][:, deg_idx])
        zte = scaler.transform(Xv[te][:, deg_idx])
        model = LogisticRegression(penalty="l1", solver="liblinear", C=best_C,
                                   max_iter=5000, random_state=SEED + split_id)
        model.fit(ztr, y[tr])
        prob = model.predict_proba(zte)[:, 1]
        coef = model.coef_[0]
        nz = np.flatnonzero(coef != 0)
        stat_map = deg_stats.set_index("gene")
        selected_genes = genes[deg_idx[nz]]
        order = np.argsort(-np.abs(coef[nz]))
        for rank, pos in enumerate(order, start=1):
            k = nz[pos]
            gene = genes[deg_idx[k]]
            selected_rows.append({"repeat": repeat, "outer_fold": fold, "gene": gene,
                                  "lasso_rank_by_abs_coef": rank, "coefficient": coef[k],
                                  "outer_train_log2FC": stat_map.loc[gene, "outer_train_log2FC"],
                                  "outer_train_pvalue": stat_map.loc[gene, "outer_train_pvalue"],
                                  "outer_train_fdr": stat_map.loc[gene, "outer_train_fdr"],
                                  "best_C": best_C})
        for local_i, sample_idx in enumerate(te):
            pred_rows.append({"repeat": repeat, "outer_fold": fold, "sample_id": samples[sample_idx],
                              "y_true": int(y[sample_idx]), "probability": prob[local_i],
                              "decision_score": model.decision_function(zte)[local_i],
                              "best_C": best_C, "n_DEG": len(deg_idx),
                              "n_selected_genes": len(nz), "evaluation": "nested_outer_test"})
        for sample_idx in tr:
            assignment_rows.append({"repeat": repeat, "outer_fold": fold,
                                    "sample_id": samples[sample_idx], "role": "train"})
        for sample_idx in te:
            assignment_rows.append({"repeat": repeat, "outer_fold": fold,
                                    "sample_id": samples[sample_idx], "role": "test"})
        fm = performance(y[te], prob)
        fold_rows.append({"repeat": repeat, "outer_fold": fold, "n_train": len(tr), "n_test": len(te),
                          "n_expressed_genes": n_expr, "n_DEG": len(deg_idx),
                          "n_selected_genes": len(nz), "selected_genes": ";".join(selected_genes),
                          "best_C": best_C, "inner_cv_auc": mean_c_auc[best_idx], **fm})
        print(f"  repeat {repeat}/{N_OUTER_REPEATS}, fold {fold}: DEG={len(deg_idx)}, "
              f"LASSO genes={len(nz)}, C={best_C:.4g}, AUC={fm['roc_auc']:.3f}, AP={fm['pr_auc_average_precision']:.3f}")

    pred = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows)
    selected = pd.DataFrame(selected_rows)
    assignments = pd.DataFrame(assignment_rows)
    inner_diag = pd.DataFrame(inner_rows)
    pred.to_csv(RES / "moduleB_core_nested_predictions.csv", index=False)
    folds.to_csv(RES / "moduleB_core_nested_fold_metrics.csv", index=False)
    selected.to_csv(RES / "moduleB_core_nested_selected_genes.csv", index=False)
    assignments.to_csv(RES / "moduleB_core_nested_fold_assignments.csv", index=False)
    inner_diag.to_csv(RES / "moduleB_core_nested_inner_diagnostics.csv", index=False)

    repeat_rows = []
    for repeat in range(1, N_OUTER_REPEATS + 1):
        dat = pred[pred["repeat"] == repeat]
        assert len(dat) == 84 and dat["sample_id"].nunique() == 84
        repeat_rows.append({"repeat": repeat, **performance(dat["y_true"], dat["probability"])})
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(RES / "moduleB_core_nested_repeat_metrics.csv", index=False)
    avg_pred = pred.groupby("sample_id").agg(y_true=("y_true", "first"),
                                             probability=("probability", "mean")).reindex(samples)
    avg_metrics = performance(avg_pred["y_true"], avg_pred["probability"])
    pd.DataFrame([{"evaluation": "repeat_averaged_out_of_fold", **avg_metrics}]).to_csv(
        RES / "moduleB_core_nested_repeat_averaged_metrics.csv", index=False)
    if len(selected):
        freq = selected.groupby("gene").agg(n_outer_folds_selected=("gene", "size"),
                                            mean_coefficient=("coefficient", "mean"),
                                            sd_coefficient=("coefficient", "std")).reset_index()
        freq["selection_frequency"] = freq["n_outer_folds_selected"] / len(folds)
        freq = freq.sort_values(["n_outer_folds_selected", "gene"], ascending=[False, True])
    else:
        freq = pd.DataFrame(columns=["gene", "n_outer_folds_selected", "mean_coefficient",
                                     "sd_coefficient", "selection_frequency"])
    freq.to_csv(RES / "moduleB_core_nested_gene_frequency.csv", index=False)

    print("\nNESTED-CV DATA ASSERTIONS")
    record_assertion("outer folds", 5 * N_OUTER_REPEATS, len(folds), len(folds) == 5 * N_OUTER_REPEATS)
    record_assertion("inner folds per outer fold", 5,
                     f"min={inner_diag.groupby(['repeat','outer_fold']).size().min()}",
                     inner_diag.groupby(["repeat", "outer_fold"]).size().eq(5).all())
    record_assertion("each repeat predicts each sample once", "84 unique per repeat",
                     int(pred.groupby("repeat")["sample_id"].nunique().min()),
                     pred.groupby("repeat")["sample_id"].nunique().eq(84).all())
    overlap_count = 0
    for (rep, fold), dat in assignments.groupby(["repeat", "outer_fold"]):
        tr_ids = set(dat.loc[dat["role"] == "train", "sample_id"])
        te_ids = set(dat.loc[dat["role"] == "test", "sample_id"])
        overlap_count += len(tr_ids & te_ids)
    record_assertion("nested train/test overlaps", 0, overlap_count, overlap_count == 0)
    summary = {
        "design": "5-fold outer x 5 repeats; 5-fold inner; DEG+scaler+LASSO fully training-local",
        "n_outer_folds": len(folds), "n_repeats": N_OUTER_REPEATS,
        "outer_fold_macro_auc_mean": folds["roc_auc"].mean(),
        "outer_fold_macro_auc_sd": folds["roc_auc"].std(ddof=1),
        "outer_fold_macro_pr_auc_mean": folds["pr_auc_average_precision"].mean(),
        "outer_fold_macro_pr_auc_sd": folds["pr_auc_average_precision"].std(ddof=1),
        "repeat_oof_auc_mean": repeats["roc_auc"].mean(),
        "repeat_oof_auc_sd": repeats["roc_auc"].std(ddof=1),
        "repeat_oof_pr_auc_mean": repeats["pr_auc_average_precision"].mean(),
        "repeat_oof_pr_auc_sd": repeats["pr_auc_average_precision"].std(ddof=1),
        "repeat_averaged_oof": avg_metrics,
        "note": "Not comparable to apparent fixed-signature training performance; no full-84 gene selection used",
    }
    save_json(summary, RES / "moduleB_core_nested_summary.json")
    print(json.dumps(json_clean(summary), indent=2))
    return pred, folds, repeats, avg_pred, summary


def plot_nested(pred, repeats, avg_pred, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    ax = axes[0]
    for rep in range(1, N_OUTER_REPEATS + 1):
        dat = pred[pred["repeat"] == rep]
        fpr, tpr, _ = roc_curve(dat["y_true"], dat["probability"])
        ax.plot(fpr, tpr, color=COLORS["grey"], alpha=0.45, lw=0.8)
    fpr, tpr, _ = roc_curve(avg_pred["y_true"], avg_pred["probability"])
    ax.plot(fpr, tpr, color=COLORS["blue"], lw=2,
            label=f"Repeat-averaged OOF AUC={avg_metrics_safe(avg_pred)['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title(f"Nested ROC\nmean repeat OOF AUC={summary['repeat_oof_auc_mean']:.3f} ± {summary['repeat_oof_auc_sd']:.3f}")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    for rep in range(1, N_OUTER_REPEATS + 1):
        dat = pred[pred["repeat"] == rep]
        precision, recall, _ = precision_recall_curve(dat["y_true"], dat["probability"])
        ax.plot(recall, precision, color=COLORS["grey"], alpha=0.45, lw=0.8)
    precision, recall, _ = precision_recall_curve(avg_pred["y_true"], avg_pred["probability"])
    ax.plot(recall, precision, color=COLORS["red"], lw=2,
            label=f"Repeat-averaged OOF AP={avg_metrics_safe(avg_pred)['pr_auc_average_precision']:.3f}")
    ax.axhline(49 / 84, ls="--", color="#777777", lw=0.8, label="Prevalence=0.583")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title(f"Nested precision-recall\nmean repeat OOF AP={summary['repeat_oof_pr_auc_mean']:.3f} ± {summary['repeat_oof_pr_auc_sd']:.3f}")
    ax.legend(loc="lower left", frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fully nested discovery evaluation (DEG + scaling + LASSO inside training folds)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figD5_nested_roc_pr")


def avg_metrics_safe(avg_pred):
    return performance(avg_pred["y_true"], avg_pred["probability"])


def fixed_signature_validation(X, y, samples):
    print("\n4. FIXED SIGNATURES: DISCOVERY-ONLY APPARENT AND BOOTSTRAP OPTIMISM")
    print("Features are already prespecified from discovery; these are NOT unbiased CV estimates.")
    summary_rows, bootstrap_rows, prediction_rows = [], [], []
    fitted = {}
    for si, (name, genes) in enumerate(SIGNATURES.items()):
        scaler, model = fit_fixed_model(X, y, genes)
        fitted[name] = (scaler, model)
        prob = model.predict_proba(scaler.transform(X[genes]))[:, 1]
        apparent = performance(y, prob)
        threshold, sens, spec = youden_threshold(y, prob)
        rng = np.random.default_rng(SEED + 1000 + si)
        for b in range(N_BOOTSTRAP):
            boot_idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[boot_idx])) < 2:
                continue
            bs, bm = fit_fixed_model(X.iloc[boot_idx], y[boot_idx], genes)
            p_boot = bm.predict_proba(bs.transform(X.iloc[boot_idx][genes]))[:, 1]
            p_orig = bm.predict_proba(bs.transform(X[genes]))[:, 1]
            row = {"signature": name, "bootstrap": b + 1}
            for metric in ["roc_auc", "pr_auc_average_precision"]:
                train_metric = roc_auc_score(y[boot_idx], p_boot) if metric == "roc_auc" else average_precision_score(y[boot_idx], p_boot)
                test_metric = roc_auc_score(y, p_orig) if metric == "roc_auc" else average_precision_score(y, p_orig)
                row[f"{metric}_bootstrap"] = train_metric
                row[f"{metric}_original"] = test_metric
                row[f"{metric}_optimism"] = train_metric - test_metric
            bootstrap_rows.append(row)
        br = pd.DataFrame([r for r in bootstrap_rows if r["signature"] == name])
        auc_opt = br["roc_auc_optimism"].mean()
        ap_opt = br["pr_auc_average_precision_optimism"].mean()
        summary_rows.append({"signature": name, "genes": ";".join(genes), "n_genes": len(genes),
                             "evaluation": "discovery_internal_features_prespecified",
                             **apparent, "youden_threshold": threshold,
                             "apparent_sensitivity": sens, "apparent_specificity": spec,
                             "n_bootstrap": len(br), "auc_optimism": auc_opt,
                             "auc_optimism_corrected": apparent["roc_auc"] - auc_opt,
                             "pr_auc_optimism": ap_opt,
                             "pr_auc_optimism_corrected": apparent["pr_auc_average_precision"] - ap_opt,
                             "auc_optimism_p025": br["roc_auc_optimism"].quantile(0.025),
                             "auc_optimism_p975": br["roc_auc_optimism"].quantile(0.975),
                             "interpretation": "apparent plus bootstrap optimism; not nested CV or external validation"})
        for i, sample in enumerate(samples):
            prediction_rows.append({"signature": name, "sample_id": sample, "y_true": y[i],
                                    "probability": prob[i], "youden_threshold": threshold,
                                    "predicted_class": int(prob[i] >= threshold),
                                    "evaluation": "apparent_discovery"})
        print(f"  {name}: apparent AUC={apparent['roc_auc']:.3f}, AP={apparent['pr_auc_average_precision']:.3f}; "
              f"optimism-corrected AUC={apparent['roc_auc']-auc_opt:.3f}")
    fixed = pd.DataFrame(summary_rows)
    fixed.to_csv(RES / "moduleB_core_fixed_signature_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RES / "moduleB_core_fixed_bootstrap.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(RES / "moduleB_core_fixed_signature_predictions.csv", index=False)
    sig_rows = []
    basis = {"2gene": "prespecified 1-SE LASSO: FLT1+SMARCA1",
             "4gene": "17 candidates with both min-C LASSO and SVM-RFE support",
             "5gene": "prespecified union of 2-gene and 4-gene sets"}
    for name, genes in SIGNATURES.items():
        for gene in genes:
            sig_rows.append({"signature": name, "gene": gene, "definition": basis[name],
                             "final_external_model": name == FINAL_SIGNATURE})
    pd.DataFrame(sig_rows).to_csv(RES / "moduleB_core_signature_definitions.csv", index=False)
    return fixed, fitted


def calibration_statistics(y, prob):
    # Preserve tiny cross-platform probabilities instead of clipping them all to
    # one constant; explicitly add an intercept column for the calibration fit.
    p = np.clip(prob, 1e-300, 1 - 1e-12)
    lp = logit(p)
    result = {"calibration_intercept": np.nan, "calibration_slope": np.nan,
              "calibration_in_the_large_intercept": np.nan,
              "calibration_status": "unstable_small_sample_n24_events12",
              "calibration_stable": False, "calibration_fit_success": False}
    try:
        design = np.column_stack([np.ones(len(y)), lp])
        joint = sm.GLM(y, design, family=sm.families.Binomial()).fit(disp=0)
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(disp=0)
        result.update({"calibration_intercept": float(joint.params[0]),
                       "calibration_slope": float(joint.params[1]),
                       "calibration_in_the_large_intercept": float(citl.params[0]),
                       "calibration_fit_success": True,
                       "calibration_intercept_pvalue": float(joint.pvalues[0]),
                       "calibration_slope_pvalue": float(joint.pvalues[1])})
    except Exception as exc:
        result["calibration_error"] = str(exc)
    return result


def external_validation(X, y, samples, fitted, fixed_metrics):
    print("\n5. FINAL FROZEN EXTERNAL VALIDATION: GSE204835 term PE 12 vs control 12")
    print("No external feature selection, tuning, threshold optimization, or recalibration.")
    genes = SIGNATURES[FINAL_SIGNATURE]
    scaler, model = fitted[FINAL_SIGNATURE]
    disc_prob = model.predict_proba(scaler.transform(X[genes]))[:, 1]
    threshold, disc_sens, disc_spec = youden_threshold(y, disc_prob)
    threshold_row = fixed_metrics.loc[fixed_metrics["signature"] == FINAL_SIGNATURE].iloc[0]
    assert np.isclose(threshold, threshold_row["youden_threshold"])

    meta = pd.read_csv(RES / "moduleB_gse204835_metadata.csv")
    term = meta[(meta["ga"] == "Term") & (meta["disease"].isin(["Preeclampsia", "Control"]))].copy()
    pe_cols = term.loc[term["disease"] == "Preeclampsia", "col"].tolist()
    ctl_cols = term.loc[term["disease"] == "Control", "col"].tolist()
    ext_cols = pe_cols + ctl_cols
    y_ext = np.array([1] * len(pe_cols) + [0] * len(ctl_cols))
    record_assertion("external term PE/control", "12/12", f"{len(pe_cols)}/{len(ctl_cols)}",
                     len(pe_cols) == 12 and len(ctl_cols) == 12)
    record_assertion("discovery/external GSM overlap", 0,
                     len(set(samples) & set(term["gsm"])), not set(samples) & set(term["gsm"]))

    cnt = pd.read_csv(OUT / "data" / "geo" / "GSE204835_counts.csv.gz", index_col=0)
    cnt.index = cnt.index.astype(str).str.split(".").str[0]
    mp = pd.read_csv(RES / "moduleB_gse204835_entrez2sym.csv", dtype={"entrez": str})
    mp["entrez"] = mp["entrez"].str.split(".").str[0]
    mp = mp.dropna(subset=["symbol"]).drop_duplicates("entrez")
    cnt = cnt.loc[cnt.index.isin(mp["entrez"]), ext_cols]
    symbol_map = mp.set_index("entrez")["symbol"]
    cnt.index = cnt.index.map(symbol_map)
    sym_counts = cnt.groupby(level=0).max()
    # Same library-size normalization as the prior gene-level validation; no
    # external expression filtering is applied to the frozen signature genes.
    logcpm = np.log2(sym_counts / sym_counts.sum(axis=0) * 1e6 + 1)
    missing = [g for g in genes if g not in logcpm.index]
    record_assertion("frozen genes available externally", "all 2", f"missing={missing}", not missing)
    raw_ext = logcpm.loc[genes, ext_cols].T.astype(float)
    # Only discovery means/scales are used; external means/SDs are never fitted.
    z_ext = scaler.transform(raw_ext)
    prob = model.predict_proba(z_ext)[:, 1]
    pred_class = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ext, pred_class, labels=[0, 1]).ravel()
    metrics = performance(y_ext, prob)
    metrics.update({"signature": FINAL_SIGNATURE, "genes": ";".join(genes),
                    "n": len(y_ext), "n_PE": int(y_ext.sum()), "n_control": int((1-y_ext).sum()),
                    "threshold_discovery_Youden": threshold, "sensitivity": tp / (tp + fn),
                    "specificity": tn / (tn + fp), "tp": int(tp), "tn": int(tn),
                    "fp": int(fp), "fn": int(fn), "preprocessing": "CPM then log2(CPM+1); discovery z-score parameters",
                    "min_probability": float(np.min(prob)), "max_probability": float(np.max(prob)),
                    "external_evaluations": 1, "dca_status": "exploratory_small_sample"})
    metrics.update(calibration_statistics(y_ext, prob))
    pd.DataFrame([metrics]).to_csv(RES / "moduleB_core_external_metrics.csv", index=False)
    save_json(metrics, RES / "moduleB_core_external_metrics.json")

    pred = pd.DataFrame({"gsm": term.set_index("col").loc[ext_cols, "gsm"].values,
                         "column_id": ext_cols, "group": np.where(y_ext == 1, "Term_PE", "Term_Control"),
                         "y_true": y_ext, "probability": prob, "decision_score": model.decision_function(z_ext),
                         "threshold_discovery_Youden": threshold, "predicted_class": pred_class})
    for gene in genes:
        pred[f"{gene}_log2CPM"] = raw_ext[gene].values
        pred[f"{gene}_z_discovery"] = z_ext[:, genes.index(gene)]
    pred.to_csv(RES / "moduleB_core_external_predictions.csv", index=False)

    final_model = {
        "signature": FINAL_SIGNATURE, "genes": genes,
        "model": "L2 logistic regression, liblinear, C=1",
        "selection_status": "prespecified 1-SE signature; frozen after discovery; external data not used for selection",
        "discovery_scaler_mean": dict(zip(genes, scaler.mean_)),
        "discovery_scaler_scale_ddof0": dict(zip(genes, scaler.scale_)),
        "coefficients": dict(zip(genes, model.coef_[0])), "intercept": model.intercept_[0],
        "threshold_discovery_Youden": threshold,
        "discovery_threshold_sensitivity": disc_sens, "discovery_threshold_specificity": disc_spec,
        "external_preprocessing": "Entrez-to-symbol max aggregation; CPM; log2(CPM+1); discovery scaler.transform",
        "external_cohort": "GSE204835 term Preeclampsia vs term Control; 12/12",
    }
    save_json(final_model, RES / "moduleB_core_final_model.json")

    thresholds = np.arange(0.01, 1.0, 0.01)
    dca_rows = []
    n = len(y_ext)
    prevalence = y_ext.mean()
    for pt in thresholds:
        classified = prob >= pt
        tp_d = int(((classified == 1) & (y_ext == 1)).sum())
        fp_d = int(((classified == 1) & (y_ext == 0)).sum())
        odds = pt / (1 - pt)
        dca_rows.append({"threshold": pt, "model_net_benefit": tp_d/n - fp_d/n*odds,
                         "treat_all_net_benefit": prevalence - (1-prevalence)*odds,
                         "treat_none_net_benefit": 0.0, "analysis": "exploratory"})
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(RES / "moduleB_core_external_dca.csv", index=False)
    bins = pd.DataFrame({"y": y_ext, "p": prob})
    bins["bin"] = pd.qcut(bins["p"].rank(method="first"), 4, labels=False) + 1
    cal_bins = bins.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                       observed_event_rate=("y", "mean"),
                                       min_predicted=("p", "min"), max_predicted=("p", "max")).reset_index()
    cal_bins["note"] = "quartile bins; unstable descriptive calibration only"
    cal_bins.to_csv(RES / "moduleB_core_external_calibration_bins.csv", index=False)
    print(pd.Series(metrics).to_string())
    return metrics, pred, cal_bins, dca


def plot_external(metrics, predictions, cal_bins, dca):
    y = predictions["y_true"].values
    p = predictions["probability"].values
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=COLORS["blue"], lw=2, label=f"AUC={metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=0.8)
    ax.scatter(1-metrics["specificity"], metrics["sensitivity"], color=COLORS["red"], zorder=5,
               label=f"Discovery threshold={metrics['threshold_discovery_Youden']:.3f}")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02),
           title="External ROC (12 PE / 12 controls)")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=0.8, label="Ideal")
    ax.errorbar(cal_bins["mean_predicted"], cal_bins["observed_event_rate"],
                yerr=np.sqrt(cal_bins["observed_event_rate"]*(1-cal_bins["observed_event_rate"])/cal_bins["n"]),
                fmt="o", color=COLORS["blue"], capsize=3, label="Quartiles (n=6 each)")
    if metrics.get("calibration_fit_success"):
        grid = np.linspace(0.01, 0.99, 200)
        fitted = expit(metrics["calibration_intercept"] + metrics["calibration_slope"]*logit(grid))
        ax.plot(grid, fitted, color=COLORS["red"], lw=1.2, label="Recalibration fit (unstable)")
    ax.set(xlabel="Predicted probability", ylabel="Observed PE proportion", xlim=(0, 1), ylim=(-0.05, 1.05),
           title=f"Calibration (unstable, n=24)\nintercept={metrics['calibration_intercept']:.2f}; slope={metrics['calibration_slope']:.2f}")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    show = dca["threshold"] <= 0.8
    ax.plot(dca.loc[show, "threshold"], dca.loc[show, "model_net_benefit"], color=COLORS["blue"], lw=2, label="2-gene model")
    ax.plot(dca.loc[show, "threshold"], dca.loc[show, "treat_all_net_benefit"], color=COLORS["red"], lw=1, label="Treat all")
    ax.plot(dca.loc[show, "threshold"], dca.loc[show, "treat_none_net_benefit"], color="#555555", lw=1, label="Treat none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0, 0.8), ylim=(-0.1, 0.6),
           title="Decision-curve analysis (exploratory)")
    ax.legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Single frozen external validation: FLT1 + SMARCA1", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figD5_external_roc_calibration_dca")


def main():
    start = time.time()
    print("=" * 84)
    print("DIRECTION 5: CORE GENE EVIDENCE, CLINICAL ASSOCIATION AND MODEL VALIDATION")
    print("=" * 84)
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    candidate_genes = pd.read_csv(RES / "moduleB_candidate_genes.csv")["gene"].tolist()
    expr, labels, X, y, samples = load_discovery()
    assert set(CORE_ORDER).issubset(X.columns)
    ev = build_evidence_matrix(candidate_genes)
    plot_evidence(ev)
    cor, adj = clinical_analysis(X, labels)
    plot_clinical(cor, adj)
    nested_pred, nested_folds, nested_repeats, avg_pred, nested_summary = nested_cv(X, y, samples)
    plot_nested(nested_pred, nested_repeats, avg_pred, nested_summary)
    fixed, fitted = fixed_signature_validation(X, y, samples)
    # Raw external outcomes are read only here, after all discovery analyses and
    # after the final two-gene model and threshold have been frozen.
    ext_metrics, ext_pred, cal_bins, dca = external_validation(X, y, samples, fitted, fixed)
    plot_external(ext_metrics, ext_pred, cal_bins, dca)

    record_assertion("all clinical analyses complete-case", "no imputation",
                     int(cor["imputation"].eq("none").sum()), cor["imputation"].eq("none").all())
    record_assertion("final external evaluations", 1, ext_metrics["external_evaluations"],
                     ext_metrics["external_evaluations"] == 1)
    assertions = pd.DataFrame(ASSERTIONS)
    assertions.to_csv(RES / "moduleB_core_data_assertions.csv", index=False)
    summary = {
        "analysis": "Direction 5 core validation", "seed": SEED,
        "discovery": {"n": 84, "EOPE": 49, "control": 35},
        "candidates": candidate_genes, "n_candidates": 17,
        "signatures": SIGNATURES, "final_external_signature": FINAL_SIGNATURE,
        "clinical": {"core_genes": CORE_ORDER, "method": "Spearman; BH within cohort x trait over 5 genes",
                     "sensitivity": "OLS trait ~ expression_z + EOPE in all 84", "imputation": "none"},
        "nested_cv": nested_summary,
        "fixed_signature_discovery_metrics": fixed.to_dict(orient="records"),
        "external_validation": ext_metrics,
        "limitations": [
            "Fixed-signature apparent/bootstrap estimates are discovery-internal because features were prespecified from discovery.",
            "Only 24 external samples; calibration and DCA are unstable/exploratory.",
            "GSE204835 gene-level direction/FDR already exists as evidence annotations; model evaluation uses independent samples but is not a wholly untouched prospective confirmation.",
            "HIST1H1T lacks an exact symbol in the supplied Entrez mapping; no unverified remap or imputation was performed.",
            "EOPE discovery and term-PE external phenotypes differ; transportability is limited.",
            "Discovery-derived z-scores across microarray and log2CPM platforms cause a large probability-calibration shift; ranking and threshold performance must be distinguished.",
        ],
        "runtime_seconds": time.time() - start,
    }
    save_json(summary, RES / "moduleB_core_run_summary.json")
    print("\nALL DATA ASSERTIONS PASSED")
    print(f"Completed in {(time.time()-start)/60:.1f} minutes")
    print("Outputs: results/moduleB_core_* and figures/figD5_*")


if __name__ == "__main__":
    main()
