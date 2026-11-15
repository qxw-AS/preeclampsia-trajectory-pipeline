"""
Module B Step 12 - independent external validation in GSE234729 (severe PE).

GSE234729 is an Illumina NovaSeq 6000 bulk RNA-seq cohort of 123 placentas:
50 severe preeclampsia (sPE) and 73 normotensive controls across African,
Asian and European ancestries. The processed matrix is DESeq2-normalised
filtered counts with Ensembl$Symbol row names.

Frozen contrast and gene sets (no re-selection, no tuning):
  - sPE vs control (all ancestries pooled); ancestry-stratified sensitivity
  - 17 discovery candidates (incl. five core genes)
  - 196 discovery DEGs
  - Hallmark Hypoxia geneset
"""
import gzip
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = Path(r"C:\Users\1\Documents\Trae-CN\temp_20261113_01")
DATA = OUT / "data" / "geo" / "gse234729"
RES = OUT / "results"
FIG = OUT / "figures"
GS = OUT / "genesets"
for folder in (RES, FIG):
    folder.mkdir(parents=True, exist_ok=True)

PREFIX = "moduleB_gse234729_"
MATRIX = DATA / "GSE234729_series_matrix.txt.gz"
EXPR = DATA / "GSE234729_Plac_samples_diff_ancestry_Deseq2_norm_filtered.txt.gz"
CORE = ["FLT1", "SMARCA1", "GUCA2A", "HIST1H1T", "KLRD1"]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
})


def parse_series_matrix(path):
    lines = gzip.open(path, "rt", encoding="utf-8", errors="replace").read().splitlines()
    attrs = {}
    for line in lines:
        if not line.startswith("!Sample_"):
            continue
        fields = [x.strip('"') for x in line.split("\t")]
        attrs.setdefault(fields[0], []).append(fields[1:])
    gsm = attrs["!Sample_geo_accession"][0]
    titles = attrs["!Sample_title"][0]

    def characteristic(name):
        for row in attrs.get("!Sample_characteristics_ch1", []):
            if row and row[0].lower().startswith(name.lower() + ":"):
                return [x.split(":", 1)[1].strip() if ":" in x else "" for x in row]
        return [""] * len(gsm)

    meta = pd.DataFrame({
        "gsm": gsm,
        "title": titles,
        "tissue": characteristic("tissue"),
        "ancestry": characteristic("maternal_ancestry_self_report"),
        "disease": characteristic("disease"),
        "fetal_sex": characteristic("fetal_sex"),
    })
    meta["group"] = meta["disease"].map({
        "Severe Preeclampsia": "PE",
        "Non-hypertensive control": "control",
    }).fillna("")
    return meta


def load_discovery_deg():
    path = RES / "moduleB_deg_significant.csv"
    deg = pd.read_csv(path)
    deg["gene_symbol"] = deg["gene_symbol"].astype(str).str.upper()
    deg = deg.set_index("gene_symbol")
    return deg["log2FC"].astype(float), deg["qvalue"].astype(float)


def effect_stats(case, control):
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    delta = float(np.mean(case) - np.mean(control))
    u, p = stats.mannwhitneyu(case, control, alternative="two-sided")
    rb = float(2 * u / (len(case) * len(control)) - 1)
    se = float(np.sqrt(np.var(case, ddof=1) / len(case) + np.var(control, ddof=1) / len(control)))
    if se > 0:
        df = se ** 4 / ((np.var(case, ddof=1) / len(case)) ** 2 / (len(case) - 1) +
                        (np.var(control, ddof=1) / len(control)) ** 2 / (len(control) - 1))
        margin = float(stats.t.ppf(0.975, df) * se)
    else:
        margin = 0.0
    return {
        "mean_pe_log2": float(np.mean(case)),
        "mean_control_log2": float(np.mean(control)),
        "delta_validation": delta,
        "ci95_low": delta - margin, "ci95_high": delta + margin,
        "rank_biserial": rb, "p_validation": float(p),
    }


def load_expression():
    raw = pd.read_csv(EXPR, sep="\t")
    first = raw.columns[0]
    # row ids are ENSG00000XXXX$SYMBOL
    parts = raw[first].astype(str).str.split("$", n=1, expand=True)
    raw["ensembl_id"] = parts[0]
    raw["symbol"] = parts[1].str.upper()
    raw = raw.dropna(subset=["symbol"])
    raw = raw[raw["symbol"] != ""]
    # collapse duplicate symbols by mean
    sample_cols = [c for c in raw.columns if c not in (first, "ensembl_id", "symbol")]
    expr = raw.groupby("symbol")[sample_cols].mean()
    # DESeq2 normalised counts -> log2(norm + 1)
    expr = np.log2(expr + 1.0)
    return expr


def validate_one(gene, log2fc_disc, qval_disc, expr, pe, control):
    if gene not in expr.index:
        return {"gene": gene, "log2FC_discovery": log2fc_disc.get(gene, np.nan),
                "qvalue_discovery": qval_disc.get(gene, np.nan),
                "n_pe": len(pe), "n_control": len(control),
                "testable": False, "status": "not_measured"}
    a = expr.loc[gene, pe].astype(float).values
    b = expr.loc[gene, control].astype(float).values
    row = {"gene": gene, "log2FC_discovery": log2fc_disc.get(gene, np.nan),
           "qvalue_discovery": qval_disc.get(gene, np.nan),
           "n_pe": len(pe), "n_control": len(control),
           "testable": True, "status": "testable"}
    row.update(effect_stats(a, b))
    return row


def validate_set(genes, log2fc_disc, qval_disc, expr, pe, control):
    rows = [validate_one(g, log2fc_disc, qval_disc, expr, pe, control) for g in genes]
    df = pd.DataFrame(rows)
    if not df.empty and df["testable"].any():
        pvals = df["p_validation"].where(df["testable"]).astype(float)
        rej, fdr, _, _ = multipletests(pvals.fillna(1.0), method="fdr_bh")
        df["FDR"] = fdr
        df["FDR"] = df["FDR"].where(df["testable"])
    else:
        df["FDR"] = np.nan
    df["direction_concordant"] = np.sign(df["log2FC_discovery"]) == np.sign(df["delta_validation"])
    df["direction_concordant"] = df["direction_concordant"].where(df["testable"])
    return df


def geneset_score(expr, genes, pe, control):
    available = [g for g in genes if g in expr.index]
    if not available:
        return {"n_available": 0, "delta": np.nan, "p": np.nan, "rb": np.nan}
    score = expr.loc[available].mean(axis=0)
    a = score[pe].values.astype(float)
    b = score[control].values.astype(float)
    delta = float(np.mean(a) - np.mean(b))
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = float(2 * u / (len(a) * len(b)) - 1)
    return {"n_available": len(available), "delta": delta, "p": float(p), "rb": rb}


def main():
    meta = parse_series_matrix(MATRIX)
    meta = meta[meta["group"].isin(["PE", "control"])].reset_index(drop=True)
    meta.to_csv(RES / f"{PREFIX}metadata.csv", index=False)

    expr = load_expression()
    common = [s for s in meta["title"] if s in expr.columns]
    meta = meta[meta["title"].isin(common)].reset_index(drop=True)
    expr = expr[meta["title"].values]
    pe = meta.loc[meta["group"] == "PE", "title"].tolist()
    control = meta.loc[meta["group"] == "control", "title"].tolist()

    log2fc_disc, qval_disc = load_discovery_deg()
    deg_genes = list(log2fc_disc.index)

    core_summary = json.loads((RES / "moduleB_core_run_summary.json").read_text(encoding="utf-8"))
    core17 = [g.upper() for g in core_summary.get("candidates", [])]
    if len(core17) != 17:
        core17 = ["CA1", "CLC", "COL17A1", "FLT1", "GPR146", "GUCA2A", "HIST1H1T",
                  "KLRD1", "LEPREL1", "LTF", "OR51B4", "SH3BP5", "SH3PXD2A",
                  "SLC16A10", "SMARCA1", "STAT4", "VNN2"]

    cand = validate_set(core17, log2fc_disc, qval_disc, expr, pe, control)
    cand.to_csv(RES / f"{PREFIX}candidate_validation.csv", index=False)

    deg = validate_set(deg_genes, log2fc_disc, qval_disc, expr, pe, control)
    deg.to_csv(RES / f"{PREFIX}deg_validation.csv", index=False)

    core = cand[cand["gene"].isin(CORE)].copy()
    core.to_csv(RES / f"{PREFIX}core_validation.csv", index=False)

    # Hallmark Hypoxia
    gmt_path = GS / "MSigDB_Hallmark_2020.gmt"
    hypoxia_genes = []
    if gmt_path.exists():
        with open(gmt_path, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if fields and fields[0].strip() == "Hypoxia":
                    hypoxia_genes = [x.strip().upper() for x in fields[2:] if x.strip()]
                    break
    hypoxia = geneset_score(expr, hypoxia_genes, pe, control)

    # Pathway validation for Fig.5 panel C
    pathway_map = {
        "Platelet activation": ("KEGG_2021_Human.gmt", "Platelet activation"),
        "Complement & coagulation cascades": ("KEGG_2021_Human.gmt", "Complement and coagulation cascades"),
        "Angiogenesis": ("MSigDB_Hallmark_2020.gmt", "Angiogenesis"),
        "Inflammatory response": ("MSigDB_Hallmark_2020.gmt", "Inflammatory Response"),
        "Hypoxia": ("MSigDB_Hallmark_2020.gmt", "Hypoxia"),
        "PPAR signalling (lipid metabolism)": ("KEGG_2021_Human.gmt", "PPAR signaling pathway"),
        "Apoptosis": ("MSigDB_Hallmark_2020.gmt", "Apoptosis"),
    }
    gmt_cache = {}
    def load_geneset(gmt_file, set_name):
        if gmt_file not in gmt_cache:
            sets = {}
            with open(GS / gmt_file, encoding="utf-8") as f:
                for line in f:
                    fields = line.rstrip("\n").split("\t")
                    sets[fields[0]] = [x.strip().upper() for x in fields[2:] if x.strip()]
            gmt_cache[gmt_file] = sets
        return gmt_cache[gmt_file].get(set_name, [])

    disc_path = pd.read_csv(RES / "moduleB_validation_concordance.csv")
    disc_path = disc_path.drop_duplicates("pathway").set_index("pathway")
    path_rows = []
    for pw, (gmt_file, set_name) in pathway_map.items():
        genes = load_geneset(gmt_file, set_name)
        score = geneset_score(expr, genes, pe, control)
        disc_delta = float(disc_path.loc[pw, "delta_discovery"]) if pw in disc_path.index else np.nan
        path_rows.append({
            "pathway": pw,
            "n_genes_available": score["n_available"],
            "delta_validation": score["delta"],
            "p_validation": score["p"],
            "rank_biserial": score["rb"],
            "delta_discovery": disc_delta,
            "direction_concordant": bool(np.sign(score["delta"]) == np.sign(disc_delta)) if pd.notna(disc_delta) and pd.notna(score["delta"]) else False,
        })
    pd.DataFrame(path_rows).to_csv(RES / f"{PREFIX}pathway_validation.csv", index=False)

    # Ancestry-stratified sensitivity for candidates + Hypoxia
    strat = []
    for anc in sorted(meta["ancestry"].unique()):
        sub = meta[meta["ancestry"] == anc]
        spe = sub.loc[sub["group"] == "PE", "title"].tolist()
        sctl = sub.loc[sub["group"] == "control", "title"].tolist()
        if len(spe) >= 2 and len(sctl) >= 2:
            sub_cand = validate_set(core17, log2fc_disc, qval_disc, expr, spe, sctl)
            sub_cand["ancestry"] = anc
            strat.append(sub_cand)
    if strat:
        pd.concat(strat, ignore_index=True).to_csv(
            RES / f"{PREFIX}candidate_validation_by_ancestry.csv", index=False)

    # Summary
    n_concordant = int(deg["direction_concordant"].fillna(False).sum())
    n_testable = int(deg["testable"].sum())
    n_fdr05 = int((deg["FDR"] < 0.05).sum())
    n_cand_concordant = int(cand["direction_concordant"].fillna(False).sum())
    n_cand_testable = int(cand["testable"].sum())
    n_cand_fdr05 = int((cand["FDR"] < 0.05).sum())

    summary = {
        "dataset": "GSE234729",
        "platform": "GPL24676 Illumina NovaSeq 6000 (RNA-seq)",
        "phenotype": "severe preeclampsia (sPE) vs normotensive control",
        "n_total": len(meta),
        "n_pe": len(pe),
        "n_control": len(control),
        "ancestry_counts": {str(k): {str(k2): int(v2) for k2, v2 in v.items()}
                             for k, v in meta.groupby(["group", "ancestry"]).size().unstack(fill_value=0).to_dict().items()},
        "gestational_age_available": False,
        "fetal_sex_available": True,
        "n_genes_after_collapse": len(expr),
        "candidates": {
            "n_requested": len(core17),
            "n_testable": n_cand_testable,
            "n_direction_concordant": n_cand_concordant,
            "pct_concordant": round(100 * n_cand_concordant / max(n_cand_testable, 1), 1),
            "n_FDR_lt_0.05": n_cand_fdr05,
        },
        "core_genes": {
            row["gene"]: {
                "delta": round(row["delta_validation"], 4),
                "FDR": round(row["FDR"], 6) if pd.notna(row["FDR"]) else None,
                "direction_concordant": bool(row["direction_concordant"]),
            }
            for _, row in core.iterrows()
        },
        "deg_validation": {
            "n_requested": len(deg_genes),
            "n_testable": n_testable,
            "n_direction_concordant": n_concordant,
            "pct_concordant": round(100 * n_concordant / max(n_testable, 1), 1),
            "n_FDR_lt_0.05": n_fdr05,
        },
        "hypoxia": {
            "n_genes_in_geneset": len(hypoxia_genes),
            "n_available": hypoxia["n_available"],
            "delta_mean_log2": round(hypoxia["delta"], 4),
            "p_value": round(hypoxia["p"], 6),
            "rank_biserial": round(hypoxia["rb"], 4),
            "direction_same_as_discovery": bool(np.sign(hypoxia["delta"]) > 0),
        },
        "note": "Frozen validation: no gene re-selection or tuning. Cross-platform (microarray discovery vs RNA-seq validation); effect sizes not directly interchangeable. GA not available; sPE may overlap with but is not identical to EOPE.",
    }
    (RES / f"{PREFIX}summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=lambda o: None if isinstance(o, float) and (np.isnan(o) or np.isinf(o)) else o),
        encoding="utf-8")
    # Second pass to sanitize any remaining NaN/inf in nested containers
    raw = (RES / f"{PREFIX}summary.json").read_text(encoding="utf-8")
    raw = raw.replace(": NaN", ": null").replace(": -Infinity", ": null").replace(": Infinity", ": null")
    (RES / f"{PREFIX}summary.json").write_text(raw, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
