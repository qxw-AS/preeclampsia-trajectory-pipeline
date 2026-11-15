"""
Module B Step 10 - independent exploratory external validation in GSE271628.

GSE271628 is audited from its GEO series matrix rather than from the proposal.
The contrast and gene sets are fixed before expression analysis: all placental
Preeclampsia vs Control samples, 17 discovery candidates (including five core
genes), the 196 discovery DEGs, and Hallmark Hypoxia. No gene selection,
parameter tuning, or expression-driven sample exclusion is performed here.
"""
import gzip
import json
import os
import re
import time
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
DATA = OUT / "data" / "geo" / "gse271628"
RES = OUT / "results"
FIG = OUT / "figures"
GS = OUT / "genesets"
for folder in (DATA, RES, FIG):
    folder.mkdir(parents=True, exist_ok=True)

PREFIX = "moduleB_gse271628_"
MATRIX = DATA / "GSE271628_series_matrix.txt.gz"
COUNTS = DATA / "GSE271628_raw_counts.txt.gz"
FILELIST = DATA / "GSE271628_filelist.txt"
LOG = DATA / "GSE271628_download_log.json"
CORE = ["FLT1", "SMARCA1", "GUCA2A", "HIST1H1T", "KLRD1"]
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE271nnn/GSE271628"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
})


def append_log(record):
    records = []
    if LOG.exists():
        try:
            records = json.loads(LOG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            records = []
    records.append(record)
    LOG.write_text(json.dumps(records, indent=2), encoding="utf-8")


def download(url, path, required=True, retries=2):
    if path.exists() and path.stat().st_size > 0:
        return True
    for attempt in range(1, retries + 1):
        rec = {
            "url": url, "file": path.name, "attempt": attempt,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                rec["http_status"] = response.status_code
                rec["content_length"] = response.headers.get("Content-Length")
                if response.status_code == 200:
                    part = path.with_name(path.name + ".part")
                    n = 0
                    with part.open("wb") as handle:
                        for chunk in response.iter_content(65536):
                            if chunk:
                                handle.write(chunk)
                                n += len(chunk)
                    part.replace(path)
                    rec.update({"status": "downloaded", "bytes": n})
                    append_log(rec)
                    print(f"Downloaded {path.name}: {n:,} bytes", flush=True)
                    return True
                rec["status"] = "http_error"
                append_log(rec)
                print(f"{path.name}: HTTP {response.status_code}", flush=True)
        except Exception as exc:
            rec.update({"status": "exception", "error": f"{type(exc).__name__}: {exc}"})
            append_log(rec)
            print(f"{path.name}: {rec['error']}", flush=True)
    return not required and not path.exists()


def write_blocked(reason, stage, extra=None):
    payload = {
        "dataset": "GSE271628", "status": "blocked", "stage": stage,
        "reason": reason, "independent_external_validation": False,
        "gestational_age_available": False,
        "note": "GSE204835 was not reused or represented as this cohort.",
    }
    if extra:
        payload.update(extra)
    (RES / f"{PREFIX}status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BLOCKED at {stage}: {reason}", flush=True)


def parse_series_matrix(path):
    lines = gzip.open(path, "rt", encoding="utf-8", errors="replace").read().splitlines()
    attrs = {}
    series = {}
    for line in lines:
        fields = [x.strip('"') for x in line.split("\t")]
        if line.startswith("!Sample_"):
            attrs.setdefault(fields[0], []).append(fields[1:])
        elif line.startswith("!Series_"):
            series.setdefault(fields[0], []).append(fields[1:])
    gsm = attrs["!Sample_geo_accession"][0]

    def sample_field(name, default=""):
        rows = attrs.get(name, [])
        return rows[0] if rows else [default] * len(gsm)

    def characteristic(name):
        for row in attrs.get("!Sample_characteristics_ch1", []):
            if row and row[0].lower().startswith(name.lower() + ":"):
                return [x.split(":", 1)[1].strip() if ":" in x else "" for x in row]
        return [""] * len(gsm)

    def relation(prefix):
        row = sample_field("!Sample_relation")
        return [next((x for x in row if x.startswith(prefix)), "")] if row else [""] * len(gsm)

    # Multiple relation rows are possible; retain both SRA and BioSample explicitly.
    relations = attrs.get("!Sample_relation", [])
    sra, biosample = [], []
    for i in range(len(gsm)):
        vals = [row[i] for row in relations if i < len(row)]
        sra.append(next((x for x in vals if x.startswith("SRA:")), ""))
        biosample.append(next((x for x in vals if x.startswith("BioSample:")), ""))

    meta = pd.DataFrame({
        "gsm": gsm,
        "description": sample_field("!Sample_description"),
        "title": sample_field("!Sample_title"),
        "condition": characteristic("condition"),
        "tissue": characteristic("tissue"),
        "source_name": sample_field("!Sample_source_name_ch1"),
        "organism": sample_field("!Sample_organism_ch1"),
        "platform": sample_field("!Sample_platform_id"),
        "instrument": sample_field("!Sample_instrument_model"),
        "library_strategy": sample_field("!Sample_library_strategy"),
        "library_source": sample_field("!Sample_library_source"),
        "molecule": sample_field("!Sample_molecule_ch1"),
        "sra_relation": sra,
        "biosample_relation": biosample,
    })
    meta["count_col"] = meta["description"].str.replace(
        r"^([A-Za-z]+)\s*(\d+)$", r"\1_\2", regex=True)
    return meta, series, attrs


def ensembl_annotation(ids):
    cache = RES / f"{PREFIX}ensembl2sym.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import mygene
    mg = mygene.MyGeneInfo()
    records = []
    ids = list(ids)
    for start in range(0, len(ids), 1000):
        batch = ids[start:start + 1000]
        result = None
        last_error = ""
        for attempt in range(1, 3):
            try:
                result = mg.querymany(
                    batch, scopes="ensembl.gene", fields="symbol,alias",
                    species="human", as_dataframe=False, verbose=False)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"Annotation batch {start // 1000 + 1}, attempt {attempt}: {last_error}")
        if result is None:
            raise RuntimeError(f"MyGene annotation failed at batch {start // 1000 + 1}: {last_error}")
        for item in result:
            if item.get("notfound") or not item.get("symbol"):
                continue
            aliases = item.get("alias", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            records.append({
                "ensembl_id": str(item["query"]).split(".")[0],
                "symbol": str(item["symbol"]).upper(),
                "aliases": ";".join(sorted({str(x).upper() for x in aliases if x})),
            })
        print(f"Annotated {min(start + 1000, len(ids)):,}/{len(ids):,} Ensembl IDs", flush=True)
    mapping = pd.DataFrame(records)
    if mapping.empty:
        raise RuntimeError("MyGene returned no Ensembl-to-symbol mappings")
    mapping = mapping.groupby("ensembl_id", as_index=False).agg({
        "symbol": "first", "aliases": lambda x: ";".join(sorted(set(";".join(x).split(";")) - {""})),
    })
    mapping.to_csv(cache, index=False)
    return mapping


def load_gmt(path):
    genesets = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 2:
                genesets[fields[0]] = sorted({x.strip().upper() for x in fields[2:] if x.strip()})
    return genesets


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
        "mean_pe_log2CPM": float(np.mean(case)),
        "mean_control_log2CPM": float(np.mean(control)),
        "delta_validation_log2FC": delta,
        "ci95_low": delta - margin, "ci95_high": delta + margin,
        "rank_biserial": rb, "p_validation": float(p),
    }


def validate_genes(requested, discovery, expression, symbol_counts, alias_index, mapping, pe, control):
    rows = []
    for gene in requested:
        discovery_row = discovery.loc[gene] if gene in discovery.index else None
        row = {
            "gene": gene,
            "log2FC_discovery": float(discovery_row["log2FC"]) if discovery_row is not None else np.nan,
            "qvalue_discovery": float(discovery_row["qvalue"]) if discovery_row is not None else np.nan,
            "n_pe": len(pe), "n_control": len(control),
            "testable": False, "status": "not_measured_or_unmapped",
            "expression_symbol": "", "ensembl_ids": "", "symbol_resolution": "",
        }
        resolved = gene if gene in expression.index else ""
        resolution = "direct_symbol" if resolved else ""
        if not resolved:
            aliases = sorted(alias_index.get(gene, set()))
            expressed_aliases = [x for x in aliases if x in expression.index]
            all_aliases = [x for x in aliases if x in symbol_counts.index]
            if len(expressed_aliases) == 1:
                resolved = expressed_aliases[0]
                resolution = "unique_alias"
            elif len(expressed_aliases) > 1:
                row["status"] = "ambiguous_alias_after_filter"
            elif len(all_aliases) == 1:
                row["status"] = "low_expression_filtered"
                row["expression_symbol"] = all_aliases[0]
            elif len(all_aliases) > 1:
                row["status"] = "ambiguous_alias"
            elif gene in symbol_counts.index:
                row["status"] = "low_expression_filtered"
        if resolved:
            a = expression.loc[resolved, pe].astype(float)
            b = expression.loc[resolved, control].astype(float)
            row.update(effect_stats(a, b))
            row.update({
                "testable": True, "status": "testable", "expression_symbol": resolved,
                "ensembl_ids": ";".join(mapping.loc[mapping["symbol"] == resolved, "ensembl_id"]),
                "symbol_resolution": resolution,
            })
            row["same_direction"] = bool(np.sign(row["delta_validation_log2FC"]) == np.sign(row["log2FC_discovery"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["fdr_validation"] = np.nan
    valid = out["p_validation"].notna() if "p_validation" in out else pd.Series(False, index=out.index)
    if valid.any():
        out.loc[valid, "fdr_validation"] = multipletests(out.loc[valid, "p_validation"], method="fdr_bh")[1]
    return out


print("=" * 78)
print("STEP 10  GSE271628 independent exploratory external validation")
print("=" * 78)

import requests

# filelist.txt is useful when present, but some GEO series omit it (HTTP 404).
download(f"{BASE}/suppl/filelist.txt", FILELIST, required=False, retries=1)
if not download(f"{BASE}/matrix/GSE271628_series_matrix.txt.gz", MATRIX, required=True, retries=2):
    write_blocked("Series matrix could not be downloaded; see GSE271628_download_log.json.", "series_matrix_download")
    raise SystemExit(0)

meta, series, attrs = parse_series_matrix(MATRIX)
conditions = set(meta["condition"])
expected_conditions = {"Control", "Preeclampsia"}
if conditions != expected_conditions:
    write_blocked(
        f"Observed conditions {sorted(conditions)} do not provide the audited PE-vs-control contrast.",
        "design_audit", {"observed_conditions": sorted(conditions)})
    raise SystemExit(0)

# Audited, expression-independent inclusion rule: every human placental RNA-seq PE/control sample.
meta["group"] = meta["condition"].map({"Preeclampsia": "PE", "Control": "Control"})
meta["included"] = (meta["tissue"].str.lower() == "placental tissue") & meta["group"].notna()
meta["exclusion_reason"] = np.where(meta["included"], "", "not_placental_PE_or_control")
meta["gestational_age"] = np.nan
meta["pe_subtype"] = ""
meta["analysis_note"] = "Included by metadata; no expression-based exclusion"
meta.to_csv(RES / f"{PREFIX}metadata.csv", index=False)

supp = series.get("!Series_supplementary_file", [])
raw_urls = [x[0].replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
            for x in supp if x and x[0].endswith("GSE271628_raw_counts.txt.gz")]
raw_url = raw_urls[0] if raw_urls else f"{BASE}/suppl/GSE271628_raw_counts.txt.gz"
if not download(raw_url, COUNTS, required=True, retries=3):
    write_blocked(
        "Series metadata were obtained, but the raw count matrix could not be downloaded.",
        "counts_download", {"n_pe": int((meta.group == "PE").sum()),
                            "n_control": int((meta.group == "Control").sum())})
    raise SystemExit(0)

counts_raw = pd.read_csv(COUNTS, sep="\t")
id_col = counts_raw.columns[0]
counts = counts_raw.set_index(id_col)
counts.columns = [str(x) for x in counts.columns]
counts.index = counts.index.astype(str).str.split(".").str[0]
if not counts.index.is_unique:
    counts = counts.groupby(level=0).sum()
expected_cols = set(meta["count_col"])
if set(counts.columns) != expected_cols:
    write_blocked(
        "Count-matrix columns do not match sample descriptions in the series matrix.",
        "sample_matching", {"count_columns": list(counts.columns),
                            "metadata_count_columns": meta["count_col"].tolist()})
    raise SystemExit(0)

count_to_gsm = meta.set_index("count_col")["gsm"].to_dict()
counts = counts.rename(columns=count_to_gsm)
ordered_gsm = meta["gsm"].tolist()
counts = counts[ordered_gsm]
meta["library_size_raw"] = meta["count_col"].map(counts_raw.set_index(id_col).sum(axis=0)).astype(np.int64)
meta.to_csv(RES / f"{PREFIX}metadata.csv", index=False)

try:
    mapping = ensembl_annotation(counts.index)
except Exception as exc:
    write_blocked(f"Ensembl annotation failed: {type(exc).__name__}: {exc}", "annotation")
    raise SystemExit(0)

mapped = counts.loc[counts.index.isin(mapping["ensembl_id"])].copy()
id_to_symbol = mapping.drop_duplicates("ensembl_id").set_index("ensembl_id")["symbol"]
mapped.index = mapped.index.map(id_to_symbol)
symbol_counts = mapped.groupby(level=0).max()  # same Entrez/Ensembl collapse rule as prior validation
cpm = symbol_counts.div(symbol_counts.sum(axis=0), axis=1) * 1e6
keep = (cpm > 1).sum(axis=1) >= 0.3 * cpm.shape[1]
logcpm = np.log2(cpm.loc[keep] + 1)
logcpm.index.name = "gene_symbol"
logcpm.to_csv(RES / f"{PREFIX}expression_log2cpm.csv.gz", compression="gzip")

alias_index = {}
for rec in mapping.itertuples(index=False):
    names = {rec.symbol} | {x for x in str(rec.aliases).split(";") if x and x != "nan"}
    for name in names:
        alias_index.setdefault(name.upper(), set()).add(rec.symbol)

pe = meta.loc[meta["included"] & (meta["group"] == "PE"), "gsm"].tolist()
control = meta.loc[meta["included"] & (meta["group"] == "Control"), "gsm"].tolist()

candidate_summary = json.loads((RES / "moduleB_candidate_summary.json").read_text(encoding="utf-8"))
candidates = candidate_summary["candidates"]
deg_all = pd.read_csv(RES / "moduleB_deg_all.csv", index_col=0)
deg196 = pd.read_csv(RES / "moduleB_deg_significant.csv", index_col=0)
assert len(deg196) == 196, f"Expected 196 frozen discovery DEGs, observed {len(deg196)}"
assert candidates == list(dict.fromkeys(candidates)) and len(candidates) == 17

candidate_validation = validate_genes(
    candidates, deg_all, logcpm, symbol_counts, alias_index, mapping, pe, control)
candidate_info = pd.read_csv(RES / "moduleB_candidate_genes.csv").set_index("gene")
candidate_validation["is_core"] = candidate_validation["gene"].isin(CORE)
candidate_validation["in_lasso"] = candidate_validation["gene"].map(candidate_info["in_lasso"])
candidate_validation["in_svm_rfe"] = candidate_validation["gene"].map(candidate_info["in_svm_rfe"])
candidate_validation.to_csv(RES / f"{PREFIX}candidate_validation.csv", index=False)
candidate_validation.loc[candidate_validation["is_core"]].to_csv(
    RES / f"{PREFIX}core_validation.csv", index=False)

deg_validation = validate_genes(
    deg196.index.tolist(), deg196, logcpm, symbol_counts, alias_index, mapping, pe, control)
deg_validation["discovery_direction"] = np.where(deg_validation["log2FC_discovery"] > 0, "up", "down")
deg_validation.to_csv(RES / f"{PREFIX}deg196_validation.csv", index=False)
testable_deg = deg_validation.loc[deg_validation["testable"]].copy()
rho, rho_p = stats.spearmanr(testable_deg["log2FC_discovery"], testable_deg["delta_validation_log2FC"])

# The pre-specified Hallmark Hypoxia set and ssGSEA settings match Module B Steps 3/5.
hallmark = load_gmt(GS / "MSigDB_Hallmark_2020.gmt")
pathways = {"Hypoxia": hallmark["Hypoxia"]}
import gseapy
ss = gseapy.ssgsea(
    data=logcpm, gene_sets=pathways, outdir=None, sample_norm_method="rank",
    min_size=15, max_size=500, permutation_num=0, no_plot=True, threads=4, seed=123)
scores = ss.res2d.pivot(index="Name", columns="Term", values="ES").loc[ordered_gsm, ["Hypoxia"]]
scores.index.name = "gsm"
scores.reset_index().merge(meta[["gsm", "description", "group"]], on="gsm").to_csv(
    RES / f"{PREFIX}ssgsea_scores.csv", index=False)
discovery_pathway = pd.read_csv(RES / "moduleB_ssgsea_group_stats.csv").set_index("pathway")
pathway_row = {"pathway": "Hypoxia", "n_pe": len(pe), "n_control": len(control)}
pathway_stats = effect_stats(scores.loc[pe, "Hypoxia"], scores.loc[control, "Hypoxia"])
pathway_row.update({
    "mean_pe_ssgsea_ES": pathway_stats["mean_pe_log2CPM"],
    "mean_control_ssgsea_ES": pathway_stats["mean_control_log2CPM"],
    "delta_validation": pathway_stats["delta_validation_log2FC"],
    "rank_biserial": pathway_stats["rank_biserial"],
    "p_validation": pathway_stats["p_validation"],
})
pathway_row["delta_discovery"] = float(discovery_pathway.loc["Hypoxia", "delta"])
pathway_row["direction_concordant"] = bool(
    np.sign(pathway_row["delta_validation"]) == np.sign(pathway_row["delta_discovery"]))
pathway_validation = pd.DataFrame([pathway_row])
pathway_validation.to_csv(RES / f"{PREFIX}pathway_validation.csv", index=False)

# Figure 1: frozen DEG concordance and candidate effects with Welch 95% CIs.
fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
colors = np.where(testable_deg["same_direction"], "#2A7F62", "#C45A32")
ax.scatter(testable_deg["log2FC_discovery"], testable_deg["delta_validation_log2FC"],
           s=20, c=colors, alpha=0.78, linewidths=0)
ax.axhline(0, color="#888888", lw=0.7)
ax.axvline(0, color="#888888", lw=0.7)
label_offsets = {"FLT1": (7, -12), "GUCA2A": (-52, 7), "SMARCA1": (7, 7), "KLRD1": (7, -3)}
for gene in CORE:
    hit = testable_deg.loc[testable_deg["gene"] == gene]
    if not hit.empty:
        r = hit.iloc[0]
        ax.annotate(gene, (r["log2FC_discovery"], r["delta_validation_log2FC"]),
                    xytext=label_offsets.get(gene, (3, 3)), textcoords="offset points",
                    fontsize=7, fontweight="bold")
ax.set_xlabel("Discovery log$_2$FC (GSE75010)")
ax.set_ylabel("Validation $\Delta$ mean log$_2$CPM (PE − control)")
ax.set_title("A  Frozen 196-DEG signature", loc="left", fontweight="bold")
ax.text(0.03, 0.97,
        f"Testable {len(testable_deg)}/196\nSpearman $\\rho$={rho:.3f}, p={rho_p:.2e}\n"
        f"Same direction {testable_deg['same_direction'].mean() * 100:.1f}%",
        transform=ax.transAxes, va="top", fontsize=8)

ax = axes[1]
plot_candidates = candidate_validation.loc[candidate_validation["testable"]].copy()
plot_candidates["gene"] = pd.Categorical(plot_candidates["gene"], categories=candidates, ordered=True)
plot_candidates = plot_candidates.sort_values("gene", ascending=False)
y = np.arange(len(plot_candidates))
point_colors = np.where(plot_candidates["same_direction"], "#2A7F62", "#C45A32")
ax.hlines(y, plot_candidates["ci95_low"], plot_candidates["ci95_high"], color=point_colors, lw=1.2)
ax.scatter(plot_candidates["delta_validation_log2FC"], y, c=point_colors, s=30, zorder=3)
ax.axvline(0, color="#888888", lw=0.7)
ax.set_yticks(y, plot_candidates["gene"].astype(str))
for tick, gene in zip(ax.get_yticklabels(), plot_candidates["gene"].astype(str)):
    if gene in CORE:
        tick.set_fontweight("bold")
ax.set_xlabel("Validation $\Delta$ mean log$_2$CPM (95% CI)")
ax.set_title(f"B  Pre-specified 17 candidates ({len(plot_candidates)}/17 testable)", loc="left", fontweight="bold")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
untestable = candidate_validation.loc[~candidate_validation["testable"], "gene"].tolist()
fig.text(0.54, 0.015,
         "Green: concordant; orange: discordant; bold: core; no retuning\n"
         f"Low-expression filtered: {', '.join(untestable)}",
         ha="center", va="bottom", fontsize=8)
fig.suptitle("GSE271628 independent external validation (PE n=7, control n=4; exploratory)", y=0.995)
fig.tight_layout(rect=(0, 0.055, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"figD7_gse271628_validation.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 2: unsupervised QC and the pre-specified hypoxia pathway.
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
pc = pca.fit_transform(logcpm.T.values)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
ax = axes[0]
for group, color in (("Control", "#4C72B0"), ("PE", "#C44E52")):
    samples = meta.loc[meta["group"] == group, "gsm"].tolist()
    idx = [ordered_gsm.index(x) for x in samples]
    ax.scatter(pc[idx, 0], pc[idx, 1], s=38, c=color, label=f"{group} (n={len(idx)})", alpha=0.85)
    for i in idx:
        description = meta.loc[meta["gsm"] == ordered_gsm[i], "description"].iloc[0]
        pca_offsets = {"PE1": (4, 8), "PE4": (4, -11), "PE6": (4, -10), "PE7": (4, 5)}
        ax.annotate(description, (pc[i, 0], pc[i, 1]),
                    xytext=pca_offsets.get(description, (3, 3)), textcoords="offset points", fontsize=7)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax.set_title("A  log$_2$CPM PCA (no exclusions)", loc="left", fontweight="bold")
ax.legend(frameon=False)

ax = axes[1]
plot_scores = scores.join(meta.set_index("gsm")[["group", "description"]])
positions = [0, 1]
for pos, group, color in zip(positions, ["Control", "PE"], ["#4C72B0", "#C44E52"]):
    vals = plot_scores.loc[plot_scores["group"] == group, "Hypoxia"].astype(float).values
    ax.boxplot(vals, positions=[pos], widths=0.42, showfliers=False,
               boxprops={"color": color}, medianprops={"color": "#222222"},
               whiskerprops={"color": color}, capprops={"color": color})
    jitter = np.linspace(-0.07, 0.07, len(vals))
    ax.scatter(np.full(len(vals), pos) + jitter, vals, s=30, c=color, alpha=0.85, zorder=3)
ax.set_xticks(positions, ["Control\n(n=4)", "PE\n(n=7)"])
ax.set_ylabel("Hallmark Hypoxia ssGSEA ES")
ax.set_title("B  Pre-specified hypoxia pathway", loc="left", fontweight="bold")
ax.text(0.03, 0.97,
        f"Mann–Whitney p={pathway_row['p_validation']:.3g}\n"
        f"PE − control={pathway_row['delta_validation']:.1f}",
        transform=ax.transAxes, va="top", fontsize=8)
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"figD7_gse271628_qc_hypoxia.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

core_test = candidate_validation.loc[candidate_validation["is_core"]]
candidate_test = candidate_validation.loc[candidate_validation["testable"]]
summary = {
    "dataset": "GSE271628", "status": "success_exploratory",
    "independent_external_validation": True,
    "platform": "GPL24676 (Illumina NovaSeq 6000), placental bulk RNA-seq, hg38",
    "n_samples": int(len(meta)), "n_pe": len(pe), "n_control": len(control),
    "n_excluded": int((~meta["included"]).sum()),
    "sample_exclusion_rule": "Exclude only samples not annotated as placental-tissue PE/control; none excluded.",
    "gestational_age_available": False, "pe_subtype_available": False,
    "comparability": "Placental PE-vs-control comparison is valid, but GA matching and EOPE-specific validation are impossible from available metadata.",
    "n_ensembl_rows": int(counts.shape[0]), "n_ensembl_annotated": int(mapped.shape[0]),
    "n_symbol_genes_before_filter": int(symbol_counts.shape[0]),
    "n_symbol_genes_after_cpm_filter": int(logcpm.shape[0]),
    "normalization": "Symbol-level maximum count collapse, CPM, retain CPM>1 in >=30% samples, log2(CPM+1)",
    "test": "Two-sided Mann-Whitney U; delta is PE minus control mean log2CPM; BH FDR within each frozen gene family",
    "n_candidates": len(candidates), "n_candidates_testable": int(len(candidate_test)),
    "n_candidates_same_direction": int(candidate_test["same_direction"].sum()),
    "n_candidates_fdr05": int((candidate_test["fdr_validation"] < 0.05).sum()),
    "n_core": len(CORE), "n_core_testable": int(core_test["testable"].sum()),
    "n_core_same_direction": int(core_test["same_direction"].sum()),
    "n_core_fdr05": int((core_test["fdr_validation"] < 0.05).sum()),
    "n_discovery_degs": int(len(deg196)), "n_degs_testable": int(len(testable_deg)),
    "n_degs_same_direction": int(testable_deg["same_direction"].sum()),
    "deg_same_direction_pct": float(testable_deg["same_direction"].mean() * 100),
    "deg_spearman_rho": float(rho), "deg_spearman_p": float(rho_p),
    "n_degs_fdr05": int((testable_deg["fdr_validation"] < 0.05).sum()),
    "hypoxia_delta_discovery": pathway_row["delta_discovery"],
    "hypoxia_delta_validation": pathway_row["delta_validation"],
    "hypoxia_p_validation": pathway_row["p_validation"],
    "hypoxia_direction_concordant": pathway_row["direction_concordant"],
    "gene_selection_or_parameter_tuning_in_gse271628": False,
    "interpretation": "Exploratory external validation only: only 11 samples, no gestational-age or PE-subtype annotation, and cross-platform effect magnitudes are not directly interchangeable.",
    "gse204835_reused_as_this_cohort": False,
}
(RES / f"{PREFIX}summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

design_audit = {
    "series_title": series.get("!Series_title", [[""]])[0][0],
    "series_overall_design": series.get("!Series_overall_design", [[""]])[0][0],
    "series_type": series.get("!Series_type", [[""]])[0][0],
    "platform_id": series.get("!Series_platform_id", [[""]])[0][0],
    "observed_groups": meta["condition"].value_counts().to_dict(),
    "observed_tissues": meta["tissue"].value_counts().to_dict(),
    "observed_instruments": meta["instrument"].value_counts().to_dict(),
    "metadata_characteristic_fields": sorted({x[0].split(":", 1)[0] for x in attrs.get("!Sample_characteristics_ch1", [])}),
    "predefined_contrast": "All placental Preeclampsia vs Control samples",
    "n_included_pe": len(pe), "n_included_control": len(control),
    "excluded_samples": meta.loc[~meta["included"], ["gsm", "exclusion_reason"]].to_dict("records"),
    "limitations": ["No gestational-age field", "No PE onset/severity subtype field", "Small control group (n=4)"],
    "filelist_txt_available": FILELIST.exists(),
    "counts_source": raw_url,
}
(RES / f"{PREFIX}design_audit.json").write_text(
    json.dumps(design_audit, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Saved GSE271628 metadata, expression, validation tables, summary and figD7 figures.")
