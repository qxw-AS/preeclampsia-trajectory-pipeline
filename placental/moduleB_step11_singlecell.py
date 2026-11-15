"""Direction 8: full eight-donor GSE290578 Deep_placenta reanalysis from raw matrices."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import math
import os
import tarfile
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from scipy import sparse, stats
from scipy.io import mmread
from sklearn.neighbors import KNeighborsTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
GEO = ROOT / "data" / "geo"
DATA = GEO / "gse290578"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SERIES = GEO / "GSE290578_series_matrix.txt.gz"
FILELIST = GEO / "GSE290578_filelist.txt"
SELECTED_DONORS = ("1", "2", "3", "4", "5", "6", "7", "8")
SELECTION_RATIONALE = (
    "All eight paired donors in GSE290578: controls Norm_D01/D04/D06/D07 and PE_D02/D03/D05/D08; "
    "offspring sex is female/male in 1/3 PE donors and 2/2 control donors. One Deep_placenta capture per donor was used so that "
    "all samples come from the same anatomical layer. Selection preceded expression analysis. PE remains fully separated from "
    "controls by gestational age (PE 28+5 to 34+1 weeks versus controls 38+2 to 39+6 weeks), which is a "
    "structural limitation of this dataset and not resolved by including more donors."
)
PREFIX = "moduleB_singlecell_"
RANDOM_STATE = 0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.linewidth": 0.7,
})
sc.settings.verbosity = 1
sc.settings.seed = RANDOM_STATE

PRIMARY_MARKERS = {
    "Trophoblast": ["KRT7", "KRT8", "KRT18", "GATA3", "TFAP2A"],
    "EVT": ["KRT7", "HLA-G", "MUC15", "ITGA5", "MCAM"],
    "Syncytiotrophoblast": ["KRT7", "ERVFRD-1", "ERVFRD1", "CGB", "CGB5", "CGB8", "CSH1", "PSG1", "PSG3"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "ERG", "FLT1"],
    "Macrophage": ["CD68", "CD14", "LYZ", "CSF1R", "AIF1", "C1QA"],
    "T_cell": ["PTPRC", "CD3D", "CD3E", "TRAC", "IL7R"],
    "NK_cell": ["PTPRC", "KLRD1", "NKG7", "GNLY", "NCR1"],
    "Neutrophil": ["S100A8", "S100A9", "FCGR3B", "CSF3R", "LTF", "LCN2", "CAMP", "RETN", "PGLYRP1"],
    "B_cell": ["PTPRC", "CD79A", "CD79B", "MS4A1", "BANK1"],
    "Stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"],
    "Mural": ["RGS5", "ACTA2", "TAGLN", "MCAM", "DES"],
    "Erythrocyte": ["HBB", "HBA1", "HBA2", "ALAS2"],
    "Platelet": ["PPBP", "PF4", "ITGA2B", "GP9"],
}
DOTPLOT_MARKERS = {
    "Trophoblast": ["KRT7", "KRT8"],
    "EVT": ["HLA-G"],
    "Syncytio": ["ERVFRD-1", "CGB5", "CGB8"],
    "Endothelial": ["PECAM1", "VWF", "FLT1"],
    "Immune": ["PTPRC"],
    "Macrophage": ["CD68", "LYZ"],
    "T/NK": ["CD3D", "KLRD1", "NKG7"],
    "Neutrophil": ["S100A8", "LTF", "FCGR3B"],
    "Stromal": ["DCN", "COL1A1"],
}
REVIEWED_CLUSTER_LABELS = {
    "2": {
        "expected_n_cells": 2099,
        "expected_automatic_label": "Platelet",
        "label": "T_cell",
        "note": "Strong PTPRC/CD3D/CD3E/TRAC support; PPBP/PF4/ITGA2B/GP9 essentially absent",
    },
    "21": {
        "expected_n_cells": 111,
        "expected_automatic_label": "Trophoblast",
        "label": "EVT",
        "note": "HLA-G/MCAM/ITGA5 support with FN1/PAPPA2/TIMP3/NOTUM enrichment; syncytiotrophoblast markers unsupported",
    },
    "32": {
        "expected_n_cells": 17,
        "expected_automatic_label": "Platelet",
        "label": "Unassigned",
        "note": "Only 17 cells with mixed endothelial, stromal and mural profiles; platelet markers unsupported",
    },
}
HYPOXIA_GENES = [
    "HIF1A", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ENO1",
    "BNIP3", "CA9", "PDK1", "HK2", "ADM", "NDRG1",
]
PROLIFERATION_GENES = ["MKI67", "TOP2A", "PCNA", "TYMS"]
INDIVIDUAL_GENES = ["HIF1A", "FLT1", "SMARCA1", "VEGFA", "VEGFB", "PGF", "KDR"]
SCORE_COLUMNS = ["hypoxia_score", "candidate_score", "proliferation_score"]
EXPRESSION_COLUMNS = [f"{g}_expression" for g in INDIVIDUAL_GENES]


def read_series_metadata() -> pd.DataFrame:
    rows = {}
    with gzip.open(SERIES, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_"):
                fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
                key = fields[0]
                values = [v.strip('"') for v in fields[1:]]
                rows.setdefault(key, []).append(values)
    n = len(rows["!Sample_geo_accession"][0])
    records = []
    for i in range(n):
        title = rows["!Sample_title"][0][i]
        donor = title.rsplit(" ", 1)[-1]
        group = "PE" if "preeclampsia" in title.lower() else "control"
        layer = "MF" if "Maternal-Fetal" in title else "Deep_placenta"
        characteristics = [values[i] for values in rows["!Sample_characteristics_ch1"]]
        sex = next(v.split(":", 1)[-1].strip() for v in characteristics if v.startswith("offspring sex:"))
        gestational_week = next(v.split(":", 1)[-1].strip() for v in characteristics if v.startswith("gestational week:"))
        birth_description = next(v.split(":", 1)[-1].strip() for v in characteristics if v.startswith("sample description:"))
        record = {
            "gsm": rows["!Sample_geo_accession"][0][i],
            "title": title,
            "donor_id": f"D{int(donor):02d}",
            "donor_number": donor,
            "group": group,
            "layer": layer,
            "offspring_sex": sex,
            "gestational_week": gestational_week,
            "birth_description": birth_description,
            "source": rows["!Sample_source_name_ch1"][0][i],
            "selected_donor": donor in SELECTED_DONORS,
            "selected": donor in SELECTED_DONORS and layer == "Deep_placenta",
        }
        for j, kind in enumerate(("barcodes", "features", "matrix"), start=1):
            url = rows[f"!Sample_supplementary_file_{j}"][0][i].replace("ftp://", "https://", 1)
            record[f"{kind}_url"] = url
            record[f"{kind}_filename"] = url.rsplit("/", 1)[-1]
        records.append(record)
    meta = pd.DataFrame(records)
    assert len(meta) == 16 and meta.gsm.nunique() == 16
    assert meta.donor_id.nunique() == 8
    assert (meta.groupby("donor_id").layer.nunique() == 2).all()
    assert meta.groupby("group").donor_id.nunique().to_dict() == {"PE": 4, "control": 4}
    return meta


def read_filelist() -> pd.DataFrame:
    files = pd.read_csv(FILELIST, sep="\t", skiprows=1,
                        names=["entry_type", "filename", "time", "bytes", "type"])
    files = files[files.entry_type == "File"].copy()
    files["bytes"] = files.bytes.astype(int)
    return files


def build_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = read_series_metadata()
    files = read_filelist()
    size_map = files.set_index("filename").bytes.to_dict()
    file_records = []
    for row in meta.itertuples(index=False):
        capture_id = f"{row.gsm}_{row.donor_id}_{row.layer}"
        for kind in ("barcodes", "features", "matrix"):
            filename = getattr(row, f"{kind}_filename")
            file_records.append({
                "gsm": row.gsm, "capture_id": capture_id, "donor_id": row.donor_id,
                "group": row.group, "layer": row.layer, "kind": kind,
                "filename": filename, "url": getattr(row, f"{kind}_url"),
                "expected_bytes": size_map[filename], "selected": row.selected,
            })
        for kind in ("barcodes", "features", "matrix"):
            meta.loc[meta.gsm == row.gsm, f"{kind}_bytes"] = size_map[getattr(row, f"{kind}_filename")]
    file_manifest = pd.DataFrame(file_records)
    assert len(file_manifest) == 48 and file_manifest.expected_bytes.notna().all()
    selected = meta[meta.selected]
    assert selected.donor_id.nunique() == 8 and len(selected) == 8
    assert set(selected.layer) == {"Deep_placenta"}
    for donor in SELECTED_DONORS:
        pair = meta[meta.donor_number == donor]
        assert int(pair.selected.sum()) == 1
        assert pair.loc[pair.selected, "layer"].iloc[0] == "Deep_placenta"
    assert selected.groupby("group").donor_id.nunique().to_dict() == {"PE": 4, "control": 4}
    assert selected.groupby(["group", "offspring_sex"]).donor_id.nunique().to_dict() == {
        ("PE", "female"): 1, ("PE", "male"): 3,
        ("control", "female"): 2, ("control", "male"): 2,
    }
    meta["capture_id"] = meta.gsm + "_" + meta.donor_id + "_" + meta.layer
    meta.to_csv(RESULTS / f"{PREFIX}sample_manifest.csv", index=False)
    file_manifest.to_csv(RESULTS / f"{PREFIX}file_manifest.csv", index=False)
    return meta, file_manifest


def gzip_header_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            return len(handle.read(16)) > 0
    except (OSError, EOFError):
        return False


def download_matrix_parallel(row: dict, destination: Path, log: dict, max_attempts: int) -> dict:
    chunk_size = 2 * 1024 * 1024
    total = int(row["expected_bytes"])
    chunk_dir = destination.parent / ".matrix_chunks"
    chunk_dir.mkdir(exist_ok=True)
    specs = []
    for index, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size - 1, total - 1)
        specs.append((index, start, end, end - start + 1, chunk_dir / f"chunk_{index:05d}"))

    def fetch(spec):
        index, start, end, expected, path = spec
        if path.exists() and path.stat().st_size == expected:
            return {"index": index, "status": "existing", "attempts": 0, "error": None}
        errors = []
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(row["url"], headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 120))
                if response.status_code != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                if len(response.content) != expected:
                    raise IOError(f"Chunk size {len(response.content)} vs {expected}")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                os.replace(temporary, path)
                return {"index": index, "status": "downloaded", "attempts": attempt, "error": None}
            except Exception as exc:
                errors.append(f"chunk {index} attempt {attempt}: {type(exc).__name__}: {exc}")
        return {"index": index, "status": "failed", "attempts": max_attempts, "error": " | ".join(errors)}

    errors = []
    attempts = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, spec): spec[0] for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            attempts.append(result["attempts"])
            if result["status"] == "failed":
                errors.append(result["error"])
                for pending in futures:
                    pending.cancel()
                break
            completed += 1
            if completed % 4 == 0 or completed == len(specs):
                print(f"  {row['capture_id']}: {completed}/{len(specs)} matrix chunks", flush=True)
    if errors:
        actual = sum(p.stat().st_size for p in chunk_dir.glob("chunk_*") if p.is_file())
        log.update(status="failed", attempts=max(attempts, default=0), actual_bytes=actual,
                   error=" | ".join(errors))
        return log
    partial = destination.with_suffix(destination.suffix + ".part")
    with open(partial, "wb") as output:
        for _, _, _, _, path in specs:
            with open(path, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    if partial.stat().st_size != total:
        log.update(status="failed", actual_bytes=partial.stat().st_size,
                   error=f"Assembled size mismatch: {partial.stat().st_size} vs {total}")
        return log
    os.replace(partial, destination)
    if not gzip_header_ok(destination):
        log.update(status="failed", actual_bytes=destination.stat().st_size, error="Invalid assembled gzip")
        return log
    for path in chunk_dir.glob("chunk_*"):
        path.unlink()
    chunk_dir.rmdir()
    log.update(status="downloaded_parallel", attempts=max(attempts, default=0), actual_bytes=destination.stat().st_size,
               error=None, parallel_workers=8, chunk_bytes=chunk_size)
    return log


def download_one(row: dict, max_attempts: int = 2) -> dict:
    destination = DATA / row["gsm"] / f"{row['kind']}.tsv.gz" if row["kind"] != "matrix" else DATA / row["gsm"] / "matrix.mtx.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "capture_id": row["capture_id"], "kind": row["kind"], "url": row["url"],
        "expected_bytes": int(row["expected_bytes"]), "destination": str(destination),
        "attempts": 0, "status": "pending", "error": None,
    }
    if destination.exists() and destination.stat().st_size == row["expected_bytes"] and gzip_header_ok(destination):
        log.update(status="existing_verified", actual_bytes=destination.stat().st_size)
        return log
    if row["kind"] == "matrix":
        return download_matrix_parallel(row, destination, log, max_attempts)
    partial = destination.with_suffix(destination.suffix + ".part")
    errors = []
    for attempt in range(1, max_attempts + 1):
        log["attempts"] = attempt
        start = partial.stat().st_size if partial.exists() else 0
        if start > row["expected_bytes"]:
            partial.unlink()
            start = 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(row["url"], headers=headers, stream=True, timeout=(30, 180)) as response:
                if start and response.status_code == 200:
                    start = 0
                response.raise_for_status()
                if start and response.status_code != 206:
                    raise RuntimeError(f"Range resume unsupported: HTTP {response.status_code}")
                mode = "ab" if start and response.status_code == 206 else "wb"
                with open(partial, mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != row["expected_bytes"]:
                raise IOError(f"Size mismatch: {partial.stat().st_size} vs {row['expected_bytes']}")
            os.replace(partial, destination)
            if not gzip_header_ok(destination):
                raise gzip.BadGzipFile("Invalid gzip header")
            log.update(status="downloaded", actual_bytes=destination.stat().st_size, error=None)
            return log
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            log["error"] = errors[-1]
    log.update(status="failed", actual_bytes=partial.stat().st_size if partial.exists() else 0,
               error=" | ".join(errors))
    return log


def bulk_fallback(error_logs: list[dict], meta: pd.DataFrame) -> None:
    associations = pd.DataFrame()
    fallback_status = "not_run"
    try:
        expr = pd.read_csv(GEO / "gse75010" / "GSE75010_data" / "GSE75010_gene_matrix.csv.gz", index_col=0)
        immune = pd.read_csv(RESULTS / "moduleB_immune_scores.csv", index_col=0)
        candidates = json.load(open(RESULTS / "moduleB_candidate_summary.json"))["candidates"]
        common = [s for s in immune.index if s in expr.columns]
        records = []
        for gene in candidates:
            if gene not in expr.index:
                records.append({"gene": gene, "cell_type": None, "status": "gene_absent"})
                continue
            for cell_type in immune.columns:
                rho, p = stats.spearmanr(expr.loc[gene, common], immune.loc[common, cell_type])
                records.append({"gene": gene, "cell_type": cell_type, "n_bulk_samples": len(common),
                                "spearman_rho": rho, "pvalue": p, "status": "bulk_association_only"})
        associations = pd.DataFrame(records)
        if "pvalue" in associations:
            valid = associations.pvalue.notna()
            p = associations.loc[valid, "pvalue"].to_numpy()
            order = np.argsort(p)
            ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
            ranked = np.minimum.accumulate(ranked[::-1])[::-1]
            fdr = np.empty(len(p)); fdr[order] = np.minimum(ranked, 1)
            associations.loc[valid, "FDR"] = fdr
        associations.to_csv(RESULTS / f"{PREFIX}bulk_marker_associations.csv", index=False)
        fallback_status = f"bulk Spearman associations only; n={len(common)} samples; not single-cell validation"
    except Exception as exc:
        fallback_status = f"bulk fallback failed: {type(exc).__name__}: {exc}"
    payload = {
        "status": "blocked", "dataset": "GSE290578", "selection_rationale": SELECTION_RATIONALE,
        "metadata_design": {
            "n_donors": int(meta.donor_id.nunique()), "n_captures": len(meta),
            "donors_per_group": meta.groupby("group").donor_id.nunique().to_dict(),
            "captures_per_donor": meta.groupby("donor_id").size().to_dict(),
            "layers": sorted(meta.layer.unique().tolist()),
            "selected_donors": meta.loc[meta.selected, ["gsm", "donor_id", "group", "layer", "offspring_sex", "gestational_week", "birth_description"]].to_dict("records"),
        },
        "download_attempts": error_logs,
        "raw_tar_avoided": {"filename": "GSE290578_RAW.tar", "bytes": 1268203520},
        "bulk_fallback_status": fallback_status,
        "interpretation": "No single-cell claim is made. Any fallback association is bulk-level evidence only.",
    }
    with open(RESULTS / f"{PREFIX}blocked.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    if not associations.empty:
        print(fallback_status)


def download_selected(file_manifest: pd.DataFrame, max_attempts: int) -> list[dict]:
    selected = file_manifest[file_manifest.selected].copy()
    kind_order = pd.Categorical(selected.kind, ["features", "barcodes", "matrix"], ordered=True)
    selected = selected.assign(_order=kind_order).sort_values(["gsm", "_order"])
    logs = []
    for row in selected.to_dict("records"):
        log = download_one(row, max_attempts=max_attempts)
        logs.append(log)
        print(f"{log['status']:18s} {row['capture_id']} {row['kind']:9s} "
              f"{log.get('actual_bytes', 0)}/{row['expected_bytes']}", flush=True)
        with open(RESULTS / f"{PREFIX}download_log.json", "w", encoding="utf-8") as handle:
            json.dump(logs, handle, indent=2, ensure_ascii=False)
    return logs


def get_expression(adata: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        return np.full(adata.n_obs, np.nan, dtype=np.float32)
    column = adata[:, gene].X
    if sparse.issparse(column):
        return np.asarray(column.todense()).ravel().astype(np.float32)
    return np.asarray(column).ravel().astype(np.float32)


def available_genes(adata: ad.AnnData, genes: list[str]) -> list[str]:
    return [g for g in genes if g in adata.var_names]


def read_geo_tsv(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        with tarfile.open(path, "r:*") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1:
                raise ValueError(f"Expected one file inside {path}, found {len(members)}")
            with archive.extractfile(members[0]) as handle:
                frame = pd.read_csv(handle, sep="\t", header=None, dtype=str)
        return frame, f"tar.gz:{members[0].name}"
    except tarfile.ReadError:
        return pd.read_csv(path, sep="\t", header=None, dtype=str), "plain.tsv.gz"


def read_geo_10x(folder: Path) -> tuple[ad.AnnData, str, str]:
    features, feature_packaging = read_geo_tsv(folder / "features.tsv.gz")
    barcodes, barcode_packaging = read_geo_tsv(folder / "barcodes.tsv.gz")
    if features.shape[1] < 2 or barcodes.shape[1] < 1:
        raise ValueError(f"Invalid 10x features/barcodes in {folder}")
    with gzip.open(folder / "matrix.mtx.gz", "rb") as handle:
        matrix = mmread(handle).tocsr()
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"Matrix dimension mismatch in {folder}: {matrix.shape} vs {(len(features), len(barcodes))}")
    a = ad.AnnData(X=matrix.T.tocsr(),
                   obs=pd.DataFrame(index=barcodes.iloc[:, 0].astype(str).to_numpy()),
                   var=pd.DataFrame(index=features.iloc[:, 1].astype(str).to_numpy()))
    a.var["gene_ids"] = features.iloc[:, 0].to_numpy()
    if features.shape[1] >= 3:
        a.var["feature_types"] = features.iloc[:, 2].to_numpy()
    return a, barcode_packaging, feature_packaging


def load_and_qc(meta: pd.DataFrame) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    adatas, qc_thresholds, raw_qc = [], [], []
    for row in meta[meta.selected].itertuples(index=False):
        capture_id = row.capture_id
        a, barcode_packaging, feature_packaging = read_geo_10x(DATA / row.gsm)
        n_barcode_rows, n_feature_rows = a.n_obs, a.n_vars
        a.var_names_make_unique()
        a.var["mt"] = a.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        genes = a.obs.n_genes_by_counts.to_numpy()
        mt = a.obs.pct_counts_mt.to_numpy()
        thresholds = {
            "min_genes": int(max(200, math.floor(np.quantile(genes, 0.01)))),
            "max_genes": int(math.ceil(np.quantile(genes, 0.99))),
            "max_pct_mt": float(np.clip(math.ceil(np.quantile(mt, 0.99)), 10, 25)),
        }
        keep = ((genes >= thresholds["min_genes"]) & (genes <= thresholds["max_genes"]) &
                (mt <= thresholds["max_pct_mt"]))
        qc_thresholds.append({
            "capture_id": capture_id, "gsm": row.gsm, "donor_id": row.donor_id,
            "group": row.group, "layer": row.layer, "raw_cells": a.n_obs,
            "barcode_rows": n_barcode_rows, "feature_rows": n_feature_rows, "matrix_genes": a.n_vars,
            "barcode_packaging": barcode_packaging, "feature_packaging": feature_packaging,
            "median_genes": float(np.median(genes)), "q01_genes": float(np.quantile(genes, 0.01)),
            "q99_genes": float(np.quantile(genes, 0.99)), "median_pct_mt": float(np.median(mt)),
            "q99_pct_mt": float(np.quantile(mt, 0.99)), **thresholds,
            "retained_cells": int(keep.sum()), "retained_fraction": float(keep.mean()),
        })
        raw = a.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].copy()
        raw["capture_id"] = capture_id; raw["status"] = "raw"
        raw_qc.append(raw)
        print(f"{capture_id}: {a.n_obs:,} raw -> {keep.sum():,} QC cells; "
              f"gene cuts {thresholds['min_genes']}-{thresholds['max_genes']}, MT <= {thresholds['max_pct_mt']}%", flush=True)
        a = a[keep].copy()
        a.obs["barcode"] = a.obs_names
        a.obs_names = capture_id + ":" + a.obs["barcode"].astype(str)
        a.obs["capture_id"] = capture_id
        a.obs["gsm"] = row.gsm
        a.obs["donor_id"] = row.donor_id
        a.obs["group"] = row.group
        a.obs["layer"] = row.layer
        a.obs["offspring_sex"] = row.offspring_sex
        a.obs["gestational_week"] = row.gestational_week
        assert a.n_obs > 100, f"Too few QC-passing cells in {capture_id}: {a.n_obs}"
        adatas.append(a)
    adata = ad.concat(adatas, join="outer", fill_value=0, index_unique=None)
    assert adata.obs_names.is_unique
    sc.pp.filter_genes(adata, min_cells=3)
    qc = pd.DataFrame(qc_thresholds)
    qc.to_csv(RESULTS / f"{PREFIX}qc_by_capture.csv", index=False)
    return adata, qc, pd.concat(raw_qc, ignore_index=True)


def annotate_clusters(adata: ad.AnnData) -> pd.DataFrame:
    signature_names = {}
    for cell_type, genes in PRIMARY_MARKERS.items():
        present = available_genes(adata, genes)
        if len(present) >= 3:
            name = f"marker_score_{cell_type}"
            sc.tl.score_genes(adata, present, score_name=name, ctrl_size=50, random_state=RANDOM_STATE)
            signature_names[cell_type] = name
    profile = adata.obs.groupby("leiden", observed=True)[list(signature_names.values())].mean()
    profile.columns = [c.replace("marker_score_", "") for c in profile.columns]
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
    assignments = {}
    records = []
    for cluster in profile.index:
        ordered = z.loc[cluster].sort_values(ascending=False)
        best = ordered.index[0]
        best_z = float(ordered.iloc[0])
        second_z = float(ordered.iloc[1]) if len(ordered) > 1 else np.nan
        best_raw = float(profile.loc[cluster, best])
        # Low/negative marker enrichment remains explicitly unassigned.
        automatic_label = best if best_raw > 0 and best_z > 0 else "Unassigned"
        label = automatic_label
        n_cells = int((adata.obs.leiden == cluster).sum())
        review = REVIEWED_CLUSTER_LABELS.get(str(cluster))
        review_note = None
        if review is not None:
            assert n_cells == review["expected_n_cells"], (cluster, n_cells, review)
            assert automatic_label == review["expected_automatic_label"], (cluster, automatic_label, review)
            label = review["label"]
            review_note = review["note"]
        assignments[cluster] = label
        records.append({"leiden": cluster, "cell_type": label, "automatic_cell_type": automatic_label,
                        "best_marker_set": best, "best_marker_score": best_raw, "best_marker_z": best_z,
                        "annotation_margin_z": best_z - second_z, "n_cells": n_cells,
                        "annotation_review": review_note or "Automatic label retained after core-marker review"})
    adata.obs["cell_type"] = adata.obs.leiden.map(assignments).astype(str)
    margin = pd.DataFrame(records).set_index("leiden").annotation_margin_z
    adata.obs["annotation_margin_z"] = adata.obs.leiden.map(margin).astype(float)
    annotation = pd.DataFrame(records).set_index("leiden")
    full = annotation.join(profile.add_prefix("mean_score_")).join(z.add_prefix("z_score_")).reset_index()
    full.to_csv(RESULTS / f"{PREFIX}cluster_annotation.csv", index=False)
    marker_evidence = []
    cluster_ids = adata.obs["leiden"]
    for cell_type, genes in PRIMARY_MARKERS.items():
        for gene in available_genes(adata, genes):
            values = get_expression(adata, gene)
            means = pd.Series(values).groupby(cluster_ids.to_numpy()).mean()
            positive = pd.Series(values > 0).groupby(cluster_ids.to_numpy()).mean() * 100
            for cluster in profile.index:
                marker_evidence.append({"leiden": cluster, "marker_set": cell_type, "gene": gene,
                                        "mean_log1p_expression": means.loc[cluster],
                                        "pct_positive": positive.loc[cluster]})
    pd.DataFrame(marker_evidence).to_csv(RESULTS / f"{PREFIX}cluster_marker_evidence.csv", index=False)
    return full


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().astype(float)
    if valid.empty:
        return result
    p = valid.to_numpy(); order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.minimum(adjusted, 1)
    result.loc[valid.index] = out
    return result


def aggregate_donor_results(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = EXPRESSION_COLUMNS + SCORE_COLUMNS
    keys = ["capture_id", "gsm", "donor_id", "group", "layer", "offspring_sex", "gestational_week", "cell_type"]
    capture = cell.groupby(keys, observed=True)[metrics].mean().reset_index()
    counts = cell.groupby(keys, observed=True).size().rename("n_cells").reset_index()
    capture = capture.merge(counts, on=keys, how="left")
    capture_totals = capture.groupby("capture_id").n_cells.sum().rename("n_capture_cells")
    capture["n_capture_cells"] = capture.capture_id.map(capture_totals).astype(int)
    capture["cell_fraction_within_capture"] = capture.n_cells / capture.n_capture_cells
    for gene in ["HIF1A", "FLT1", "SMARCA1", "VEGFA", "PGF"]:
        positive = (cell[f"{gene}_expression"] > 0).astype(float)
        pct = positive.groupby([cell[k] for k in keys], observed=True).mean().mul(100).rename(f"pct_{gene}_positive").reset_index()
        capture = capture.merge(pct, on=keys, how="left")
    capture.to_csv(RESULTS / f"{PREFIX}capture_celltype_pseudobulk.csv", index=False)

    donor_keys = ["donor_id", "group", "cell_type"]
    donor = capture.groupby(donor_keys, observed=True)[metrics].mean().reset_index()
    donor_counts = capture.groupby(donor_keys, observed=True).agg(
        n_cells=("n_cells", "sum"), n_captures=("capture_id", "nunique"),
        capture_ids=("capture_id", lambda x: ";".join(sorted(x)))).reset_index()
    donor = donor.merge(donor_counts, on=donor_keys, how="left")
    all_capture = cell.groupby(["capture_id", "gsm", "donor_id", "group", "layer", "offspring_sex", "gestational_week"], observed=True)[metrics].mean().reset_index()
    all_capture["cell_type"] = "ALL"
    all_donor = all_capture.groupby(["donor_id", "group", "cell_type"], observed=True)[metrics].mean().reset_index()
    all_counts = cell.groupby(["donor_id", "group"], observed=True).agg(
        n_cells=("cell_id", "size"), n_captures=("capture_id", "nunique"),
        capture_ids=("capture_id", lambda x: ";".join(sorted(set(x))))).reset_index()
    all_counts["cell_type"] = "ALL"
    all_donor = all_donor.merge(all_counts, on=["donor_id", "group", "cell_type"], how="left")
    donor = pd.concat([all_donor, donor], ignore_index=True)
    donor_totals = donor[donor.cell_type == "ALL"].set_index("donor_id").n_cells.rename("n_donor_cells")
    donor["n_donor_cells"] = donor.donor_id.map(donor_totals).astype(int)
    donor["cell_fraction_within_donor"] = donor.n_cells / donor.n_donor_cells
    donor.to_csv(RESULTS / f"{PREFIX}donor_celltype_summary.csv", index=False)

    comparison_metrics = metrics + ["cell_fraction_within_donor"]
    comparisons = []
    for (cell_type, metric), frame in donor.melt(id_vars=donor_keys + ["n_cells", "n_captures"], value_vars=comparison_metrics,
                                                   var_name="metric", value_name="value").groupby(["cell_type", "metric"]):
        pe = frame.loc[frame.group == "PE", "value"].dropna()
        ctrl = frame.loc[frame.group == "control", "value"].dropna()
        p = stats.ttest_ind(pe, ctrl, equal_var=False).pvalue if len(pe) >= 2 and len(ctrl) >= 2 else np.nan
        comparisons.append({"cell_type": cell_type, "metric": metric, "n_PE_donors": len(pe),
                            "n_control_donors": len(ctrl), "PE_mean": pe.mean(), "control_mean": ctrl.mean(),
                            "PE_minus_control": pe.mean() - ctrl.mean(), "welch_pvalue_exploratory": p})
    comparison = pd.DataFrame(comparisons)
    comparison["FDR_exploratory"] = bh_fdr(comparison.welch_pvalue_exploratory)
    comparison["inference_note"] = "Donor-level n=4/group; descriptive/exploratory and not confirmatory because PE is completely confounded with gestational age"
    comparison.to_csv(RESULTS / f"{PREFIX}donor_group_comparison.csv", index=False)
    return capture, donor, comparison


def plot_qc(raw_qc: pd.DataFrame, cell: pd.DataFrame, qc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    positions = np.arange(len(qc))
    axes[0].bar(positions - 0.2, qc.raw_cells, width=0.4, color="#B8C4CE", label="Raw")
    axes[0].bar(positions + 0.2, qc.retained_cells, width=0.4, color="#287D8E", label="QC pass")
    axes[0].set_xticks(positions); axes[0].set_xticklabels(qc.donor_id + "\n" + qc.layer, rotation=45, ha="right")
    axes[0].set_ylabel("Cells"); axes[0].legend(frameon=False); axes[0].set_title("A  Retained cells by capture", loc="left")
    sample = raw_qc.sample(min(20000, len(raw_qc)), random_state=RANDOM_STATE)
    axes[1].scatter(sample.n_genes_by_counts, sample.pct_counts_mt, s=1, alpha=0.12, color="#58656F", rasterized=True)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Raw genes per cell (log scale)"); axes[1].set_ylabel("Raw mitochondrial reads (%)")
    axes[1].set_title("B  Raw QC distribution (20k-cell sample)", loc="left")
    x = np.arange(len(qc))
    axes[2].plot(x, qc.q01_genes, "o--", color="#6B7280", label="1st percentile genes")
    axes[2].plot(x, qc.q99_genes, "o--", color="#E07A3F", label="99th percentile genes")
    axes[2].set_xticks(x); axes[2].set_xticklabels(qc.donor_id + "\n" + qc.layer, rotation=45, ha="right")
    axes[2].set_ylabel("Genes per cell"); axes[2].legend(frameon=False, fontsize=7)
    axes[2].set_title("C  Capture-specific adaptive thresholds", loc="left")
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES / "figD8_qc.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_umap(cell: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("cell_type", "A  Marker-based annotation"), ("group", "B  Clinical group"),
              ("donor_id", "C  Donor")]
    for ax, (column, title) in zip(axes, panels):
        categories = sorted(cell[column].unique())
        palette = plt.colormaps["tab20" if column != "group" else "Set1"]
        colors = {c: palette(i % palette.N) for i, c in enumerate(categories)}
        if column == "group": colors = {"control": "#287D8E", "PE": "#C44E52"}
        for category in categories:
            f = cell[cell[column] == category]
            ax.scatter(f.UMAP1, f.UMAP2, s=0.4, alpha=0.55, color=colors[category], label=category,
                       linewidths=0, rasterized=True)
        ax.set_title(title, loc="left"); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.spines[["top", "right"]].set_visible(False)
        if column != "donor_id":
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6, markerscale=8)
    fig.tight_layout(); fig.savefig(FIGURES / "figD8_umap.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_marker_dotplot(adata: ad.AnnData) -> None:
    markers = {k: available_genes(adata, v) for k, v in DOTPLOT_MARKERS.items()}
    markers = {k: v for k, v in markers.items() if v}
    dot = sc.pl.dotplot(adata, markers, groupby="cell_type", dendrogram=False, show=False, return_fig=True)
    dot.savefig(FIGURES / "figD8_marker_dotplot.png", dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_candidate_violins(adata: ad.AnnData) -> None:
    keys = ["HIF1A_expression", "FLT1_expression", "SMARCA1_expression", "hypoxia_score", "candidate_score"]
    categories = sorted(adata.obs.cell_type.unique())
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, key in zip(axes.ravel(), keys):
        values = [adata.obs.loc[adata.obs.cell_type == cell_type, key].to_numpy(dtype=float)
                  for cell_type in categories]
        parts = ax.violinplot(values, showextrema=False, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("#287D8E"); body.set_alpha(0.55); body.set_edgecolor("none")
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.7)
        ax.set_xticks(np.arange(1, len(categories) + 1))
        ax.set_xticklabels(categories, rotation=90)
        ax.set_ylabel(key.replace("_", " ")); ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(FIGURES / "figD8_candidate_violin.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_candidate_umap(cell: pd.DataFrame) -> None:
    keys = ["HIF1A_expression", "FLT1_expression", "SMARCA1_expression", "hypoxia_score", "candidate_score"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, key in zip(axes.ravel(), keys):
        values = cell[key].to_numpy(dtype=float)
        vmax = np.nanquantile(values, 0.99)
        vmax = max(vmax, np.finfo(float).eps)
        points = ax.scatter(cell.UMAP1, cell.UMAP2, c=values, cmap="viridis", vmin=0, vmax=vmax,
                            s=0.4, alpha=0.7, linewidths=0, rasterized=True)
        ax.set_title(key.replace("_", " "), loc="left"); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(points, ax=ax, shrink=0.7, label="log1p normalized" if key.endswith("expression") else "score")
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(FIGURES / "figD8_candidate_umap.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_donor_scores(donor: pd.DataFrame) -> None:
    metrics = ["hypoxia_score", "candidate_score", "FLT1_expression", "SMARCA1_expression"]
    frame = donor[donor.cell_type == "ALL"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        for pos, (group, color) in enumerate([("control", "#287D8E"), ("PE", "#C44E52")]):
            values = frame.loc[frame.group == group, metric].dropna().to_numpy()
            ax.boxplot(values, positions=[pos], widths=0.4, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": color, "alpha": 0.25}, medianprops={"color": "black"})
            ax.scatter(np.full(len(values), pos) + np.linspace(-0.07, 0.07, len(values)), values,
                       color=color, s=28, zorder=3, label=group if metric == metrics[0] else None)
            for donor_id, value in zip(frame.loc[frame.group == group, "donor_id"], values):
                ax.annotate(donor_id, (pos, value), xytext=(5, 3), textcoords="offset points", fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "PE"]); ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Donor-level {metric}", loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GSE290578: 4 donors/group; points are biological replicates", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "figD8_donor_scores.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def run_analysis(meta: pd.DataFrame) -> None:
    adata, qc, raw_qc = load_and_qc(meta)
    print(f"QC: {qc.raw_cells.sum():,} raw -> {adata.n_obs:,} retained; {adata.n_vars:,} genes", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="capture_id", flavor="seurat")
    sc.tl.pca(adata, n_comps=50, mask_var="highly_variable", zero_center=False, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=RANDOM_STATE,
                    transformer=KNeighborsTransformer(n_neighbors=15, metric="euclidean", n_jobs=-1))
    sc.tl.umap(adata, min_dist=0.3, random_state=RANDOM_STATE)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden", flavor="igraph", n_iterations=2,
                 directed=False, random_state=RANDOM_STATE)
    annotation = annotate_clusters(adata)

    candidates = json.load(open(RESULTS / "moduleB_candidate_summary.json"))["candidates"]
    signature_genes = {
        "hypoxia_score": available_genes(adata, HYPOXIA_GENES),
        "candidate_score": available_genes(adata, candidates),
        "proliferation_score": available_genes(adata, PROLIFERATION_GENES),
    }
    for score, genes in signature_genes.items():
        if len(genes) < 3:
            raise RuntimeError(f"Insufficient genes for {score}: {genes}")
        sc.tl.score_genes(adata, genes, score_name=score, ctrl_size=50, random_state=RANDOM_STATE)
    for gene in INDIVIDUAL_GENES:
        adata.obs[f"{gene}_expression"] = get_expression(adata, gene)
    adata.obs["UMAP1"] = adata.obsm["X_umap"][:, 0]
    adata.obs["UMAP2"] = adata.obsm["X_umap"][:, 1]
    adata.obs["cell_id"] = adata.obs_names

    cell_columns = ["cell_id", "barcode", "gsm", "capture_id", "donor_id", "group", "layer",
                    "offspring_sex", "gestational_week", "n_genes_by_counts", "total_counts", "pct_counts_mt", "leiden",
                    "cell_type", "annotation_margin_z", "UMAP1", "UMAP2"] + EXPRESSION_COLUMNS + SCORE_COLUMNS
    cell = adata.obs[cell_columns].copy()
    cell.to_csv(RESULTS / f"{PREFIX}cell_level.csv.gz", index=False, compression="gzip")
    capture, donor, comparison = aggregate_donor_results(cell)
    lr_columns = ["capture_id", "gsm", "donor_id", "group", "layer", "cell_type", "n_cells",
                  "VEGFA_expression", "VEGFB_expression", "PGF_expression", "FLT1_expression", "KDR_expression",
                  "pct_VEGFA_positive", "pct_PGF_positive", "pct_FLT1_positive"]
    capture[lr_columns].to_csv(RESULTS / f"{PREFIX}vegf_flt1_capture_pseudobulk.csv", index=False)

    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=20, use_raw=False)
    de = sc.get.rank_genes_groups_df(adata, group=None)
    de.to_csv(RESULTS / f"{PREFIX}cluster_top_markers.csv", index=False)

    plot_qc(raw_qc, cell, qc)
    plot_umap(cell)
    plot_marker_dotplot(adata)
    plot_candidate_violins(adata)
    plot_candidate_umap(cell)
    plot_donor_scores(donor)

    expected_cells = int(qc.retained_cells.sum())
    assertions = {
        "16_unique_GSM_in_metadata": len(meta) == 16 and meta.gsm.nunique() == 16,
        "8_donors_paired_layers_in_metadata": meta.donor_id.nunique() == 8 and bool((meta.groupby("donor_id").layer.nunique() == 2).all()),
        "selected_8_donors_8_captures": meta[meta.selected].donor_id.nunique() == 8 and len(meta[meta.selected]) == 8,
        "selected_4_donors_per_group": meta[meta.selected].groupby("group").donor_id.nunique().to_dict() == {"PE": 4, "control": 4},
        "raw_cell_counts_match_barcodes": bool((qc.raw_cells == qc.barcode_rows).all()),
        "matrix_gene_dimensions_match_features": bool((qc.matrix_genes == qc.feature_rows).all()),
        "cell_level_rows_equal_qc_retained": len(cell) == expected_cells == adata.n_obs,
        "unique_cell_ids": cell.cell_id.is_unique,
        "all_8_captures_retained": cell.capture_id.nunique() == 8,
        "all_8_donors_retained": cell.donor_id.nunique() == 8,
        "donor_summary_has_ALL_for_8_donors": donor[donor.cell_type == "ALL"].donor_id.nunique() == 8,
        "no_unassigned_donor_or_group": not cell[["donor_id", "group"]].isna().any().any(),
    }
    assert all(assertions.values()), assertions
    summary = {
        "status": "success_singlecell_deep_placenta_8donor", "dataset": "GSE290578",
        "selection_rationale": SELECTION_RATIONALE,
        "selected_donors": meta.loc[meta.selected, ["donor_id", "group", "offspring_sex", "gestational_week"]].drop_duplicates().to_dict("records"),
        "n_metadata_donors": int(meta.donor_id.nunique()), "n_metadata_captures": len(meta),
        "n_analyzed_donors": int(cell.donor_id.nunique()), "n_analyzed_captures": int(cell.capture_id.nunique()),
        "n_raw_cells": int(qc.raw_cells.sum()), "n_qc_cells": int(adata.n_obs),
        "n_removed_cells": int(qc.raw_cells.sum() - adata.n_obs),
        "qc_retained_fraction": float(adata.n_obs / qc.raw_cells.sum()), "n_genes_after_filter": int(adata.n_vars),
        "n_leiden_clusters": int(adata.obs.leiden.nunique()),
        "cell_type_counts": {str(k): int(v) for k, v in cell.cell_type.value_counts().items()},
        "cells_by_capture": {"|".join(map(str, k)): int(v) for k, v in cell.groupby(["capture_id", "donor_id", "group"], observed=True).size().items()},
        "qc_rule": "Per capture: genes >= max(200, 1st percentile), genes <= 99th percentile, MT% <= clip(99th percentile, 10, 25)",
        "signature_genes_used": signature_genes,
        "signature_genes_missing": {
            "hypoxia": sorted(set(HYPOXIA_GENES) - set(signature_genes["hypoxia_score"])),
            "candidates": sorted(set(candidates) - set(signature_genes["candidate_score"])),
        },
        "annotation_method": "Leiden cluster-level known-marker scores followed by manual core-marker review; reviewed relabelling is recorded per cluster and unsupported or mixed-lineage clusters are labelled Unassigned",
        "reviewed_cluster_labels": REVIEWED_CLUSTER_LABELS,
        "gene_symbol_aliases": {"ERVFRD1": "ERVFRD-1 in the GEO Cell Ranger reference"},
        "geo_container_handling": "Tar-wrapped barcodes.tsv.gz files are read by extracting the inner barcodes.tsv in memory; original downloads are unchanged",
        "donor_aggregation": "One selected Deep_placenta capture per donor; capture-cell-type means are the donor-level values",
        "neighbors_method": "Exact sklearn KNeighborsTransformer, 15 nearest neighbors on 30 PCs; avoids PyNNDescent csr_array incompatibility",
        "group_comparison_note": "Four donors per group; p-values remain exploratory because PE is completely separated from gestational age and four versus four donors is still underpowered for confirmatory inference",
        "design_confound": "All PE samples are preterm and all controls are full term; gestational age and PE cannot be separated in this dataset",
        "ligand_receptor_status": "No formal genome-wide LR inference; targeted VEGFA/VEGFB/PGF-FLT1/KDR capture pseudobulk saved",
        "assertions": assertions,
        "software": {"scanpy": sc.__version__, "anndata": ad.__version__},
    }
    with open(RESULTS / f"{PREFIX}summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    (RESULTS / f"{PREFIX}blocked.json").unlink(missing_ok=True)
    print(json.dumps({k: summary[k] for k in ["status", "n_raw_cells", "n_qc_cells", "n_leiden_clusters", "cell_type_counts"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-download-attempts", type=int, default=2)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    meta, file_manifest = build_manifests()
    if not args.skip_download:
        logs = download_selected(file_manifest, args.max_download_attempts)
        failed = [x for x in logs if x["status"] == "failed"]
        if failed:
            bulk_fallback(logs, meta)
            raise RuntimeError(f"Download blocked for {len(failed)} files; see {PREFIX}blocked.json")
    if args.download_only:
        selected = file_manifest[file_manifest.selected]
        print(f"Download complete: {len(selected)} files, {selected.expected_bytes.sum():,} expected bytes")
        return
    try:
        run_analysis(meta)
    except Exception as exc:
        if isinstance(exc, (requests.RequestException, gzip.BadGzipFile, EOFError)):
            logs = json.load(open(RESULTS / f"{PREFIX}download_log.json")) if (RESULTS / f"{PREFIX}download_log.json").exists() else []
            logs.append({"status": "matrix_validation_failed", "error": f"{type(exc).__name__}: {exc}"})
            bulk_fallback(logs, meta)
        raise


if __name__ == "__main__":
    main()
