"""
Module A Step 8: landmark robustness, incremental value and internal validation.

Primary 48-h incremental models use the existing 701 complete-cluster admissions
and the prespecified whole-admission composite outcome, reproducing Step 3.
Landmark sensitivity cohorts are independently rebuilt at 24/48/72 h, require
in-hospital follow-up beyond the landmark, use only pre-landmark laboratories,
and count timestamped composite components only after the landmark.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
BASE = Path(r"D:\Program Files\mimic-iv-3.1")
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
PREFIX = RES / "moduleA_robustness_"

SEED = 20261114
LANDMARKS = (24, 48, 72)
BASE_COLS = ["anchor_age", "platelet_first", "ast_first", "alt_first"]
FEATURE_COLS = [
    "platelet_slope", "platelet_intercept", "platelet_first", "platelet_mean",
    "ast_slope", "ast_intercept", "ast_first", "ast_mean",
    "alt_slope", "alt_intercept", "alt_first", "alt_mean",
]
CLASS_ORDER = [
    "Stable",
    "Low-grade",
    "Liver-dominant decompensation",
    "Platelet-dominant decline",
]
CLASS_SHORT = {
    "Stable": "Stable",
    "Low-grade": "Low-grade",
    "Liver-dominant decompensation": "Liver-dominant",
    "Platelet-dominant decline": "Platelet-dominant",
}
COLORS = {"base": "#4C72B0", "extended": "#C44E52", "extended_fold_gmm": "#55A868"}
CORE_ITEMIDS = {
    51265: "platelet", 50826: "platelet",
    50878: "ast", 51256: "ast",
    50861: "alt", 51006: "alt",
}
CACHE_FILES = {
    "admissions": str(PREFIX) + "cache_admissions.csv.gz",
    "core_labs": str(PREFIX) + "cache_core_labs.csv.gz",
    "creatinine": str(PREFIX) + "cache_creatinine.csv.gz",
    "events": str(PREFIX) + "cache_event_candidates.csv.gz",
    "metadata": str(PREFIX) + "cache_metadata.json",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save_json(obj, path):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.ndarray):
            return convert(x.tolist())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    Path(path).write_text(json.dumps(convert(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def hours_between(end, start):
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.total_seconds() / 3600


def prepare_raw_data():
    if all(Path(p).exists() for p in CACHE_FILES.values()):
        print("Loading cached raw extracts...", flush=True)
        adm = pd.read_csv(CACHE_FILES["admissions"], parse_dates=["admittime", "dischtime", "deathtime"])
        core = pd.read_csv(CACHE_FILES["core_labs"], parse_dates=["charttime"])
        cr = pd.read_csv(CACHE_FILES["creatinine"], parse_dates=["charttime"])
        events = pd.read_csv(CACHE_FILES["events"])
        metadata = json.loads(Path(CACHE_FILES["metadata"]).read_text(encoding="utf-8"))
        return adm, core, cr, events, metadata

    start = time.time()
    print("Reading diagnoses and admissions...", flush=True)
    diag = pd.read_csv(
        BASE / "hosp/diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )
    pe_mask = (
        diag["icd_code"].str.startswith("O14", na=False)
        | diag["icd_code"].str.startswith("O11", na=False)
        | diag["icd_code"].str.startswith("O10", na=False)
        | ((diag["icd_version"] == 9) & diag["icd_code"].str.startswith("642", na=False))
    )
    pe_hadms = set(diag.loc[pe_mask, "hadm_id"].unique())
    adm = pd.read_csv(
        BASE / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"],
    )
    adm = adm[adm["hadm_id"].isin(pe_hadms)].copy()
    patients = pd.read_csv(BASE / "hosp/patients.csv.gz", usecols=["subject_id", "anchor_age", "gender"])
    adm = adm.merge(patients, on="subject_id", how="left")
    for c in ["admittime", "dischtime", "deathtime"]:
        adm[c] = pd.to_datetime(adm[c])
    adm["los_hours"] = hours_between(adm["dischtime"], adm["admittime"])
    print(f"PE admissions: {len(adm)}", flush=True)

    wanted = set(CORE_ITEMIDS) | {50912}
    core_parts, cr_parts = [], []
    print("Scanning labevents once for trajectory markers and creatinine...", flush=True)
    for i, chunk in enumerate(pd.read_csv(
        BASE / "hosp/labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        low_memory=False,
        chunksize=2_000_000,
    ), 1):
        mask = chunk["itemid"].isin(wanted) & chunk["hadm_id"].isin(pe_hadms) & chunk["valuenum"].notna()
        if mask.any():
            sub = chunk.loc[mask].copy()
            core_parts.append(sub[sub["itemid"].isin(CORE_ITEMIDS)])
            cr_parts.append(sub[sub["itemid"] == 50912])
        if i % 10 == 0:
            print(f"  lab chunks={i}; retained={sum(map(len, core_parts)) + sum(map(len, cr_parts)):,}", flush=True)
    core = pd.concat(core_parts, ignore_index=True)
    core["marker"] = core["itemid"].map(CORE_ITEMIDS)
    core["charttime"] = pd.to_datetime(core["charttime"])
    cr = pd.concat(cr_parts, ignore_index=True)
    cr["charttime"] = pd.to_datetime(cr["charttime"])
    print(f"Core labs={len(core):,}; creatinine={len(cr):,}", flush=True)

    adm_time = adm.set_index("hadm_id")["admittime"]
    candidates = []

    def add_candidates(frame, component):
        if frame is None or len(frame) == 0:
            return
        x = frame.copy()
        x["component"] = component
        candidates.append(x[["hadm_id", "component", "event_hours"]])

    death = adm[adm["hospital_expire_flag"] == 1].copy()
    death["event_time"] = death["deathtime"].fillna(death["dischtime"])
    death["event_hours"] = (death["event_time"] - death["admittime"]).dt.total_seconds() / 3600
    add_candidates(death, "death")

    print("Reading ICU stays and procedure events...", flush=True)
    icu = pd.read_csv(BASE / "icu/icustays.csv.gz", usecols=["hadm_id", "intime", "outtime", "los"])
    icu = icu[icu["hadm_id"].isin(pe_hadms)].copy()
    icu["intime"] = pd.to_datetime(icu["intime"])
    total_los = icu.groupby("hadm_id")["los"].sum()
    first_icu = icu.sort_values("intime").groupby("hadm_id", as_index=False).first()
    first_icu = first_icu[first_icu["hadm_id"].map(total_los).fillna(0) >= 3].copy()
    first_icu["event_time"] = first_icu["intime"] + pd.Timedelta(hours=72)
    first_icu["event_hours"] = (first_icu["event_time"] - first_icu["hadm_id"].map(adm_time)).dt.total_seconds() / 3600
    add_candidates(first_icu, "icu_long")

    d_items = pd.read_csv(BASE / "icu/d_items.csv.gz")
    vent_ids = set(d_items.loc[d_items["label"].str.contains("ventilation", case=False, na=False), "itemid"])
    proc = pd.read_csv(
        BASE / "icu/procedureevents.csv.gz",
        usecols=["hadm_id", "itemid", "starttime"], low_memory=False,
    )
    proc = proc[proc["hadm_id"].isin(pe_hadms) & proc["itemid"].isin(vent_ids)].copy()
    proc["starttime"] = pd.to_datetime(proc["starttime"])
    proc["event_hours"] = (proc["starttime"] - proc["hadm_id"].map(adm_time)).dt.total_seconds() / 3600
    add_candidates(proc, "ventilation")

    blood_labels = [
        "Fresh Frozen Plasma", "Packed Red Cells", "Whole Blood", "Cryoprecipitate",
        "Platelet Concentrate", "Fibrinogen Concentrate",
    ]
    blood_ids = set(d_items.loc[d_items["label"].isin(blood_labels), "itemid"])
    vaso_ids = set(d_items.loc[d_items["label"].str.contains(
        "norepinephrine|epinephrine|dopamine|phenylephrine|vasopressin|angiotensin",
        case=False, na=False,
    ), "itemid"])
    input_parts = []
    print("Scanning inputevents for transfusion and vasopressors...", flush=True)
    for chunk in pd.read_csv(
        BASE / "icu/inputevents.csv.gz",
        usecols=["hadm_id", "itemid", "starttime"], low_memory=False, chunksize=2_000_000,
    ):
        mask = chunk["hadm_id"].isin(pe_hadms) & chunk["itemid"].isin(blood_ids | vaso_ids)
        if mask.any():
            input_parts.append(chunk.loc[mask].copy())
    inputs = pd.concat(input_parts, ignore_index=True) if input_parts else pd.DataFrame(
        columns=["hadm_id", "itemid", "starttime"]
    )
    inputs["starttime"] = pd.to_datetime(inputs["starttime"])
    inputs["event_hours"] = (inputs["starttime"] - inputs["hadm_id"].map(adm_time)).dt.total_seconds() / 3600
    add_candidates(inputs[inputs["itemid"].isin(blood_ids)], "transfusion")
    add_candidates(inputs[inputs["itemid"].isin(vaso_ids)], "vasopressors")

    print("Deriving first creatinine-based AKI onset per admission...", flush=True)
    cr_clean = cr[(cr["valuenum"] > 0) & (cr["valuenum"] < 30)].sort_values(["hadm_id", "charttime"])
    aki_rows = []
    for hadm, group in cr_clean.groupby("hadm_id", sort=False):
        group = group.drop_duplicates("charttime")
        values = group["valuenum"].to_numpy()
        baseline = np.minimum.accumulate(values)
        hit = np.flatnonzero(values >= 1.5 * baseline)
        if len(hit):
            row = group.iloc[int(hit[0])]
            aki_rows.append({"hadm_id": hadm, "event_time": row["charttime"]})
    aki = pd.DataFrame(aki_rows)
    if len(aki):
        aki["event_hours"] = (pd.to_datetime(aki["event_time"]) - aki["hadm_id"].map(adm_time)).dt.total_seconds() / 3600
    else:
        aki["event_hours"] = []
    add_candidates(aki, "aki")

    events = pd.concat(candidates, ignore_index=True)
    events = events[np.isfinite(events["event_hours"]) & (events["event_hours"] >= 0)].copy()
    events = events.merge(adm[["hadm_id", "los_hours"]], on="hadm_id", how="left")
    events = events[events["event_hours"] <= events["los_hours"] + 1e-8].drop(columns="los_hours")

    metadata = {
        "source": str(BASE),
        "pe_admissions": len(adm),
        "core_lab_rows": len(core),
        "creatinine_rows": len(cr),
        "event_candidate_rows": len(events),
        "event_definition": "Earliest post-landmark death, ICU 72 h (if total ICU LOS >=3 d), procedure ventilation, transfusion, vasopressor, or first creatinine >=1.5 x admission cumulative nadir; censor at discharge.",
        "core_itemids": CORE_ITEMIDS,
        "elapsed_seconds": time.time() - start,
    }
    adm.to_csv(CACHE_FILES["admissions"], index=False, compression="gzip")
    core.to_csv(CACHE_FILES["core_labs"], index=False, compression="gzip")
    cr.to_csv(CACHE_FILES["creatinine"], index=False, compression="gzip")
    events.to_csv(CACHE_FILES["events"], index=False, compression="gzip")
    save_json(metadata, CACHE_FILES["metadata"])
    print(f"Raw extract cached in {(time.time() - start) / 60:.1f} min", flush=True)
    return adm, core, cr, events, metadata


def extract_features(window_labs):
    stats_names = ["n", "first", "last", "min", "max", "mean", "slope", "intercept", "std", "auc", "range", "time_span"]
    rows = []
    for (hadm, marker), group in window_labs.groupby(["hadm_id", "marker"], sort=False):
        group = group.sort_values("hours_since_adm", kind="mergesort")
        t = group["hours_since_adm"].to_numpy(dtype=float)
        y = group["valuenum"].to_numpy(dtype=float)
        values = {name: np.nan for name in stats_names}
        if len(group) >= 2:
            slope, intercept = np.polyfit(t, y, 1)
            auc = np.trapezoid(y, t) if hasattr(np, "trapezoid") else np.trapz(y, t)
            values = {
                "n": len(group), "first": y[0], "last": y[-1], "min": y.min(), "max": y.max(),
                "mean": y.mean(), "slope": slope, "intercept": intercept, "std": y.std(),
                "auc": auc, "range": y.max() - y.min(), "time_span": t[-1] - t[0],
            }
        rows.append({"hadm_id": hadm, "marker": marker, **values})
    long = pd.DataFrame(rows)
    wide = long.set_index(["hadm_id", "marker"])[stats_names].unstack("marker")
    wide.columns = [f"{marker}_{stat}" for stat, marker in wide.columns]
    wide = wide.reset_index()
    max_hours = window_labs.groupby("hadm_id")["hours_since_adm"].max().rename("max_feature_hours")
    return wide.merge(max_hours, on="hadm_id", how="left")


def phenotype_names(frame, label_col):
    summary = frame.groupby(label_col)[FEATURE_COLS].mean()
    remaining = list(summary.index)
    liver_score = summary["ast_mean"].fillna(0) / 70 + summary["alt_mean"].fillna(0) / 70
    liver = int(liver_score.idxmax())
    remaining.remove(liver)
    platelet = int(summary.loc[remaining, "platelet_slope"].idxmin())
    remaining.remove(platelet)
    scale = frame[["platelet_slope", "ast_slope", "alt_slope"]].std().replace(0, 1).fillna(1)
    stability = (
        summary.loc[remaining, "platelet_slope"].abs() / scale["platelet_slope"]
        + summary.loc[remaining, "ast_slope"].abs() / scale["ast_slope"]
        + summary.loc[remaining, "alt_slope"].abs() / scale["alt_slope"]
    )
    stable = int(stability.idxmin())
    remaining.remove(stable)
    mapping = {
        stable: "Stable",
        platelet: "Platelet-dominant decline",
        liver: "Liver-dominant decompensation",
    }
    if remaining:
        mapping[int(remaining[0])] = "Low-grade"
    return mapping


def fit_gmm(full, landmark):
    clean = full.dropna(subset=FEATURE_COLS).copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(clean[FEATURE_COLS])
    rows = []
    for k in range(2, 6):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", n_init=10,
            max_iter=500, random_state=42,
        ).fit(x)
        labels = gmm.predict(x)
        counts = pd.Series(labels).value_counts()
        rows.append({
            "landmark_hours": landmark,
            "n_classes": k,
            "BIC": gmm.bic(x),
            "AIC": gmm.aic(x),
            "min_class_size": int(counts.min()),
            "max_class_size": int(counts.max()),
            "converged": bool(gmm.converged_),
            "selected": k == 4,
        })
        print(f"  {landmark}h k={k}: BIC={rows[-1]['BIC']:.1f}; sizes={counts.sort_index().to_dict()}", flush=True)
    gmm = GaussianMixture(
        n_components=4, covariance_type="full", n_init=20,
        max_iter=500, random_state=42,
    ).fit(x)
    clean["gmm_component"] = gmm.predict(x)
    clean["max_posterior_prob"] = gmm.predict_proba(x).max(axis=1)
    mapping = phenotype_names(clean, "gmm_component")
    clean["class_name"] = clean["gmm_component"].map(mapping)
    assert set(clean["class_name"].unique()) == set(CLASS_ORDER)
    clean["landmark_hours"] = landmark
    return clean, pd.DataFrame(rows), {str(k): v for k, v in mapping.items()}


def design_matrix(frame, extended):
    x = frame[BASE_COLS].astype(float).copy()
    if extended:
        for name in CLASS_ORDER[1:]:
            x[f"class::{name}"] = (frame["class_name"] == name).astype(float)
    return sm.add_constant(x, has_constant="add")


def fit_full_models(frame):
    y = frame["composite_outcome"].astype(int)
    base = sm.Logit(y, design_matrix(frame, False)).fit(disp=0, maxiter=300)
    extended = sm.Logit(y, design_matrix(frame, True)).fit(disp=0, maxiter=300)
    rows = []
    for model_name, model in [("base", base), ("extended", extended)]:
        for term in model.params.index:
            rows.append({
                "model": model_name,
                "term": term,
                "coefficient": model.params[term],
                "SE": model.bse[term],
                "OR": np.exp(model.params[term]),
                "CI_low": np.exp(model.params[term] - 1.96 * model.bse[term]),
                "CI_high": np.exp(model.params[term] + 1.96 * model.bse[term]),
                "p_value": model.pvalues[term],
            })
    comparison = {
        "n": len(frame),
        "events": int(y.sum()),
        "base_loglik": base.llf,
        "extended_loglik": extended.llf,
        "LR_chi2": 2 * (extended.llf - base.llf),
        "LR_df": extended.df_model - base.df_model,
        "LR_p_value": stats.chi2.sf(2 * (extended.llf - base.llf), extended.df_model - base.df_model),
        "base_AUC": roc_auc_score(y, base.predict()),
        "extended_AUC": roc_auc_score(y, extended.predict()),
        "delta_AUC": roc_auc_score(y, extended.predict()) - roc_auc_score(y, base.predict()),
        "base_PR_AUC": average_precision_score(y, base.predict()),
        "extended_PR_AUC": average_precision_score(y, extended.predict()),
    }
    return base, extended, pd.DataFrame(rows), comparison


def fit_sklearn_design(train, test, extended):
    medians = train[BASE_COLS].median()
    xtr = train[BASE_COLS].fillna(medians).to_numpy(dtype=float)
    xte = test[BASE_COLS].fillna(medians).to_numpy(dtype=float)
    scaler = StandardScaler().fit(xtr)
    xtr = scaler.transform(xtr)
    xte = scaler.transform(xte)
    if extended:
        tr_class = np.column_stack([(train["class_name"] == c).astype(float) for c in CLASS_ORDER[1:]])
        te_class = np.column_stack([(test["class_name"] == c).astype(float) for c in CLASS_ORDER[1:]])
        xtr = np.column_stack([xtr, tr_class])
        xte = np.column_stack([xte, te_class])
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-8)
    model.fit(xtr, train["composite_outcome"].astype(int))
    return model.predict_proba(xte)[:, 1], model


def bootstrap_primary(frame, n_boot):
    rng = np.random.default_rng(SEED)
    rows = []
    n = len(frame)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = frame.iloc[idx].copy()
        row = {"replicate": b + 1, "status": "ok"}
        try:
            if boot["composite_outcome"].nunique() < 2:
                raise ValueError("single outcome class")
            pb, _ = fit_sklearn_design(boot, boot, False)
            pe, model = fit_sklearn_design(boot, boot, True)
            row.update({
                "platelet_dominant_OR": float(np.exp(model.coef_[0][len(BASE_COLS) + 2])),
                "delta_AUC": roc_auc_score(boot["composite_outcome"], pe) - roc_auc_score(boot["composite_outcome"], pb),
                "base_AUC": roc_auc_score(boot["composite_outcome"], pb),
                "extended_AUC": roc_auc_score(boot["composite_outcome"], pe),
            })
        except Exception as exc:
            row.update({"status": f"failed: {type(exc).__name__}", "platelet_dominant_OR": np.nan, "delta_AUC": np.nan})
        rows.append(row)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    boot_df = pd.DataFrame(rows)
    summary = []
    for metric in ["platelet_dominant_OR", "delta_AUC"]:
        values = boot_df[metric].dropna()
        summary.append({
            "metric": metric,
            "requested_replicates": n_boot,
            "valid_replicates": len(values),
            "mean": values.mean(),
            "CI_low": values.quantile(0.025),
            "CI_high": values.quantile(0.975),
            "method": "patient-level nonparametric bootstrap; refit base and extended models in each resample; fixed trajectory labels; apparent within-resample AUC; percentile CI",
        })
    return boot_df, pd.DataFrame(summary)


def fold_gmm_labels(train, test, seed=42):
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    xtr = scaler.transform(train[FEATURE_COLS])
    xte = scaler.transform(test[FEATURE_COLS])
    gmm = GaussianMixture(
        n_components=4, covariance_type="full", n_init=5,
        max_iter=500, random_state=seed,
    ).fit(xtr)
    train = train.copy()
    test = test.copy()
    train["gmm_component"] = gmm.predict(xtr)
    test["gmm_component"] = gmm.predict(xte)
    mapping = phenotype_names(train, "gmm_component")
    train["class_name"] = train["gmm_component"].map(mapping)
    test["class_name"] = test["gmm_component"].map(mapping)
    if train["class_name"].isna().any() or test["class_name"].isna().any():
        raise ValueError("unmapped GMM component")
    return train, test, {str(k): v for k, v in mapping.items()}


def cross_validate(frame, repeats):
    y = frame["composite_outcome"].astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    oof_rows, metric_rows, label_maps = [], [], []
    for split, (tr, te) in enumerate(cv.split(frame, y)):
        repeat, fold = split // 5 + 1, split % 5 + 1
        train = frame.iloc[tr].copy()
        test = frame.iloc[te].copy()
        assert set(train["hadm_id"]).isdisjoint(set(test["hadm_id"]))
        pred_base, _ = fit_sklearn_design(train, test, False)
        pred_ext, _ = fit_sklearn_design(train, test, True)
        train_gmm, test_gmm, mapping = fold_gmm_labels(train, test)
        pred_gmm, _ = fit_sklearn_design(train_gmm, test_gmm, True)
        label_maps.append({"repeat": repeat, "fold": fold, "component_to_name": mapping})
        out = test[["hadm_id", "composite_outcome"]].copy()
        out["repeat"] = repeat
        out["fold"] = fold
        out["p_base"] = pred_base
        out["p_extended"] = pred_ext
        out["p_extended_fold_gmm"] = pred_gmm
        out["fold_gmm_class"] = test_gmm["class_name"].to_numpy()
        oof_rows.append(out)
        for model, pred in [("base", pred_base), ("extended", pred_ext), ("extended_fold_gmm", pred_gmm)]:
            metric_rows.append({
                "repeat": repeat, "fold": fold, "model": model,
                "n_test": len(test), "events_test": int(test["composite_outcome"].sum()),
                "ROC_AUC": roc_auc_score(test["composite_outcome"], pred),
                "PR_AUC": average_precision_score(test["composite_outcome"], pred),
            })
        print(f"  CV repeat={repeat} fold={fold}", flush=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    assert len(oof) == len(frame) * repeats
    assert (oof.groupby("repeat")["hadm_id"].nunique() == len(frame)).all()
    paired = fold_metrics.pivot(index=["repeat", "fold"], columns="model", values=["ROC_AUC", "PR_AUC"]).reset_index()
    paired["delta_ROC_extended_minus_base"] = paired[("ROC_AUC", "extended")] - paired[("ROC_AUC", "base")]
    paired["delta_PR_extended_minus_base"] = paired[("PR_AUC", "extended")] - paired[("PR_AUC", "base")]
    paired["delta_ROC_fold_gmm_minus_base"] = paired[("ROC_AUC", "extended_fold_gmm")] - paired[("ROC_AUC", "base")]
    paired["delta_PR_fold_gmm_minus_base"] = paired[("PR_AUC", "extended_fold_gmm")] - paired[("PR_AUC", "base")]
    paired.columns = ["_".join([str(v) for v in col if str(v)]) for col in paired.columns]
    summaries = []
    for model in ["base", "extended", "extended_fold_gmm"]:
        sub = fold_metrics[fold_metrics["model"] == model]
        for metric in ["ROC_AUC", "PR_AUC"]:
            values = sub[metric]
            summaries.append({
                "model": model, "metric": metric, "n_folds": len(values),
                "mean": values.mean(), "SD": values.std(),
                "mean_CI_low": values.mean() - stats.t.ppf(0.975, len(values) - 1) * values.std() / np.sqrt(len(values)),
                "mean_CI_high": values.mean() + stats.t.ppf(0.975, len(values) - 1) * values.std() / np.sqrt(len(values)),
            })
    for col in [c for c in paired.columns if c.startswith("delta_")]:
        values = paired[col]
        summaries.append({
            "model": "paired_fold_difference", "metric": col, "n_folds": len(values),
            "mean": values.mean(), "SD": values.std(),
            "mean_CI_low": values.mean() - stats.t.ppf(0.975, len(values) - 1) * values.std() / np.sqrt(len(values)),
            "mean_CI_high": values.mean() + stats.t.ppf(0.975, len(values) - 1) * values.std() / np.sqrt(len(values)),
        })
    save_json(label_maps, str(PREFIX) + "cv_fold_gmm_label_maps.json")
    return oof, fold_metrics, paired, pd.DataFrame(summaries)


def calibration_statistics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    intercept_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit).fit()
    joint = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)
    return {
        "calibration_intercept_offset": intercept_model.params[0],
        "calibration_intercept_joint": joint.params[0],
        "calibration_slope": joint.params[1],
        "Brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }


def oof_diagnostics(oof, frame):
    y = frame.set_index("hadm_id")["composite_outcome"]
    mean_pred = oof.groupby("hadm_id")[["p_base", "p_extended", "p_extended_fold_gmm"]].mean()
    mean_pred["y"] = y
    assert mean_pred["y"].notna().all()
    stats_rows, curve_rows, dca_rows = [], [], []
    for model, col in [("base", "p_base"), ("extended", "p_extended"), ("extended_fold_gmm", "p_extended_fold_gmm")]:
        p = mean_pred[col].to_numpy()
        yy = mean_pred["y"].to_numpy(dtype=int)
        stats_rows.append({
            "model": model, "n": len(yy), "events": int(yy.sum()),
            "ROC_AUC_mean_OOF": roc_auc_score(yy, p),
            "PR_AUC_mean_OOF": average_precision_score(yy, p),
            **calibration_statistics(yy, p),
        })
        frac, pred = calibration_curve(yy, p, n_bins=10, strategy="quantile")
        counts = pd.DataFrame({"p": p, "y": yy}).assign(
            bin=lambda z: pd.qcut(z["p"], q=10, duplicates="drop")
        ).groupby("bin", observed=True).size()
        for i, (f, m) in enumerate(zip(frac, pred)):
            curve_rows.append({"model": model, "bin": i + 1, "observed": f, "mean_predicted": m})
        for threshold in np.arange(0.01, 0.3001, 0.01):
            positive = p >= threshold
            tp = int(((positive == 1) & (yy == 1)).sum())
            fp = int(((positive == 1) & (yy == 0)).sum())
            weight = threshold / (1 - threshold)
            nb = tp / len(yy) - fp / len(yy) * weight
            dca_rows.append({"model": model, "threshold": threshold, "TP": tp, "FP": fp, "net_benefit": nb})
    for threshold in np.arange(0.01, 0.3001, 0.01):
        prevalence = mean_pred["y"].mean()
        weight = threshold / (1 - threshold)
        dca_rows.append({"model": "treat_all", "threshold": threshold, "net_benefit": prevalence - (1 - prevalence) * weight})
        dca_rows.append({"model": "treat_none", "threshold": threshold, "net_benefit": 0.0})
    return mean_pred.reset_index(), pd.DataFrame(stats_rows), pd.DataFrame(curve_rows), pd.DataFrame(dca_rows)


def landmark_outcomes(frame, events, landmark):
    eligible_events = events[(events["event_hours"] > landmark) & events["hadm_id"].isin(frame["hadm_id"])]
    first = eligible_events.groupby("hadm_id")["event_hours"].min()
    out = frame.copy()
    out["event_hours_from_admission"] = out["hadm_id"].map(first)
    out["post_landmark_event"] = out["event_hours_from_admission"].notna().astype(int)
    out["event_hours_from_admission"] = out["event_hours_from_admission"].fillna(out["los_hours"])
    out["event_hours_from_admission"] = np.minimum(out["event_hours_from_admission"], out["los_hours"])
    out["time_days"] = (out["event_hours_from_admission"] - landmark) / 24
    out["event"] = out["post_landmark_event"]
    assert (out.loc[out["event"] == 1, "event_hours_from_admission"] > landmark).all()
    assert (out["time_days"] > 0).all()
    return out


def fit_landmark_model(frame, landmark):
    model_data = frame.rename(columns={"post_landmark_event": "composite_outcome"})
    y = model_data["composite_outcome"]
    x_base = design_matrix(model_data, False)
    x_ext = design_matrix(model_data, True)
    base = sm.GLM(y, x_base, family=sm.families.Binomial()).fit(maxiter=300)
    ext = sm.GLM(y, x_ext, family=sm.families.Binomial()).fit(maxiter=300)
    base_method = ext_method = "Binomial GLM IRLS (logit link)"
    print(f"  {landmark}h fit methods: base={base_method}; extended={ext_method}; converged={base.converged}/{ext.converged}", flush=True)
    rows = []
    for term in ext.params.index:
        rows.append({
            "landmark_hours": landmark, "model": "extended", "term": term,
            "fit_method": ext_method, "converged": bool(ext.converged),
            "coefficient": ext.params[term], "SE": ext.bse[term],
            "OR": np.exp(ext.params[term]), "CI_low": np.exp(ext.params[term] - 1.96 * ext.bse[term]),
            "CI_high": np.exp(ext.params[term] + 1.96 * ext.bse[term]), "p_value": ext.pvalues[term],
        })
    lr = multivariate_logrank_test(model_data["time_days"], model_data["class_name"], model_data["event"])
    lr_stat = 2 * (ext.llf - base.llf)
    logrank = {
        "landmark_hours": landmark, "chi2": lr.test_statistic, "df": 3, "p_value": lr.p_value,
        "base_fit_method": base_method, "extended_fit_method": ext_method,
        "base_converged": bool(base.converged), "extended_converged": bool(ext.converged),
        "LR_model_chi2": lr_stat, "LR_model_df": ext.df_model - base.df_model,
        "LR_model_p_value": stats.chi2.sf(lr_stat, ext.df_model - base.df_model),
    }
    return pd.DataFrame(rows), logrank


def build_landmarks(adm, core, events):
    lab = core.merge(adm[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="left")
    lab["hours_since_adm"] = (lab["charttime"] - lab["admittime"]).dt.total_seconds() / 3600
    lab = lab[(lab["hours_since_adm"] >= 0)].copy()
    all_assignments, all_bic, all_coef, summaries, logranks, mappings = [], [], [], [], [], {}
    assertions = {}
    for landmark in LANDMARKS:
        print(f"Building independent {landmark}-h landmark cohort...", flush=True)
        window = lab[lab["hours_since_adm"] <= landmark].copy()
        at_risk = set(adm.loc[adm["los_hours"] > landmark, "hadm_id"])
        window = window[window["hadm_id"].isin(at_risk)].copy()
        counts = window.groupby(["hadm_id", "marker"]).size().unstack(fill_value=0)
        has_plt2 = counts.get("platelet", pd.Series(0, index=counts.index)) >= 2
        has_liver2 = (counts.get("ast", pd.Series(0, index=counts.index)) >= 2) | (counts.get("alt", pd.Series(0, index=counts.index)) >= 2)
        eligible = set(counts.index[has_plt2 & has_liver2])
        agg = window[window["hadm_id"].isin(eligible)].pivot_table(
            index="hadm_id", columns="marker", values="valuenum", aggfunc=["min", "max"]
        )
        plt_low = agg.get(("min", "platelet"), pd.Series(False, index=agg.index)) < 100
        ast_high = agg.get(("max", "ast"), pd.Series(-np.inf, index=agg.index)) > 70
        alt_high = agg.get(("max", "alt"), pd.Series(-np.inf, index=agg.index)) > 70
        prevalent = set(agg.index[plt_low & (ast_high | alt_high)])
        eligible = sorted(eligible - prevalent)
        cohort = adm[adm["hadm_id"].isin(eligible)].copy()
        window = window[window["hadm_id"].isin(eligible)].copy()
        features = extract_features(window)
        full = cohort.merge(features, on="hadm_id", how="inner")
        clean, bic, mapping = fit_gmm(full, landmark)
        clean = landmark_outcomes(clean, events, landmark)
        clean["class_name"] = pd.Categorical(clean["class_name"], categories=CLASS_ORDER, ordered=True)
        coef, logrank = fit_landmark_model(clean, landmark)
        platelet = coef[coef["term"] == "class::Platelet-dominant decline"].iloc[0]
        summaries.append({
            "landmark_hours": landmark,
            "n_eligible_before_complete_features": len(cohort),
            "n": len(clean),
            "events": int(clean["post_landmark_event"].sum()),
            "event_rate": clean["post_landmark_event"].mean(),
            "Stable_n": int((clean["class_name"] == "Stable").sum()),
            "Low_grade_n": int((clean["class_name"] == "Low-grade").sum()),
            "Liver_dominant_n": int((clean["class_name"] == "Liver-dominant decompensation").sum()),
            "Platelet_dominant_n": int((clean["class_name"] == "Platelet-dominant decline").sum()),
            "Platelet_dominant_events": int(clean.loc[clean["class_name"] == "Platelet-dominant decline", "event"].sum()),
            "Platelet_dominant_adjusted_OR": platelet["OR"],
            "CI_low": platelet["CI_low"], "CI_high": platelet["CI_high"], "p_value": platelet["p_value"],
            "fit_method": platelet["fit_method"], "model_converged": bool(platelet["converged"]),
            "logrank_chi2": logrank["chi2"], "logrank_p_value": logrank["p_value"],
        })
        all_assignments.append(clean)
        all_bic.append(bic)
        all_coef.append(coef)
        logranks.append(logrank)
        mappings[str(landmark)] = mapping
        assertions[str(landmark)] = {
            "max_feature_hours": float(clean["max_feature_hours"].max()),
            "all_features_at_or_before_landmark": bool(clean["max_feature_hours"].max() <= landmark + 1e-9),
            "all_discharge_after_landmark": bool((clean["los_hours"] > landmark).all()),
            "all_event_times_strictly_after_landmark": bool((clean.loc[clean["event"] == 1, "event_hours_from_admission"] > landmark).all()),
            "all_followup_times_positive": bool((clean["time_days"] > 0).all()),
            "unique_hadm": bool(clean["hadm_id"].is_unique),
            "model_converged": bool(logrank["base_converged"] and logrank["extended_converged"]),
        }
        print(pd.DataFrame(summaries[-1:]).to_string(index=False), flush=True)
    return (
        pd.concat(all_assignments, ignore_index=True), pd.concat(all_bic, ignore_index=True),
        pd.concat(all_coef, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(logranks),
        mappings, assertions,
    )


def save_figure(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_incremental(mean_pred, calib_curves, dca):
    y = mean_pred["y"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for model, col, label in [
        ("base", "p_base", "Base"), ("extended", "p_extended", "Extended"),
        ("extended_fold_gmm", "p_extended_fold_gmm", "Fold-refit GMM"),
    ]:
        p = mean_pred[col]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=COLORS[model], lw=1.5,
                     label=f"{label}: {roc_auc_score(y, p):.3f}")
        precision, recall, _ = precision_recall_curve(y, p)
        axes[1].plot(recall, precision, color=COLORS[model], lw=1.5,
                     label=f"{label}: {average_precision_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#999999", lw=.8)
    axes[0].set(xlabel="1 − specificity", ylabel="Sensitivity", title="A  Out-of-fold ROC", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].axhline(y.mean(), ls="--", color="#999999", lw=.8, label=f"Prevalence={y.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="B  Out-of-fold precision–recall", xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best")
    save_figure(fig, "figD6_incremental_roc_pr")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for model, col, label in [
        ("base", "p_base", "Base"), ("extended", "p_extended", "Extended"),
        ("extended_fold_gmm", "p_extended_fold_gmm", "Fold-refit GMM"),
    ]:
        curve = calib_curves[calib_curves["model"] == model]
        axes[0].plot(curve["mean_predicted"], curve["observed"], marker="o", ms=3.5, lw=1.3,
                     color=COLORS[model], label=label)
        sub = dca[dca["model"] == model]
        axes[1].plot(sub["threshold"], sub["net_benefit"], lw=1.5, color=COLORS[model], label=label)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#888888", lw=.8)
    axes[0].set(xlabel="Mean predicted risk", ylabel="Observed event fraction", title="A  Mean out-of-fold calibration", xlim=(0, .55), ylim=(0, .55))
    all_dca = dca[dca["model"] == "treat_all"]
    axes[1].plot(all_dca["threshold"], all_dca["net_benefit"], color="#777777", lw=1, label="Treat all")
    axes[1].axhline(0, color="#222222", lw=.8, ls="--", label="Treat none")
    axes[1].set(xlabel="Threshold probability", ylabel="Net benefit", title="B  Decision curve analysis", xlim=(.01, .30))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best")
    save_figure(fig, "figD6_calibration_dca")


def plot_landmarks(summary, bic):
    plot_data = summary.sort_values("landmark_hours", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ypos = np.arange(len(plot_data))
    for i, row in plot_data.iterrows():
        ax.plot([row["CI_low"], row["CI_high"]], [i, i], color="#4C72B0", lw=1.6)
        ax.plot(row["Platelet_dominant_adjusted_OR"], i, "o", color="#C44E52", ms=6)
    ax.axvline(1, ls="--", lw=.8, color="#777777")
    ax.set_xscale("log")
    ax.set_xlim(0.45, 5.0)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{int(h)} h" for h in plot_data["landmark_hours"]])
    ax.set_xlabel("Adjusted odds ratio vs Stable (95% CI, log scale)")
    ax.set_title("Platelet-dominant decline and post-landmark severe maternal composite outcome", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, row in plot_data.iterrows():
        ax.text(1.03, i, f"{row['Platelet_dominant_adjusted_OR']:.2f} ({row['CI_low']:.2f}–{row['CI_high']:.2f}); "
                         f"n={int(row['n'])}, events={int(row['events'])}; log-rank P={row['logrank_p_value']:.3g}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.3)
    ax.set_ylim(-.6, len(plot_data) - .4)
    ax.text(0.0, -0.31, "Log-rank tests are unadjusted; each independent GMM includes a singleton liver-dominant class.",
            transform=ax.transAxes, va="top", fontsize=6.5, color="#555555")
    save_figure(fig, "figD6_landmark_forest")

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for landmark, color in zip(LANDMARKS, ["#4C72B0", "#C44E52", "#55A868"]):
        sub = bic[bic["landmark_hours"] == landmark]
        ax.plot(sub["n_classes"], sub["BIC"], marker="o", label=f"{landmark} h", color=color)
        selected = sub[sub["n_classes"] == 4]
        ax.scatter(selected["n_classes"], selected["BIC"], s=90, facecolors="none", edgecolors=color, linewidths=1.5)
    ax.set(xlabel="Number of GMM components", ylabel="Bayesian information criterion", title="Landmark-specific GMM fit (k=4 prespecified; circles)")
    ax.set_xticks([2, 3, 4, 5])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, "figD6_landmark_bic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    assert args.bootstrap >= 1000, "At least 1000 bootstrap replicates are required"
    assert args.repeats >= 5, "At least 5 CV repeats are required"
    started = time.time()

    print("=" * 78, "\nMODULE A STEP 8  Robustness and incremental prediction\n" + "=" * 78, flush=True)
    cohort = pd.read_csv(ROOT / "moduleA_landmark_cohort.csv")
    clusters = pd.read_csv(ROOT / "moduleA_cluster_assignments.csv")
    name_map = {1: "Stable", 2: "Low-grade", 0: "Liver-dominant decompensation", 3: "Platelet-dominant decline"}
    primary = cohort.merge(
        clusters[["hadm_id", "trajectory_class"] + FEATURE_COLS],
        on="hadm_id", how="inner",
    )
    primary["class_name"] = primary["trajectory_class"].map(name_map)
    primary["class_name"] = pd.Categorical(primary["class_name"], categories=CLASS_ORDER, ordered=True)
    assert len(primary) == 701 and primary["hadm_id"].is_unique
    assert primary["composite_outcome"].sum() == 80

    print("Fitting prespecified full-data base and extended models...", flush=True)
    base, extended, coefficients, comparison = fit_full_models(primary)
    platelet_or = np.exp(extended.params["class::Platelet-dominant decline"])
    assert np.isclose(platelet_or, 3.1971811980268843, rtol=1e-6)
    coefficients.to_csv(str(PREFIX) + "full_model_coefficients.csv", index=False)
    save_json(comparison, str(PREFIX) + "full_model_comparison.json")
    print(json.dumps(comparison, indent=2), flush=True)

    print(f"Bootstrapping {args.bootstrap} times with fixed class labels...", flush=True)
    boot, boot_summary = bootstrap_primary(primary, args.bootstrap)
    boot.to_csv(str(PREFIX) + "bootstrap_replicates.csv", index=False)
    boot_summary.to_csv(str(PREFIX) + "bootstrap_summary.csv", index=False)
    assert len(boot) == args.bootstrap and boot_summary["valid_replicates"].min() >= 0.95 * args.bootstrap

    print(f"Repeated stratified 5-fold CV ({args.repeats} repeats), including fold-specific GMM sensitivity...", flush=True)
    oof, fold_metrics, paired, cv_summary = cross_validate(primary, args.repeats)
    oof.to_csv(str(PREFIX) + "oof_predictions.csv", index=False)
    fold_metrics.to_csv(str(PREFIX) + "cv_fold_metrics.csv", index=False)
    paired.to_csv(str(PREFIX) + "cv_paired_fold_differences.csv", index=False)
    cv_summary.to_csv(str(PREFIX) + "cv_metrics_summary.csv", index=False)

    mean_pred, calibration, calib_curves, dca = oof_diagnostics(oof, primary)
    mean_pred.to_csv(str(PREFIX) + "oof_patient_mean_predictions.csv", index=False)
    calibration.to_csv(str(PREFIX) + "calibration_metrics.csv", index=False)
    calib_curves.to_csv(str(PREFIX) + "calibration_curve.csv", index=False)
    dca.to_csv(str(PREFIX) + "dca.csv", index=False)
    plot_incremental(mean_pred, calib_curves, dca)

    adm, core, cr, events, raw_metadata = prepare_raw_data()
    assignments, bic, landmark_coef, landmark_summary, logrank, mappings, landmark_assertions = build_landmarks(adm, core, events)
    assignments.to_csv(str(PREFIX) + "landmark_assignments.csv", index=False)
    bic.to_csv(str(PREFIX) + "landmark_bic.csv", index=False)
    landmark_coef.to_csv(str(PREFIX) + "landmark_model_coefficients.csv", index=False)
    landmark_summary.to_csv(str(PREFIX) + "landmark_summary.csv", index=False)
    logrank.to_csv(str(PREFIX) + "landmark_logrank.csv", index=False)
    save_json(mappings, str(PREFIX) + "landmark_class_label_maps.json")
    plot_landmarks(landmark_summary, bic)

    assertions = {
        "primary_48h": {
            "n": len(primary), "events": int(primary["composite_outcome"].sum()),
            "unique_hadm": primary["hadm_id"].is_unique,
            "four_classes": primary["class_name"].nunique() == 4,
            "platelet_OR_reproduces_step3": bool(np.isclose(platelet_or, 3.1971811980268843, rtol=1e-6)),
            "bootstrap_replicates_requested": args.bootstrap,
            "bootstrap_replicates_valid_OR": int(boot["platelet_dominant_OR"].notna().sum()),
            "trajectory_labels_fixed_in_bootstrap_and_primary_CV": True,
        },
        "cross_validation": {
            "repeats": args.repeats, "folds_per_repeat": 5,
            "all_train_test_hadm_disjoint": True,
            "one_prediction_per_hadm_per_repeat": bool((oof.groupby("repeat")["hadm_id"].nunique() == 701).all()),
            "all_numeric_imputation_and_scaling_fit_within_training_fold": True,
            "fold_gmm_and_phenotype_mapping_fit_within_training_fold": True,
        },
        "landmarks": landmark_assertions,
        "all_passed": all(v for x in landmark_assertions.values() for k, v in x.items() if k.startswith("all_") or k in {"unique_hadm", "model_converged"}),
    }
    save_json(assertions, str(PREFIX) + "assertions.json")
    assert assertions["all_passed"]

    summary = {
        "primary_48h": comparison,
        "primary_platelet_dominant": coefficients[
            (coefficients["model"] == "extended") & (coefficients["term"] == "class::Platelet-dominant decline")
        ].iloc[0].to_dict(),
        "bootstrap": boot_summary.to_dict("records"),
        "cv_summary": cv_summary.to_dict("records"),
        "oof_diagnostics": calibration.to_dict("records"),
        "landmarks": landmark_summary.to_dict("records"),
        "optimism_note": "The primary extended CV uses trajectory classes fitted unsupervised on all 701 admissions, so its OOF estimates retain slight optimistic bias. The extended_fold_gmm analysis refits scaling, GMM and phenotype mapping within each training fold and is the less optimistic sensitivity analysis.",
        "outcome_note": "The prespecified 48-h incremental analysis uses the original whole-admission composite to reproduce Step 3 (OR=3.20). Landmark sensitivity models instead use timestamped components strictly after each landmark and discharge after the landmark; their estimates are therefore not directly interchangeable with the primary whole-admission analysis.",
        "landmark_fit_note": "All landmark regressions use the same binomial GLM with logit link fitted by IRLS. Each independently fitted k=4 GMM produced a singleton Liver-dominant class (and two Low-grade admissions at 48 h), indicating local-mode/separation instability; Wald intervals and class-specific estimates require cautious interpretation. Log-rank tests are unadjusted.",
        "cv_inference_note": "Fold means and t intervals describe 25 paired folds; repeated folds are correlated and intervals are not independent-replicate confirmatory inference.",
        "elapsed_minutes": (time.time() - started) / 60,
    }
    save_json(summary, str(PREFIX) + "summary.json")
    print("\nBootstrap:\n", boot_summary.to_string(index=False), flush=True)
    print("\nOOF diagnostics:\n", calibration.to_string(index=False), flush=True)
    print("\nLandmarks:\n", landmark_summary.to_string(index=False), flush=True)
    print(f"\nCompleted in {(time.time() - started) / 60:.1f} min. Outputs: results/moduleA_robustness_*; figures/figD6_*", flush=True)


if __name__ == "__main__":
    main()
