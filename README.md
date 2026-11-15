# Analysis code for: Platelet, AST and ALT trajectories and placental FLT1 programs in preeclampsia: a multicohort translational study

This repository contains the analysis code used to generate the clinical trajectory,
placental transcriptomic, candidate-prioritisation, single-cell and reanalysis results
reported in the manuscript submitted to the *Journal of Translational Medicine*.

No participant-level data are included. All datasets are publicly available under their
original access conditions (see Data sources).

## Contents

```
clinical/            MIMIC-IV cohort construction, trajectory modelling, outcome models
placental/           GEO discovery and validation analyses (bulk and single-cell)
figures/             Rendering of the five main figures and eleven supplementary figures
reanalysis/          Post-review sensitivity analyses reported in the manuscript
requirements.txt     Python packages and versions used
MANIFEST.sha256      SHA-256 checksums of every file in this archive
```

## Data sources

| Dataset | Use | Access |
|---|---|---|
| MIMIC-IV v3.1 (hosp module) | Clinical cohort (701 preeclampsia-related admissions) | PhysioNet, credentialed access under a data use agreement |
| GSE75010 | Placental discovery cohort (49 early-onset preeclampsia, 35 controls) | GEO, public |
| GSE234729 | External directional validation (43 severe preeclampsia, 68 controls) | GEO, public |
| GSE204835 | Exploratory cross-platform check (12 term preeclampsia, 12 controls) | GEO, public |
| GSE271628 | Exploratory bulk validation | GEO, public |
| GSE290578 | Single-cell localisation (eight donors) | GEO, public |

Downloaded files are expected under `data/geo/`. MIMIC-IV files are expected under a
local `hosp/` directory; the path is set in the `ROOT`/`MIMIC` constants at the top of each
clinical script and must be edited before running.

## Environment

Python 3.10.4. Package versions are pinned in `requirements.txt`. Analyses were run on
Windows 10 with a single CPU; no GPU is required.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

Clinical branch (MIMIC-IV):

```
python clinical/mimic_query.py
python clinical/mimic_composite_outcomes.py
python clinical/mimic_verify_simplified.py
python clinical/moduleA_step1_cohort.py
python clinical/moduleA_step2_clustering.py
python clinical/moduleA_step3_analysis.py
python clinical/moduleA_step5_table1_summary.py
python clinical/moduleA_step6_event_times.py
python clinical/moduleA_step8_robustness.py
python clinical/panel_revision_analyses.py
```

Placental branch (GEO):

```
python placental/moduleB_download_geo.py
python placental/moduleB_step1_deg.py
python placental/moduleB_step2_wgcna.py
python placental/moduleB_step2b_wgcna_deepening.py
python placental/moduleB_step3_gsea.py
python placental/moduleB_step4_ppi.py
python placental/moduleB_step7_candidate_genes.py
python placental/moduleB_step8_immune.py
python placental/moduleB_step9_core_validation.py
python placental/moduleB_step10_gse271628.py
python placental/moduleB_step11_singlecell.py
python placental/moduleB_step12_gse234729.py
```

Figures:

```
python figures/build_submission_figures.py
```

Post-review reanalyses:

```
python reanalysis/reanalysis_era_groupcv.py
python reanalysis/reanalysis_c1_m4_minor.py
```

Scripts write to a `results/` and `figures/` directory relative to the working directory.
Each script defines its output path in a top-level constant and can be run independently
once the inputs listed above exist.

## Reproducibility notes

- The Gaussian mixture model is fitted with `n_init=20` and `random_state=42` in the primary
  analysis. Refits inside cross-validation folds use `n_init=10` and a fixed repeat seed.
- The split-half module stability analysis uses 200 stratified splits with seed 20261115.
- Bootstrap and permutation analyses use fixed seeds and are reproducible on the same
  package versions.
- MIMIC-IV dates are subject-level date-shifted by the data provider; calendar year cannot be
  recovered, so ICD coding system is used as the era stratifier in the reanalysis.
- Absolute paths in the scripts reflect the original analysis environment. Only the input and
  output roots need to be changed to run elsewhere.

## License

MIT License. See `LICENSE`.

## Contact

Xiaowei Qiu, Department of Obstetrics and Gynecology, Nanfang Hospital, Southern Medical
University, Guangzhou, China. Email: qxw24657@163.com. ORCID: 0009-0003-8966-8073.
