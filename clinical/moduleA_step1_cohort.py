"""
Module A: MIMIC-IV 数据清洗 + landmark 队列构建 + 轨迹特征提取
输出: moduleA_landmark_cohort.csv, moduleA_trajectory_features.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = r"D:\Program Files\mimic-iv-3.1"
OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
Path(OUT).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Module A Step 1: Data cleaning & landmark cohort construction")
print("=" * 70)

# ── 1. PE cohort from diagnoses ──
diag = pd.read_csv(f"{BASE}\\hosp\\diagnoses_icd.csv.gz",
                   usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
pe_mask = (
    diag['icd_code'].str.startswith('O14', na=False) |
    diag['icd_code'].str.startswith('O11', na=False) |
    diag['icd_code'].str.startswith('O10', na=False) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('642', na=False))
)
pe_hadms = set(diag[pe_mask]['hadm_id'].unique())
print(f"PE-related hospitalizations: {len(pe_hadms)}")

# ── 2. Admissions ──
adm = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz",
                  usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                           'hospital_expire_flag', 'admission_type', 'insurance',
                           'marital_status', 'race'])
adm['admittime'] = pd.to_datetime(adm['admittime'])
adm['dischtime'] = pd.to_datetime(adm['dischtime'])
pe_adm = adm[adm['hadm_id'].isin(pe_hadms)].copy()
print(f"PE admissions: {len(pe_adm)}")

# ── 3. Patients (age, gender) ──
patients = pd.read_csv(f"{BASE}\\hosp\\patients.csv.gz",
                       usecols=['subject_id', 'anchor_age', 'gender'])
pe_adm = pe_adm.merge(patients, on='subject_id', how='left')

# ── 4. Lab items ──
LAB_ITEMIDS = {
    'platelet': [51265, 50826],
    'ast': [50878, 51256],
    'alt': [50861, 51006],
    'ldh': [50813, 51002],
    'bilirubin_total': [50885, 51508],
    'creatinine': [50912, 50821],
    'hemoglobin': [51222, 50811],
    'wbc': [51301, 50818],
    'albumin': [50862],
    'lactate': [50001, 50025],
}
ID_TO_MARKER = {}
for marker, ids in LAB_ITEMIDS.items():
    for i in ids:
        ID_TO_MARKER[i] = marker
ITEM_IDS = set(ID_TO_MARKER.keys())

# ── 5. Load lab events for PE hadms ──
print("Loading lab events for PE cohort...")
chunks = []
for chunk in pd.read_csv(f"{BASE}\\hosp\\labevents.csv.gz",
                         usecols=['subject_id', 'hadm_id', 'itemid', 'charttime',
                                  'valuenum', 'valueuom'],
                         low_memory=False, chunksize=2000000):
    m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(ITEM_IDS)) & chunk['valuenum'].notna()
    if m.any():
        chunks.append(chunk[m])
lab_pe = pd.concat(chunks, ignore_index=True)
lab_pe['marker'] = lab_pe['itemid'].map(ID_TO_MARKER)
lab_pe['charttime'] = pd.to_datetime(lab_pe['charttime'])
print(f"Lab records: {len(lab_pe):,}  |  hadms with any lab: {lab_pe['hadm_id'].nunique()}")

# ── 6. Build landmark cohort ──
# For each hadm: compute time since admission (hours), keep 0-48h window
print("\nBuilding 48h landmark window...")
pe_adm = pe_adm.set_index('hadm_id')
lab_pe = lab_pe.merge(pe_adm[['admittime']], on='hadm_id', how='left')
lab_pe['hours_since_adm'] = (lab_pe['charttime'] - lab_pe['admittime']).dt.total_seconds() / 3600

# Keep only 0-48h window
window_lab = lab_pe[(lab_pe['hours_since_adm'] >= 0) & (lab_pe['hours_since_adm'] <= 48)].copy()

# Core markers for trajectory: platelet, ast, alt
core_markers = ['platelet', 'ast', 'alt']
core_lab = window_lab[window_lab['marker'].isin(core_markers)]

# hadms with at least 2 measurements for platelet AND (ast OR alt) within 48h
hadm_marker_counts = core_lab.groupby(['hadm_id', 'marker']).size().unstack(fill_value=0)
has_plt2 = hadm_marker_counts.get('platelet', 0) >= 2
has_liver2 = (hadm_marker_counts.get('ast', 0) >= 2) | (hadm_marker_counts.get('alt', 0) >= 2)
eligible_hadms = hadm_marker_counts[has_plt2 & has_liver2].index.tolist()
print(f"Hadms with >=2 platelet AND >=2 (ast or alt) in 48h: {len(eligible_hadms)}")

# Exclude prevalent HELLP in 48h window (plt<100 AND liver>70)
HELLP_PLT = 100
HELLP_LIVER = 70
prevalent_hellp = set()
for hadm in eligible_hadms:
    h = core_lab[core_lab['hadm_id'] == hadm]
    pv = h.pivot_table(index='charttime', columns='marker', values='valuenum', aggfunc='max')
    has_plt_low = ('platelet' in pv.columns) and (pv['platelet'] < HELLP_PLT).any()
    has_liver_high = False
    if 'ast' in pv.columns: has_liver_high = has_liver_high or (pv['ast'] > HELLP_LIVER).any()
    if 'alt' in pv.columns: has_liver_high = has_liver_high or (pv['alt'] > HELLP_LIVER).any()
    if has_plt_low and has_liver_high:
        prevalent_hellp.add(hadm)

eligible_hadms = [h for h in eligible_hadms if h not in prevalent_hellp]
print(f"After excluding prevalent HELLP: {len(eligible_hadms)}")

# ── 7. Outcome: severe maternal composite outcome ──
print("\nComputing severe maternal composite outcome...")
# Sub-outcomes:
# 1. In-hospital death
# 2. AKI (N17 / 584)
# 3. ICU LOS >= 3 days
# 4. ICU blood transfusion
# 5. ICU vasopressors
# 6. Mechanical ventilation

# Death
death_flag = pe_adm.loc[eligible_hadms, 'hospital_expire_flag'].fillna(0).astype(int)

# AKI
aki_mask = (diag['icd_code'].str.startswith('N17', na=False) |
            ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('584', na=False)))
aki_hadms = set(diag[aki_mask]['hadm_id'])
aki_flag = pd.Series([1 if h in aki_hadms else 0 for h in eligible_hadms], index=eligible_hadms)

# ICU LOS >= 3 days
icustays = pd.read_csv(f"{BASE}\\icu\\icustays.csv.gz", usecols=['hadm_id', 'los'])
icu_los = icustays.groupby('hadm_id')['los'].sum()
icu_long = pd.Series([1 if icu_los.get(h, 0) >= 3 else 0 for h in eligible_hadms], index=eligible_hadms)

# Mechanical ventilation (from chartevents or inputevents - use oxygen devices)
# Simpler: use procedureevents for ventilation
try:
    proc = pd.read_csv(f"{BASE}\\hosp\\procedures_icd.csv.gz",
                       usecols=['hadm_id', 'icd_code', 'icd_version'])
    vent_mask = (proc['icd_code'].str.startswith('967', na=False) |  # ICD-9 mechanical ventilation
                 proc['icd_code'].str.startswith('5A19', na=False))  # ICD-10 ventilation
    vent_hadms = set(proc[vent_mask]['hadm_id'])
except:
    vent_hadms = set()
vent_flag = pd.Series([1 if h in vent_hadms else 0 for h in eligible_hadms], index=eligible_hadms)

# Blood transfusion + vasopressors from ICU inputevents
d_items = pd.read_csv(f"{BASE}\\icu\\d_items.csv.gz")
blood_labels = ['Fresh Frozen Plasma', 'Packed Red Cells', 'Whole Blood',
                'Cryoprecipitate', 'Platelet Concentrate', 'Fibrinogen Concentrate']
blood_items = d_items[d_items['label'].isin(blood_labels)]['itemid'].unique()
vaso_items = d_items[d_items['label'].str.contains(
    'norepinephrine|epinephrine|dopamine|phenylephrine|vasopressin|angiotensin',
    case=False, na=False)]['itemid'].unique()

blood_hadms = set()
vaso_hadms = set()
for chunk in pd.read_csv(f"{BASE}\\icu\\inputevents.csv.gz",
                          usecols=['hadm_id', 'itemid'],
                          low_memory=False, chunksize=2000000):
    m = chunk['hadm_id'].isin(eligible_hadms)
    if m.any():
        sub = chunk[m]
        blood_hadms.update(sub[sub['itemid'].isin(blood_items)]['hadm_id'].unique())
        vaso_hadms.update(sub[sub['itemid'].isin(vaso_items)]['hadm_id'].unique())

blood_flag = pd.Series([1 if h in blood_hadms else 0 for h in eligible_hadms], index=eligible_hadms)
vaso_flag = pd.Series([1 if h in vaso_hadms else 0 for h in eligible_hadms], index=eligible_hadms)

# Composite outcome: >=1 of (death, AKI, ICU>=3d, ventilation, transfusion, vasopressors)
composite = pd.DataFrame({
    'hadm_id': eligible_hadms,
    'death': death_flag.values,
    'aki': aki_flag.values,
    'icu_long': icu_long.values,
    'ventilation': vent_flag.values,
    'transfusion': blood_flag.values,
    'vasopressors': vaso_flag.values,
})
composite['composite_outcome'] = (composite[['death', 'aki', 'icu_long', 'ventilation', 'transfusion', 'vasopressors']].sum(axis=1) >= 1).astype(int)
composite['n_components'] = composite[['death', 'aki', 'icu_long', 'ventilation', 'transfusion', 'vasopressors']].sum(axis=1)

print(f"\nComposite outcome breakdown:")
for col in ['death', 'aki', 'icu_long', 'ventilation', 'transfusion', 'vasopressors']:
    print(f"  {col}: {composite[col].sum()} ({100*composite[col].mean():.1f}%)")
print(f"  COMPOSITE (>=1): {composite['composite_outcome'].sum()} ({100*composite['composite_outcome'].mean():.1f}%)")

# ── 8. Merge with admissions info ──
cohort = pe_adm.loc[eligible_hadms].reset_index()
cohort = cohort.merge(composite, on='hadm_id', how='left')
cohort['los_hours'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / 3600

print(f"\nFinal landmark cohort: n={len(cohort)}")
print(f"  Age: mean={cohort['anchor_age'].mean():.1f}, median={cohort['anchor_age'].median():.0f}")
print(f"  Female (F in MIMIC-IV): {(cohort['gender']=='F').sum()}")
print(f"  LOS: median={cohort['los_hours'].median()/24:.1f} days")

# Save cohort
cohort.to_csv(f"{OUT}\\moduleA_landmark_cohort.csv", index=False)
print(f"\nSaved cohort: moduleA_landmark_cohort.csv (n={len(cohort)})")

# ── 9. Extract trajectory features ──
print("\nExtracting trajectory features (platelet, ast, alt)...")

def extract_trajectory_features(hadm_id, marker):
    """Extract trajectory features for a single marker in 0-48h window."""
    h = core_lab[(core_lab['hadm_id'] == hadm_id) & (core_lab['marker'] == marker)].sort_values('hours_since_adm')
    if len(h) < 2:
        return None
    t = h['hours_since_adm'].values
    y = h['valuenum'].values
    if len(t) >= 2:
        # Linear regression slope
        coeffs = np.polyfit(t, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
    else:
        slope = 0
        intercept = y[0]
    return {
        f'{marker}_n': len(h),
        f'{marker}_first': y[0],
        f'{marker}_last': y[-1],
        f'{marker}_min': y.min(),
        f'{marker}_max': y.max(),
        f'{marker}_mean': y.mean(),
        f'{marker}_slope': slope,
        f'{marker}_intercept': intercept,
        f'{marker}_std': y.std(),
        f'{marker}_auc': np.trapz(y, t),
        f'{marker}_range': y.max() - y.min(),
        f'{marker}_time_span': t[-1] - t[0],
    }

features = []
for hadm in eligible_hadms:
    row = {'hadm_id': hadm}
    for marker in core_markers:
        feat = extract_trajectory_features(hadm, marker)
        if feat:
            row.update(feat)
        else:
            # fill with NaN
            for k in ['n', 'first', 'last', 'min', 'max', 'mean', 'slope', 'intercept', 'std', 'auc', 'range', 'time_span']:
                row[f'{marker}_{k}'] = np.nan
    features.append(row)

feat_df = pd.DataFrame(features)
print(f"Trajectory features extracted: {feat_df.shape[0]} hadms x {feat_df.shape[1]} features")

# Merge with cohort for outcome
feat_df = feat_df.merge(cohort[['hadm_id', 'composite_outcome', 'anchor_age', 'gender', 'los_hours']], on='hadm_id', how='left')
feat_df.to_csv(f"{OUT}\\moduleA_trajectory_features.csv", index=False)
print(f"Saved features: moduleA_trajectory_features.csv")

print("\n" + "=" * 70)
print("Step 1 complete.")
print("=" * 70)
