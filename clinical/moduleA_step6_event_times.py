"""
Module A Step 6 - time-to-composite-event derivation for the cumulative incidence panel.

Event time = earliest of:
  death (admissions.deathtime)
  ICU stay reaching 72 h (icustays.intime + 72 h, only if icu_long == 1)
  invasive/non-invasive ventilation start (icu.procedureevents)
  vasopressor start (icu.inputevents)
  blood-product transfusion start (icu.inputevents)
  AKI onset: first creatinine >= 1.5 x the admission nadir (hosp.labevents, itemid 50912)
Censoring: hospital discharge (los_hours).
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
BASE = r"D:\Program Files\mimic-iv-3.1"
RES = rf"{OUT}\results"
os.makedirs(RES, exist_ok=True)

print("=" * 78)
print("MODULE A STEP 6  Candidate event timestamps")
print("=" * 78)

cohort = pd.read_csv(rf"{OUT}\moduleA_landmark_cohort.csv")
clusters = pd.read_csv(rf"{OUT}\moduleA_cluster_assignments.csv")
cohort['admittime'] = pd.to_datetime(cohort['admittime'])
cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
hadms = set(cohort['hadm_id'])
print(f"cohort: {len(cohort)} admissions")

event_time = pd.Series(np.nan, index=cohort['hadm_id'].values)
def update(h, t):
    if pd.isna(t):
        return
    cur = event_time.get(h, np.nan)
    if pd.isna(cur) or t < cur:
        event_time[h] = t

def hours_from(h, ts):
    a = cohort.loc[cohort['hadm_id'] == h, 'admittime']
    if len(a) == 0 or pd.isna(ts):
        return np.nan
    return (pd.to_datetime(ts) - a.iloc[0]).total_seconds() / 3600

# ── death ────────────────────────────────────────────────────────────────
adm = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz",
                  usecols=['hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
adm = adm[adm['hadm_id'].isin(hadms)].copy()
adm['deathtime'] = pd.to_datetime(adm['deathtime'])
adm['dischtime'] = pd.to_datetime(adm['dischtime'])
for _, r in adm[adm['hospital_expire_flag'] == 1].iterrows():
    ts = r['deathtime'] if pd.notna(r['deathtime']) else r['dischtime']
    update(r['hadm_id'], (ts - r['admittime']).total_seconds() / 3600)
print("death times done")

# ── ICU 72 h ─────────────────────────────────────────────────────────────
icu = pd.read_csv(f"{BASE}\\icu\\icustays.csv.gz", usecols=['hadm_id', 'intime', 'outtime', 'los'])
icu = icu[icu['hadm_id'].isin(hadms)].copy()
icu['intime'] = pd.to_datetime(icu['intime'])
icu_long = cohort.set_index('hadm_id')['icu_long']
for h, g in icu.groupby('hadm_id'):
    if int(icu_long.get(h, 0)) == 1:
        first = g.sort_values('intime').iloc[0]
        update(h, hours_from(h, first['intime']) + 72)
print("ICU 72-h times done")

# ── ventilation start ────────────────────────────────────────────────────
pe = pd.read_csv(f"{BASE}\\icu\\procedureevents.csv.gz",
                 usecols=['hadm_id', 'itemid', 'starttime'], low_memory=False)
pe = pe[pe['hadm_id'].isin(hadms)]
d_items = pd.read_csv(f"{BASE}\\icu\\d_items.csv.gz")
vent_ids = d_items[d_items['label'].str.contains('ventilation', case=False, na=False)]['itemid'].unique()
pev = pe[pe['itemid'].isin(vent_ids)].copy()
pev['starttime'] = pd.to_datetime(pev['starttime'])
for _, r in pev.iterrows():
    update(r['hadm_id'], hours_from(r['hadm_id'], r['starttime']))
print(f"ventilation events: {pev['hadm_id'].nunique()}")

# ── vasopressors / transfusion ───────────────────────────────────────────
blood_labels = ['Fresh Frozen Plasma', 'Packed Red Cells', 'Whole Blood',
                'Cryoprecipitate', 'Platelet Concentrate', 'Fibrinogen Concentrate']
blood_items = set(d_items[d_items['label'].isin(blood_labels)]['itemid'])
vaso_items = set(d_items[d_items['label'].str.contains(
    'norepinephrine|epinephrine|dopamine|phenylephrine|vasopressin|angiotensin',
    case=False, na=False)]['itemid'])

n_vaso = n_blood = 0
for chunk in pd.read_csv(f"{BASE}\\icu\\inputevents.csv.gz",
                         usecols=['hadm_id', 'itemid', 'starttime'],
                         low_memory=False, chunksize=2000000):
    chunk = chunk[chunk['hadm_id'].isin(hadms)]
    if len(chunk) == 0:
        continue
    chunk['starttime'] = pd.to_datetime(chunk['starttime'])
    for _, r in chunk[chunk['itemid'].isin(vaso_items | blood_items)].iterrows():
        update(r['hadm_id'], hours_from(r['hadm_id'], r['starttime']))
    n_vaso += chunk[chunk['itemid'].isin(vaso_items)]['hadm_id'].nunique()
    n_blood += chunk[chunk['itemid'].isin(blood_items)]['hadm_id'].nunique()
print(f"inputevents scanned (vasopressor admissions {n_vaso}, transfusion {n_blood})")

# ── AKI onset from creatinine ────────────────────────────────────────────
print("scanning labevents for creatinine (this takes a few minutes)...")
cr_chunks = []
for chunk in pd.read_csv(f"{BASE}\\hosp\\labevents.csv.gz",
                         usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'],
                         low_memory=False, chunksize=2000000):
    m = (chunk['itemid'] == 50912) & chunk['hadm_id'].isin(hadms) & chunk['valuenum'].notna()
    if m.any():
        cr_chunks.append(chunk[m])
cr = pd.concat(cr_chunks, ignore_index=True)
cr['charttime'] = pd.to_datetime(cr['charttime'])
cr = cr[(cr['valuenum'] > 0) & (cr['valuenum'] < 30)]
cr = cr.sort_values(['hadm_id', 'charttime'])
print(f"creatinine measurements: {len(cr):,} in {cr['hadm_id'].nunique()} admissions")

aki_time = {}
for h, g in cr.groupby('hadm_id'):
    g = g.drop_duplicates('charttime')
    v = g['valuenum'].values
    t = g['charttime'].values
    base = np.minimum.accumulate(v)
    hit = np.where(v >= 1.5 * base)[0]
    if len(hit):
        aki_time[h] = hours_from(h, pd.Timestamp(t[hit[0]]))
for h, t in aki_time.items():
    update(h, t)
print(f"AKI onsets derived: {len(aki_time)}")

# ── assemble ─────────────────────────────────────────────────────────────
out = cohort[['hadm_id', 'admittime', 'dischtime', 'los_hours', 'composite_outcome']].copy()
out['event_hours'] = out['hadm_id'].map(event_time)
out = out.merge(clusters[['hadm_id', 'trajectory_class', 'class_name']], on='hadm_id', how='left')
if out.loc[out['trajectory_class'].notna(), 'class_name'].isna().any():
    raise ValueError('Missing trajectory class labels')

# events without a timestamp are censored at discharge
out['event_hours'] = out['event_hours'].clip(lower=0)
out.loc[out['event_hours'].isna(), 'event_hours'] = out.loc[out['event_hours'].isna(), 'los_hours']
out['event_hours'] = np.minimum(out['event_hours'], out['los_hours'])
out['time_days'] = out['event_hours'] / 24.0
out['los_days'] = out['los_hours'] / 24.0
out['event'] = out['composite_outcome'].astype(int)

out.to_csv(rf"{RES}\moduleA_time_to_event.csv", index=False)

print("\nEvent-time completeness by outcome status:")
print(out.groupby('event')['event_hours'].describe()[['count', 'mean', '50%', 'max']].to_string())
print(f"\nevents with derived timestamp: {int(out.loc[out.event==1,'event_hours'].notna().sum())} / {int(out.event.sum())}")
print(out.groupby('class_name')['event'].agg(['sum', 'count', 'mean']).to_string())
