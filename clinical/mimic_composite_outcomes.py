"""
验证：严重母体复合结局 v2（修复 pharmacy 报错 + 修正 ICU 血液制品 itemid）
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = r"D:\Program Files\mimic-iv-3.1"
print("=" * 70)
print("PE 队列严重母体复合结局事件数（修复版）")
print("=" * 70)

# ── PE 队列 ──
diag = pd.read_csv(f"{BASE}\\hosp\\diagnoses_icd.csv.gz",
                   usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
pe_mask = (
    diag['icd_code'].str.startswith('O14', na=False) |
    diag['icd_code'].str.startswith('O11', na=False) |
    diag['icd_code'].str.startswith('O10', na=False) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('642', na=False))
)
pe_hadms = set(diag[pe_mask]['hadm_id'])
print(f"PE 住院 hadm: {len(pe_hadms)}")

adm = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz")
adm['admittime'] = pd.to_datetime(adm['admittime'])
adm['dischtime'] = pd.to_datetime(adm['dischtime'])
pe_adm = adm[adm['hadm_id'].isin(pe_hadms)].copy()

d_items = pd.read_csv(f"{BASE}\\icu\\d_items.csv.gz")

# ── 子结局 1: 院内死亡 ──
death = pe_adm['hospital_expire_flag'].fillna(0).astype(int)

# ── 子结局 2: AKI ICD ──
aki_mask = (diag['icd_code'].str.startswith('N17', na=False) |
            ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('584', na=False)))
aki_hadms = set(diag[aki_mask]['hadm_id'])
aki_icd = pe_adm['hadm_id'].isin(aki_hadms).astype(int)

# ── 子结局 3: ICU 长住院 ──
icustays = pd.read_csv(f"{BASE}\\icu\\icustays.csv.gz",
                       usecols=['hadm_id', 'los'])
icu_stay_per_hadm = icustays.groupby('hadm_id')['los'].sum()
icu_long = pe_adm['hadm_id'].map(lambda h: int(icu_stay_per_hadm.get(h, 0) >= 3))

# ── 子结局 4: ICU 血液制品 ──
# 真实血液制品 itemid（d_items 中 label 精确匹配 + itemtype 为 input 或 fluid）
blood_labels = ['Fresh Frozen Plasma', 'Packed Red Cells', 'Whole Blood',
                'Cryoprecipitate', 'Platelet Concentrate', 'ESDEP',
                'Fibrinogen Concentrate', 'Prothrombin Complex']
blood_items = d_items[d_items['label'].isin(blood_labels)]
print(f"\nICU 血液制品真实 itemid ({len(blood_items)} 个):")
print(blood_items[['itemid', 'label']].to_string())

chunks = []
for chunk in pd.read_csv(f"{BASE}\\icu\\inputevents.csv.gz",
                          usecols=['hadm_id', 'itemid'],
                          low_memory=False, chunksize=2000000):
    m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(blood_items['itemid'].unique()))
    if m.any(): chunks.append(chunk[m])
blood_hadms = set(pd.concat(chunks)['hadm_id']) if chunks else set()
blood_trans = pe_adm['hadm_id'].isin(blood_hadms).astype(int)
print(f"ICU 血液制品输注 hadm: {blood_trans.sum()}")

# ── 子结局 5: ICU 升压药 ──
vaso_labels = ['Norepinephrine', 'Epinephrine', 'Dopamine', 'Phenylephrine',
               'Vasopressin', 'Vasopressor', 'Angiotensin II', 'Metoprolol',
               'Esmolol', 'Labetalol', 'Hydralazine', 'Nifedipine', 'Magnesium Sulfate']
vaso_items = d_items[d_items['label'].str.contains(
    'vasopressin|norepinephrine|epinephrine|dopamine|phenylephrine|angiotensin',
    case=False, na=False)]
print(f"\nICU 升压药 itemid ({len(vaso_items)} 个)")
vaso_hadms = set()
if len(vaso_items) > 0:
    for chunk in pd.read_csv(f"{BASE}\\icu\\inputevents.csv.gz",
                              usecols=['hadm_id', 'itemid'],
                              low_memory=False, chunksize=2000000):
        m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(vaso_items['itemid']))
        if m.any(): vaso_hadms.update(chunk[m]['hadm_id'].unique())
vaso = pe_adm['hadm_id'].isin(vaso_hadms).astype(int)
print(f"ICU 升压药 hadm: {vaso.sum()}")

# ── 子结局 6: ICU 机械通气 ──
# 真实机械通气 itemid: chartevents 中 Ventilator Mode / Respiratory Rate（intubated）
vent_items = d_items[d_items['label'].str.contains(
    'ventilator mode|respiratory support|endotracheal|tracheostomy tube',
    case=False, na=False)]
print(f"\nICU 机械通气 itemid ({len(vent_items)} 个)")
vent_hadms = set()
if len(vent_items) > 0:
    for chunk in pd.read_csv(f"{BASE}\\icu\\chartevents.csv.gz",
                              usecols=['hadm_id', 'itemid'],
                              low_memory=False, chunksize=3000000):
        m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(vent_items['itemid']))
        if m.any(): vent_hadms.update(chunk[m]['hadm_id'].unique())
mechvent = pe_adm['hadm_id'].isin(vent_hadms).astype(int)
print(f"ICU 机械通气 hadm: {mechvent.sum()}")

# ── 汇总 ──
print("\n" + "=" * 70)
print("【严重母体复合结局汇总】")
print("=" * 70)
outcomes = pd.DataFrame({
    'death': death, 'aki_icd': aki_icd, 'icu_long': icu_long,
    'blood_trans': blood_trans, 'vaso': vaso, 'mechvent': mechvent
})

print(f"\n{'结局':>20s} | {'n':>6s} | {'%':>7s}")
print("-" * 42)
for col, label in [('death', '院内死亡'), ('aki_icd', 'AKI (ICD N17/584)'),
                    ('icu_long', 'ICU LOS ≥ 3天'), ('blood_trans', '血液制品输注'),
                    ('vaso', 'ICU 升压药'), ('mechvent', 'ICU 机械通气')]:
    n = outcomes[col].sum()
    pct = 100*n/len(outcomes)
    print(f"{label:>20s} | {n:>6d} | {pct:>6.1f}%")

# 复合 = 至少 1 个
comp = (outcomes.sum(axis=1) >= 1).astype(int)
print(f"\n{'★ 复合结局 (≥1/6)':>20s} | {comp.sum():>6d} | {100*comp.mean():>6.1f}%")

# 复合结局 (≥2/6) — 更严格
comp2 = (outcomes.sum(axis=1) >= 2).astype(int)
print(f"{'  复合结局 (≥2/6)':>20s} | {comp2.sum():>6d} | {100*comp2.mean():>6.1f}%")

# 分层
print(f"\n  按满足结局数分层:")
for k in range(0, 5):
    cnt = (outcomes.sum(axis=1) == k).sum()
    print(f"    {k} 个: {cnt} ({100*cnt/len(outcomes):.1f}%)")
cnt5plus = (outcomes.sum(axis=1) >= 5).sum()
print(f"    ≥5 个: {cnt5plus}")

# ── 48h landmark 后 ──
print("\n" + "=" * 70)
print("【48h landmark 后发生的复合结局】")
print("=" * 70)

# 简化的 landmark 有效队列（有 plt+liver 检测且排除 prevalent）
# 先快速重新做 lab 和 landmark 队列
print("\n构建 landmark 队列...")
LAB_TARGET = {51265: 'platelet', 50878: 'ast', 51256: 'ast', 50861: 'alt'}
chunks = []
for chunk in pd.read_csv(f"{BASE}\\hosp\\labevents.csv.gz",
                          usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'],
                          low_memory=False, chunksize=2000000):
    m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(LAB_TARGET)) & chunk['valuenum'].notna()
    if m.any(): chunks.append(chunk[m])
lab = pd.concat(chunks)
lab['charttime'] = pd.to_datetime(lab['charttime'])

strict_eligible = []
for _, row in pe_adm.iterrows():
    hadm = row['hadm_id']; admtime = row['admittime']
    if pd.isna(admtime): continue
    e48 = admtime + pd.Timedelta(hours=48)
    h_lab = lab[lab['hadm_id'] == hadm]
    w = h_lab[(h_lab['charttime'] >= admtime) & (h_lab['charttime'] <= e48)]
    plt_n = (w['itemid'] == 51265).sum()
    liver_n = w['itemid'].isin([50878, 51256, 50861]).sum()
    if plt_n < 1 or liver_n < 1: continue
    plt_min = w[w['itemid'] == 51265]['valuenum'].min()
    liver_max = w[w['itemid'].isin([50878, 51256, 50861])]['valuenum'].max()
    if pd.notna(plt_min) and pd.notna(liver_max) and plt_min < 100 and liver_max > 70:
        continue  # prevalent
    strict_eligible.append(hadm)

mask2 = pe_adm['hadm_id'].isin(strict_eligible)
print(f"  排除 prevalent 的简化 landmark 队列: {mask2.sum()}")

# 复合结局是"整个住院期间"的，在 landmark 设计下近似 = 住院时长 > 48h 的复合结局
pe_adm['los_h'] = (pe_adm['dischtime'] - pe_adm['admittime']).dt.total_seconds() / 3600
post48_mask = pe_adm['los_h'] > 48
post48_comp = (comp.values & post48_mask.values & mask2.values).sum()
post48_comp2 = (comp2.values & post48_mask.values & mask2.values).sum()
print(f"\n  ⭐ landmark 队列中，48h 后发生复合结局 (≥1): {post48_comp}")
print(f"  ⭐ landmark 队列中，48h 后发生复合结局 (≥2): {post48_comp2}")
print(f"  复合结局事件率 (≥1): {100*post48_comp/max(mask2.sum(),1):.1f}%")
print(f"  复合结局事件率 (≥2): {100*post48_comp2/max(mask2.sum(),1):.1f}%")

# 各子结局在 landmark 队列的事件数
print(f"\n  landmark 队列中各子结局事件数（48h 后）:")
post48_mask_arr = post48_mask.values & mask2.values
for col, label in [('death', '院内死亡'), ('aki_icd', 'AKI'),
                    ('icu_long', 'ICU ≥3天'), ('blood_trans', '血液制品'),
                    ('vaso', '升压药'), ('mechvent', '机械通气')]:
    n = outcomes[col].values[post48_mask_arr].sum()
    print(f"    {label}: {n}")

# ── 与 HELLP 编码的关联 ──
hellp_mask = diag['icd_code'].str.startswith('O142', na=False) | (
    (diag['icd_version'] == 9) & diag['icd_code'].str.startswith('6425', na=False))
hellp_hadms = set(diag[hellp_mask]['hadm_id'])
print(f"\nHELLP 编码住院中:")
print(f"  复合结局 (≥1): {outcomes.loc[pe_adm['hadm_id'].isin(hellp_hadms)].sum().sum()} / {sum(pe_adm['hadm_id'].isin(hellp_hadms))}")
print(f"  HELLP 编码 hadm 中各子结局:")
h_idx = pe_adm.index[pe_adm['hadm_id'].isin(hellp_hadms)]
for col, label in [('death', '死亡'), ('aki_icd', 'AKI'), ('icu_long', 'ICU≥3d'),
                    ('blood_trans', '输血'), ('vaso', '升压药'), ('mechvent', '通气')]:
    print(f"    {label}: {outcomes.loc[h_idx, col].sum()} / {len(h_idx)}")

print("\n" + "=" * 70)
print("完成。复合结局事件数充足。")
print("=" * 70)
