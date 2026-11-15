"""
MIMIC-IV v3.1 子痫前期—HELLP 综合征 预查询脚本 v2
修复 admissions 列名；增加更全面的 PE/HELLP 编码覆盖
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = r"D:\Program Files\mimic-iv-3.1"
print("=" * 70)
print("MIMIC-IV v3.1 PE/HELLP 预查询 v2")
print("=" * 70)

# ── 先探查各表列 ──
adm_head = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz", nrows=1)
diag_head = pd.read_csv(f"{BASE}\\hosp\\diagnoses_icd.csv.gz", nrows=1)
d_lab_head = pd.read_csv(f"{BASE}\\hosp\\d_labitems.csv.gz", nrows=1)
print(f"\nadmissions 列: {list(adm_head.columns)}")
print(f"diagnoses_icd 列: {list(diag_head.columns)}")

# ── 读取诊断 ──
print("\n【1】诊断编码查询...")
diag = pd.read_csv(f"{BASE}\\hosp\\diagnoses_icd.csv.gz",
                   usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version', 'seq_num'])

# PE 相关 ICD-10
pe_icd10_codes = [
    'O140', 'O1400', 'O1401', 'O1402', 'O1403',  # 轻度 PE
    'O141', 'O1410', 'O1411', 'O1412', 'O1413',  # 重度 PE
    'O142',                                      # HELLP
    'O143', 'O144', 'O145', 'O149',              # 其他/未特指
    'O11', 'O110', 'O119',                       # 先存高血压合并 PE
    'O100', 'O101', 'O102', 'O109', 'O10',       # 妊娠高血压
]
# ICD-9 PE 编码
pe_icd9_codes = ['6424', '6425', '6426', '6427', '6428', '6429',
                 '64240', '64250', '64260', '64270', '64280', '64290',
                 '642', '6420', '6421', '6422', '6423', '64240', '64250']

# 用 startswith 匹配前缀（处理 ICD 编码后可能有扩展数字）
pe_mask = (
    diag['icd_code'].str.startswith('O14', na=False) |
    diag['icd_code'].str.startswith('O11', na=False) |
    diag['icd_code'].str.startswith('O10', na=False) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('642', na=False))
)
hellp_mask = (
    diag['icd_code'].str.startswith('O142', na=False) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('6425', na=False))
)
severe_pe_mask = (
    diag['icd_code'].str.startswith('O141', na=False) |
    diag['icd_code'].str.startswith('O142', na=False) |
    diag['icd_code'].str.startswith('O149', na=False) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('6425', na=False)) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('6426', na=False)) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('6427', na=False)) |
    ((diag['icd_version'] == 9) & diag['icd_code'].str.startswith('6429', na=False))
)

pe_diag = diag[pe_mask].copy()
hellp_diag = diag[hellp_mask].copy()
severe_pe_diag = diag[severe_pe_mask].copy()

# 进一步：在诊断编码中搜索 "hellp" 相关编码，看看有没有其他写法
all_icd_codes = diag['icd_code'].unique()
hellp_related = [c for c in all_icd_codes if isinstance(c, str) and ('O142' in str(c) or '6425' in str(c))]
print(f"\n  诊断表全部编码中 O142/6425 相关的唯一编码值: {hellp_related}")
# 也搜索完整的 icd_code 包含这些的
pe_subjects = set(pe_diag['subject_id'])
pe_hadms = set(pe_diag['hadm_id'])
hellp_subjects = set(hellp_diag['subject_id'])
hellp_hadms = set(hellp_diag['hadm_id'])
severe_pe_subjects = set(severe_pe_diag['subject_id'])
pe_severity_only = diag[diag['icd_code'].str.startswith('O142', na=False)]
print(f"\n  PE/O14.x + O11.x + O10.x + 642.x 编码患者: {len(pe_subjects)}")
print(f"  对应住院 hadm: {len(pe_hadms)}")
print(f"  HELLP(O142/6425) 编码患者: {len(hellp_subjects)}")
print(f"  HELLP(O142/6425) 编码住院: {len(hellp_hadms)}")
print(f"  重度 PE 编码患者（O141/O142/O149 + ICD-9）: {len(severe_pe_subjects)}")

# 输出 PE 编码的频率分布，看看主要用的是哪几个
print(f"\n  PE 相关编码频率分布 (top 15):")
top_codes = pe_diag['icd_code'].value_counts().head(15)
for code, cnt in top_codes.items():
    print(f"    {code:10s} n={cnt}")

# ── 读取 admissions ──
print("\n【2】关联住院信息...")
# 查看 admissions 有哪些列
_adm_cols_first = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz", nrows=1).columns.tolist()
print(f"  admissions 完整列: {_adm_cols_first}")

adm = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz")
adm['admittime'] = pd.to_datetime(adm['admittime'])
adm['dischtime'] = pd.to_datetime(adm['dischtime'])
pe_adm = adm[adm['hadm_id'].isin(pe_hadms)].copy()

print(f"  PE 相关住院数: {len(pe_adm)}")
print(f"  PE 患者年龄分布:")
patients = pd.read_csv(f"{BASE}\\hosp\\patients.csv.gz", usecols=['subject_id', 'anchor_age', 'gender'])
pe_patients = patients[patients['subject_id'].isin(pe_subjects)]
print(f"    年龄: mean={pe_patients['anchor_age'].mean():.1f}, "
      f"median={pe_patients['anchor_age'].median():.0f}, "
      f"n={pe_patients['anchor_age'].notna().sum()}")
print(f"    性别: F={(pe_patients['gender']=='F').sum()}, "
      f"M={(pe_patients['gender']=='M').sum()}")  # MIMIC 中 M=女, F=男

# ── 3. 关键 lab itemid ──
print("\n【3】实验室指标 itemid 映射...")
d_lab = pd.read_csv(f"{BASE}\\hosp\\d_labitems.csv.gz")

# 手动构建 MIMIC-IV 中 PE/HELLP 常用 itemid
# 参考 MIMIC-IV labevents 标准 itemid
LAB_ITEMIDS = {
    'platelet': [51265, 50826],          # Platelet Count
    'ast': [50878, 51256],               # Aspartate Aminotransferase
    'alt': [50861, 51006],               # Alanine Aminotransferase
    'ldh': [50813, 51002],               # Lactate Dehydrogenase
    'bilirubin_total': [50885, 51508],   # Bilirubin, Total
    'bilirubin_direct': [50809],         # Bilirubin, Direct
    'pt': [51274, 50822],                # Prothrombin Time
    'inr': [51237, 50823],               # INR
    'aptt': [51275, 50824],              # APTT
    'fibrinogen': [51230, 50817],        # Fibrinogen
    'ddimer': [51544, 51543, 50995],     # D-dimer
    'albumin': [50862],                  # Albumin
    'creatinine': [50912, 50821],        # Creatinine (AKI)
    'lactate': [50001, 50025],           # Lactate
}
# 打印 item 详情用于核对
all_ids = [i for ids in LAB_ITEMIDS.values() for i in ids]
lab_items_info = d_lab[d_lab['itemid'].isin(all_ids)][['itemid', 'label', 'fluid', 'category']]
print(lab_items_info.to_string())

# 反向映射 itemid → marker
ID_TO_MARKER = {}
for marker, ids in LAB_ITEMIDS.items():
    for i in ids:
        ID_TO_MARKER[i] = marker
ITEM_IDS = set(ID_TO_MARKER.keys())

# ── 4. 读 labevents (PE 住院) ──
print(f"\n【4】实验室数据加载（PE 住院）...")
print("  注意：labevents.csv.gz 可能很大，只读取 PE hadm + 关键 itemid")

# 只读取 PE 住院相关的
chunks = []
for chunk in pd.read_csv(f"{BASE}\\hosp\\labevents.csv.gz",
                         usecols=['subject_id', 'hadm_id', 'itemid', 'charttime',
                                  'valuenum', 'valueuom'],
                         low_memory=False, chunksize=2000000):
    m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(ITEM_IDS)) & chunk['valuenum'].notna()
    if m.any():
        chunks.append(chunk[m])
    print(f"  已读 {sum(len(c) for c in chunks)} 行...", end='\r')

lab_pe = pd.concat(chunks, ignore_index=True)
lab_pe['marker'] = lab_pe['itemid'].map(ID_TO_MARKER)
lab_pe['charttime'] = pd.to_datetime(lab_pe['charttime'])
print(f"\n  PE 住院中关键检测记录数（去空值）: {len(lab_pe):,}")
print(f"  覆盖 hadm 数: {lab_pe['hadm_id'].nunique()} / {len(pe_hadms)}")

# 各 marker 的覆盖情况
print(f"\n  各指标覆盖:")
for marker in ['platelet', 'ast', 'alt', 'ldh', 'bilirubin_total', 'bilirubin_direct',
               'pt', 'inr', 'aptt', 'fibrinogen', 'ddimer', 'albumin', 'creatinine', 'lactate']:
    m_lab = lab_pe[lab_pe['marker'] == marker]
    if len(m_lab) > 0:
        hadm_with = m_lab['hadm_id'].nunique()
        vals = m_lab['valuenum']
        print(f"    {marker:20s}: hadm={hadm_with:4d}/{len(pe_hadms):4d} ({100*hadm_with/len(pe_hadms):5.1f}%), "
              f"n={len(vals):6d}, median={vals.median():7.1f}, mean={vals.mean():7.1f}")

# ── 5. PE 住院中 ≥2 次检测的比例 ──
print(f"\n【5】有 ≥2 次关键检测的 PE 住院...")
core = ['platelet', 'ast', 'alt', 'ldh', 'bilirubin_total']
core_lab = lab_pe[lab_pe['marker'].isin(core)]
hadm_total_counts = core_lab.groupby('hadm_id').size()
hadm_by_marker = core_lab.groupby(['hadm_id', 'marker']).size().unstack(fill_value=0)

print(f"  PE 住院总数: {len(pe_hadms)}")
print(f"  有任何核心检测: {hadm_total_counts.index.nunique()}")
print(f"  总核心检测次数分布:")
print(f"    0 次: {len(pe_hadms) - hadm_total_counts.index.nunique()}")
print(f"    1 次: {(hadm_total_counts == 1).sum()}")
print(f"    2 次: {(hadm_total_counts == 2).sum()}")
print(f"    3-4 次: {((hadm_total_counts >= 3) & (hadm_total_counts <= 4)).sum()}")
print(f"    ≥5 次: {(hadm_total_counts >= 5).sum()}")

# ── 6. 48h landmark HELLP 事件 ──
print(f"\n【6】48h landmark 设计：HELLP 实验室判定...")

# HELLP 阈值（经典 Tennessee 标准）
# 溶血: LDH > 600 U/L 或胆红素 > 1.2 mg/dL
# 肝酶: AST > 70 U/L 或 ALT > 70 U/L（> 2× ULN）
# 血小板: < 100×10^9/L
HELLP_PLT = 100
HELLP_LDH = 600
HELLP_AST_ALT = 70
HELLP_BILI = 1.2  # mg/dL

results = []
for idx, row in pe_adm.iterrows():
    hadm = row['hadm_id']
    admtime = row['admittime']
    if pd.isna(admtime):
        continue
    end_48h = admtime + pd.Timedelta(hours=48)
    end_7d = admtime + pd.Timedelta(days=7)

    hadm_lab = core_lab[core_lab['hadm_id'] == hadm].copy()
    if len(hadm_lab) == 0:
        continue

    w = hadm_lab[(hadm_lab['charttime'] >= admtime) & (hadm_lab['charttime'] <= end_48h)]
    o = hadm_lab[(hadm_lab['charttime'] > end_48h) & (hadm_lab['charttime'] <= end_7d)]

    # 观察窗（prevalent）已满足 HELLP → 排除
    def hellp_status(lab_df):
        if len(lab_df) == 0:
            return False
        pv = lab_df.pivot_table(index='charttime', columns='marker', values='valuenum', aggfunc='max')
        has_plt = False
        has_liver = False
        has_hem = False
        if 'platelet' in pv.columns:
            has_plt = (pv['platelet'] < HELLP_PLT).any()
        if 'ast' in pv.columns:
            has_liver = has_liver or (pv['ast'] > HELLP_AST_ALT).any()
        if 'alt' in pv.columns:
            has_liver = has_liver or (pv['alt'] > HELLP_AST_ALT).any()
        if 'ldh' in pv.columns:
            has_hem = has_hem or (pv['ldh'] > HELLP_LDH).any()
        if 'bilirubin_total' in pv.columns:
            has_hem = has_hem or (pv['bilirubin_total'] > HELLP_BILI).any()
        return has_plt and has_liver and has_hem

    results.append({
        'hadm_id': hadm,
        'subject_id': row['subject_id'],
        'window_n': len(w),
        'outcome_n': len(o),
        'hellp_in_window': hellp_status(w),
        'incident_hellp_48h_7d': hellp_status(o) and not hellp_status(w),
    })

res_df = pd.DataFrame(results)
print(f"\n  PE 住院总 hadm: {len(pe_hadms)}")
print(f"  有核心 lab 的 hadm: {res_df.shape[0]}")
print(f"  观察窗内已满足 HELLP（prevalent, 排除）: {res_df['hellp_in_window'].sum()}")
print(f"  观察窗有 ≥2 次检测且排除 prevalent 的有效 landmark 队列: "
      f"{res_df[(res_df['window_n'] >= 2) & (~res_df['hellp_in_window'])].shape[0]}")

eligible = res_df[(res_df['window_n'] >= 2) & (~res_df['hellp_in_window'])]
print(f"\n  === 关键数字 ===")
print(f"  有效 landmark 队列: n={len(eligible)}")
print(f"  landmark 后 7d 内新发 HELLP 事件: n={eligible['incident_hellp_48h_7d'].sum()}")
print(f"  事件率: {100*eligible['incident_hellp_48h_7d'].sum()/max(len(eligible),1):.1f}%")

# ── 7. 不同 HELLP 阈值敏感性 ──
print(f"\n【7】HELLP 事件数的阈值敏感性...")
for plt, ldh, liver in [(100, 600, 70), (100, 340, 70), (150, 600, 40), (100, 300, 40)]:
    cnt = 0
    for _, r in eligible.iterrows():
        hadm = r['hadm_id']
        admtime = pe_adm[pe_adm['hadm_id'] == hadm]['admittime'].iloc[0]
        e48 = admtime + pd.Timedelta(hours=48)
        e7d = admtime + pd.Timedelta(days=7)
        o = core_lab[(core_lab['hadm_id'] == hadm) &
                     (core_lab['charttime'] > e48) & (core_lab['charttime'] <= e7d)]
        if len(o) == 0:
            continue
        pv = o.pivot_table(index='charttime', columns='marker', values='valuenum', aggfunc='max')
        hp = (pv['platelet'] < plt).any() if 'platelet' in pv.columns else False
        hl = False
        if 'ast' in pv.columns: hl = hl or (pv['ast'] > liver).any()
        if 'alt' in pv.columns: hl = hl or (pv['alt'] > liver).any()
        hh = False
        if 'ldh' in pv.columns: hh = hh or (pv['ldh'] > ldh).any()
        if 'bilirubin_total' in pv.columns: hh = hh or (pv['bilirubin_total'] > 1.2).any()
        if hp and hl and hh:
            cnt += 1
    print(f"  Plt<{plt}, LDH>{ldh}, AST/ALT>{liver}: 新发 HELLP = {cnt}")

print("\n" + "=" * 70)
print("MIMIC-IV 预查询完成")
print("=" * 70)
