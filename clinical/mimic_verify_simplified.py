"""
验证：只用 Platelet + AST/ALT（不拉 LDH/胆红素）
"血小板降 + 肝酶升" 作为 PE 恶化 endpoint 的可行性
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BASE = r"D:\Program Files\mimic-iv-3.1"

# ── 读诊断表，拿 PE hadm + HELLP 编码 hadm ──
print("加载诊断表...")
diag = pd.read_csv(f"{BASE}\\hosp\\diagnoses_icd.csv.gz",
                   usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])

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
pe_hadms = set(diag[pe_mask]['hadm_id'])
hellp_hadms = set(diag[hellp_mask]['hadm_id'])

# ── 读 admissions ──
print("加载 admissions...")
adm = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz")
adm['admittime'] = pd.to_datetime(adm['admittime'])
pe_adm = adm[adm['hadm_id'].isin(pe_hadms)].copy()

# ── 读 labevents（只保留 Platelet + AST + ALT 三个 itemid）─────────────────────────────────
# 只看这三个 item，缩小读取范围
print("加载 Platelet + AST + ALT 检测（三个 item）...")
TARGET_ITEMS = {
    51265: 'platelet',       # Platelet Count
    50878: 'ast',            # Asparate Aminotransferase
    51256: 'ast2',           # AST alternative
    50861: 'alt',            # ALT
    51006: 'alt2',           # ALT alternative（但之前发现是 Urea Nitrogen？再确认）
}
# 51006 在 d_labitems 里实际是 Urea Nitrogen，排除
TARGET_ITEMS = {51265: 'platelet', 50878: 'ast', 51256: 'ast2', 50861: 'alt'}

chunks = []
for chunk in pd.read_csv(f"{BASE}\\hosp\\labevents.csv.gz",
                         usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
                         low_memory=False, chunksize=2000000):
    m = (chunk['hadm_id'].isin(pe_hadms)) & (chunk['itemid'].isin(TARGET_ITEMS.keys())) & chunk['valuenum'].notna()
    if m.any():
        chunks.append(chunk[m])
lab_pe = pd.concat(chunks, ignore_index=True)
lab_pe['marker'] = lab_pe['itemid'].map(TARGET_ITEMS)
# 合并 ast2/alt2 标记
lab_pe.loc[lab_pe['marker'] == 'ast2', 'marker'] = 'ast'
lab_pe['charttime'] = pd.to_datetime(lab_pe['charttime'])
print(f"  加载完成: {len(lab_pe):,} 行，覆盖 {lab_pe['hadm_id'].nunique()} hadm")

# ── 1. 覆盖率：同时有 Platelet 且有 AST/ALT 的 hadm ──
print("\n" + "=" * 70)
print("【1】三项指标的覆盖（核心可行性验证）")
print("=" * 70)
hadm_by_marker = lab_pe.groupby(['hadm_id', 'marker']).size().unstack(fill_value=0)
print(f"  PE 住院总 hadm: {len(pe_hadms)}")

for m in ['platelet', 'ast', 'alt']:
    if m in hadm_by_marker.columns:
        n = (hadm_by_marker[m] > 0).sum()
        print(f"    有 {m}: {n} ({100*n/len(pe_hadms):.1f}%)")

# 关键：同时有 platelet AND (ast OR alt)
has_plt = hadm_by_marker.get('platelet', pd.Series(dtype=float)) > 0
has_any_liver = (hadm_by_marker.get('ast', pd.Series(dtype=float)) > 0) | (hadm_by_marker.get('alt', pd.Series(dtype=float)) > 0)
has_both = has_plt & has_any_liver
print(f"\n  ⭐ 同时有 Platelet + (AST 或 ALT)：{has_both.sum()} hadm ({100*has_both.sum()/len(pe_hadms):.1f}%)")
print(f"    => 这就是 [简化版] 的可分析队列")

# ── 2. 48h landmark 下，用简化定义能抓多少事件 ──
print("\n" + "=" * 70)
print("【2】Landmark 设计 + 简化 endpoint（Platelet↓ + Liver↑）")
print("=" * 70)

# 简化 endpoint 阈值（参考 HELLP 标准，但只看这两个）
PLT_THRESH = 100      # ×10^9/L
AST_ALT_THRESH = 70   # U/L

results = []
for idx, row in pe_adm.iterrows():
    hadm = row['hadm_id']
    admtime = row['admittime']
    if pd.isna(admtime):
        continue
    e48 = admtime + pd.Timedelta(hours=48)
    e7d = admtime + pd.Timedelta(days=7)

    hadm_lab = lab_pe[lab_pe['hadm_id'] == hadm].copy()
    # 合并 ast/alt
    hadm_lab['is_liver'] = hadm_lab['marker'].isin(['ast', 'alt'])

    # 每个时间点的最差值（platelet 取最低，liver 取最高）
    w = hadm_lab[(hadm_lab['charttime'] >= admtime) & (hadm_lab['charttime'] <= e48)]
    o = hadm_lab[(hadm_lab['charttime'] > e48) & (hadm_lab['charttime'] <= e7d)]

    def simplify_endpoint(lab_df):
        if len(lab_df) == 0:
            return False, False, False
        plt_min = lab_df[lab_df['marker'] == 'platelet']['valuenum'].min()
        liver_max = lab_df[lab_df['is_liver']]['valuenum'].max()
        plt_low = pd.notna(plt_min) and plt_min < PLT_THRESH
        liver_high = pd.notna(liver_max) and liver_max > AST_ALT_THRESH
        return plt_low, liver_high, (plt_low and liver_high)

    # 观察窗（prevalent）
    w_plt, w_liver, w_combined = simplify_endpoint(w)
    # 结局窗
    o_plt, o_liver, o_combined = simplify_endpoint(o)

    results.append({
        'hadm_id': hadm,
        'subject_id': row['subject_id'],
        'window_plt_n': (w['marker'] == 'platelet').sum(),
        'window_liver_n': w['is_liver'].sum(),
        'hellp_coded': hadm in hellp_hadms,
        'plt_low_in_window': w_plt,
        'liver_high_in_window': w_liver,
        'prevalent_combined': w_combined,
        'plt_low_post48h': o_plt,
        'liver_high_post48h': o_liver,
        'incident_combined_48h_7d': o_combined and not w_combined,
    })

res = pd.DataFrame(results)
print(f"\n  PE 住院总 hadm: {len(res)}")
print(f"  观察窗已满足简化 endpoint（prevalent 排除）: {res['prevalent_combined'].sum()}")

# 有效队列 = 观察窗有 platelet ≥1 次 + 有 liver ≥1 次 + 排除 prevalent
eligible = res[
    (res['window_plt_n'] >= 1) &
    (res['window_liver_n'] >= 1) &
    (~res['prevalent_combined'])
].copy()
print(f"  🔑 有效简化 landmark 队列（有 plt+liver 检测且排除 prevalent）: {len(eligible)}")

# 新发事件数
n_incident = eligible['incident_combined_48h_7d'].sum()
print(f"\n  ⭐ 48h landmark 后 7d 内新发 [Plt<{PLT_THRESH} 且 AST/ALT>{AST_ALT_THRESH}]：{n_incident} 例")
print(f"  事件率: {100*n_incident/max(len(eligible),1):.1f}%")

# ── 3. 敏感性分析：不同阈值组合 ──
print("\n【3】简化 endpoint 的阈值敏感性分析")
print(f"  {'Plt':>5s} {'AST/ALT':>7s} | {'eligible':>10s} {'incident':>8s} | 事件与 HELLP 编码重叠")
for plt, liver in [(100, 70), (120, 50), (150, 40), (80, 80), (100, 40), (150, 70)]:
    cnt = 0
    overlap = 0
    for _, r in res.iterrows():
        hadm = r['hadm_id']
        admtime = pe_adm[pe_adm['hadm_id'] == hadm]['admittime'].iloc[0]
        e48 = admtime + pd.Timedelta(hours=48)
        e7d = admtime + pd.Timedelta(days=7)
        hadm_lab = lab_pe[lab_pe['hadm_id'] == hadm].copy()
        hadm_lab['is_liver'] = hadm_lab['marker'].isin(['ast', 'alt'])
        w = hadm_lab[(hadm_lab['charttime'] >= admtime) & (hadm_lab['charttime'] <= e48)]
        o = hadm_lab[(hadm_lab['charttime'] > e48) & (hadm_lab['charttime'] <= e7d)]
        # 要求观察窗同时有 plt+liver 检测（有效队列）
        if (w['marker'] == 'platelet').sum() < 1 or w['is_liver'].sum() < 1:
            continue
        w_plt_low = w[w['marker'] == 'platelet']['valuenum'].min() < plt if len(w[w['marker'] == 'platelet']) > 0 else False
        w_liver_high = w[w['is_liver']]['valuenum'].max() > liver if w['is_liver'].sum() > 0 else False
        if w_plt_low and w_liver_high:
            continue  # prevalent, 排除
        # 结局窗
        if len(o) == 0 or (o['marker'] == 'platelet').sum() < 1 or o['is_liver'].sum() < 1:
            continue
        o_plt_low = o[o['marker'] == 'platelet']['valuenum'].min() < plt
        o_liver_high = o[o['is_liver']]['valuenum'].max() > liver
        if o_plt_low and o_liver_high:
            cnt += 1
            if r['hellp_coded']:
                overlap += 1
    print(f"  Plt<{plt:>3d}, AST/ALT>{liver:>3d} | {len(eligible):>10d} {cnt:>8d} | {overlap} 例是 HELLP 编码")

# ── 4. 简化 endpoint vs 真正 HELLP 编码的重叠度 ──
print("\n【4】简化 endpoint 与 HELLP 编码的诊断性能")
hellp_coded_hadms = res[res['hellp_coded']]['hadm_id']
print(f"\n  HELLP 编码 hadm 总数: {len(hellp_coded_hadms)}")

# 在 HELLP 编码患者中，有多少在结局窗满足简化 endpoint？
hellp_with_lab = res[res['hadm_id'].isin(hellp_coded_hadms)]
hellp_with_both = hellp_with_lab[
    (hellp_with_lab['window_plt_n'] >= 1) &
    (hellp_with_lab['window_liver_n'] >= 1)
]
print(f"  HELLP 编码患者中有 plt+liver 检测的: {len(hellp_with_both)}")

sens_list = []
for plt, liver in [(100, 70), (120, 50), (150, 40), (100, 40)]:
    sens = 0
    for _, r in hellp_with_both.iterrows():
        hadm = r['hadm_id']
        admtime = pe_adm[pe_adm['hadm_id'] == hadm]['admittime'].iloc[0]
        e48 = admtime + pd.Timedelta(hours=48)
        e7d = admtime + pd.Timedelta(days=7)
        hadm_lab = lab_pe[lab_pe['hadm_id'] == hadm].copy()
        hadm_lab['is_liver'] = hadm_lab['marker'].isin(['ast', 'alt'])
        # 找整个住院期间的最差值（看能不能抓到 HELLP 编码中有的恶化）
        all_post = hadm_lab[hadm_lab['charttime'] >= admtime]
        if len(all_post) == 0:
            continue
        plt_min = all_post[all_post['marker'] == 'platelet']['valuenum'].min()
        liver_max = all_post[all_post['is_liver']]['valuenum'].max()
        if pd.notna(plt_min) and pd.notna(liver_max) and plt_min < plt and liver_max > liver:
            sens += 1
    print(f"    Plt<{plt}, AST/ALT>{liver}: 在 {len(hellp_with_both)} 例 HELLP 编码中的检出率 = {sens}/{len(hellp_with_both)} = {100*sens/max(len(hellp_with_both),1):.0f}%")

# ── 5. 每例的 platelet / liver 检测次数分布（够不够做轨迹） ──
print("\n【5】有效队列中 platelet 和 AST/ALT 各被检测了多少次（够不够画轨迹）")
eligible_lab_counts = hadm_by_marker.loc[eligible['hadm_id']]
print(f"\n  Platelet 检测次数分布（eligible 队列）:")
for n in range(1, 6):
    cnt = ((eligible_lab_counts['platelet'] >= n) & (eligible_lab_counts['platelet'] < n+1)).sum()
    print(f"    {n} 次: {cnt}")
print(f"    ≥5 次: {(eligible_lab_counts['platelet'] >= 5).sum()}")

eligible_lab_counts['liver_max'] = eligible_lab_counts.get('ast', 0) + eligible_lab_counts.get('alt', 0)
print(f"\n  AST+ALT 合计检测次数分布:")
for n in range(1, 6):
    cnt = ((eligible_lab_counts['liver_max'] >= n) & (eligible_lab_counts['liver_max'] < n+1)).sum()
    print(f"    {n} 次: {cnt}")
print(f"    ≥5 次: {(eligible_lab_counts['liver_max'] >= 5).sum()}")

# ── 6. 最关键的：LCMM 做联合轨迹需要的每 hadm 重复检测次数 ──
print("\n【6】够不够做 LCMM 联合轨迹？（关键结论）")
# LCMM 至少要求每 hadm 有 ≥2 次的纵向数据点
both_at_least_2 = (
    (eligible_lab_counts['platelet'] >= 2) &
    (eligible_lab_counts['liver_max'] >= 2)
)
print(f"  eligible 队列中 Platelet ≥2 次 且 AST/ALT 合计 ≥2 次: {both_at_least_2.sum()}")
print(f"  → 这就是**真正能做联合轨迹建模**的有效样本量！")

print("\n" + "=" * 70)
print("验证完成。数字在上面。")
print("=" * 70)
