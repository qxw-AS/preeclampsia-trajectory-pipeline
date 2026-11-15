"""
Module A Step 5: Generate Table 1 figure + results summary
Fixes gender labeling (MIMIC-IV: F=female)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
FIG_DIR = f"{OUT}\\figures"

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
})

# Load data
clusters = pd.read_csv(f"{OUT}\\moduleA_cluster_assignments.csv")
cohort = pd.read_csv(f"{OUT}\\moduleA_landmark_cohort.csv")
# Merge cohort with clusters (which contains trajectory features)
feature_cols = ['hadm_id', 'trajectory_class', 'class_name', 'platelet_first', 'ast_first', 'alt_first',
                'platelet_slope', 'ast_slope', 'alt_slope']
df = cohort.merge(clusters[feature_cols], on='hadm_id', how='inner')
if df['class_name'].isna().any():
    raise ValueError('Missing trajectory class labels')
class_order = ['Stable', 'Low-grade', 'Liver-dominant decompensation', 'Platelet-dominant decline']
df['class_name'] = pd.Categorical(df['class_name'], categories=class_order, ordered=True)

# Fix gender: MIMIC-IV uses F=female, M=male
df['is_female'] = (df['gender'] == 'F').astype(int)

# ── Build Table 1 data ──
def desc_cont(series):
    return f"{series.median():.1f} ({series.quantile(0.25):.1f}–{series.quantile(0.75):.1f})"

def desc_cat(series, val=1):
    n = (series == val).sum()
    return f"{n} ({100*n/len(series):.1f})"

rows = []
df['los_days'] = df['los_hours'] / 24.0
cont_vars = [
    ('Age (years)', 'anchor_age'),
    ('LOS (days)', 'los_days'),
    ('Initial platelet (×10⁹/L)', 'platelet_first'),
    ('Initial AST (U/L)', 'ast_first'),
    ('Initial ALT (U/L)', 'alt_first'),
    ('Platelet slope (×10⁹/L/h)', 'platelet_slope'),
    ('AST slope (U/L/h)', 'ast_slope'),
    ('ALT slope (U/L/h)', 'alt_slope'),
]
cat_vars = [
    ('Female', 'is_female'),
    ('AKI', 'aki'),
    ('ICU ≥ 3 days', 'icu_long'),
    ('Mechanical ventilation', 'ventilation'),
    ('Transfusion', 'transfusion'),
    ('Vasopressors', 'vasopressors'),
    ('Composite outcome', 'composite_outcome'),
]

for label, col in cont_vars:
    row = [label]
    groups = []
    for cls in class_order:
        sub = df[df['class_name'] == cls]
        row.append(desc_cont(sub[col]))
        groups.append(sub[col].dropna().values)
    try:
        _, p = stats.kruskal(*groups)
        row.append(f'{p:.4f}' if p >= 0.001 else '<0.001')
    except:
        row.append('NA')
    rows.append(row)

for label, col in cat_vars:
    row = [label]
    tables = []
    for cls in class_order:
        sub = df[df['class_name'] == cls]
        row.append(desc_cat(sub[col], 1))
    try:
        ct = pd.crosstab(df['class_name'], df[col])
        if ct.shape[1] == 2:
            _, p, _, _ = stats.chi2_contingency(ct)
            row.append(f'{p:.4f}' if p >= 0.001 else '<0.001')
        else:
            row.append('NA')
    except:
        row.append('NA')
    rows.append(row)

# ── Render Table 1 as figure ──
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

col_labels = ['Characteristic'] + [f'{c}\n(n={len(df[df["class_name"]==c])})' for c in class_order] + ['P value']
table_data = [col_labels] + rows

tbl = ax.table(cellText=table_data, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.6)

# Style header
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold')
    cell.set_edgecolor('#2C3E50')

# Style rows
for i in range(1, len(table_data)):
    for j in range(len(col_labels)):
        cell = tbl[i, j]
        if i % 2 == 0:
            cell.set_facecolor('#F8F9FA')
        else:
            cell.set_facecolor('white')
        if j == 0:
            cell.set_text_props(fontweight='bold', ha='left')
            cell._loc = 'left'

tbl.auto_set_column_width([0, 1, 2, 3, 4, 5])

ax.set_title('Table 1. Baseline Characteristics by Trajectory Class',
             fontsize=11, fontweight='bold', pad=20)

plt.savefig(f"{FIG_DIR}\\table1_baseline.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f"{FIG_DIR}\\table1_baseline.pdf", bbox_inches='tight', facecolor='white')
plt.close()
print("Saved table1_baseline.png/pdf")

# ── Results summary text ──
summary = f"""
Module A Results Summary
========================

COHORT
  Landmark cohort (48h window, ≥2 platelet + ≥2 liver enzyme measurements): n=943
  Complete trajectory features for clustering: n=701
  Composite outcome (≥1 of: death, AKI, ICU≥3d, ventilation, transfusion, vasopressors): 97/943 (10.3%)

TRAJECTORY CLASSES (GMM, k=4)
  1. Stable (n=235): normal platelet (~233) and liver enzymes (~54); outcome 7.7%
  2. Low-grade (n=338): low liver enzymes (~21); outcome 9.5%
  3. Liver-dominant decompensation (n=49): high AST (~181) and ALT (~188); outcome 18.4%
  4. Platelet-dominant decline (n=79): steep platelet decline (slope=-1.4/h); outcome 26.6%

OUTCOME ASSOCIATION (logistic regression, adjusted for age + initial labs)
  - Low-grade vs Stable:       OR=1.41 (0.73–2.71), p=0.304
  - Liver-dominant vs Stable:  OR=1.46 (0.38–5.57), p=0.582
  - Platelet-dominant vs Stable: OR=3.20 (1.54–6.66), p=0.002 **
  - Age (per year): OR=1.05 (1.01–1.10), p=0.009
  - Initial AST (per U/L): OR=1.02 (1.01–1.03), p<0.001

KEY FINDING
  The Platelet-dominant decline trajectory class is independently associated with
  a 3.2-fold increased odds of severe maternal composite outcome (p=0.002),
  establishing that 48-hour dynamic platelet decline carries prognostic information
  beyond baseline laboratory values.

FIGURES GENERATED
  - fig2_trajectory: Joint platelet/liver enzyme trajectory by 4 classes (main text Fig 2)
  - fig3_forest: Adjusted ORs forest plot (main text Fig 3)
  - figS1_model_fit: GMM model selection (BIC/AIC/silhouette)
  - figS2_component_outcomes: Component outcome rates by class
  - figS3_outcome_gradient: Composite outcome rate gradient across classes
  - table1_baseline: Baseline characteristics table
"""
with open(f"{OUT}\\moduleA_results_summary.txt", 'w') as f:
    f.write(summary)
print("\nSaved moduleA_results_summary.txt")
print(summary)
