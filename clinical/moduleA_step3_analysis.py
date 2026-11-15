"""
Module A Step 3: Statistical analysis
- Logistic regression for composite outcome (since no time-to-event in 48h-7d window)
- Table 1: baseline characteristics by trajectory class
- Forest plot of adjusted ORs
Output: moduleA_regression_results.csv, moduleA_table1.csv
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"

print("=" * 70)
print("Module A Step 3: Statistical analysis")
print("=" * 70)

# Load data
cohort = pd.read_csv(f"{OUT}\\moduleA_landmark_cohort.csv")
clusters = pd.read_csv(f"{OUT}\\moduleA_cluster_assignments.csv")

# Merge
df = cohort.merge(clusters[['hadm_id', 'trajectory_class', 'class_name', 
                            'platelet_first', 'ast_first', 'alt_first',
                            'platelet_slope', 'ast_slope', 'alt_slope']], 
                   on='hadm_id', how='inner')
print(f"Analysis set: n={len(df)}")

# Use the measurement-based labels assigned during clustering.
if df['class_name'].isna().any():
    raise ValueError('Missing trajectory class labels')

# Reorder for display (lowest to highest risk)
class_order = ['Stable', 'Low-grade', 'Liver-dominant decompensation', 'Platelet-dominant decline']
df['class_name'] = pd.Categorical(df['class_name'], categories=class_order, ordered=True)

print(f"\nClass distribution:")
print(df['class_name'].value_counts().to_string())
print(f"\nOutcome rate by class:")
print(df.groupby('class_name')['composite_outcome'].agg(['count', 'sum', 'mean']).round(3).to_string())

# ── Table 1: Baseline characteristics by class ──
print("\nGenerating Table 1...")

def desc_continuous(series):
    return f"{series.median():.1f} ({series.quantile(0.25):.1f}-{series.quantile(0.75):.1f})"

def desc_categorical(series, value=1):
    n = (series == value).sum()
    pct = 100 * n / len(series)
    return f"{n} ({pct:.1f}%)"

# Variables for Table 1
df['los_days'] = df['los_hours'] / 24.0
cont_vars = {
    'Age': 'anchor_age',
    'LOS (days)': 'los_days',
    'Initial platelet (x10^9/L)': 'platelet_first',
    'Initial AST (U/L)': 'ast_first',
    'Initial ALT (U/L)': 'alt_first',
    'Platelet slope (x10^9/L/h)': 'platelet_slope',
    'AST slope (U/L/h)': 'ast_slope',
    'ALT slope (U/L/h)': 'alt_slope',
}
cat_vars = {
    'Female': 'gender',  # MIMIC-IV: F=female
    'AKI': 'aki',
    'ICU >= 3 days': 'icu_long',
    'Mechanical ventilation': 'ventilation',
    'Transfusion': 'transfusion',
    'Vasopressors': 'vasopressors',
    'Composite outcome': 'composite_outcome',
}

table1_rows = []
for var_label, var_col in cont_vars.items():
    row = {'Characteristic': var_label}
    groups = []
    for cls in class_order:
        sub = df[df['class_name'] == cls]
        row[f'{cls} (n={len(sub)})'] = desc_continuous(sub[var_col])
        groups.append(sub[var_col].values)
    # Kruskal-Wallis test
    valid_groups = [g[~np.isnan(g)] for g in groups]
    if all(len(g) > 0 for g in valid_groups):
        stat, p = stats.kruskal(*valid_groups)
        row['P value'] = f'{p:.4f}' if p >= 0.001 else '<0.001'
    else:
        row['P value'] = 'NA'
    table1_rows.append(row)

for var_label, var_col in cat_vars.items():
    row = {'Characteristic': var_label}
    tables = []
    for cls in class_order:
        sub = df[df['class_name'] == cls]
        if var_col == 'gender':
            row[f'{cls} (n={len(sub)})'] = desc_categorical(sub[var_col], 'F')
        else:
            row[f'{cls} (n={len(sub)})'] = desc_categorical(sub[var_col], 1)
        tables.append(sub[var_col].value_counts())
    # Chi-square test
    try:
        contingency = pd.crosstab(df['class_name'], df[var_col])
        if contingency.shape == (4, 2):
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            row['P value'] = f'{p:.4f}' if p >= 0.001 else '<0.001'
        else:
            row['P value'] = 'NA'
    except:
        row['P value'] = 'NA'
    table1_rows.append(row)

table1 = pd.DataFrame(table1_rows)
table1.to_csv(f"{OUT}\\moduleA_table1.csv", index=False)
print(f"Table 1 saved: moduleA_table1.csv")

# ── Logistic regression for composite outcome ──
print("\nRunning logistic regression...")

# Create numeric class variable (0=Stable reference, 1=Low-grade, 2=Liver-dominant, 3=Platelet-dominant)
class_to_num = {name: i for i, name in enumerate(class_order)}
df['class_num'] = df['class_name'].map(class_to_num)

# Covariates
covariates = ['anchor_age', 'gender']  # gender is all female, so only age
# Also use initial lab values as confounders
covariates += ['platelet_first', 'ast_first', 'alt_first']

# Model 1: unadjusted
print("\nModel 1: Unadjusted (class only)")
X = pd.get_dummies(df['class_name'], drop_first=True, prefix='class')
X = sm.add_constant(X)
y = df['composite_outcome']
model1 = sm.Logit(y, X.astype(float)).fit(disp=0)
print(model1.summary().tables[1])

# Model 2: adjusted for age + initial labs
print("\nModel 2: Adjusted (class + age + initial labs)")
X2 = pd.get_dummies(df['class_name'], drop_first=True, prefix='class')
X2['age'] = df['anchor_age']
X2['plt0'] = df['platelet_first']
X2['ast0'] = df['ast_first']
X2['alt0'] = df['alt_first']
X2 = sm.add_constant(X2.astype(float))
model2 = sm.Logit(y, X2).fit(disp=0)
print(model2.summary().tables[1])

# Extract ORs and CIs
def extract_or(model, class_prefix='class_'):
    results = []
    for col in model.params.index:
        if col == 'const':
            continue
        coef = model.params[col]
        se = model.bse[col]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)
        p = model.pvalues[col]
        name = col.replace(class_prefix, '').replace('_', ' ').title()
        results.append({
            'variable': name,
            'OR': or_val,
            'CI_low': ci_low,
            'CI_high': ci_high,
            'p_value': p,
        })
    return results

or_unadj = extract_or(model1)
or_adj = extract_or(model2)

# Add reference class
for res_list in [or_unadj, or_adj]:
    res_list.insert(0, {
        'variable': 'Stable (reference)',
        'OR': 1.0, 'CI_low': 1.0, 'CI_high': 1.0, 'p_value': np.nan,
    })

or_df_unadj = pd.DataFrame(or_unadj)
or_df_adj = pd.DataFrame(or_adj)

print("\nUnadjusted ORs:")
print(or_df_unadj.to_string(index=False))
print("\nAdjusted ORs:")
print(or_df_adj.to_string(index=False))

# Save
or_df_adj.to_csv(f"{OUT}\\moduleA_regression_results.csv", index=False)
print(f"\nRegression results saved: moduleA_regression_results.csv")

# ── Component outcomes by class ──
print("\nComponent outcomes by class:")
for cls in class_order:
    sub = df[df['class_name'] == cls]
    print(f"\n  {cls} (n={len(sub)}):")
    for comp in ['aki', 'icu_long', 'ventilation', 'transfusion', 'vasopressors', 'composite_outcome']:
        n = sub[comp].sum()
        pct = 100 * n / len(sub)
        print(f"    {comp}: {n} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("Step 3 complete.")
print("=" * 70)
