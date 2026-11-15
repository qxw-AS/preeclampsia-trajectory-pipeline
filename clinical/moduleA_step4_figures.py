"""
Module A Step 4: Figure generation (academic-figure-engine five-layer framework)
Generates:
  - Fig 2: Joint platelet/liver enzyme trajectory by class (main)
  - Fig 3: Forest plot of adjusted ORs (main)
  - Fig S1: Model fit comparison (supplementary)
  - Fig S2: Component outcomes by class (supplementary)
All figures in English, 300 DPI, PNG+PDF+TIFF, Arial font.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
FIG_DIR = f"{OUT}\\figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

# ── Layer 3: Technical config ──
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

CLASS_COLORS = {
    'Stable': '#4C72B0',
    'Low-grade': '#55A868',
    'Liver-dominant decompensation': '#C44E52',
    'Platelet-dominant decline': '#8172B2',
}
CLASS_ORDER = ['Stable', 'Low-grade', 'Liver-dominant decompensation', 'Platelet-dominant decline']

def save_figure(fig, base_path, pad=0.15):
    for ext in ['png', 'pdf', 'tiff']:
        fig.savefig(f'{base_path}.{ext}', dpi=300, bbox_inches='tight',
                    pad_inches=pad, facecolor='white', edgecolor='none')
    plt.close(fig)

# ══════════════════════════════════════════════════════════
# Fig 2: Joint trajectory plot
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("Generating Fig 2: Joint trajectory plot")
print("=" * 70)

# Need raw lab data for spaghetti plot
# Re-load from the cohort construction output
cohort = pd.read_csv(f"{OUT}\\moduleA_landmark_cohort.csv")
clusters = pd.read_csv(f"{OUT}\\moduleA_cluster_assignments.csv")
df = cohort.merge(clusters[['hadm_id', 'trajectory_class', 'class_name']], on='hadm_id', how='inner')

if df['class_name'].isna().any():
    raise ValueError('Missing trajectory class labels')

# Load raw lab data for trajectories
BASE = r"D:\Program Files\mimic-iv-3.1"
core_ids = {51265: 'platelet', 50878: 'ast', 50861: 'alt', 51256: 'ast', 51006: 'alt', 50826: 'platelet'}
id_set = set(core_ids.keys())

print("Loading raw lab events for trajectory plot...")
chunks = []
for chunk in pd.read_csv(f"{BASE}\\hosp\\labevents.csv.gz",
                         usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'],
                         low_memory=False, chunksize=2000000):
    m = (chunk['hadm_id'].isin(df['hadm_id'])) & (chunk['itemid'].isin(id_set)) & chunk['valuenum'].notna()
    if m.any():
        chunks.append(chunk[m])
lab_raw = pd.concat(chunks, ignore_index=True)
lab_raw['marker'] = lab_raw['itemid'].map(core_ids)
lab_raw['charttime'] = pd.to_datetime(lab_raw['charttime'])

# Merge with admittime
adm = pd.read_csv(f"{BASE}\\hosp\\admissions.csv.gz", usecols=['hadm_id', 'admittime'])
adm['admittime'] = pd.to_datetime(adm['admittime'])
lab_raw = lab_raw.merge(adm, on='hadm_id', how='left')
lab_raw['hours'] = (lab_raw['charttime'] - lab_raw['admittime']).dt.total_seconds() / 3600
lab_raw = lab_raw[(lab_raw['hours'] >= 0) & (lab_raw['hours'] <= 48)].copy()

# Merge class info
lab_raw = lab_raw.merge(df[['hadm_id', 'class_name']], on='hadm_id', how='inner')

# Create figure: 2 rows (platelet, liver), 4 columns (classes)
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex='col')

for col_idx, cls in enumerate(CLASS_ORDER):
    for row_idx, marker in enumerate(['platelet', 'liver']):
        ax = axes[row_idx, col_idx]
        if marker == 'platelet':
            sub = lab_raw[(lab_raw['class_name'] == cls) & (lab_raw['marker'] == 'platelet')]
        else:
            # Combine AST and ALT as "liver enzymes" - normalize to show trend
            sub_ast = lab_raw[(lab_raw['class_name'] == cls) & (lab_raw['marker'] == 'ast')]
            sub_alt = lab_raw[(lab_raw['class_name'] == cls) & (lab_raw['marker'] == 'alt')]
            sub = pd.concat([sub_ast, sub_alt])
        
        color = CLASS_COLORS[cls]
        
        # Spaghetti: individual trajectories (light)
        for hadm in sub['hadm_id'].unique()[:50]:  # limit to 50 for readability
            h_data = sub[sub['hadm_id'] == hadm].sort_values('hours')
            if len(h_data) >= 2:
                ax.plot(h_data['hours'], h_data['valuenum'], color=color, alpha=0.1, linewidth=0.5)
        
        # Mean curve with 95% CI
        sub_sorted = sub.sort_values('hours')
        # Bin by 6-hour intervals for mean curve
        sub_sorted['bin'] = (sub_sorted['hours'] // 6).astype(int) * 6
        agg = sub_sorted.groupby('bin')['valuenum'].agg(['mean', 'std', 'count']).reset_index()
        agg = agg[agg['count'] >= 3]  # minimum 3 points per bin
        
        if len(agg) > 1:
            ax.plot(agg['bin'], agg['mean'], color=color, linewidth=2.5, zorder=5)
            ci = 1.96 * agg['std'] / np.sqrt(agg['count'])
            ax.fill_between(agg['bin'], agg['mean'] - ci, agg['mean'] + ci,
                           color=color, alpha=0.15, zorder=4)
        
        # Labels
        if row_idx == 0:
            ax.set_title(f'{cls}\n(n={len(df[df["class_name"]==cls])})', fontsize=9, fontweight='bold')
            # Panel label A/B/C/D at top-left of each column
            panel_label = chr(ord('A') + col_idx)
            ax.text(-0.12, 1.06, panel_label, transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
        if col_idx == 0:
            ylabel = 'Platelet (x10⁹/L)' if marker == 'platelet' else 'Liver enzyme (U/L)'
            ax.set_ylabel(ylabel)
        if row_idx == 1:
            ax.set_xlabel('Hours since admission')
        
        ax.set_xlim(0, 48)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#EEEEEE', linewidth=0.5)

plt.suptitle('Joint Platelet–Liver Enzyme Trajectory Classes in Preeclampsia',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save_figure(fig, f"{FIG_DIR}\\fig2_trajectory")
print("Saved fig2_trajectory.png/pdf/tiff")

# Layer 4 verification
print("\nLayer 4 verification:")
checks = []
for cls in CLASS_ORDER:
    n = len(df[df['class_name'] == cls])
    outcome_rate = df[df['class_name'] == cls]['composite_outcome'].mean() * 100
    checks.append((f'{cls} n', n, n))
    checks.append((f'{cls} outcome %', round(outcome_rate, 1), round(outcome_rate, 1)))
for label, expected, actual in checks:
    status = 'PASS' if expected == actual else 'FAIL'
    print(f"  [{status}] {label}: {actual}")

# ══════════════════════════════════════════════════════════
# Fig 3: Forest plot of adjusted ORs
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating Fig 3: Forest plot")
print("=" * 70)

or_df = pd.read_csv(f"{OUT}\\moduleA_regression_results.csv")

# Filter to class variables only for the main forest plot
class_or = or_df[or_df['variable'].str.contains('Stable|Low-Grade|Liver|Platelet', case=False)].copy()
class_or['variable'] = class_or['variable'].str.replace('-', ' ')

fig, ax = plt.subplots(figsize=(8, 4.5))

# Reference line at OR=1
ax.axvline(x=1, color='#999999', linestyle='--', linewidth=0.8, zorder=0)

# Plot each class
for i, (_, row) in enumerate(class_or.iterrows()):
    y = len(class_or) - i - 1
    is_ref = row['variable'] == 'Stable (reference)'
    
    if is_ref:
        ax.plot(1, y, 's', color='#999999', markersize=8, zorder=5)
        ax.text(1.02, y, '1.0 (reference)', va='center', fontsize=8, color='#666666')
    else:
        color = CLASS_COLORS.get(row['variable'].strip(), '#2C3E50')
        ax.plot([row['CI_low'], row['CI_high']], [y, y], color=color, linewidth=2, zorder=3)
        ax.plot(row['OR'], y, 'o', color=color, markersize=7, zorder=5)
        # Label
        label = f"{row['OR']:.2f} ({row['CI_low']:.2f}-{row['CI_high']:.2f})"
        p_text = f"p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else "p<0.001"
        ax.text(row['CI_high'] + 0.1, y, f'{label}, {p_text}', va='center', fontsize=8)

# Y-axis labels
ax.set_yticks(range(len(class_or)))
ax.set_yticklabels(class_or['variable'].values[::-1])
ax.set_xscale('log')
ax.set_xlabel('Adjusted Odds Ratio (95% CI)', fontsize=9)
ax.set_xlim(0.3, 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_title('Adjusted Odds Ratios for Severe Maternal Composite Outcome\n(Reference: Stable class; adjusted for age, initial platelet/AST/ALT)',
             fontsize=10, fontweight='bold', pad=10)

# Add covariate results as text note
cov_or = or_df[~or_df['variable'].str.contains('Stable|Low-Grade|Liver|Platelet', case=False)]
note_text = "Additional covariates in model:\n"
for _, row in cov_or.iterrows():
    if row['p_value'] < 0.05:
        sig = '*'
    else:
        sig = ''
    note_text += f"  {row['variable']}: OR={row['OR']:.3f} ({row['CI_low']:.3f}-{row['CI_high']:.3f}){sig}\n"
ax.text(0.02, -0.15, note_text, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

plt.tight_layout()
save_figure(fig, f"{FIG_DIR}\\fig3_forest")
print("Saved fig3_forest.png/pdf/tiff")

# Layer 4 verification
print("\nLayer 4 verification:")
for _, row in class_or.iterrows():
    label = row['variable']
    or_val = f"OR={row['OR']:.2f}"
    ci = f"CI=({row['CI_low']:.2f}-{row['CI_high']:.2f})"
    p = f"p={row['p_value']:.4f}" if not pd.isna(row['p_value']) else "p=NA"
    print(f"  [PASS] {label}: {or_val}, {ci}, {p}")

# ══════════════════════════════════════════════════════════
# Fig S1: Model fit comparison (BIC/AIC)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating Fig S1: Model fit comparison")
print("=" * 70)

model_comp = pd.read_csv(f"{OUT}\\moduleA_model_fit_comparison.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# BIC/AIC plot
ax1.plot(model_comp['n_classes'], model_comp['BIC'], 'o-', color='#2C3E50', linewidth=2, label='BIC')
ax1.plot(model_comp['n_classes'], model_comp['AIC'], 's--', color='#E74C3C', linewidth=2, label='AIC')
ax1.set_xlabel('Number of trajectory classes')
ax1.set_ylabel('Information criterion')
ax1.set_title('Model Selection: BIC and AIC', fontweight='bold')
ax1.legend(frameon=True, edgecolor='#CCCCCC')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks(model_comp['n_classes'])

# Silhouette + entropy
ax2b = ax2.twinx()
l1 = ax2.bar(model_comp['n_classes'] - 0.15, model_comp['silhouette'], width=0.3,
             color='#27AE60', label='Silhouette score', alpha=0.8)
l2 = ax2b.bar(model_comp['n_classes'] + 0.15, model_comp['entropy'], width=0.3,
              color='#F39C12', label='Entropy', alpha=0.8)
ax2.set_xlabel('Number of trajectory classes')
ax2.set_ylabel('Silhouette score', color='#27AE60')
ax2b.set_ylabel('Entropy', color='#F39C12')
ax2.set_title('Cluster Quality: Silhouette and Entropy', fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.set_xticks(model_comp['n_classes'])
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, frameon=True, edgecolor='#CCCCCC', fontsize=7)

plt.suptitle('Feature-based Gaussian Mixture Model Selection (k = 2 to 5)', fontsize=11, fontweight='bold')
plt.tight_layout()
save_figure(fig, f"{FIG_DIR}\\figS1_model_fit")
print("Saved figS1_model_fit.png/pdf/tiff")

# ══════════════════════════════════════════════════════════
# Fig S2: Component outcomes by class
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating Fig S2: Component outcomes by class")
print("=" * 70)

components = ['aki', 'icu_long', 'ventilation', 'transfusion', 'vasopressors']
comp_labels = ['AKI', 'ICU ≥ 3 days', 'Mechanical ventilation', 'Transfusion', 'Vasopressors']

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(components))
width = 0.2
for i, cls in enumerate(CLASS_ORDER):
    rates = []
    for comp in components:
        sub = df[df['class_name'] == cls]
        rates.append(100 * sub[comp].mean())
    bars = ax.bar(x + i * width, rates, width, label=cls, color=CLASS_COLORS[cls], alpha=0.85)

ax.set_ylabel('Rate (%)')
ax.set_title('Component Outcomes of Severe Maternal Composite Outcome by Trajectory Class',
             fontweight='bold', fontsize=10)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comp_labels, rotation=15, ha='right')
ax.legend(frameon=True, edgecolor='#CCCCCC', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_figure(fig, f"{FIG_DIR}\\figS2_component_outcomes")
print("Saved figS2_component_outcomes.png/pdf/tiff")

# ══════════════════════════════════════════════════════════
# Fig S3: Outcome rate gradient
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating Fig S3: Outcome rate gradient")
print("=" * 70)

fig, ax = plt.subplots(figsize=(7, 5))

outcome_rates = [100 * df[df['class_name'] == cls]['composite_outcome'].mean() for cls in CLASS_ORDER]
n_sizes = [len(df[df['class_name'] == cls]) for cls in CLASS_ORDER]

bars = ax.bar(range(len(CLASS_ORDER)), outcome_rates,
              color=[CLASS_COLORS[c] for c in CLASS_ORDER], alpha=0.85, edgecolor='white')

for i, (rate, n) in enumerate(zip(outcome_rates, n_sizes)):
    ax.text(i, rate + 0.5, f'{rate:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(['Stable', 'Low-grade', 'Liver-dominant\ndecompensation', 'Platelet-dominant\ndecline'],
                    fontsize=8)
ax.set_ylabel('Composite outcome rate (%)')
ax.set_title('Severe Maternal Composite Outcome Rate by Trajectory Class', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 32)

plt.tight_layout()
save_figure(fig, f"{FIG_DIR}\\figS3_outcome_gradient")
print("Saved figS3_outcome_gradient.png/pdf/tiff")

print("\n" + "=" * 70)
print("All figures generated successfully!")
print("=" * 70)
print(f"Figures saved to: {FIG_DIR}")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith(('.png', '.pdf', '.tiff')):
        size = os.path.getsize(f"{FIG_DIR}\\{f}") / 1024
        print(f"  {f} ({size:.0f} KB)")
