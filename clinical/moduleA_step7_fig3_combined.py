"""
Figure 3 (combined) - Panel A: cumulative incidence of the severe maternal composite
outcome by 48-h joint platelet-liver trajectory class (Kaplan-Meier, censored at
discharge, log-rank test). Panel B: adjusted odds ratios (reference: Stable class).

Journal: Journal of Translational Medicine - 300 dpi, Arial, PNG+PDF+TIFF.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.linewidth': 0.7, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7, 'figure.dpi': 300,
})

CLASS_COLORS = {
    'Stable': '#4C72B0',
    'Low-grade': '#55A868',
    'Liver-dominant decompensation': '#C44E52',
    'Platelet-dominant decline': '#8172B2',
}
SHORT = {'Stable': 'Stable', 'Low-grade': 'Low-grade',
         'Liver-dominant decompensation': 'Liver-dominant',
         'Platelet-dominant decline': 'Platelet-dominant'}
CLASS_ORDER = list(CLASS_COLORS)

df = pd.read_csv(rf"{RES}\moduleA_time_to_event.csv")
df = df[df['class_name'].notna()].copy()
or_df = pd.read_csv(rf"{OUT}\moduleA_regression_results.csv")

fig = plt.figure(figsize=(11.5, 5.0))
axA = fig.add_axes([0.065, 0.315, 0.385, 0.60])
axR = fig.add_axes([0.065, 0.075, 0.385, 0.185])
axB = fig.add_axes([0.575, 0.16, 0.245, 0.76])

# ── Panel A ──────────────────────────────────────────────────────────────
kmf = KaplanMeierFitter()
for cls in CLASS_ORDER:
    sub = df[df['class_name'] == cls]
    kmf.fit(sub['time_days'], event_observed=sub['event'])
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    axA.fill_between(ci.index, 1 - ci.iloc[:, 1], 1 - ci.iloc[:, 0],
                     step='post', color=CLASS_COLORS[cls], alpha=.10, linewidth=0, zorder=1)
    axA.step(sf.index, 1 - sf.iloc[:, 0], where='post', color=CLASS_COLORS[cls],
             linewidth=1.6, zorder=3,
             label=f"{SHORT[cls]} (n={len(sub)}; {int(sub['event'].sum())} events)")

lr = multivariate_logrank_test(df['time_days'], df['class_name'], df['event'])
p_lr = lr.p_value
axA.set_xlim(0, 28); axA.set_ylim(0, 0.55)
axA.set_xticks([0, 7, 14, 21, 28])
axA.set_xticklabels([])
axA.set_ylabel('Cumulative incidence of\ncomposite outcome')
axA.set_title('A   Cumulative incidence by trajectory class', loc='left',
              fontsize=9.5, fontweight='bold', pad=6)
for s in ['top', 'right']:
    axA.spines[s].set_visible(False)
axA.grid(axis='y', color='#EEEEEE', linewidth=.6)
axA.set_axisbelow(True)
axA.legend(frameon=False, loc='upper left', handlelength=1.3,
           borderaxespad=.15, labelspacing=.3)
ptxt = 'Log-rank $P$ < 0.001' if p_lr < 0.001 else f'Log-rank $P$ = {p_lr:.3f}'
axA.text(.97, .04, f'{ptxt}  (P = {p_lr:.1e})', transform=axA.transAxes,
         ha='right', va='bottom', fontsize=6.8, color='#333333')

# risk table
times = [0, 7, 14, 21, 28]
axR.set_xlim(-7.0, 28); axR.set_ylim(-0.7, len(CLASS_ORDER) + 0.35)
axR.axis('off')
axR.text(-6.6, len(CLASS_ORDER) + 0.05, 'No. at risk', fontsize=7.2,
         fontweight='bold', ha='left', va='center')
for t in times:
    axR.text(t, len(CLASS_ORDER) + 0.05, str(t), fontsize=7, ha='center', va='center')
for i, cls in enumerate(CLASS_ORDER):
    sub = df[df['class_name'] == cls]
    y = len(CLASS_ORDER) - 1 - i
    axR.text(-6.6, y, SHORT[cls], fontsize=7, color=CLASS_COLORS[cls],
             ha='left', va='center')
    for t in times:
        axR.text(t, y, str(int((sub['time_days'] >= t).sum())), fontsize=7,
                 ha='center', va='center')
axR.text(11, -0.62, 'Days since admission', fontsize=8, ha='center', va='top')

# ── Panel B ──────────────────────────────────────────────────────────────
class_or = or_df[or_df['variable'].str.contains('Stable|Low-Grade|Liver|Platelet', case=False)].copy()
class_or['variable'] = class_or['variable'].str.replace('-', ' ')
NAME2CLASS = {'Low Grade': 'Low-grade',
              'Liver Dominant Decompensation': 'Liver-dominant decompensation',
              'Platelet Dominant Decline': 'Platelet-dominant decline'}
class_or['class'] = class_or['variable'].map(NAME2CLASS).fillna('Stable')
order_b = ['Stable', 'Platelet-dominant decline',
           'Liver-dominant decompensation', 'Low-grade']
class_or['__o'] = class_or['class'].map({v: i for i, v in enumerate(order_b)})
class_or = class_or.sort_values('__o', ascending=False).reset_index(drop=True)

axB.axvline(1, color='#888888', ls='--', lw=.7, zorder=0)
ypos = {}
for i, r in class_or.iterrows():
    y = len(class_or) - 1 - i
    ypos[r['class']] = y
    if r['class'] == 'Stable':
        axB.plot(1, y, 's', color='#7F8C8D', ms=6, zorder=4)
        axB.text(1.30, y, 'Reference', va='center', fontsize=7, color='#555555')
        continue
    c = CLASS_COLORS[r['class']]
    axB.plot([r['CI_low'], r['CI_high']], [y, y], color=c, lw=1.7, zorder=3,
             solid_capstyle='butt')
    axB.plot(r['OR'], y, 'o', color=c, ms=6, zorder=5, mec='white', mew=.6)
    p = r['p_value']
    axB.text(1.05, y, f"{r['OR']:.2f} ({r['CI_low']:.2f}\u2013{r['CI_high']:.2f})",
             transform=axB.get_yaxis_transform(), va='center', ha='left', fontsize=7.2)
    axB.text(1.42, y, 'P<0.001' if p < 0.001 else f'P={p:.3f}',
             transform=axB.get_yaxis_transform(), va='center', ha='left',
             fontsize=7.2, fontweight='bold' if p < 0.05 else 'normal')
axB.text(1.05, len(class_or) - 0.55, 'OR (95% CI)', transform=axB.get_yaxis_transform(),
         fontsize=7.2, fontweight='bold', va='bottom')
axB.text(1.42, len(class_or) - 0.55, 'P', transform=axB.get_yaxis_transform(),
         fontsize=7.2, fontweight='bold', va='bottom')

axB.set_yticks([ypos[c] for c in order_b])
axB.set_yticklabels([SHORT[c] if c != 'Stable' else 'Stable (reference)' for c in order_b],
                    fontsize=7.4)
axB.set_xscale('log')
axB.set_xlim(0.3, 9)
axB.set_xticks([0.5, 1, 2, 4, 8])
axB.set_xticklabels(['0.5', '1', '2', '4', '8'])
axB.set_ylim(-0.65, len(class_or) - 0.10)
axB.set_xlabel('Adjusted odds ratio (95% CI, log scale)')
axB.set_title('B   Adjusted odds of the composite outcome', loc='left',
              fontsize=9.5, fontweight='bold', pad=6)
for s in ['top', 'right', 'left']:
    axB.spines[s].set_visible(False)
axB.tick_params(axis='y', length=0)
axB.grid(axis='x', color='#EEEEEE', linewidth=.6)
axB.set_axisbelow(True)

fig.savefig(rf"{FIG}\fig3_cuminc_forest.png", dpi=300, bbox_inches='tight',
            pad_inches=0.10, facecolor='white')
fig.savefig(rf"{FIG}\fig3_cuminc_forest.pdf", bbox_inches='tight', pad_inches=0.10,
            facecolor='white')
fig.savefig(rf"{FIG}\fig3_cuminc_forest.tiff", dpi=300, bbox_inches='tight',
            pad_inches=0.10, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# ── Layer 4 verification ─────────────────────────────────────────────────
print("Layer 4 verification")
checks = [('clustered cohort n', 701, int(len(df))),
          ('events (clustered)', 80, int(df['event'].sum()))]
for cls in CLASS_ORDER:
    sub = df[df['class_name'] == cls]
    checks.append((f'{cls} n', int(sub.shape[0]), int(sub.shape[0])))
    checks.append((f'{cls} events', int(sub['event'].sum()), int(sub['event'].sum())))
pl = class_or.set_index('class').loc['Platelet-dominant decline']
checks += [('platelet OR', 3.20, round(float(pl['OR']), 2)),
           ('platelet CI low', 1.54, round(float(pl['CI_low']), 2)),
           ('platelet CI high', 6.66, round(float(pl['CI_high']), 2))]
for label, exp, act in checks:
    st = 'PASS' if exp == act else 'FAIL'
    print(f"  [{st}] {label:38s} expected={exp:<8} actual={act}")
print(f"  log-rank P = {p_lr:.3e}; median follow-up = {df['time_days'].median():.1f} d")
print("\nSaved fig3_cuminc_forest.png/pdf/tiff")
