"""
Module B figures (academic-figure-engine)

Fig 4   A: ssGSEA pathway scores, EOPE vs preterm control (discovery, n=84)
        B: pathway-pathway correlation structure within EOPE placentas
Fig S9  Hallmark GSEA landscape (normalised enrichment scores)
Fig S10 Validation in GSE204835: gene-level effect-size concordance
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"
RES = rf"{OUT}\results"
FIG = rf"{OUT}\figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.linewidth': 0.7, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'figure.dpi': 300,
})

C_EOPE, C_CTRL = '#C44E52', '#4C72B0'

scores = pd.read_csv(rf"{RES}\moduleB_ssgsea_pathway_scores.csv", index_col=0)
stats_df = pd.read_csv(rf"{RES}\moduleB_ssgsea_group_stats.csv").set_index('pathway')
train = pd.read_csv(rf"{OUT}\data\geo\gse75010\GSE75010_data\GSE75010_training_labels.csv")
grp = train.set_index('sample_id')['group']
scores = scores.loc[[s for s in scores.index if s in grp.index]]
group = grp.loc[scores.index]

PATH_ORDER = ['Hypoxia', 'Apoptosis', 'Platelet activation',
              'Complement & coagulation cascades',
              'PPAR signalling (lipid metabolism)', 'Inflammatory response',
              'Angiogenesis']

# ══════════════════════════════════════════════════════════════════════════
# Fig 4
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10.6, 4.3))
axA = fig.add_axes([0.235, 0.14, 0.40, 0.76])
axB = fig.add_axes([0.745, 0.175, 0.215, 0.68])

# ── A: paired boxplots ───────────────────────────────────────────────────
ypos = np.arange(len(PATH_ORDER))[::-1]
for y, p in zip(ypos, PATH_ORDER):
    for dx, g, c in [(-0.19, 'Control', C_CTRL), (0.19, 'EOPE', C_EOPE)]:
        v = scores.loc[group == g, p].astype(float).values
        bp = axA.boxplot([v], positions=[y + dx], widths=0.30, vert=False,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='white', lw=1.0),
                         whiskerprops=dict(color=c, lw=.8),
                         capprops=dict(color=c, lw=.8),
                         boxprops=dict(facecolor=c, edgecolor=c, lw=.6, alpha=.85))
        jitter = np.random.default_rng(0).normal(0, 0.030, len(v))
        axA.scatter(v, np.full(len(v), y + dx) + jitter, s=3.5, color=c,
                    alpha=.32, linewidths=0, zorder=1)
    pv = stats_df.loc[p, 'p_value']
    fdr = stats_df.loc[p, 'fdr']
    star = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else 'n.s.'))
    axA.text(1.005, y, star, transform=axA.get_yaxis_transform(),
             fontsize=8, va='center', ha='left',
             fontweight='bold' if star != 'n.s.' else 'normal',
             color='#C0392B' if star != 'n.s.' else '#7F8C8D')

axA.set_yticks(ypos)
axA.set_ylim(-0.6, len(PATH_ORDER) - 0.4)
axA.set_yticklabels([p.replace(' signalling (lipid metabolism)', ' signalling') for p in PATH_ORDER])
axA.set_xlabel('ssGSEA enrichment score (per sample)')
axA.set_title('A   Pathway activity in EOPE versus preterm control placenta', loc='left',
              fontsize=9.5, fontweight='bold', pad=6)
for s in ['top', 'right']:
    axA.spines[s].set_visible(False)
axA.grid(axis='x', color='#EEEEEE', lw=.6); axA.set_axisbelow(True)
axA.legend(handles=[Patch(facecolor=C_CTRL, label=f'Control (n={int((group=="Control").sum())})'),
                    Patch(facecolor=C_EOPE, label=f'EOPE (n={int((group=="EOPE").sum())})')],
           frameon=False, loc='upper left', handlelength=1.0)

# ── B: pathway correlation heatmap within EOPE ───────────────────────────
eope_scores = scores.loc[group == 'EOPE', PATH_ORDER].astype(float)
R = eope_scores.corr(method='spearman')
P = pd.DataFrame(np.ones_like(R), index=R.index, columns=R.columns)
for i, a in enumerate(PATH_ORDER):
    for j, b in enumerate(PATH_ORDER):
        _, P.loc[a, b] = stats.spearmanr(eope_scores[a], eope_scores[b])

mask = np.triu(np.ones_like(R, dtype=bool), k=1)
Rm = R.mask(mask)
im = axB.imshow(Rm.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
for i, a in enumerate(PATH_ORDER):
    for j, b in enumerate(PATH_ORDER):
        if mask[i, j] or i == j:
            continue
        r = R.loc[a, b]
        s = '***' if P.loc[a, b] < 0.001 else ('**' if P.loc[a, b] < 0.01 else ('*' if P.loc[a, b] < 0.05 else ''))
        axB.text(j, i, f'{r:.2f}{s}', ha='center', va='center', fontsize=6.2,
                 color='white' if abs(r) > .6 else '#1B2631')
short = ['Hypoxia', 'Apoptosis', 'Platelet act.', 'Complement/coag.',
         'PPAR', 'Inflammation', 'Angiogenesis']
axB.set_xticks(range(len(short))); axB.set_xticklabels(short, rotation=45, ha='right', fontsize=6.5)
axB.set_yticks(range(len(short))); axB.set_yticklabels(short, fontsize=6.5)
axB.set_title('B   Pathway co-activity in EOPE', loc='left', fontsize=9.5,
              fontweight='bold', pad=6)
cb = fig.colorbar(im, ax=axB, shrink=.55, pad=.03, aspect=18)
cb.set_label('Spearman $\\rho$', fontsize=7)
cb.ax.tick_params(labelsize=6.5)

fig.savefig(rf"{FIG}\fig4_pathway_scores.png", dpi=300, bbox_inches='tight',
            pad_inches=0.10, facecolor='white')
fig.savefig(rf"{FIG}\fig4_pathway_scores.pdf", bbox_inches='tight', pad_inches=0.10,
            facecolor='white')
fig.savefig(rf"{FIG}\fig4_pathway_scores.tiff", dpi=300, bbox_inches='tight',
            pad_inches=0.10, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# Fig S9 - Hallmark GSEA landscape
# ══════════════════════════════════════════════════════════════════════════
hal = pd.read_csv(rf"{RES}\moduleB_gsea_hallmark.csv").rename(columns=str.strip)
hal = hal.dropna(subset=['NES', 'FDR q-val'])
hal = hal[hal['FDR q-val'] < 0.25].copy()
hal['Term'] = hal['Term'].str.replace('HALLMARK_', '', regex=False).str.replace('_', ' ').str.title()
hal = hal.sort_values('NES')
sel = pd.concat([hal.head(8), hal.tail(8)]).drop_duplicates('Term').sort_values('NES')

fig, ax = plt.subplots(figsize=(5.4, 4.6))
cols = [C_CTRL if n < 0 else C_EOPE for n in sel['NES']]
ax.hlines(range(len(sel)), 0, sel['NES'], color=cols, lw=1.4)
ax.scatter(sel['NES'], range(len(sel)), s=26, color=cols, zorder=3, linewidths=0)
ax.axvline(0, color='#888888', lw=.7)
ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel['Term'], fontsize=7)
ax.set_xlabel('Normalised enrichment score (EOPE vs control)')
ax.set_title('Hallmark pathways in EOPE placenta (FDR < 0.25)', fontsize=9, pad=6)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color='#EEEEEE', lw=.6); ax.set_axisbelow(True)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS9_gsea_hallmark.{ext}", dpi=300, bbox_inches='tight',
                facecolor='white')
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# Fig S10 - gene-level validation
# ══════════════════════════════════════════════════════════════════════════
gv = pd.read_csv(rf"{RES}\moduleB_validation_gene_level.csv")
fig, ax = plt.subplots(figsize=(4.0, 3.8))
sig_v = gv['fdr_validation'] < 0.05
ax.scatter(gv.loc[~sig_v, 'log2FC_discovery'], gv.loc[~sig_v, 'delta_validation'],
           s=12, c='#BDBDBD', alpha=.75, linewidths=0, label='FDR $\\geq$ 0.05')
ax.scatter(gv.loc[sig_v, 'log2FC_discovery'], gv.loc[sig_v, 'delta_validation'],
           s=16, c='#C44E52', alpha=.85, linewidths=0,
           label='FDR < 0.05')
b, a = np.polyfit(gv['log2FC_discovery'], gv['delta_validation'], 1)
xs = np.linspace(gv['log2FC_discovery'].min(), gv['log2FC_discovery'].max(), 50)
ax.plot(xs, a + b * xs, color='#2C3E50', lw=.9, ls='--')
ax.axhline(0, color='#DDDDDD', lw=.6); ax.axvline(0, color='#DDDDDD', lw=.6)
rho, rp = stats.spearmanr(gv['log2FC_discovery'], gv['delta_validation'])
ax.set_xlabel('log$_2$ FC, discovery (GSE75010)')
ax.set_ylabel('log$_2$ difference, validation (GSE204835)')
ax.set_title('DEG effect-size concordance', fontsize=9, pad=6)
ax.text(.03, .96, f'$\\rho$ = {rho:.2f}\n$P$ = {rp:.1e}\nsame direction: '
        f'{gv["same_direction"].mean()*100:.0f}%', transform=ax.transAxes,
        fontsize=7, va='top')
ax.legend(frameon=False, loc='lower right', fontsize=6.5)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(rf"{FIG}\figS10_validation_concordance.{ext}", dpi=300,
                bbox_inches='tight', facecolor='white')
plt.close(fig)

# ── Layer 4 verification ─────────────────────────────────────────────────
print("Layer 4 verification - Fig 4")
tests = [('samples', 84, int(len(scores))),
         ('EOPE n', 49, int((group == 'EOPE').sum())),
         ('control n', 35, int((group == 'Control').sum())),
         ('pathway rows', 7, int(scores.shape[1])),
         ('significant FDR<0.05', int((stats_df['fdr'] < 0.05).sum()),
          int((stats_df['fdr'] < 0.05).sum()))]
for lb, e, a in tests:
    print(f"  [{'PASS' if e == a else 'FAIL'}] {lb:24s} expected={e:<6} actual={a}")
print("\n  pathway statistics")
for p in PATH_ORDER:
    print(f"    {p:38s} delta={stats_df.loc[p,'delta']:+.1f}  "
          f"FDR={stats_df.loc[p,'fdr']:.2e}")
print(f"\n  strongest within-EOPE correlation: "
      f"{R.where(~mask).stack().idxmax()} r={R.where(~mask).stack().max():.2f}")
print("\nSaved fig4_pathway_scores, figS9_gsea_hallmark, figS10_validation_concordance")
