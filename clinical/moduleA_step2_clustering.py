"""
Module A Step 2: GMM trajectory clustering (2-5 classes, BIC selection)
Uses trajectory features extracted in Step 1.
Output: moduleA_cluster_assignments.csv, moduleA_model_fit_comparison.csv
"""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

OUT = r"C:\Users\1\Documents\Trae-CN\temp_20261113_01"

print("=" * 70)
print("Module A Step 2: GMM trajectory clustering")
print("=" * 70)

# Load features
feat = pd.read_csv(f"{OUT}\\moduleA_trajectory_features.csv")
print(f"Loaded: {feat.shape[0]} hadms x {feat.shape[1]} columns")

# ── Feature selection for clustering ──
# Use slope, intercept, first, mean for each of platelet, ast, alt
# These capture trajectory shape without redundancy
feature_cols = [
    'platelet_slope', 'platelet_intercept', 'platelet_first', 'platelet_mean',
    'ast_slope', 'ast_intercept', 'ast_first', 'ast_mean',
    'alt_slope', 'alt_intercept', 'alt_first', 'alt_mean',
]

# Drop rows with NaN in these features
feat_clean = feat.dropna(subset=feature_cols).copy()
print(f"After dropping NaN in trajectory features: {len(feat_clean)} hadms")

X = feat_clean[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Fit GMM for 2-5 classes ──
print("\nFitting GMM for 2-5 classes...")
results = []
models = {}

for n_classes in range(2, 6):
    gmm = GaussianMixture(n_components=n_classes,
                          covariance_type='full',
                          n_init=20,
                          max_iter=500,
                          random_state=42)
    gmm.fit(X_scaled)
    models[n_classes] = gmm
    
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    sil = silhouette_score(X_scaled, gmm.predict(X_scaled))
    
    # Class sizes
    labels = gmm.predict(X_scaled)
    class_sizes = pd.Series(labels).value_counts().sort_index()
    min_class_size = class_sizes.min()
    max_class_size = class_sizes.max()
    
    # Entropy (posterior probability quality)
    probs = gmm.predict_proba(X_scaled)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1).mean()
    
    results.append({
        'n_classes': n_classes,
        'BIC': bic,
        'AIC': aic,
        'silhouette': sil,
        'entropy': entropy,
        'min_class_size': min_class_size,
        'max_class_size': max_class_size,
        'converged': gmm.converged_,
    })
    
    print(f"  k={n_classes}: BIC={bic:.1f}, AIC={aic:.1f}, "
          f"silhouette={sil:.3f}, entropy={entropy:.3f}, "
          f"class sizes: {dict(class_sizes)}")
    
# Save model comparison
model_comp = pd.DataFrame(results)
model_comp['min_class_fraction'] = model_comp['min_class_size'] / len(feat_clean)
model_comp['eligible'] = ((model_comp['min_class_fraction'] >= 0.05) &
                          model_comp['converged'])

# Select by BIC only among converged solutions with every class >=5% of the cohort.
# This prevents full-covariance degenerate solutions with tiny outlier classes.
eligible = model_comp[model_comp['eligible']]
if eligible.empty:
    raise RuntimeError('No GMM solution met the 5% minimum-class-size criterion')
optimal_k = int(eligible.loc[eligible['BIC'].idxmin(), 'n_classes'])
model_comp['selected'] = model_comp['n_classes'].eq(optimal_k)
print(f"\nSelected k={optimal_k} (lowest BIC among non-degenerate solutions)")
print(model_comp[['n_classes', 'BIC', 'min_class_size', 'eligible', 'selected']].to_string(index=False))
model_comp.to_csv(f"{OUT}\\moduleA_model_fit_comparison.csv", index=False)
print(f"\nModel comparison saved: moduleA_model_fit_comparison.csv")

gmm_final = models[optimal_k]
labels = gmm_final.predict(X_scaled)
probs = gmm_final.predict_proba(X_scaled)
feat_clean['trajectory_class'] = labels
feat_clean['max_posterior_prob'] = probs.max(axis=1)

# ── Name classes by clinical phenotype ──
# Determine class characteristics
class_summary = feat_clean.groupby('trajectory_class').agg({
    'platelet_first': 'mean',
    'platelet_slope': 'mean',
    'platelet_mean': 'mean',
    'ast_first': 'mean',
    'ast_slope': 'mean',
    'ast_mean': 'mean',
    'alt_first': 'mean',
    'alt_slope': 'mean',
    'alt_mean': 'mean',
    'composite_outcome': 'mean',
}).round(2)

print(f"\nClass characteristics (k={optimal_k}):")
print(class_summary.to_string())

# Name the four phenotypes from laboratory measurements only; outcomes are not used.
if optimal_k == 4:
    remaining = set(class_summary.index)
    liver_cls = (class_summary['ast_first'] + class_summary['alt_first']).idxmax()
    remaining.remove(liver_cls)
    platelet_cls = class_summary.loc[list(remaining), 'platelet_slope'].idxmin()
    remaining.remove(platelet_cls)
    stable_cls = class_summary.loc[list(remaining), 'platelet_slope'].abs().idxmin()
    remaining.remove(stable_cls)
    low_grade_cls = remaining.pop()
    class_names = {
        stable_cls: 'Stable',
        low_grade_cls: 'Low-grade',
        liver_cls: 'Liver-dominant decompensation',
        platelet_cls: 'Platelet-dominant decline',
    }
else:
    class_names = {cls: f'Trajectory class {i + 1}'
                   for i, cls in enumerate(sorted(class_summary.index))}

print(f"\nClass name mapping: {class_names}")
feat_clean['class_name'] = feat_clean['trajectory_class'].map(class_names)

# ── Save assignments ──
feat_clean.to_csv(f"{OUT}\\moduleA_cluster_assignments.csv", index=False)
print(f"\nSaved cluster assignments: moduleA_cluster_assignments.csv (k={optimal_k})")
print(f"Class distribution:")
print(feat_clean['class_name'].value_counts().to_string())

print("\n" + "=" * 70)
print("Step 2 complete.")
print("=" * 70)
