
# Linear Discriminant Analysis — From Scratch

# In this project, I used python to walk through the step-by-step of LDA without the power of sklearn and only using numpy and pandas for the math and matplotlib for plotting.
# LDA is implemented manually hence, our matrices and vectors are easily visible and explainable.

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Styles
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (7, 5),
})

COLORS = {"setosa": "#3b82f6", "versicolor": "#f97316"}   # blue / orange

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 1 — LOAD & PLOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CLASSES: 0 (setosa) and 1 (versicolor)
# FEATURES: petal_length (feature 2) and petal_width (feature 3).

print("~" * 60)
print("STEP 1 — Load & Plot")
print("~" * 60)

iris = load_iris()
mask = iris.target < 2                          # keep setosa (0) & versicolor (1)
X = iris.data[mask][:, 2:4]                     # columns 2,3 = petal_length, petal_width
y = iris.target[mask]

feature_names = ["petal_length", "petal_width"]
class_names   = {0: "setosa", 1: "versicolor"}

df = pd.DataFrame(X, columns=feature_names)
df["class"] = [class_names[label] for label in y]

print(f"Samples: {len(df)}  |  Features: {feature_names}")
print(df.groupby("class").size())
print()

fig, ax = plt.subplots()
for cls, color in COLORS.items():
    sub = df[df["class"] == cls]
    ax.scatter(sub["petal_length"], sub["petal_width"],
               c=color, label=cls, edgecolors="k", linewidths=0.4, s=50, alpha=0.85)
ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.set_title("Step 1 — Iris: Setosa vs Versicolor")
ax.legend()
plt.tight_layout()
plt.savefig("step1_scatter.png")
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````
# STEP 2 — MARGINAL (NAIVE) PROJECTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````````````````
# If we only looked at one feature at a time, how well could we separate the classes?  
# Histograms on each axis show the overlap.

print("~" * 60)
print("STEP 2 — Marginal (naive) projections")
print("~" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, feat in enumerate(feature_names):
    ax = axes[i]
    for cls, color in COLORS.items():
        vals = df.loc[df["class"] == cls, feat]
        ax.hist(vals, bins=15, alpha=0.55, color=color, label=cls, edgecolor="k", linewidth=0.4)
        # rug plot (small ticks at the bottom)
        ax.plot(vals, np.full_like(vals, -0.3), "|", color=color, markersize=8, alpha=0.7)
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.set_title(f"Step 2 — Projection onto {feat}")
    ax.legend()

plt.tight_layout()
plt.savefig("step2_marginals.png")
plt.show()

# Overlap analysis
setosa_pl = df.loc[df["class"] == "setosa", "petal_length"]
versic_pl = df.loc[df["class"] == "versicolor", "petal_length"]
setosa_pw = df.loc[df["class"] == "setosa", "petal_width"]
versic_pw = df.loc[df["class"] == "versicolor", "petal_width"]

overlap_pl = max(0, min(setosa_pl.max(), versic_pl.max()) - max(setosa_pl.min(), versic_pl.min()))
overlap_pw = max(0, min(setosa_pw.max(), versic_pw.max()) - max(setosa_pw.min(), versic_pw.min()))

print(f"  petal_length range  → setosa [{setosa_pl.min():.1f}, {setosa_pl.max():.1f}]  "
      f"versicolor [{versic_pl.min():.1f}, {versic_pl.max():.1f}]  "
      f"overlap width = {overlap_pl:.2f} cm")
print(f"  petal_width  range  → setosa [{setosa_pw.min():.1f}, {setosa_pw.max():.1f}]  "
      f"versicolor [{versic_pw.min():.1f}, {versic_pw.max():.1f}]  "
      f"overlap width = {overlap_pw:.2f} cm")
print()
print("  ➜ Neither single feature separates the classes perfectly.")
print("    LDA finds the ONE direction in 2D that maximises the ratio")
print("    of between-class spread to within-class spread — a combined")
print("    discriminant axis that does better than either feature alone.")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 — CLASS STATISTICS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mean vectors for each class and the grand mean (overall mean of our data).
#LDA tries to push centroids/means apart (between-clsas)
# and keep the intracluster variance as small as possible (within-class)

print("~" * 60)
print("STEP 3 — Class Statistics (mean vectors)")
print("~" * 60)

X0 = X[y == 0]   # setosa
X1 = X[y == 1]   # versicolor

mu0 = X0.mean(axis=0)          # mean vector for class 0
mu1 = X1.mean(axis=0)          # mean vector for class 1
mu  = X.mean(axis=0)           # grand mean (over all samples)

print(f"  μ_setosa     = {mu0}   (shape {mu0.shape})")
print(f"  μ_versicolor = {mu1}   (shape {mu1.shape})")
print(f"  μ_overall    = {mu}    (shape {mu.shape})")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 4 — SCATTER MATRICES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sw (within-class scatter):  sum over classes of the scatter of
#     each class's points around its OWN mean.
#     Sw = Σ_c  (X_c − μ_c)ᵀ (X_c − μ_c)
#
# Sb (between-class scatter): how far apart the class means are,
#     weighted by class size.
#     Sb = Σ_c  n_c (μ_c − μ)(μ_c − μ)ᵀ
# %%
print("=" * 60)
print("STEP 4 — Scatter Matrices")
print("=" * 60)

# Within-class scatter  Sw = Σ_c (X_c − μ_c)ᵀ (X_c − μ_c)
# Each (X_c − μ_c) is an (n_c × 2) centred matrix.
# The product gives a (2 × 2) scatter matrix per class.
S0 = (X0 - mu0).T @ (X0 - mu0)     # scatter for setosa
S1 = (X1 - mu1).T @ (X1 - mu1)     # scatter for versicolor
Sw = S0 + S1

print("  Within-class scatter  Sw  (2×2):")
print(f"    {Sw[0]}")
print(f"    {Sw[1]}")
print()

# Between-class scatter  Sb = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ
n0, n1 = len(X0), len(X1)
diff0 = (mu0 - mu).reshape(-1, 1)   # column vector
diff1 = (mu1 - mu).reshape(-1, 1)
Sb = n0 * (diff0 @ diff0.T) + n1 * (diff1 @ diff1.T)

print("  Between-class scatter Sb  (2×2):")
print(f"    {Sb[0]}")
print(f"    {Sb[1]}")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 5 — SOLVE FOR THE DISCRIMINANT AXIS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LDA maximises the "generalised Rayleigh quotient":
#     J(w) = wᵀ Sb w  /  wᵀ Sw w
#
# The solution is the eigenvector of  Sw⁻¹ Sb  with the largest
# eigenvalue.  For a 2-class problem, there is at most 1 non-zero
# eigenvalue, so we get exactly one discriminant direction.
# %%
print("=" * 60)
print("STEP 5 — Eigenvalue Problem  →  Discriminant Axis w")
print("=" * 60)

Sw_inv = np.linalg.inv(Sw)
A      = Sw_inv @ Sb                       # the matrix whose eigenvectors we need

eigenvalues, eigenvectors = np.linalg.eig(A)

# Sort descending by eigenvalue magnitude
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx].real        # ensure real (tiny imaginary parts from float noise)
eigenvectors = eigenvectors[:, idx].real

# The first eigenvector is the discriminant direction
w = eigenvectors[:, 0]
w = w / np.linalg.norm(w)                  # normalise to unit length

print(f"  Sw⁻¹ @ Sb =")
print(f"    {A[0]}")
print(f"    {A[1]}")
print()
print(f"  Eigenvalues  : {eigenvalues}")
print(f"  Eigenvectors (columns):")
print(f"    {eigenvectors}")
print()
print(f"  ★ Discriminant axis  w = {w}  (unit length)")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````
# STEP 6 — DRAW THE DISCRIMINANT AXIS ON THE 2D SCATTER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````````````````
# The axis passes through the grand mean, in the direction of w.
# We draw a line from  μ − t·w  to  μ + t·w  for a suitable t.
# %%
print("=" * 60)
print("STEP 6 — Overlay discriminant axis on scatter")
print("=" * 60)

fig, ax = plt.subplots()
for cls, color in COLORS.items():
    sub = df[df["class"] == cls]
    ax.scatter(sub["petal_length"], sub["petal_width"],
               c=color, label=cls, edgecolors="k", linewidths=0.4, s=50, alpha=0.85)

# Compute a nice line extent
t_range = 3.0
p1 = mu - t_range * w
p2 = mu + t_range * w
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=2, label="LDA axis (w)")

# Mark the grand mean
ax.plot(*mu, marker="X", color="red", markersize=12, zorder=5, label="Grand mean")

ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.set_title("Step 6 — Discriminant Axis through Grand Mean")
ax.legend()
plt.tight_layout()
plt.savefig("step6_axis.png")
plt.show()
print("  (see plot)\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 7 — PROJECT ALL POINTS ONTO w
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The 1D projection score for a point x is simply:
#     score = x · w   (dot product)
#
# This collapses the 2D data onto the single direction that
# maximally separates the two classes.
# %%
print("=" * 60)
print("STEP 7 — 1D Projected Scores")
print("=" * 60)

scores = X @ w                             # shape (100,)
scores_0 = scores[y == 0]                  # setosa projections
scores_1 = scores[y == 1]                  # versicolor projections

print(f"  Setosa     projected scores: min={scores_0.min():.3f}  max={scores_0.max():.3f}  mean={scores_0.mean():.3f}")
print(f"  Versicolor projected scores: min={scores_1.min():.3f}  max={scores_1.max():.3f}  mean={scores_1.mean():.3f}")
print()

fig, ax = plt.subplots(figsize=(9, 3.5))
bins = np.linspace(scores.min() - 0.2, scores.max() + 0.2, 30)
ax.hist(scores_0, bins=bins, alpha=0.55, color=COLORS["setosa"],
        label="setosa", edgecolor="k", linewidth=0.4)
ax.hist(scores_1, bins=bins, alpha=0.55, color=COLORS["versicolor"],
        label="versicolor", edgecolor="k", linewidth=0.4)
# rug ticks
ax.plot(scores_0, np.full_like(scores_0, -0.3), "|",
        color=COLORS["setosa"], markersize=10, alpha=0.7)
ax.plot(scores_1, np.full_like(scores_1, -0.3), "|",
        color=COLORS["versicolor"], markersize=10, alpha=0.7)

ax.set_xlabel("Projection score  (x · w)")
ax.set_ylabel("Count")
ax.set_title("Step 7 — 1D Histogram of Projected Scores")
ax.legend()
plt.tight_layout()
plt.savefig("step7_projection.png")
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# STEP 8 — DECISION THRESHOLD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We draw a line to depict our threshold which is also the midpoint between the two projected class means.  
# Point score < threshold; point is predicted as "setosa".  
# Point score >= threshold; point is predicted as "versicolor".
# %%
print("=" * 60)
print("STEP 8 — Decision Threshold")
print("=" * 60)

proj_mu0 = mu0 @ w      # projected mean of setosa
proj_mu1 = mu1 @ w      # projected mean of versicolor
threshold = (proj_mu0 + proj_mu1) / 2.0

print(f"  Projected mean (setosa)     = {proj_mu0:.4f}")
print(f"  Projected mean (versicolor) = {proj_mu1:.4f}")
print(f"  Threshold (midpoint)        = {threshold:.4f}")
print()

# Determine which side is which class
# (if proj_mu0 < threshold, then scores < threshold → setosa)
if proj_mu0 < proj_mu1:
    rule = "score < threshold → setosa,  score ≥ threshold → versicolor"
    predict = lambda s: 0 if s < threshold else 1
else:
    rule = "score < threshold → versicolor,  score ≥ threshold → setosa"
    predict = lambda s: 1 if s < threshold else 0

print(f"  Decision rule:  {rule}")
print()

# Re-draw the projection plot with threshold marked
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(scores_0, bins=bins, alpha=0.55, color=COLORS["setosa"],
        label="setosa", edgecolor="k", linewidth=0.4)
ax.hist(scores_1, bins=bins, alpha=0.55, color=COLORS["versicolor"],
        label="versicolor", edgecolor="k", linewidth=0.4)
ax.axvline(threshold, color="red", linewidth=2, linestyle="--", label=f"Threshold = {threshold:.2f}")
ax.plot(proj_mu0, 0, marker="D", color=COLORS["setosa"], markersize=10,
        zorder=5, label=f"μ_setosa proj = {proj_mu0:.2f}")
ax.plot(proj_mu1, 0, marker="D", color=COLORS["versicolor"], markersize=10,
        zorder=5, label=f"μ_versicolor proj = {proj_mu1:.2f}")
ax.set_xlabel("Projection score  (x · w)")
ax.set_ylabel("Count")
ax.set_title("Step 8 — Decision Threshold on Projected Scores")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("step8_threshold.png")
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# STEP 9 — TEST / CLASSIFY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# We classify a few sample points plus synthetic "new" points.
# Then we report accuracy on the full training set.
# %%
print("=" * 60)
print("STEP 9 — Classify Sample & Synthetic Points")
print("=" * 60)

# A few points from the dataset (indices 0, 25, 49, 55, 75, 99)
sample_indices = [0, 25, 49, 55, 75, 99]
# Plus two synthetic points someone might measure in the field
synthetic_points = np.array([
    [2.0, 0.8],   # should look like setosa
    [4.5, 1.4],   # should look like versicolor
])
synthetic_labels = ["(synthetic)", "(synthetic)"]

print(f"  {'Point':>22s}  {'Score':>7s}  {'Predicted':>12s}  {'Actual':>12s}")
print("  " + "-" * 60)

for idx in sample_indices:
    point = X[idx]
    score = point @ w
    pred  = predict(score)
    actual = int(y[idx])
    tag = "✓" if pred == actual else "✗"
    print(f"  {str(point):>22s}  {score:7.3f}  {class_names[pred]:>12s}  "
          f"{class_names[actual]:>12s}  {tag}")

for i, point in enumerate(synthetic_points):
    score = point @ w
    pred  = predict(score)
    print(f"  {str(point):>22s}  {score:7.3f}  {class_names[pred]:>12s}  "
          f"{'—':>12s}  {synthetic_labels[i]}")

print()

# Full-dataset accuracy
preds_full = np.array([predict(s) for s in scores])
accuracy = np.mean(preds_full == y) * 100
n_correct = int(np.sum(preds_full == y))

print(f"  Full-dataset accuracy:  {n_correct}/{len(y)} = {accuracy:.1f}%")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# BONUS — DECISION BOUNDARY IN 2D
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# The decision boundary in the original 2D space is the set of
# points where  x · w = threshold, which is a line perpendicular
# to w.  Let's overlay it on the scatter for a complete picture.
# %%
print("=" * 60)
print("BONUS — Decision boundary in 2D")
print("=" * 60)

fig, ax = plt.subplots()
for cls, color in COLORS.items():
    sub = df[df["class"] == cls]
    ax.scatter(sub["petal_length"], sub["petal_width"],
               c=color, label=cls, edgecolors="k", linewidths=0.4, s=50, alpha=0.85)

# LDA axis (through grand mean)
t_range = 3.0
p1 = mu - t_range * w
p2 = mu + t_range * w
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=1.5, label="LDA axis (w)")

# Decision boundary: all x where x · w = threshold
# In 2D this is a line.  We need a direction perpendicular to w.
w_perp = np.array([-w[1], w[0]])           # 90° rotation of w
# A point on the boundary: find t such that (mu + t*w) · w = threshold
# mu·w + t = threshold  →  t = threshold − mu·w
t_boundary = threshold - mu @ w
boundary_point = mu + t_boundary * w
bp1 = boundary_point - 3.0 * w_perp
bp2 = boundary_point + 3.0 * w_perp
ax.plot([bp1[0], bp2[0]], [bp1[1], bp2[1]], "r-", linewidth=2, label="Decision boundary")

# Grand mean
ax.plot(*mu, marker="X", color="red", markersize=12, zorder=5, label="Grand mean")

# Synthetic points
ax.scatter(synthetic_points[:, 0], synthetic_points[:, 1],
           marker="*", s=200, c="limegreen", edgecolors="k", linewidths=0.6,
           zorder=6, label="Synthetic test pts")

ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.set_title("Bonus — Full LDA picture: axis + decision boundary")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("bonus_full_picture.png")
plt.show()
print("  (see plot)\n")

print("=" * 60)
print("Done!  All steps of LDA computed from scratch.")
print("=" * 60)
