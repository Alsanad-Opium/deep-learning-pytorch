# K-Means Clustering — Complete Notes

## 1. What K-Means Is

K-Means is an **unsupervised, centroid-based, partitional** clustering algorithm. It divides `n` observations into `k` clusters such that each point belongs to the cluster whose centroid (mean) it is closest to.

- **Unsupervised**: no labels are used or needed.
- **Partitional**: every point belongs to exactly one cluster (hard assignment), unlike fuzzy/soft clustering.
- **Centroid-based**: clusters are represented by their mean point, not by density or connectivity (contrast with DBSCAN, hierarchical clustering).

---

## 2. The Objective Function (The Math)

K-Means minimizes **within-cluster sum of squares (WCSS)**, also called **inertia**:

```
J = Σ (i=1 to k) Σ (x in Cᵢ) ||x - μᵢ||²
```

Where:
- `k` = number of clusters
- `Cᵢ` = set of points assigned to cluster i
- `μᵢ` = centroid (mean vector) of cluster i
- `||x - μᵢ||²` = squared Euclidean distance between point x and centroid μᵢ

**Interpretation**: minimize the total squared distance between every point and the centroid it's assigned to. Lower inertia = tighter, more compact clusters.

**Important limitation baked into the math**: because it uses squared Euclidean distance, K-Means implicitly assumes:
- Clusters are **roughly spherical/convex** in shape
- Clusters are of **similar size and density**
- Features are on **comparable scales** (hence why you scale first)

This is an **NP-hard** problem to solve exactly (finding the *global* optimum), so K-Means uses an iterative heuristic (Lloyd's Algorithm) that converges to a *local* optimum, not guaranteed to be global.

---

## 3. The Algorithm — Lloyd's Algorithm (Standard K-Means)

**Step 0 — Choose k** (number of clusters — see Section 5 for how)

**Step 1 — Initialize** k centroids (see Section 4 for initialization strategies)

**Step 2 — Assignment step**: assign each point to the nearest centroid (by Euclidean distance)
```
Cᵢ = { x : ||x - μᵢ|| ≤ ||x - μⱼ|| for all j }
```

**Step 3 — Update step**: recompute each centroid as the mean of all points assigned to it
```
μᵢ = (1/|Cᵢ|) Σ (x in Cᵢ) x
```

**Step 4 — Repeat** Steps 2–3 until:
- Centroids stop moving (converged), or
- Assignments no longer change, or
- Max iterations reached, or
- Improvement in inertia falls below a tolerance `tol`

**Convergence guarantee**: inertia is guaranteed to monotonically decrease (or stay the same) each iteration — it will converge, but only to a **local minimum**, not necessarily the global one. This is why initialization matters so much (Section 4) and why you should run it with multiple random starts (`n_init` in sklearn).

---

## 4. Initialization Strategies

Poor initialization → poor local minimum → bad clusters. Two main approaches:

### a) Random initialization
Pick k random points from the dataset as starting centroids. Simple but prone to bad local minima — e.g. two initial centroids landing close together in the same true cluster.

### b) K-Means++ (the modern default, `init="k-means++"` in sklearn)
Smarter seeding that spreads initial centroids apart:
1. Choose the first centroid uniformly at random from the data points.
2. For each remaining point, compute its squared distance `D(x)²` to the nearest already-chosen centroid.
3. Choose the next centroid randomly, with probability proportional to `D(x)²` (points farther from existing centroids are more likely to be picked).
4. Repeat until k centroids are chosen.

**Why it matters**: spreads centroids out logically instead of leaving it to chance, which empirically leads to faster convergence and better, more consistent final clusters. This is why `k-means++` is sklearn's default.

### c) Multiple restarts (`n_init`)
Because even k-means++ has randomness, sklearn runs the entire algorithm `n_init` times with different seeds and keeps the run with the lowest final inertia. `n_init="auto"` (newer sklearn) picks a sensible number automatically (usually 1 for k-means++, more for random init).

---

## 5. Choosing the Optimal k

There is **no single correct k** — you're balancing model fit against interpretability/business need. Several complementary methods:

### a) Elbow Method (using Inertia)
- Fit K-Means for a range of k (e.g. 1 to 12).
- Plot k (x-axis) vs inertia (y-axis).
- Inertia always decreases as k increases (more clusters = tighter fit, trivially inertia → 0 when k = n).
- Look for the "elbow" — the point where the rate of decrease sharply flattens. Beyond that point, adding more clusters gives diminishing returns.
- **Weakness**: the "elbow" is often subjective/ambiguous — real data rarely has a crisp bend.

### b) Silhouette Score
For each point i:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where:
- `a(i)` = mean distance from i to all other points in its **own** cluster (cohesion)
- `b(i)` = mean distance from i to all points in the **nearest other** cluster (separation)

- Ranges from **-1 to +1**.
  - Close to **+1**: point is well-matched to its own cluster, far from neighboring clusters (good).
  - Close to **0**: point is on/near the boundary between two clusters (ambiguous).
  - **Negative**: point is probably in the wrong cluster (likely closer to a different cluster's mean than its own).
- Average silhouette score across all points gives one number per k. Plot k vs average silhouette and pick the k that **maximizes** it (unlike inertia, this doesn't monotonically decrease, so it can give a genuine peak).
- Can also inspect **per-cluster silhouette plots** (`silhouette_samples`) — reveals if a low average score is due to one bad cluster or uniformly mediocre separation across all of them.

### c) Gap Statistic
Compares the inertia of your actual clustering against the expected inertia under a **null reference distribution** (uniformly random data with no real cluster structure). The optimal k is where the gap between observed and expected inertia is largest. More statistically rigorous than the elbow method but more computationally expensive (requires generating reference datasets).

### d) Domain knowledge / business constraints
Sometimes k is dictated by practical need — e.g. "marketing only has budget for 4 customer segments" — regardless of what the math says is "optimal."

### e) Davies-Bouldin Index (less common, good to know)
Measures average similarity between each cluster and its most similar other cluster (based on within-cluster scatter vs between-cluster separation). **Lower is better** (0 = perfect separation).

**Best practice**: use elbow + silhouette together. If they roughly agree, you have confidence in k. If they disagree, investigate why (e.g. plot the clusters and look at them visually via PCA/t-SNE).

---

## 6. Preprocessing — Why Scaling Matters

K-Means uses Euclidean distance, which is **not scale-invariant**. A feature ranging 0–100,000 (like income) will dominate the distance calculation over a feature ranging 0–100 (like age) purely because of its magnitude, not because it's actually more important.

**Standard fix**: `StandardScaler` (z-score normalization) — transforms each feature to mean 0, std 1:
```
z = (x - μ) / σ
```
Alternative: `MinMaxScaler` (scales to a [0,1] range) — useful if you want bounded values, but more sensitive to outliers than StandardScaler.

**Rule**: always fit the scaler on training data only, then use the *same fitted scaler* (`.transform()`, not `.fit_transform()`) on any new/test data — otherwise you leak information and get inconsistent scaling between train and inference time.

**Other preprocessing considerations**:
- **Categorical variables**: K-Means doesn't natively handle categorical data (no meaningful "distance" between categories). Either one-hot encode (with caution — can dilute distance in high dimensions) or use a variant like **K-Modes** / **K-Prototypes** designed for mixed/categorical data.
- **Outliers**: K-Means is sensitive to outliers since centroids are means, and means are pulled by extreme values. Consider removing or capping outliers, or use **K-Medoids (PAM)**, which uses actual data points as cluster centers (medians in effect) and is more robust.
- **Dimensionality**: in high dimensions, Euclidean distance becomes less meaningful ("curse of dimensionality" — distances between points converge). Consider dimensionality reduction (PCA) before clustering if you have many features.

---

## 7. Visualizing Clusters (High-Dimensional Data)

You typically cluster on the full feature space (e.g. 3+ scaled features) but can't visualize more than 2-3 dimensions directly. Common approach:

1. **Fit K-Means on the full scaled feature space** (not on reduced data — you want the real distances).
2. **Reduce to 2D for plotting only**, using PCA (or t-SNE/UMAP for more complex nonlinear structure):
   ```
   pca = PCA(n_components=2)
   data_2d = pca.fit_transform(scaled_data)
   centers_2d = pca.transform(kmeans.cluster_centers_)
   ```
3. Scatter-plot `data_2d`, colored by cluster label, and overlay `centers_2d` as the centroids.

**Important nuance**: the 2D plot is a *projection* — two points that look close in the PCA plot might actually be farther apart in the true high-dimensional space (and vice versa), since PCA only preserves the directions of maximum variance. Don't over-interpret visual "overlap" in a PCA scatterplot as proof clusters are badly separated — cross-check with silhouette score, which is computed on the true feature space, not the 2D projection.

---

## 8. Evaluating & "Testing" a Fitted Model

Unlike supervised learning, there's no ground-truth label to score against, so "testing" means something different:

| Goal | Method |
|---|---|
| Overall cluster quality | Silhouette score (aggregate) |
| Per-cluster diagnosis | Silhouette samples/plot per cluster |
| Internal compactness | Inertia (lower = tighter, but always decreases with k, so only comparable at fixed k) |
| Stability check | Refit with different `random_state` seeds — do assignments stay consistent? |
| Predicting new/unseen points | `kmeans.predict()` on new data — **must** be transformed with the *same fitted scaler* first |
| Interpretability check | `groupby(cluster).mean()` — do the resulting cluster profiles make real-world sense? |
| External validation (if labels *do* exist) | Adjusted Rand Index, Normalized Mutual Information — compares clustering to a known ground truth, if you happen to have one (rare in real unsupervised settings) |

**No train/test split needed** in the supervised sense — K-Means fits on the full dataset because the goal is discovering structure in *all* your data, not generalizing to predict a held-out label. (You *can* split if you specifically want to test stability/generalization of cluster assignments to new incoming data, but it's a different motivation than supervised train/test splitting.)

---

## 9. Strengths and Weaknesses

**Strengths**
- Simple, fast, scales well to large datasets (`O(n·k·i·d)` — linear in n, where i = iterations, d = dimensions).
- Easy to interpret centroids.
- Guaranteed to converge (to *a* local minimum).
- Works well when clusters are roughly spherical and similarly sized.

**Weaknesses**
- Must specify k in advance.
- Sensitive to initialization (mitigated by k-means++ and multiple restarts).
- Assumes spherical, equally-sized, equally-dense clusters — fails on elongated, nested, or unevenly-sized/density clusters (e.g. can't separate two concentric circles).
- Sensitive to outliers (mean-based centroids get pulled).
- Only works on numeric data natively.
- Local minimum, not global — different runs can give different results without proper initialization strategy.

---

## 10. Related Variants (Good to Know)

- **MiniBatchKMeans**: uses small random batches instead of the full dataset per iteration — much faster on very large datasets, at a small cost to cluster quality.
- **K-Medoids / PAM**: uses actual data points as cluster centers instead of means — more robust to outliers, works with arbitrary distance metrics (not just Euclidean), but more computationally expensive.
- **K-Modes**: K-Means adapted for purely categorical data (uses mode instead of mean, and a matching-based dissimilarity instead of Euclidean distance).
- **K-Prototypes**: handles mixed numeric + categorical data by combining K-Means and K-Modes logic.
- **Gaussian Mixture Models (GMM)**: soft/probabilistic clustering — each point gets a probability of belonging to each cluster rather than a hard assignment; clusters can be elliptical, not just spherical (via covariance matrices).
- **Hierarchical Clustering**: builds a tree (dendrogram) of nested clusters, doesn't require specifying k upfront, but is more computationally expensive on large datasets.
- **DBSCAN**: density-based, doesn't require specifying k, can find arbitrarily-shaped clusters and naturally handles noise/outliers as "unclustered" — but struggles with clusters of varying density.

---

## 11. Typical End-to-End Workflow Summary

1. Load and clean data
2. EDA — understand feature distributions, relationships
3. Select relevant features for clustering
4. Handle categorical variables if needed
5. **Scale features** (StandardScaler, fit on the data you're clustering)
6. Determine optimal k (elbow + silhouette, cross-checked)
7. Fit final K-Means with chosen k (`k-means++` init, sufficient `n_init`)
8. Reduce dimensions (PCA) *only* for visualization
9. Evaluate: silhouette score, visual inspection, stability across seeds
10. **Profile clusters**: `groupby(cluster).mean()` on original (unscaled) features to interpret what each cluster represents in real-world terms
11. Assign human-readable personas/labels to clusters
12. Translate cluster insights into actionable business recommendations