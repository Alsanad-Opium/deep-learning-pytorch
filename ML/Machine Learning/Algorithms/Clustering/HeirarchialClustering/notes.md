# Agglomerative Clustering: Master Notes (Theory, sklearn, Metrics, Workflow, Interview Prep)

This file merges and extends the earlier notes. It adds the corrections and deeper points that came out of interview practice.

---

## Part 1. Core theory

### 1.1 What it is
Agglomerative clustering is **bottom-up hierarchical clustering**.

1. Every point starts as its own cluster (n clusters).
2. Compute distances between all pairs of clusters.
3. Merge the **two closest clusters**.
4. Update the distances from the new cluster to the rest.
5. Repeat until one cluster remains, or until k clusters remain.

The full merge history is stored as a tree called a **dendrogram**. Divisive clustering is the top-down opposite (rarely used).

### 1.2 Two separate choices (a very common confusion)

| Choice | What it defines | Examples |
|---|---|---|
| **Distance metric** | Distance between two *points* | Euclidean, Manhattan, cosine, Hamming |
| **Linkage** | Distance between two *clusters*, built on top of the metric | single, complete, average, Ward |

Saying "linkage decides the distance metric" is wrong. The metric measures points; the linkage decides how to combine point distances into a cluster distance.

### 1.3 Linkage in depth

```
Cluster A and cluster B, with points a in A and b in B:

single    d(A,B) = min over a,b of d(a,b)          closest pair
complete  d(A,B) = max over a,b of d(a,b)          farthest pair
average   d(A,B) = mean over a,b of d(a,b)         all pairs
ward      d(A,B) = increase in total within-cluster sum of squares if A and B merge
```

| Linkage | Cluster shapes | Strength | Weakness |
|---|---|---|---|
| **Single** | Elongated, stringy, non-convex | Can follow rings and chains | **Chaining**: joins clusters through a thin bridge of points; noise-sensitive |
| **Complete** | Compact, similar diameter | Avoids chaining | One outlier inflates distances; can split large clusters |
| **Average** | In between | Balanced, less noise-sensitive than single | No strong shape preference |
| **Ward** | Compact, balanced, similar size | Resists chaining; behaves like KMeans; usual default | Euclidean only; dislikes very unequal cluster sizes; outlier-sensitive |

**Why Ward needs Euclidean distance:** its criterion is variance, meaning the sum of squared distances to a cluster **mean**. The mean is the point that minimises squared *Euclidean* distance. With Manhattan or cosine distance the centroid is no longer the natural centre, so "increase in variance" stops meaning what it should. sklearn enforces `metric='euclidean'` for Ward.

### 1.4 The dendrogram
- **y-axis:** merge distance (height at which two clusters joined). **x-axis:** points or clusters.
- Cutting with a horizontal line gives clusters; the number of vertical lines it crosses is k.
- A **long vertical line** means clusters that were far apart were forced to merge, so cutting just below it is a natural choice of k.
- If the top merge only isolates a tiny group, those points are outliers, not structure.
- The dendrogram is the tree of the whole merge history, not "a representation of the linkage".

**Cophenetic correlation:** correlation between original pairwise distances and the distances implied by the tree. Closer to 1 means the tree represents the data faithfully. Use it to compare linkages.

### 1.5 Properties
- Deterministic (no random initialisation).
- No k needed upfront; one fit gives every k.
- Memory O(n²), time roughly O(n²) to O(n³): practical up to a few thousand points.
- **Greedy:** a bad early merge is never undone.
- No `predict` for new points.

---

## Part 2. Why outliers wreck the result (the core lesson)

Distance-based linkage means an outlier is far from everything, so it does not join any cluster until late, and it joins at a **large height**.

Consequences:
1. The top of the dendrogram shows "outliers vs everyone else".
2. k=2 gives one tiny cluster and one giant cluster.
3. For larger k, Ward keeps splitting off the next most extreme point, so the smallest cluster shrinks to **1 point**.
4. Silhouette gets inflated (see Part 4), so metrics look good while the clustering is useless.

**Root cause in many real datasets:** heavy right skew in positive features (spending, income, counts). A few huge values dominate the distance calculation.

### The fix: diagnose, then transform
1. **Diagnose** with histograms (skew) and boxplots (individual extreme points).
2. **Transform** right-skewed positive features with `np.log1p`.
   - `log1p(x) = log(1 + x)`, so zeros are safe.
   - Log **reduces** skew and pulls in the tail. It does **not** make data exactly normal. Say "less skewed / more symmetric".
3. **Scale after the log** with `StandardScaler` (log can't handle negatives, and scaled data has negatives).
4. **Other options** depending on what the outliers are:
   - data errors: remove
   - real but extreme: cap (winsorise), use `RobustScaler`, or analyse separately
   - many features: consider PCA
5. **Re-check:** the minimum cluster size should no longer collapse to 1 to 6.

---

## Part 3. sklearn implementation

### 3.1 Pipeline
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

features = df.drop(columns=['Channel', 'Region', 'Cluster'], errors='ignore')  # real features only
logged = np.log1p(features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(logged)

labels = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(X_scaled)
```

| Parameter | Meaning |
|---|---|
| `n_clusters` | Number of clusters (or `None` with `distance_threshold`) |
| `linkage` | `'ward'`, `'complete'`, `'average'`, `'single'` |
| `metric` | Point distance (`'euclidean'`, `'manhattan'`, `'cosine'`...). Ward needs `'euclidean'`. Older sklearn used `affinity`. |
| `distance_threshold` | Cut the tree at a height instead of fixing k |
| `compute_full_tree` | Must be `True` with `distance_threshold` |

### 3.2 Dendrogram, cutting and cophenetic (scipy)
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
plt.show()

labels = fcluster(Z, t=3, criterion='maxclust') - 1     # cut by k
labels = fcluster(Z, t=20, criterion='distance') - 1    # cut by height
c, _ = cophenet(Z, pdist(X_scaled))                     # closer to 1 is better
```

### 3.3 Sweep over k
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

rows = []
for k in range(2, 11):
    lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_scaled)
    rows.append({
        'k': k,
        'silhouette': silhouette_score(X_scaled, lab),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, lab),
        'davies_bouldin': davies_bouldin_score(X_scaled, lab),
        'min_cluster_size': pd.Series(lab).value_counts().min(),
    })
print(pd.DataFrame(rows).round(3))
```

Common bugs:
- `linkage=` passed to `silhouette_score` (it belongs only in the clustering model).
- `range(1, ...)`: every metric needs at least 2 clusters.
- Leaving label or ID columns (for example a previous `Cluster` column) in the features.
- Plotting with `plt.hist(df)` (all columns overlaid) instead of `df.hist()` (one plot per column).

### 3.4 Assigning new data (no `predict` on agglomerative)

Always apply the **same fitted** preprocessing: `np.log1p`, then `scaler.transform(...)`. Never call `fit_transform` on new data: one row would learn a new mean and standard deviation and the result would be meaningless. Save the fitted scaler (for example with `joblib`) and keep the column order identical.

Approaches:

| Approach | How | Notes |
|---|---|---|
| **Nearest centroid** | Compute each cluster's mean in scaled space; assign to the closest | Simple, fast |
| **Classifier on cluster labels** | Train kNN, logistic regression or decision tree with the labels as targets, then `predict` | Most common; a decision tree gives readable rules |
| **Refit periodically** | Add new data and re-cluster | Accurate but expensive; labels may change between runs |

```python
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid().fit(X_scaled, labels)
x_new = scaler.transform(np.log1p(new_customer))      # transform, never fit
print(nc.predict(x_new))
```

Also flag points far from every centroid ("unusual customer") and monitor drift; refit the model and scaler if the customer mix changes.

---

## Part 4. Metrics in depth

All three below are **internal** metrics (only the data and the labels).

### 4.1 Silhouette score
For each point *i*:
- `a(i)` = mean distance to the **other points in its own cluster**
- `b(i)` = mean distance to the points in the **nearest other cluster**
- `s(i) = (b(i) - a(i)) / max(a(i), b(i))`

The score is the mean of `s(i)` over all points.

- Range -1 to 1. Higher is better. Near 0: overlapping clusters. Negative: points probably in the wrong cluster.
- In a good clustering `b` is larger than `a`, so `b - a` is positive.
- Rough guide: above 0.5 strong, 0.25 to 0.5 weak-to-moderate, below 0.25 little structure.
- Cost O(n²); use `sample_size=` on large data. `silhouette_samples` gives per-point values for per-cluster plots.

**How outliers inflate it:** if a few extreme outliers form their own far-away cluster, `b` is huge for almost every ordinary point, so nearly every `s(i)` approaches 1 and the average is high. The score then says "outliers are far from everyone", not "the data has clear groups".

**Not comparable across preprocessing:** a log transform changes all distances. A raw-data silhouette (0.79) and a logged-data silhouette (0.26) come from different spaces, so the drop does not mean the clustering got worse.

### 4.2 Calinski-Harabasz index
```
CH = [ B / (k - 1) ] / [ W / (n - k) ]      B = between-cluster, W = within-cluster dispersion
```
Higher is better. Fast. Favours compact, convex clusters and often keeps rising with k, so a monotonic rise is uninformative.

### 4.3 Davies-Bouldin index
For each cluster, take its worst neighbour by `(spread_i + spread_j) / distance between centroids`, then average. **Lower is better**, 0 is ideal. Centroid-based, so it favours compact, roughly spherical clusters.

### 4.4 Minimum cluster size
Not a formal metric, but essential: tiny clusters (1 to 10 points) usually mean isolated outliers, not real segments. Discard any k whose smallest cluster is negligible.

### 4.5 Other tools
| Tool | Use |
|---|---|
| Elbow (WCSS) | Bend in within-cluster variance versus k; often ambiguous |
| Gap statistic | Compares compactness to random data; can indicate k=1 |
| Bootstrap stability + ARI | Re-cluster subsamples; consistent labelling supports the k |
| Inconsistency coefficient | Flags merges unusually large compared with nearby merges |
| PCA plot | Shows whether clusters occupy separate regions or slice through one blob |

---

## Part 5. External validation (comparing clusters to known labels)

If a true label exists (for example `Channel`), use it as an **outside check** that no internal metric can provide.

```python
pd.crosstab(df.Clusters_k2, df['Channel'])
```

### Why "87% agreement" is not "87% accuracy"
1. **Cluster IDs are arbitrary.** Cluster 0 vs 1 carries no meaning, so a naive comparison depends on which cluster you map to which label (label permutation). Agreement must be computed with the best mapping, or with a label-invariant metric.
2. **Class imbalance sets a baseline.** In the Wholesale data, Horeca has 298 customers and Retail 142, so always guessing "Horeca" already scores about 68%. Compare 87% against that baseline, not against 50%.
3. **Channel is not necessarily the "true" structure.** Clustering is unsupervised. It may find real structure (for example spending volume) that is not the same as Channel. Disagreement is not automatically an error.
4. **There is no train/test notion here.** "Accuracy" implies predicting a target; clustering only found groups.

### Better measures (label-invariant)
| Metric | Meaning | Range |
|---|---|---|
| **Adjusted Rand Index (ARI)** | Pair-counting agreement, corrected for chance; 0 = random, 1 = perfect | about -0.5 to 1 |
| **Normalized / Adjusted Mutual Information (NMI/AMI)** | Shared information between clusters and labels | 0 to 1 |
| **Homogeneity / Completeness / V-measure** | Each cluster contains one class / each class sits in one cluster / their harmonic mean | 0 to 1 |
| **Fowlkes-Mallows** | Geometric mean of pairwise precision and recall | 0 to 1 |
| **Purity** | Share of points in their cluster's majority class (biased toward large k) | 0 to 1 |

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
adjusted_rand_score(df['Channel'], labels)
```

### Computed on the Wholesale project crosstabs
| | Agreement with best mapping | ARI | NMI |
|---|---|---|---|
| k=2 | about 87% | 0.54 | 0.45 |
| k=3 | about 87% | 0.55 | 0.41 |

Reading: agreement with Channel is well above the 68% baseline and ARI is clearly above 0, so the clusters carry real channel information. But ARI around 0.5 (not near 1) shows the match is moderate, not perfect. k=3 does not improve on k=2 in a meaningful way.

---

## Part 6. Choosing k when metrics tie or disagree

Use metrics only to **shortlist** k. Then decide with:

1. **Profiling** (`groupby` means or medians): does the extra cluster add a new customer type, or just split an old one by volume?
2. **Cluster sizes:** is the extra cluster large enough to act on, or a small noisy group?
3. **External validation** (crosstab, ARI): does the extra cluster align with a known label?
4. **Stability:** bootstrap ARI across resamples.
5. **Dendrogram gap:** a large jump before the cut supports that k.
6. **Purpose:** simple two-way split vs targeting a specific high-volume segment.
7. **Parsimony:** near-tied scores and no new information means prefer the simpler k.
8. **Visual check:** PCA scatter coloured by cluster.

Each check has limits (in the Wholesale data both k=2 and k=3 matched Channel about equally, so profiling was the tiebreaker). They work together.

### Profiling rules
- Compare **down a column** (cluster vs cluster), not across a row, because feature scales differ.
- Only call a feature "high" or "low" when the gap between clusters is large.
- Use medians of the **original** values for readable units; means of logged values still compare gaps fine.
- Describe clusters only by what the features show ("high Fresh, low Detergents_Paper"). Do **not** invent stories (loyalty, travellers, visit frequency) that the columns cannot support.

---

## Part 7. Agglomerative vs KMeans vs DBSCAN vs GMM

| | Agglomerative | KMeans | DBSCAN | GMM |
|---|---|---|---|---|
| Needs k upfront | No (cut later) | Yes | No | Yes |
| Randomness | None (deterministic) | Random **initial centroids** (use `n_init`, `random_state`) | None | Random init |
| Cluster shape | Depends on linkage | Convex, spherical | Arbitrary | Elliptical |
| Outliers | Sensitive | Sensitive | Labels noise | Moderate |
| Scalability | Poor (O(n²) memory) | Good | Moderate | Moderate |
| Gives hierarchy | Yes | No | No | No |
| Predicts new points | No | Yes | No | Yes |
| Metric flexibility | Many metrics and linkages | Euclidean centroids | Many | Gaussian |

Points to get right:
- **You choose k in KMeans.** What is random is the initial centroid placement.
- **Concentric circles:** KMeans cuts them in half. Among agglomerative options only **single linkage** can separate them (via chaining). Ward and complete favour compact blobs and also fail. **DBSCAN** or spectral clustering are the standard tools for non-convex shapes.
- Agglomerative is not specifically "for intermingled data"; overlapping clusters are hard for nearly every method.
- **Pick KMeans** for large data, when you need to `predict` new samples, or when clusters look roughly spherical.
- **Pick agglomerative** for small to medium data when you want the dendrogram, need a deterministic result, or want to inspect several k from one fit.

---

## Part 8. Full workflow checklist

1. Understand the data: `info`, `describe`, missing values; drop IDs, labels, previous cluster columns.
2. Plot distributions: `df.hist(bins=30)` and boxplots.
3. `log1p` skewed positive features, then `StandardScaler`. Save the fitted scaler.
4. Handle outliers: inspect what they are before removing.
5. Dendrogram with Ward; note large gaps; check cophenetic correlation across linkages.
6. Sweep k: silhouette, CH, DB, **minimum cluster size**.
7. Shortlist k; drop k values with tiny clusters.
8. Profile clusters; compare with external labels using ARI/NMI, not "accuracy".
9. Test stability (bootstrap ARI) and visualise with PCA.
10. Decide k and justify with several pieces of evidence.
11. Plan deployment: saved scaler, nearest-centroid or classifier for new points, drift monitoring.
12. Write up: preprocessing, linkage, metrics, chosen k and why, profiles, validation, limitations.

---

## Part 9. Wholesale project: end-to-end story

| Stage | Result |
|---|---|
| Raw data, k=2 | Silhouette 0.79, but smallest cluster was 6 points (outliers vs everyone) |
| Raw data, k>=4 | Smallest cluster size 1 |
| Diagnosis | Heavy right skew in every spending feature (histograms) |
| Fix | `log1p` then `StandardScaler` |
| After fix, k=2 | Silhouette 0.258, CH 134.6, DB 1.60, smallest cluster 178 |
| After fix, k=3 | Silhouette 0.255, CH 116.8, DB 1.54, smallest cluster 53 |
| k=2 profile | Cluster 0: high Grocery, Milk, Detergents_Paper. Cluster 1: high Fresh and Frozen, lowest Detergents_Paper |
| k=3 profile | Fresh/Frozen cluster unchanged; Grocery/Milk group split by volume; extra cluster small and mixed-channel |
| External check | About 87% agreement with Channel (baseline 68%); ARI about 0.54 |
| Decision | **k=2** |

One-paragraph summary you can say aloud:

> "Ward clustering on the raw data isolated a handful of outliers because of heavy right skew. I diagnosed that with histograms, applied log1p and standard scaling, and the smallest cluster at k=2 grew from 6 to 178. Silhouette dropped from 0.79 to 0.26, but the earlier score was inflated by outliers. k=2 and k=3 were nearly tied on metrics, so I profiled the clusters: k=3 only split the Grocery-heavy group by volume into a small mixed-channel cluster. The k=2 clusters matched the Horeca/Retail channel well above the 68% majority baseline, with ARI around 0.54, so I chose k=2. For new customers I would apply the saved log and scaler, then assign by nearest centroid or a classifier trained on the labels."

---

## Part 10. Interview Q&A cheat sheet

**1. How does agglomerative clustering work?**
Bottom-up: every point starts as a cluster and the two closest clusters merge repeatedly. Closeness needs a point metric and a linkage. The merge history is the dendrogram, and merge height is the distance at which the merge happened.

**2. Single vs complete vs Ward? Why Euclidean for Ward?**
Single = closest pair (chaining, elongated shapes, noise-sensitive). Complete = farthest pair (compact, outlier-sensitive). Ward = smallest increase in within-cluster variance (compact, balanced). Ward needs Euclidean because variance is defined by squared Euclidean distance to the centroid.

**3. Why did tiny clusters appear, and how did you fix it?**
Outliers are far from everything so they merge last and high, dominating the top of the tree; larger k keeps isolating the next extreme point. Confirmed by histogram and boxplot, fixed with log1p plus scaling (or remove, cap, or analyse separately).

**4. Silhouette dropped from 0.79 to 0.26. Did clustering get worse?**
No. `s = (b - a) / max(a, b)`, averaged. Outliers in their own cluster inflate `b` for nearly every point. Scores from different preprocessing are not comparable. Evidence of improvement: smallest cluster 6 to 178, and match with Channel.

**5. Agglomerative vs KMeans?**
Deterministic vs random initial centroids; full hierarchy vs one fit per k; many linkages vs Euclidean centroids; predicts no new points vs can predict; O(n²) memory vs scalable. Limitation: memory and irreversible greedy merges. Concentric circles need single linkage or DBSCAN, not Ward.

**6. Metrics tie. How do you choose k?**
Profiling, cluster sizes, external validation, stability, dendrogram gap, purpose, parsimony, PCA plot.

**7. How do you assign a new customer?**
Same log1p, then `scaler.transform` (never `fit`), then nearest centroid or a classifier trained on cluster labels; flag far-away points; refit if data drifts.

**8. Is 87% agreement "accuracy"?**
No. Cluster IDs are arbitrary, class imbalance gives a 68% baseline, Channel is not necessarily the structure clustering should find, and clustering is unsupervised. Use ARI, NMI or V-measure, and read them alongside the baseline.

---

## Part 11. Common mistakes (final list)

1. Clustering unscaled or unlogged skewed data.
2. Trusting a high silhouette produced by a few outliers.
3. Ignoring the smallest cluster size.
4. Confusing point metric with linkage.
5. Ward with a non-Euclidean metric.
6. Saying log makes data "normal".
7. Calling `fit_transform` on new data instead of `transform`.
8. Comparing silhouette across different preprocessing.
9. Reporting "accuracy" against an external label instead of ARI/NMI plus the baseline.
10. Describing clusters with stories the data cannot support.
11. Leaving label, ID or previous cluster columns in the features.
12. Claiming agglomerative fixes non-convex shapes regardless of linkage.