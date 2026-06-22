# 📘 K-Nearest Neighbors (KNN)

## 🔹 What is KNN?

K-Nearest Neighbors (KNN) is a **supervised learning algorithm** used for:
- Classification
- Regression

It works by:
> Finding the **K closest data points** and making predictions based on them.

---

## 🔹 Intuition

- Similar points exist close to each other
- A new point is classified based on its **neighbors**

---

## 🔹 Algorithm Steps

1. Choose value of **K**
2. Compute distance from new point to all training points
3. Select **K nearest neighbors**
4. Perform:
   - **Classification** → Majority voting
   - **Regression** → Average of values

---

## 🔹 Distance Metrics

### 1. Euclidean Distance

d = √Σ(xi - yi)²


### 2. Manhattan Distance

d = Σ|xi - yi|


### 3. Minkowski Distance

d = (Σ|xi - yi|^p)^(1/p)


---

## 🔹 Choosing K

| K Value | Behavior |
|--------|---------|
| Small K | Overfitting (high variance) |
| Large K | Underfitting (high bias) |

👉 Optimal K is found using **cross-validation**

---

## 🔹 Advantages

- Simple to understand
- No training phase (lazy learner)
- Works well on small datasets
- Non-linear decision boundary

---

## 🔹 Disadvantages

- Slow for large datasets (O(n))
- Memory intensive
- Sensitive to:
  - Noise
  - Outliers
  - Feature scaling
- Suffers from **curse of dimensionality**

---

## 🔹 Important Concepts

### 1. Feature Scaling (VERY IMPORTANT)
KNN is distance-based → scaling is required

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
2. Curse of Dimensionality
As features increase:
Distance loses meaning
Performance degrades
3. Decision Boundary
KNN creates non-linear boundaries
Highly flexible depending on K
🔹 Time Complexity
Phase	Complexity
Training	O(1)
Prediction	O(n × d)
🔹 Sklearn Implementation
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
🔹 Hyperparameter Tuning
from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5)
    scores.append(score.mean())

optimal_k = k_values[scores.index(max(scores))]
🔹 When to Use KNN

✅ Small datasets
✅ Non-linear patterns
✅ Baseline model

🔹 When NOT to Use

❌ Large datasets
❌ High-dimensional data
❌ Real-time systems

🔹 Key Takeaways
KNN is simple but powerful
Scaling is mandatory
Choice of K is critical
Not suitable for large-scale problems