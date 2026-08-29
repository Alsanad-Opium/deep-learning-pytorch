Yes. Let’s combine **Part 4 → Gradient & Hessian**, **Part 5 → Tree Construction & Split Gain**, **Part 6 → Regularization**, and **Part 7 → Important Hyperparameters** into one continuous set of notes.

# XGBoost — Gradient, Hessian, Tree Construction, Regularization & Hyperparameters

---

# 1. Gradient and Hessian in XGBoost

This is one of the biggest differences between traditional Gradient Boosting and XGBoost.

### Gradient Boosting

Traditional Gradient Boosting primarily uses the **first derivative** of the loss:

$$
g_i = \frac{\partial L(y_i,\hat y_i)}{\partial \hat y_i}
$$

The gradient tells us:

> **How should the prediction change to reduce the loss?**

---

### XGBoost

XGBoost uses both:

$$
\boxed{\text{Gradient + Hessian}}
$$

The gradient:

$$
g_i =
\frac{\partial L(y_i,\hat y_i)}
{\partial \hat y_i}
$$

The Hessian:

$$
h_i =
\frac{\partial^2 L(y_i,\hat y_i)}
{\partial \hat y_i^2}
$$

The Hessian tells us about the **curvature** of the loss.

A simple intuition:

```text
Gradient → direction of improvement
Hessian  → curvature / rate of change of gradient
```

---

# 2. Why Does XGBoost Need the Hessian?

Suppose you're standing on a hill and trying to reach the lowest point.

The gradient tells you:

> "Go this direction."

But it doesn't tell you very much about how sharply the landscape curves.

The Hessian provides that additional information.

Therefore:

```text
Gradient
   +
Hessian
   ↓
Better approximation of loss
   ↓
Better tree/leaf updates
```

This comes from a **second-order Taylor approximation**.

---

# 3. Taylor Expansion in XGBoost

Suppose the current prediction is:

$$
\hat y_i
$$

and the new tree gives:

$$
f_t(x_i)
$$

Then:

$$
\hat y_i^{(t)}
=
\hat y_i^{(t-1)}
+
f_t(x_i)
$$

The loss becomes:

$$
L(y_i,\hat y_i+f_t(x_i))
$$

XGBoost approximates this using Taylor expansion:

$$
L(y_i,\hat y_i+f_t(x_i))
\approx
L(y_i,\hat y_i)
+
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t^2(x_i)
$$

where:

$$
g_i = \frac{\partial L}{\partial \hat y_i}
$$

and

$$
h_i = \frac{\partial^2L}{\partial\hat y_i^2}
$$

The first term is the current loss.

The other terms tell XGBoost how the new tree will change the loss.

---

# 4. XGBoost Objective Function

The complete objective is:

$$
Obj =
\sum_{i=1}^{n}L(y_i,\hat y_i)
+
\sum_{k=1}^{K}\Omega(f_k)
$$

where:

### First part

$$
\sum L(y_i,\hat y_i)
$$

is the **training loss**.

### Second part

$$
\sum\Omega(f_k)
$$

is the **regularization penalty**.

Therefore:

$$
\boxed{
Objective = Training\ Loss + Model\ Complexity
}
$$

XGBoost tries to minimize this entire objective.

---

# 5. How XGBoost Builds a Tree

Now let's understand the actual process.

Suppose we have:

```text
Dataset
   ↓
Initial prediction
   ↓
Calculate Gradient
   ↓
Calculate Hessian
   ↓
Find possible splits
   ↓
Calculate gain
   ↓
Choose best split
   ↓
Create leaves
   ↓
Calculate optimal leaf weights
   ↓
Add tree to ensemble
```

This happens repeatedly.

---

# 6. Step 1 — Initial Prediction

Suppose we're doing regression.

Initially, the model might predict approximately the same value for every observation.

For example:

```text
Actual:

10
15
20
25

Initial prediction:

17.5
17.5
17.5
17.5
```

Then XGBoost calculates the loss for each observation.

---

# 7. Step 2 — Calculate Gradient and Hessian

For every observation, XGBoost calculates:

$$
g_i
$$

and

$$
h_i
$$

We can represent it as:

| Sample | Actual | Prediction | Gradient | Hessian |
| ------ | -----: | ---------: | -------: | ------: |
| 1      |     10 |       17.5 |  \(g_1\) | \(h_1\) |
| 2      |     15 |       17.5 |  \(g_2\) | \(h_2\) |
| 3      |     20 |       17.5 |  \(g_3\) | \(h_3\) |
| 4      |     25 |       17.5 |  \(g_4\) | \(h_4\) |

These values guide the construction of the next tree.

---

# 8. Step 3 — Consider a Split

Suppose we have:

```text
Feature = Age
```

Potential split:

$$
Age < 30
$$

This produces:

```text
             Age < 30?
              /     \
            Yes      No
             |        |
          Left      Right
```

Now XGBoost calculates the gradients and Hessians inside each child.

---

# 9. Gradient and Hessian Aggregation

For a leaf \(j\):

$$
G_j = \sum_{i\in I_j}g_i
$$

and:

$$
H_j = \sum_{i\in I_j}h_i
$$

where \(I_j\) represents observations belonging to that leaf.

So instead of looking at every observation independently, XGBoost aggregates:

```text
Individual gradients
        ↓
       G_j

Individual Hessians
        ↓
       H_j
```

These values are then used to calculate the optimal leaf weight and split gain.

---

# 10. Optimal Leaf Weight

The optimal weight of a leaf is:

$$
\boxed{
w_j^*
=
-\frac{G_j}{H_j+\lambda}
}
$$

where:

* \(G_j\) = sum of gradients
* \(H_j\) = sum of Hessians
* \(\lambda\) = L2 regularization

This equation is extremely important.

---

# 11. Simple Numerical Example

Suppose one leaf contains several observations.

After aggregation:

$$
G_j=20
$$

$$
H_j=10
$$

and:

$$
\lambda=2
$$

Then:

$$
w_j^*
=
-\frac{20}{10+2}
$$

$$
w_j^*
=
-\frac{20}{12}
$$

$$
\boxed{w_j^*=-1.667}
$$

So that leaf contributes approximately:

```text
-1.667
```

to the prediction before accounting for the learning rate.

---

# 12. Why the Negative Sign?

Remember that the gradient points toward increasing loss.

We want to move in the **opposite direction**.

Therefore:

$$
-\text{Gradient}
$$

is used.

That's why:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

has a negative sign.

---

# 13. How Does XGBoost Choose the Best Split?

This is where **split gain** comes in.

For every possible split, XGBoost asks:

> "How much will this split improve the objective?"

A commonly presented gain formula is:

$$
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
$$

where:

* \(G_L\) = gradient sum in left child
* \(H_L\) = Hessian sum in left child
* \(G_R\) = gradient sum in right child
* \(H_R\) = Hessian sum in right child
* \(G\) = gradient sum in parent
* \(H\) = Hessian sum in parent
* \(\lambda\) = L2 regularization
* \(\gamma\) = split penalty

---

# 14. Understanding the Gain Formula

Think of it as:

```text
Gain
 =
Benefit from left child
+
Benefit from right child
-
Benefit before split
-
Complexity penalty
```

So XGBoost doesn't simply ask:

> "Can I split?"

It asks:

> **"Is this split worth the additional complexity?"**

---

# 15. Example of Split Gain

Suppose:

$$
G_L=10,\quad H_L=5
$$

$$
G_R=20,\quad H_R=10
$$

Parent:

$$
G=30,\quad H=15
$$

and:

$$
\lambda=1
$$

Ignoring gamma temporarily:

$$
Gain =
\frac12
\left[
\frac{10^2}{5+1}
+
\frac{20^2}{10+1}
-
\frac{30^2}{15+1}
\right]
$$

$$
=
\frac12
\left[
16.67+36.36-56.25
\right]
$$

$$
\approx -1.61
$$

Negative gain means this split isn't beneficial under these values.

Therefore XGBoost would reject it.

---

# 16. What Does Gamma Do?

Now we introduce:

$$
\gamma
$$

In XGBoost:

```python
gamma
```

represents the minimum loss reduction required for a split.

If:

$$
Gain > \gamma
$$

the split is worthwhile.

Conceptually:

```text
Gain
  ↓
Is improvement large enough?
  ↓
Yes → Split
No  → Don't split
```

---

# 17. Effect of Increasing Gamma

Suppose:

```python
gamma = 0
```

Almost any positive improvement can justify a split.

But:

```python
gamma = 5
```

requires substantially more improvement.

Therefore:

```text
gamma ↑
   ↓
Harder to split
   ↓
Fewer leaves
   ↓
Simpler trees
   ↓
Lower model complexity
```

This can help reduce overfitting.

---

# 18. Regularization in XGBoost

This is one of the reasons XGBoost is so powerful.

XGBoost doesn't just optimize:

$$
Loss
$$

It optimizes:

$$
\boxed{
Loss + Complexity
}
$$

The regularization term is approximately:

$$
\Omega(f)
=
\gamma T
+
\frac12\lambda\sum_{j=1}^{T}w_j^2
$$

where:

* \(T\) = number of leaves
* \(\gamma\) = penalty for additional leaves
* \(\lambda\) = L2 regularization
* \(w_j\) = leaf weight

---

# 19. Two Main Types of Regularization

XGBoost supports:

### L1

```python
reg_alpha
```

### L2

```python
reg_lambda
```

And tree complexity is controlled by parameters such as:

```python
gamma
max_depth
min_child_weight
```

---

# 20. L1 Regularization — `reg_alpha`

L1 regularization adds a penalty based on the absolute magnitude of leaf weights.

Conceptually:

$$
\alpha\sum |w_j|
$$

where:

$$
\alpha = reg\_alpha
$$

L1 can encourage some weights toward zero.

Use it when you want stronger regularization and potentially a sparser model.

---

# 21. L2 Regularization — `reg_lambda`

L2 adds a squared penalty:

$$
\frac12\lambda\sum w_j^2
$$

where:

```python
reg_lambda
```

controls the strength.

Increasing it:

```text
reg_lambda ↑
       ↓
Larger leaf weights penalized more
       ↓
More conservative model
```

---

# 22. Why Does Lambda Appear in Leaf Weight?

Remember:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

If:

$$
\lambda=0
$$

then:

$$
w_j^*
=
-\frac{G_j}{H_j}
$$

But if:

$$
\lambda
$$

becomes larger, the denominator increases.

Therefore the magnitude of the leaf weight decreases.

Example:

$$
G=20,\ H=10
$$

Without regularization:

$$
w=-2
$$

With:

$$
\lambda=10
$$

$$
w=-\frac{20}{20}=-1
$$

So regularization makes the update more conservative.

---

# 23. Important XGBoost Hyperparameters

Now let's organize the parameters properly.

```text
XGBoost Parameters
│
├── Learning
│   ├── learning_rate
│   └── n_estimators
│
├── Tree Complexity
│   ├── max_depth
│   ├── min_child_weight
│   └── gamma
│
├── Sampling
│   ├── subsample
│   └── colsample_bytree
│
├── Regularization
│   ├── reg_alpha
│   └── reg_lambda
│
└── Objective / Performance
    ├── objective
    ├── eval_metric
    └── n_jobs
```

---

# 24. `n_estimators`

Number of boosting rounds / trees.

```python
n_estimators=300
```

means the model can build up to approximately 300 boosting trees.

Increasing:

```text
n_estimators ↑
      ↓
More opportunities to improve
      ↓
Higher model capacity
```

But too many trees can eventually overfit.

---

# 25. `learning_rate`

Controls how much each tree contributes.

Mathematically:

$$
F_t(x)
=
F_{t-1}(x)
+
\eta f_t(x)
$$

where:

$$
\eta = learning\_rate
$$

Example:

```python
learning_rate=0.1
```

means each tree's contribution is scaled down.

---

# 26. Learning Rate + Number of Trees

These parameters should be considered together.

Example:

```python
learning_rate=0.1
n_estimators=100
```

versus:

```python
learning_rate=0.03
n_estimators=500
```

The second model learns more gradually.

General principle:

$$
\boxed{
learning\_rate\downarrow
\Rightarrow
n\_estimators\uparrow
}
$$

This isn't a strict mathematical rule, but it's a useful practical tuning strategy.

---

# 27. `max_depth`

Controls maximum depth of each tree.

Example:

```python
max_depth=3
```

produces relatively shallow trees.

Increasing it:

```text
max_depth ↑
      ↓
More complex trees
      ↓
More interactions captured
      ↓
Higher overfitting risk
```

For XGBoost, shallow trees are often effective because boosting combines many of them.

---

# 28. `min_child_weight`

Controls the minimum amount of Hessian/instance weight required for a child node.

Higher:

```text
min_child_weight ↑
        ↓
Harder to create small child nodes
        ↓
Fewer splits
        ↓
More conservative model
```

If your model is heavily overfitting, increasing this parameter can help.

---

# 29. `subsample`

Controls the fraction of training rows used for each boosting round.

Example:

```python
subsample=0.8
```

Approximately 80% of training samples are used for each tree.

```text
subsample = 1.0
       ↓
All rows

subsample = 0.8
       ↓
~80% rows

subsample = 0.6
       ↓
~60% rows
```

This introduces randomness and can reduce overfitting.

---

# 30. `colsample_bytree`

Controls the fraction of features considered for each tree.

Example:

```python
colsample_bytree=0.8
```

means roughly 80% of features are sampled for each tree.

This is conceptually similar to feature subsampling in Random Forest.

---

# 31. `gamma`

We already saw this mathematically.

```python
gamma
```

controls the minimum loss reduction required to make a split.

Increasing:

```text
gamma ↑
   ↓
Splits need stronger evidence
   ↓
Simpler trees
```

---

# 32. `reg_alpha`

L1 regularization.

```python
reg_alpha=0.1
```

Increasing it increases the L1 penalty.

Useful when you want stronger regularization.

---

# 33. `reg_lambda`

L2 regularization.

```python
reg_lambda=5
```

Increasing it penalizes large leaf weights more strongly.

The default behavior already includes L2 regularization, so you don't need to set it to a nonzero value just to "turn regularization on."

---

# 34. Parameter Effects Cheat Sheet

| Parameter          | Increase it →              |
| ------------------ | -------------------------- |
| `n_estimators`     | More trees / capacity      |
| `learning_rate`    | Larger updates             |
| `max_depth`        | More complex trees         |
| `min_child_weight` | More conservative splits   |
| `gamma`            | Fewer splits               |
| `subsample`        | More data per tree         |
| `colsample_bytree` | More features per tree     |
| `reg_alpha`        | Stronger L1 regularization |
| `reg_lambda`       | Stronger L2 regularization |

---

# 35. How Parameters Control Overfitting

Suppose:

```text
Training R² = 0.99
Validation R² = 0.70
```

This suggests overfitting.

Possible actions:

```text
max_depth ↓
min_child_weight ↑
gamma ↑
subsample ↓
colsample_bytree ↓
reg_alpha ↑
reg_lambda ↑
```

You can also reduce model capacity by lowering the effective number of trees or using early stopping.

---

# 36. If the Model Is Underfitting

Suppose:

```text
Training R² = 0.65
Validation R² = 0.60
```

The model may be too simple.

Possible actions:

```text
max_depth ↑
n_estimators ↑
learning_rate ↑
min_child_weight ↓
gamma ↓
```

But don't blindly increase everything.

The goal is to find the right bias-variance balance.

---

# 37. Complete XGBRegressor Example

Here's a good baseline for your California Housing experiments.

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Then:

```python
from sklearn.metrics import r2_score, mean_squared_error

print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# 38. Complete XGBClassifier Example

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Probability:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

---

# 39. The Entire XGBoost Algorithm

Now put everything together.

### Step 1

Start with an initial prediction.

### Step 2

Calculate loss.

### Step 3

Calculate:

$$
g_i
$$

and:

$$
h_i
$$

### Step 4

Consider possible tree splits.

### Step 5

For every candidate split, calculate:

$$
G_L,H_L,G_R,H_R
$$

### Step 6

Calculate split gain:

$$
Gain =
\frac12
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
$$

### Step 7

Choose the best useful split.

### Step 8

Calculate leaf weights:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

### Step 9

Build the tree.

### Step 10

Scale the tree contribution:

$$
\eta f_t(x)
$$

where:

$$
\eta = learning\_rate
$$

### Step 11

Update predictions.

### Step 12

Repeat for the next boosting round.

---

# 40. Complete Mental Model

You should be able to visualize XGBoost like this:

```text
                    XGBoost
                       │
                       ↓
                Initial Prediction
                       │
                       ↓
                 Calculate Loss
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
          Gradient             Hessian
             │                   │
             └─────────┬─────────┘
                       ↓
                Candidate Splits
                       │
                       ↓
                  Calculate Gain
                       │
              ┌────────┴────────┐
              ↓                 ↓
         Gain sufficient?    Gain insufficient?
              │                 │
             YES                NO
              ↓                 ↓
         Create Split       Reject Split
              │
              ↓
          Create Leaves
              │
              ↓
       Calculate Leaf Weights
              │
              ↓
       Apply Regularization
              │
              ↓
        Apply Learning Rate
              │
              ↓
        Update Predictions
              │
              ↓
        Next Boosting Round
```

---

# 41. The 4 Concepts You Must Remember

If you're asked in an interview:

> **How does XGBoost improve Gradient Boosting?**

Your answer should revolve around these four things:

### 1. Second-order optimization

Uses:

$$
Gradient + Hessian
$$

instead of relying only on the gradient.

### 2. Regularization

Uses:

$$
L1 + L2 + tree\ complexity
$$

to control overfitting.

### 3. Shrinkage

Uses:

$$
learning\_rate
$$

to make each tree's contribution smaller.

### 4. Subsampling

Uses:

```text
subsample
colsample_bytree
```

to introduce randomness and improve generalization.

---

## One-line memory trick

Remember XGBoost as:

$$
\boxed{
XGBoost =
Gradient + Hessian + Trees + Regularization + Shrinkage + Sampling
}
$$

And the three equations worth memorizing are:

$$
\boxed{
w_j^*=-\frac{G_j}{H_j+\lambda}
}
$$

$$
\boxed{
Gain=
\frac12
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
}
$$

and:

$$
\boxed{
Objective = Loss + Regularization
}
$$

These three equations give you most of the mathematical intuition behind **how XGBoost actually constructs and regularizes its trees**.
