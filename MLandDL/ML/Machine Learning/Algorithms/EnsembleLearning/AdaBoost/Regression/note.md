# AdaBoost Regressor — Detailed Notes

## 1. What is AdaBoost?

**AdaBoost = Adaptive Boosting**

AdaBoost is a **boosting ensemble algorithm** that combines many **weak learners** to create a stronger regression model.

For regression:

```text
Weak Learner 1
      ↓
Weak Learner 2
      ↓
Weak Learner 3
      ↓
      ...
      ↓
Weak Learner N
      ↓
Weighted combination
      ↓
Final prediction
```

The key idea is:

> **Build models sequentially, where later models focus more on the observations that previous models predicted poorly.**

---

# 2. What is Boosting?

Before AdaBoost, understand the general idea of boosting.

There are two major ensemble approaches you've already encountered:

### Bagging

Models are generally trained **independently/in parallel**.

```text
             ┌── Model 1
             ├── Model 2
Data ────────┼── Model 3
             ├── Model 4
             └── Model 5
                    ↓
                 Average
```

Example:

**Random Forest**

---

### Boosting

Models are trained **sequentially**.

```text
Data
 ↓
Model 1
 ↓
Errors
 ↓
Model 2 focuses on errors
 ↓
Errors
 ↓
Model 3 focuses on errors
 ↓
...
 ↓
Final model
```

Examples:

* AdaBoost
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

---

# 3. What is a Weak Learner?

A **weak learner** is a model that performs only slightly better than random guessing/baseline but can contribute useful information when combined with other learners.

For AdaBoost, decision trees are commonly used as weak learners.

For example:

```python
DecisionTreeRegressor(max_depth=1)
```

This is called a **decision stump**.

It might look like:

```text
             Feature 3 < 5?
                 /     \
               Yes      No
               /         \
             10           20
```

It's a very simple model.

Individually, it may not perform very well.

But AdaBoost combines many of them.

---

# 4. Basic Idea of AdaBoost Regressor

Suppose you have:

```text
100 training samples
```

Initially, every sample gets equal importance.

```text
Sample:    1  2  3  4  5  ... 100
Weight:   .01 .01 .01 .01 .01 ... .01
```

Train the first weak learner.

Some samples will be predicted well:

```text
Sample 1 → good prediction
Sample 2 → good prediction
Sample 3 → bad prediction
Sample 4 → good prediction
Sample 5 → bad prediction
```

AdaBoost gives more attention to the difficult samples.

Conceptually:

```text
Before:

Sample 1 → 0.01
Sample 2 → 0.01
Sample 3 → 0.01
Sample 4 → 0.01
Sample 5 → 0.01

After Model 1:

Sample 1 → 0.005
Sample 2 → 0.005
Sample 3 → 0.025  ← harder
Sample 4 → 0.005
Sample 5 → 0.025  ← harder
```

Then the next learner focuses more on those difficult observations.

---

# 5. AdaBoost Regressor Workflow

The simplified workflow is:

```text
                 Training Data
                      ↓
            Assign equal weights
                      ↓
                Train Model 1
                      ↓
               Calculate errors
                      ↓
        Increase importance of difficult
                 observations
                      ↓
                Train Model 2
                      ↓
               Calculate errors
                      ↓
        Increase importance of difficult
                 observations
                      ↓
                    ...
                      ↓
               Train Model N
                      ↓
           Weighted combination
                      ↓
              Final prediction
```

---

# 6. How the Algorithm Works

Let's understand the process more carefully.

Suppose:

```text
X = features
y = target
```

and we have:

```text
n_estimators = 3
```

So we're going to create:

```text
Tree 1
Tree 2
Tree 3
```

---

## Step 1 — Initialize sample weights

Initially, all training observations have equal weights.

If there are `N` observations:

[
w_i = \frac{1}{N}
]

For 5 observations:

```text
w1 = 0.2
w2 = 0.2
w3 = 0.2
w4 = 0.2
w5 = 0.2
```

---

# 7. Train the First Weak Learner

Train a weak regression tree:

```python
DecisionTreeRegressor(max_depth=1)
```

It produces:

```text
Actual    Prediction
  10          11
  20          18
  30          31
  40          25
  50          48
```

Now we calculate how well the model performed.

---

# 8. Calculate Errors

For regression, we have continuous prediction errors.

A simple absolute error is:

[
e_i = |y_i - \hat{y}_i|
]

For example:

```text
Actual = 40
Prediction = 25

Error = |40 - 25|
      = 15
```

AdaBoost uses a normalized error measure internally to determine how well the weak learner performed.

The important concept is:

> **Samples with larger errors receive more attention in later iterations.**

---

# 9. Increase Attention on Difficult Samples

Suppose:

```text
Sample 1 → predicted well
Sample 2 → predicted well
Sample 3 → predicted poorly
Sample 4 → predicted well
Sample 5 → predicted poorly
```

The next learner focuses more on:

```text
Sample 3
Sample 5
```

So:

```text
Model 1
   ↓
Find difficult observations
   ↓
Increase their influence
   ↓
Model 2
```

This is where the **"adaptive"** part of AdaBoost comes from.

---

# 10. Train the Next Weak Learner

Now train another weak learner using the updated sample weights.

```text
Model 1
    ↓
Errors
    ↓
Update weights
    ↓
Model 2
    ↓
Errors
    ↓
Update weights
    ↓
Model 3
```

Each new learner attempts to improve the ensemble's performance.

---

# 11. Combine the Learners

At the end, we have:

```text
Tree 1
Tree 2
Tree 3
...
Tree N
```

The final prediction is a **weighted combination** of the individual learners.

Conceptually:

[
F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)
]

where:

* (h_m(x)) = prediction of weak learner (m)
* (\alpha_m) = weight given to that learner
* (M) = number of learners

The exact implementation details of `AdaBoostRegressor` use a regression-specific error/weighting procedure, so don't assume it is identical to classification AdaBoost.

---

# 12. Important Parameters

The main parameters you'll tune are:

```python
AdaBoostRegressor(
    estimator=...,
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',
    random_state=42
)
```

Let's understand them.

---

## `estimator`

This is the weak learner.

Example:

```python
DecisionTreeRegressor(max_depth=1)
```

So:

```python
AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1)
)
```

means:

> Use decision trees as the weak learners.

In current scikit-learn versions, the parameter is called:

```python
estimator
```

Older versions used:

```python
base_estimator
```

---

# 13. `n_estimators`

Controls the number of weak learners.

Example:

```python
n_estimators=50
```

means:

```text
Tree 1
Tree 2
Tree 3
...
Tree 50
```

Increasing it can allow the ensemble to learn more complex relationships.

For example:

```python
n_estimators = [50, 100, 200, 300]
```

### Too low

```text
Too few learners
      ↓
Underfitting
```

### Too high

```text
Too many learners
      ↓
More training time
      ↓
Potential overfitting
```

The effect also depends strongly on `learning_rate` and the base estimator.

---

# 14. `learning_rate`

This controls how much each new weak learner contributes to the final model.

Typical values:

```python
learning_rate = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
```

Conceptually:

```text
Small learning rate
       ↓
Each learner has smaller influence
       ↓
Usually need more estimators
```

while:

```text
Large learning rate
       ↓
Each learner has greater influence
       ↓
May need fewer estimators
```

There is generally a trade-off:

```text
learning_rate ↓
       ↕
n_estimators ↑
```

For example:

```text
learning_rate = 0.1
n_estimators = 300
```

can sometimes work similarly to:

```text
learning_rate = 0.5
n_estimators = 100
```

but they are **not equivalent** and should be validated rather than assumed interchangeable.

---

# 15. `loss`

This is particularly important for **AdaBoostRegressor**.

Scikit-learn provides:

```python
loss = [
    'linear',
    'square',
    'exponential'
]
```

### `linear`

Uses a linear loss.

This is the default.

```python
loss='linear'
```

### `square`

Penalizes larger errors more strongly.

```python
loss='square'
```

Large errors receive substantially more penalty.

### `exponential`

Uses exponential loss and puts even stronger emphasis on large errors.

```python
loss='exponential'
```

You can tune it:

```python
params_ada = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.5, 1.0],
    'loss': ['linear', 'square', 'exponential']
}
```

---

# 16. `random_state`

Controls reproducibility.

```python
random_state=42
```

If you run the same experiment again, you can reproduce the same random behavior.

It's good practice to set it while learning and experimenting.

---

# 17. Base Tree Depth

This is very important.

Consider:

```python
DecisionTreeRegressor(max_depth=1)
```

This is a stump.

```text
Depth 1
    ↓
Very weak learner
```

You can increase it:

```python
max_depth=2
```

or:

```python
max_depth=3
```

Now each learner is stronger.

### The trade-off

```text
max_depth ↑
      ↓
individual trees become stronger
      ↓
model becomes more complex
      ↓
potential overfitting
```

So AdaBoost has **two levels of hyperparameters**:

```text
AdaBoost parameters
    │
    ├── n_estimators
    ├── learning_rate
    └── loss
             +
Base Decision Tree parameters
    │
    ├── max_depth
    ├── min_samples_split
    ├── min_samples_leaf
    └── ccp_alpha
```

---

# 18. Example AdaBoost Regressor

Basic version:

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

base_tree = DecisionTreeRegressor(
    max_depth=1,
    random_state=42
)

model = AdaBoostRegressor(
    estimator=base_tree,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

---

# 19. Evaluate with R²

For regression, you can use:

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print(r2)
```

Remember:

[
R^2 = 1-\frac{SS_{res}}{SS_{tot}}
]

R² interpretation:

```text
R² = 1.0   → perfect
R² = 0.8   → strong
R² = 0.5   → moderate
R² = 0     → no improvement over baseline mean
R² < 0     → worse than predicting the mean
```

These are rough interpretations, not universal thresholds.

---

# 20. AdaBoost with RandomizedSearchCV

This is what you were doing.

For example:

```python
from sklearn.model_selection import RandomizedSearchCV

params_ada = {
    'n_estimators': [50, 100, 150, 250, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.5, 1.0],
    'loss': ['linear', 'square', 'exponential']
}
```

Base model:

```python
base_tree = DecisionTreeRegressor(
    max_depth=1,
    random_state=42
)

ada = AdaBoostRegressor(
    estimator=base_tree,
    random_state=42
)
```

Then:

```python
random_search = RandomizedSearchCV(
    estimator=ada,
    param_distributions=params_ada,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

random_search.fit(x_train, y_train)
```

Then:

```python
print(random_search.best_params_)
print(random_search.best_score_)
```

---

# 21. Tuning the Base Tree Too

This is more advanced and important for your situation.

You can tune the AdaBoost parameters **and** the underlying decision tree.

```python
base_tree = DecisionTreeRegressor(
    random_state=42
)

ada = AdaBoostRegressor(
    estimator=base_tree,
    random_state=42
)
```

Parameter grid:

```python
params_ada = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.5],
    'loss': ['linear', 'square', 'exponential'],
    'estimator__max_depth': [1, 2, 3],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4]
}
```

Notice:

```python
'estimator__max_depth'
```

The double underscore means:

```text
AdaBoost
   ↓
estimator
   ↓
DecisionTreeRegressor
   ↓
max_depth
```

---

# 22. Why Your Decision Tree Beat AdaBoost

You recently got approximately:

```text
Decision Tree R²  → 0.67
AdaBoost R²       → 0.35
```

This makes sense given your configuration.

You used:

```python
DecisionTreeRegressor(
    max_depth=1,
    ...
)
```

as the AdaBoost base learner.

That means your AdaBoost was using extremely weak trees.

Meanwhile, your standalone Decision Tree was allowed to learn a much more complex structure.

So:

```text
Standalone Tree
       ↓
strong learner
       ↓
R² ≈ 0.67
```

versus:

```text
AdaBoost
   ↓
many depth-1 trees
   ↓
weak learners
   ↓
R² ≈ 0.35
```

This does **not** mean AdaBoost is inherently worse.

It means the particular AdaBoost configuration was not suitable or sufficiently tuned for your dataset.

---

# 23. AdaBoost vs Decision Tree

| Property                        | Decision Tree         | AdaBoost                         |
| ------------------------------- | --------------------- | -------------------------------- |
| Number of models                | 1                     | Many                             |
| Training                        | Single tree           | Sequential learners              |
| Main idea                       | Recursive splitting   | Correct previous errors          |
| Complexity                      | Depends on tree depth | Depends on learners + estimators |
| Overfitting control             | Depth/pruning         | Learning rate + weak learners    |
| Training speed                  | Usually faster        | Usually slower                   |
| Interpretability                | High                  | Lower                            |
| Handles nonlinear relationships | ✅                     | ✅                                |
| Regression                      | ✅                     | ✅                                |

---

# 24. AdaBoost vs Bagging

This distinction is very important because you've already studied Bagging.

### Bagging

```text
             Dataset
                ↓
      ┌─────────┼─────────┐
      ↓         ↓         ↓
   Tree 1    Tree 2    Tree 3
      ↓         ↓         ↓
      └─────────┼─────────┘
                ↓
              Average
```

Models are mostly independent.

Example:

**Random Forest**

---

### AdaBoost

```text
Dataset
   ↓
Tree 1
   ↓
Errors
   ↓
Tree 2 focuses on difficult observations
   ↓
Errors
   ↓
Tree 3
   ↓
...
   ↓
Weighted combination
```

Models are sequentially related.

---

# 25. Advantages of AdaBoost Regressor

### 1. Can improve weak learners

Instead of one weak model:

```text
weak model → mediocre performance
```

AdaBoost combines many:

```text
weak + weak + weak + ...
        ↓
stronger ensemble
```

### 2. Captures nonlinear relationships

Decision-tree-based AdaBoost can model nonlinear patterns.

### 3. Doesn't require feature scaling

Tree-based models generally don't require:

```python
StandardScaler()
```

for numerical features.

### 4. Relatively simple

The API is straightforward:

```python
AdaBoostRegressor(...)
```

---

# 26. Disadvantages

### 1. Sensitive to noisy data

Because boosting focuses on difficult observations, extremely noisy observations can receive too much attention.

```text
Noise/outlier
      ↓
large error
      ↓
more attention
      ↓
future learners focus on it
```

This can hurt performance.

### 2. Can overfit

Especially with:

```text
too many estimators
+
high learning rate
+
complex base learners
```

### 3. Can be slower than one tree

You're training many models instead of one.

### 4. Not always the best boosting algorithm

For tabular regression, algorithms such as:

* Gradient Boosting
* HistGradientBoosting
* XGBoost
* LightGBM
* CatBoost

can often be stronger depending on the dataset.

---

# 27. Important Hyperparameter Relationships

Remember this:

### `n_estimators`

```text
↑
more weak learners
↑
complexity
↑
training time
```

### `learning_rate`

```text
↑
each learner has more influence
↑
model can learn faster
↑
potential overfitting
```

### `max_depth`

```text
↑
each weak learner becomes stronger
↑
model complexity
↑
potential overfitting
```

So:

```text
High learning_rate
+
High max_depth
+
High n_estimators
        ↓
Potential overfitting
```

Whereas:

```text
Very low learning_rate
+
Very few estimators
+
depth=1
        ↓
Potential underfitting
```

---

# 28. A Good Starting Search

For your current learning:

```python
params_ada = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2, 0.5, 1.0],
    'loss': ['linear', 'square', 'exponential'],
    'estimator__max_depth': [1, 2, 3]
}
```

Then use:

```python
RandomizedSearchCV(
    ada,
    params_ada,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)
```

This is preferable to blindly trying every combination if the search space becomes large.

---

# 29. The Most Important Concept to Remember

If you remember only one thing about AdaBoost:

> **AdaBoost builds weak learners sequentially and gives subsequent learners more influence from observations that previous learners handled poorly, then combines the learners into a final prediction.**

For regression:

```text
Data
 ↓
Equal/initial sample influence
 ↓
Weak regression tree
 ↓
Measure prediction errors
 ↓
Emphasize difficult observations
 ↓
Next weak tree
 ↓
Repeat
 ↓
Weighted ensemble
 ↓
Final regression prediction
```

And the three parameters you should remember first are:

```text
n_estimators
      ↓
How many learners?

learning_rate
      ↓
How strongly does each learner contribute?

estimator
      ↓
What weak learner are we using?
```

For your current stage, I'd focus on those three first, then learn `loss` and tuning the underlying `DecisionTreeRegressor`.
