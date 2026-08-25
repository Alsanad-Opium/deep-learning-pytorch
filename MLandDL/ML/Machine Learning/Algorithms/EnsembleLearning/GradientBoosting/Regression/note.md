# Gradient Boosting Regressor — Complete Notes

Since you just went through the mathematical working of **GradientBoostingClassifier**, let's do **GradientBoostingRegressor** in exactly the same style.

The key idea is:

> **Gradient Boosting Regressor builds decision trees sequentially, where each new tree tries to reduce the loss made by the previous ensemble by learning the negative gradient of the loss function.**

---

# 1. What is Gradient Boosting Regressor?

`GradientBoostingRegressor` is a **supervised ensemble learning algorithm** used for regression problems.

Instead of training one large decision tree, it builds many smaller trees sequentially:

```text
Tree 1
  ↓
Tree 2 corrects Tree 1
  ↓
Tree 3 corrects previous trees
  ↓
Tree 4 corrects previous trees
  ↓
...
Final prediction
```

Each new tree is trained to improve the predictions made by the existing ensemble.

---

# 2. Why is it called "Gradient Boosting"?

There are two words:

### Boosting

Trees are built **sequentially**, with each new tree improving the previous model.

### Gradient

The new tree is trained using the **negative gradient of the loss function**.

So:

[
\boxed{\text{Gradient Boosting}=\text{Sequential learning + Gradient-based error correction}}
]

---

# 3. Basic regression problem

Suppose we have:

[
X = \text{input features}
]

and:

[
y = \text{continuous target}
]

For example:

```text
House features → House price
```

We want:

[
\hat y = F(X)
]

where (F(X)) is the final prediction.

---

# 4. The fundamental idea

Gradient Boosting constructs the model as an additive combination of trees:

[
\boxed{
F_M(X)=F_0(X)+\eta\sum_{m=1}^{M}f_m(X)
}
]

where:

* (F_0(X)) = initial prediction
* (f_m(X)) = the (m^{th}) decision tree
* (M) = number of trees
* (\eta) = learning rate

So instead of:

```text
One giant tree
```

we have:

```text
Small tree + small tree + small tree + ...
```

---

# 5. Step 1 — Initial prediction

The first thing Gradient Boosting does is create an initial prediction.

For **squared error loss**, the initial prediction is the **mean of the target values**.

Suppose:

[
y=[10,20,30,40,50]
]

Then:

[
F_0=\frac{10+20+30+40+50}{5}
]

[
F_0=30
]

Therefore, initially:

```text
Every prediction = 30
```

So:

| Actual | Initial prediction |
| -----: | -----------------: |
|     10 |                 30 |
|     20 |                 30 |
|     30 |                 30 |
|     40 |                 30 |
|     50 |                 30 |

Obviously, this isn't very good.

So Gradient Boosting starts building trees to correct these errors.

---

# 6. Step 2 — Calculate the loss

For the default regression loss:

```python
loss='squared_error'
```

the loss for one observation is:

[
L(y,F)=\frac{1}{2}(y-F)^2
]

The factor (1/2) is mathematically convenient because it disappears when taking the derivative.

For the whole dataset:

[
L=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{2}(y_i-F(x_i))^2
]

Our objective is:

[
\boxed{\text{Minimize squared error}}
]

---

# 7. Step 3 — Calculate the gradient

This is where **Gradient Boosting** gets its name.

Our loss is:

[
L_i=\frac{1}{2}(y_i-F(x_i))^2
]

Take the derivative with respect to the current prediction (F(x_i)):

[
\frac{\partial L_i}{\partial F(x_i)}
====================================

F(x_i)-y_i
]

But Gradient Boosting wants the **negative gradient**:

[
-\frac{\partial L_i}{\partial F(x_i)}
]

Therefore:

[
\boxed{
r_i=y_i-F(x_i)
}
]

This is the **pseudo-residual**.

And in the case of squared error:

> **The negative gradient is exactly the ordinary residual.**

That's why Gradient Boosting Regression is often explained using residuals.

---

# 8. Example

Suppose:

| Actual (y) | Prediction (F(x)) |
| ---------: | ----------------: |
|         10 |                30 |
|         20 |                30 |
|         30 |                30 |
|         40 |                30 |
|         50 |                30 |

Calculate:

[
r_i=y_i-F(x_i)
]

### Observation 1

[
r=10-30=-20
]

### Observation 2

[
r=20-30=-10
]

### Observation 3

[
r=30-30=0
]

### Observation 4

[
r=40-30=10
]

### Observation 5

[
r=50-30=20
]

Therefore:

| Actual | Prediction | Residual |
| -----: | ---------: | -------: |
|     10 |         30 |      -20 |
|     20 |         30 |      -10 |
|     30 |         30 |        0 |
|     40 |         30 |      +10 |
|     50 |         30 |      +20 |

The model needs to:

```text
Decrease predictions for samples with negative residuals
Increase predictions for samples with positive residuals
```

---

# 9. Step 4 — Train a decision tree on the residuals

Now we train a **DecisionTreeRegressor**.

Important:

> The tree is not initially trying to predict the original target (y).

Instead, it tries to learn:

[
X \rightarrow r
]

where:

[
r=y-F(X)
]

So:

```text
X
↓
Pseudo-residual
↓
Decision Tree
```

For example:

```text
House features
       ↓
Tree
       ↓
Residual correction
```

---

# 10. Step 5 — Add the new tree

Suppose the first tree predicts a correction:

[
f_1(X)
]

We update the model:

[
\boxed{
F_1(X)=F_0(X)+\eta f_1(X)
}
]

where:

[
\eta=\text{learning rate}
]

---

# 11. Example of the update

Suppose:

[
F_0=30
]

and the first tree predicts:

[
f_1(X)=10
]

with:

[
\eta=0.1
]

Then:

[
F_1=30+(0.1)(10)
]

[
F_1=31
]

So instead of jumping from 30 to 40, the model only moves:

```text
30 → 31
```

This controlled update helps prevent overfitting.

---

# 12. Step 6 — Calculate new residuals

Now the model has improved.

Suppose:

```text
Actual = 40
Old prediction = 30
```

Residual was:

[
40-30=10
]

After the tree:

[
New prediction=31
]

New residual:

[
40-31=9
]

The error became smaller.

Now another tree is trained to correct the remaining errors.

---

# 13. The complete boosting process

The process becomes:

```text
                 Training Data
                      ↓
              Initial prediction
                      ↓
                Calculate loss
                      ↓
             Calculate gradient
                      ↓
              Negative gradient
                      ↓
               Train Tree 1
                      ↓
              Update prediction
                      ↓
             Calculate new error
                      ↓
               Train Tree 2
                      ↓
              Update prediction
                      ↓
             Calculate new error
                      ↓
                     ...
                      ↓
               Train Tree M
                      ↓
                Final prediction
```

---

# 14. Mathematical algorithm

Let's write the entire algorithm mathematically.

### Step 1 — Initialize

For squared error:

[
\boxed{
F_0(x)=\frac{1}{N}\sum_{i=1}^{N}y_i
}
]

So the initial prediction is the mean.

---

### Step 2 — Calculate negative gradient

At iteration (m):

[
g_{im}
======

-\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]*{F=F*{m-1}}
]

For squared error:

[
L(y,F)=\frac{1}{2}(y-F)^2
]

therefore:

[
\boxed{
g_{im}=y_i-F_{m-1}(x_i)
}
]

This is the residual.

---

### Step 3 — Fit a tree

Fit a regression tree:

[
f_m(x)\approx g_{im}
]

The tree learns the pattern in the residuals.

---

### Step 4 — Update the model

[
\boxed{
F_m(x)=F_{m-1}(x)+\eta f_m(x)
}
]

---

### Step 5 — Repeat

Repeat this for:

[
m=1,2,\ldots,M
]

Final model:

[
\boxed{
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}f_m(x)
}
]

---

# 15. Why does Gradient Boosting use many trees?

Imagine the first tree makes this prediction:

```text
Actual = 100
Prediction = 70
```

Error:

[
30
]

Tree 2 tries to correct that.

Suppose it adds:

[
20
]

With learning rate:

[
\eta=0.1
]

the actual correction is only:

[
2
]

So:

[
70\rightarrow72
]

Then another tree might add:

[
1.5
]

Then:

[
72\rightarrow73.5
]

And so on.

Conceptually:

```text
70
 ↓
72
 ↓
73.5
 ↓
75
 ↓
...
 ↓
98
 ↓
99
 ↓
100
```

The model gradually approaches the correct prediction.

---

# 16. Learning rate

The learning rate controls how much each tree contributes.

The update is:

[
F_m=F_{m-1}+\eta f_m
]

### Small learning rate

```python
learning_rate=0.01
```

Each tree makes a tiny correction.

Usually requires more trees.

### Larger learning rate

```python
learning_rate=0.2
```

Each tree makes a larger correction.

Usually requires fewer trees.

This creates an important relationship:

[
\boxed{
\text{learning rate} \downarrow
\Rightarrow
\text{usually n_estimators} \uparrow
}
]

---

# 17. `n_estimators`

This represents the number of boosting stages.

```python
n_estimators=100
```

means approximately:

```text
Tree 1
Tree 2
Tree 3
...
Tree 100
```

More trees give the model more opportunities to correct errors.

But too many trees can cause overfitting, especially with a large learning rate or complex trees.

---

# 18. `max_depth`

This controls the depth of the individual decision trees.

For example:

```python
max_depth=2
```

creates shallow trees.

```python
max_depth=8
```

allows much more complex trees.

For Gradient Boosting, shallow trees are often useful because:

> **We want many weak learners making small corrections rather than a few extremely powerful trees.**

This is why your earlier experiment with:

```python
max_depth=8, 10, 12, ..., 20
```

was potentially problematic.

Your default model uses a much smaller tree depth.

---

# 19. Different loss functions

This is an important part of GradientBoostingRegressor.

You were experimenting with:

```python
loss='squared_error'
```

and:

```python
loss='absolute_error'
```

The loss function determines **what error the gradient is trying to reduce**.

---

## A. Squared Error

[
L(y,F)=\frac{1}{2}(y-F)^2
]

Gradient:

[
\frac{\partial L}{\partial F}=F-y
]

Negative gradient:

[
\boxed{y-F}
]

Therefore:

> With squared error, Gradient Boosting learns the ordinary residuals.

This is the standard/default loss.

---

# 20. Absolute Error

Absolute error:

[
L(y,F)=|y-F|
]

Unlike squared error, it does not heavily punish large errors.

For example:

```text
Error = 2
Squared loss → 4
Absolute loss → 2

Error = 10
Squared loss → 100
Absolute loss → 10
```

Therefore:

> **Squared error is more sensitive to outliers.**

Absolute error is generally more robust to outliers.

Its derivative involves the sign of the residual:

[
\frac{\partial L}{\partial F}
=============================

\begin{cases}
-1 & y>F\
+1 & y<F
\end{cases}
]

So the negative gradient is approximately:

[
\operatorname{sign}(y-F)
]

---

# 21. Huber Loss

Huber loss combines the advantages of squared and absolute error.

For a residual (r=y-F):

[
L_\delta(r)=
\begin{cases}
\frac{1}{2}r^2 & |r|\leq\delta\
\delta(|r|-\frac{1}{2}\delta) & |r|>\delta
\end{cases}
]

So:

```text
Small errors
     ↓
Squared error behavior

Large errors
     ↓
Absolute error behavior
```

This makes Huber loss useful when you have **some outliers but don't want to completely ignore them**.

---

# 22. Quantile loss

Quantile loss is used when you're interested in predicting a particular quantile rather than simply the conditional mean.

For example:

```text
50th percentile → median
90th percentile → upper prediction level
```

It is useful for things such as:

```text
Prediction intervals
Risk estimation
Demand forecasting
```

---

# 23. Gradient Boosting vs Random Forest

Since you've already studied Random Forest, this distinction is important.

### Random Forest

Trees are generally trained **independently**.

```text
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┤→ Average
Tree 4 ──┤
Tree 5 ──┘
```

### Gradient Boosting

Trees are trained **sequentially**.

```text
Tree 1
  ↓
Tree 2
  ↓
Tree 3
  ↓
Tree 4
  ↓
Final model
```

So:

| Random Forest                     | Gradient Boosting                                  |
| --------------------------------- | -------------------------------------------------- |
| Parallel/independent trees        | Sequential trees                                   |
| Bagging                           | Boosting                                           |
| Reduces variance                  | Primarily reduces bias, while controlling variance |
| Random subsets of data/features   | Each tree corrects previous errors                 |
| Usually harder to overfit         | Can overfit if too complex                         |
| Less sensitive to hyperparameters | More sensitive to hyperparameters                  |

---

# 24. Gradient Boosting vs AdaBoost

Since you've also studied AdaBoost:

### AdaBoost

Focuses on difficult observations by changing their weights.

```text
Wrong samples
     ↓
Higher weights
     ↓
Next tree focuses on them
```

### Gradient Boosting

Calculates:

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Negative gradient
    ↓
New tree learns correction
```

So the core difference is:

[
\boxed{\text{AdaBoost → reweight samples}}
]

[
\boxed{\text{Gradient Boosting → follow the loss gradient}}
]

---

# 25. The complete mathematical picture

This is the most important section for your notes.

We start with:

[
F_0(x)
]

Then for each iteration (m):

### Calculate current prediction

[
\hat y_i=F_{m-1}(x_i)
]

### Calculate loss

[
L(y_i,F_{m-1}(x_i))
]

### Calculate negative gradient

[
\boxed{
r_{im}
======

-\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\bigg|*{F=F*{m-1}}
}
]

### Train a tree

[
f_m(x)\approx r_{im}
]

### Update

[
\boxed{
F_m(x)=F_{m-1}(x)+\eta f_m(x)
}
]

Finally:

[
\boxed{
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}f_m(x)
}
]

For **squared error**, this simplifies to:

[
\boxed{
r_{im}=y_i-F_{m-1}(x_i)
}
]

which means:

> **The next tree learns the residuals left by the current model.**

---

# 26. One complete numerical example

Suppose:

[
y=[10,20,30,40]
]

### Initial prediction

Mean:

[
F_0=\frac{10+20+30+40}{4}=25
]

So:

| Actual | Prediction |
| -----: | ---------: |
|     10 |         25 |
|     20 |         25 |
|     30 |         25 |
|     40 |         25 |

Residuals:

[
[-15,-5,5,15]
]

Tree 1 learns these residuals.

Suppose for one observation the tree predicts:

[
f_1(x)=10
]

with:

[
\eta=0.1
]

Then:

[
F_1(x)=25+(0.1)(10)
]

[
F_1(x)=26
]

Now calculate the new residual:

[
40-26=14
]

Tree 2 learns the remaining errors.

Then:

[
F_2=F_1+\eta f_2
]

And this continues.

After many trees:

[
F_M(x)
]

becomes a much better approximation of the target.

---

# 27. Important parameters in `GradientBoostingRegressor`

For your practical work, remember these:

```python
GradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    max_features=None,
    random_state=42
)
```

### `loss`

Controls the objective being minimized.

Common choices:

```text
squared_error
absolute_error
huber
quantile
```

### `learning_rate`

Controls the contribution of each tree.

### `n_estimators`

Number of boosting stages.

### `max_depth`

Complexity of each individual tree.

### `max_features`

Number of features considered when splitting.

### `subsample`

Fraction of training samples used for each tree.

If:

```python
subsample=1.0
```

all samples are used.

If:

```python
subsample=0.8
```

80% are used for each boosting stage.

This can introduce randomness and can help reduce overfitting.

---

# 28. The one diagram you should remember

```text
                   X, y
                    │
                    ↓
             Initial prediction
                    │
                    ↓
               Calculate loss
                    │
                    ↓
          Calculate negative gradient
                    │
                    ↓
             Train Tree 1
                    │
                    ↓
           Add learning-rate-scaled
                 Tree 1
                    │
                    ↓
          New predictions/errors
                    │
                    ↓
             Train Tree 2
                    │
                    ↓
           Add Tree 2 correction
                    │
                    ↓
                   ...
                    │
                    ↓
             Train Tree M
                    │
                    ↓
             Final prediction
```

### The core equation

If you remember only one equation:

[
\boxed{
F_m(x)=F_{m-1}(x)+\eta f_m(x)
}
]

And if you're using the default squared-error loss, remember:

[
\boxed{
\text{Residual}=y-F(x)
}
]

So the entire intuition is:

> **Make an initial prediction → calculate how wrong it is → train a tree to predict that error → add a small part of that tree's prediction → repeat.**

That's the fundamental working of **GradientBoostingRegressor**.
