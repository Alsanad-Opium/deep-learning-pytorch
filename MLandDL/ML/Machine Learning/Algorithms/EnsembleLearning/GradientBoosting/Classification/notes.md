### Binary GradientBoostingClassifier

---

# Gradient Boosting Classifier — Theory + Mathematics

## 1. The main idea

Gradient Boosting is an **ensemble learning algorithm**.

It builds many weak decision trees **sequentially**:

```text
Tree 1
  ↓
Tree 2 corrects Tree 1
  ↓
Tree 3 corrects Tree 1 + Tree 2
  ↓
Tree 4 corrects previous trees
  ↓
...
Final prediction
```

The key idea is:

> **Each new tree tries to correct the errors made by the existing ensemble.**

But there is an important mathematical difference from simple residual fitting:

> Gradient Boosting doesn't necessarily fit the ordinary residuals.
> It fits the **negative gradient of the loss function**.

That's where the word **"Gradient"** comes from.

---

# 2. Start with a binary classification problem

Suppose:

```text
y ∈ {0, 1}
```

For example:

```text
SMS → Spam / Ham
```

We want:

[
P(y=1|X)
]

The final prediction should be a probability between 0 and 1.

We use the sigmoid function:

[
p = \frac{1}{1+e^{-F(X)}}
]

where:

[
F(X)
]

is the score produced by the boosting model.

---

# 3. The model is built as an additive model

Instead of having one giant tree, Gradient Boosting creates:

[
F_M(X)=F_0(X)+\eta f_1(X)+\eta f_2(X)+...+\eta f_M(X)
]

where:

* (F_0(X)) = initial prediction
* (f_1(X),f_2(X),...) = decision trees
* (M) = number of trees
* (\eta) = learning rate

So:

[
\boxed{F_M(X)=F_0(X)+\eta\sum_{m=1}^{M}f_m(X)}
]

This is the fundamental equation behind Gradient Boosting.

---

# 4. Why don't we directly predict 0 and 1?

Suppose the model predicts:

```text
0.8
```

for an observation whose true label is:

```text
1
```

The model is doing fairly well.

But if it predicts:

```text
0.2
```

for a true label:

```text
1
```

that's a much worse prediction.

Therefore, classification needs a loss function that measures **how good the predicted probability is**.

---

# 5. Log Loss

For binary classification, GradientBoostingClassifier commonly uses **log loss**.

For one observation:

[
L(y,p)
======

-[y\log(p)+(1-y)\log(1-p)]
]

where:

* (y) = actual class
* (p) = predicted probability

For the entire dataset:

[
L =
-\frac{1}{N}
\sum_{i=1}^{N}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
]

Our objective is:

[
\boxed{\text{Minimize Log Loss}}
]

---

# 6. Example of log loss

Suppose:

```text
Actual y = 1
```

### Model A

```text
p = 0.9
```

Loss:

[
-\log(0.9)\approx0.105
]

Very good.

### Model B

```text
p = 0.5
```

[
-\log(0.5)\approx0.693
]

Worse.

### Model C

```text
p = 0.1
```

[
-\log(0.1)\approx2.303
]

Very bad.

So log loss strongly penalizes **confident incorrect predictions**.

---

# 7. Now comes the "Gradient" part

Suppose our current model is:

[
F_{m-1}(x)
]

We calculate the current probability:

[
p_i = \sigma(F_{m-1}(x_i))
]

where:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

Now we calculate the gradient of the loss.

For log loss:

[
L_i=-[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
]

The derivative with respect to (F) simplifies to:

[
\frac{\partial L_i}{\partial F_i}=p_i-y_i
]

Therefore the **negative gradient** is:

[
\boxed{r_i=y_i-p_i}
]

This is extremely important.

For binary Gradient Boosting Classification:

> **The pseudo-residual is essentially actual class − predicted probability.**

---

# 8. Example

Suppose we have:

| Actual (y) | Predicted probability (p) |
| ---------: | ------------------------: |
|          1 |                       0.8 |
|          1 |                       0.6 |
|          0 |                       0.3 |
|          0 |                       0.1 |

Calculate:

[
r_i=y_i-p_i
]

### Observation 1

[
r=1-0.8=0.2
]

### Observation 2

[
r=1-0.6=0.4
]

### Observation 3

[
r=0-0.3=-0.3
]

### Observation 4

[
r=0-0.1=-0.1
]

So:

| Actual | Probability | Pseudo-residual |
| -----: | ----------: | --------------: |
|      1 |         0.8 |            +0.2 |
|      1 |         0.6 |            +0.4 |
|      0 |         0.3 |            −0.3 |
|      0 |         0.1 |            −0.1 |

Notice something interesting:

The model gives a **large positive residual** to the observation where:

```text
actual = 1
prediction = 0.6
```

because it needs to increase its prediction.

Similarly:

```text
actual = 0
prediction = 0.3
```

gets:

[
-0.3
]

meaning the model needs to push the prediction downward.

---

# 9. The next decision tree learns these residuals

Now we train a decision tree on:

```text
X → pseudo-residual
```

rather than directly predicting the class.

Conceptually:

```text
Original data
     ↓
Current model
     ↓
Predicted probabilities
     ↓
Calculate negative gradient
     ↓
Pseudo-residuals
     ↓
Decision Tree
     ↓
Add tree to existing model
```

This is the central mechanism.

---

# 10. Updating the model

Suppose the current model is:

[
F_{m-1}(x)
]

We train a tree:

[
f_m(x)
]

Then update:

[
\boxed{
F_m(x)=F_{m-1}(x)+\eta f_m(x)
}
]

where:

[
\eta
]

is the learning rate.

For example:

```text
Current score = 0.8

Tree correction = 0.4

learning_rate = 0.1
```

Then:

[
F_{new}=0.8+(0.1)(0.4)
]

[
F_{new}=0.84
]

The model makes only a **small correction**.

---

# 11. Why do we need the learning rate?

Without a learning rate:

[
F_m=F_{m-1}+f_m
]

A tree could make a very large correction.

Instead:

[
F_m=F_{m-1}+\eta f_m
]

where typically:

```text
η = 0.01
η = 0.05
η = 0.1
η = 0.2
```

Small learning rate:

```text
slow learning
      +
more trees
      ↓
potentially better generalization
```

Large learning rate:

```text
fast learning
      +
fewer trees
```

This is why:

[
\boxed{\text{learning_rate and n_estimators are closely related}}
]

---

# 12. Complete mathematical algorithm

Now let's put everything together.

### Step 1 — Initialize the model

Find an initial constant prediction:

[
F_0
]

For binary log-loss classification, this corresponds to the **log-odds of the positive class**:

[
F_0=
\log\left(\frac{p_0}{1-p_0}\right)
]

where:

[
p_0=\frac{\text{number of positive samples}}{N}
]

For example, if:

```text
70% → class 1
30% → class 0
```

then:

[
p_0=0.7
]

and:

[
F_0=\log\left(\frac{0.7}{0.3}\right)
]

[
F_0\approx0.847
]

---

# 13. Step 2 — Calculate probability

Convert the current score into probability:

[
p_i=\frac{1}{1+e^{-F_{m-1}(x_i)}}
]

---

# 14. Step 3 — Calculate negative gradient

For log loss:

[
\boxed{r_i=y_i-p_i}
]

These are called:

**pseudo-residuals**

or

**negative gradients**.

---

# 15. Step 4 — Fit a decision tree

Train a regression tree on:

[
X_i \rightarrow r_i
]

Notice:

### It's a regression tree!

Even though you're doing **classification**, the individual trees are used to model the gradient values.

---

# 16. Step 5 — Add the tree

Update:

[
F_m(x)=F_{m-1}(x)+\eta f_m(x)
]

---

# 17. Step 6 — Repeat

Repeat:

```text
Calculate probability
        ↓
Calculate negative gradient
        ↓
Fit tree
        ↓
Add tree
        ↓
Calculate new probability
        ↓
Calculate new gradient
        ↓
Fit another tree
        ↓
...
```

After (M) iterations:

[
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}f_m(x)
]

---

# 18. Finally convert score to probability

After all trees:

[
p(x)=\frac{1}{1+e^{-F_M(x)}}
]

Then classification:

[
\hat y=
\begin{cases}
1 & p\geq0.5\
0 & p<0.5
\end{cases}
]

So the entire process is:

```text
                 Gradient Boosting Classifier

                         Training Data
                              ↓
                    Initial prediction F₀
                              ↓
                    Calculate probability
                              ↓
                       Calculate loss
                              ↓
                  Calculate negative gradient
                         r = y - p
                              ↓
                  Train decision-tree f₁
                              ↓
                F₁ = F₀ + ηf₁
                              ↓
                    Calculate new errors
                              ↓
                  Train decision-tree f₂
                              ↓
                F₂ = F₁ + ηf₂
                              ↓
                             ...
                              ↓
                  Fₘ = Fₘ₋₁ + ηfₘ
                              ↓
                       Sigmoid function
                              ↓
                     Final probability
                              ↓
                       Class 0 or 1
```

---

# 19. How is this different from AdaBoost?

This is a very useful distinction since you've already studied AdaBoost.

### AdaBoost

AdaBoost focuses on **misclassified observations** by increasing their weights.

Conceptually:

```text
Wrong prediction
       ↓
Increase weight
       ↓
Next classifier focuses more on it
```

### Gradient Boosting

Gradient Boosting focuses on the **gradient of the loss function**.

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

So:

| AdaBoost                         | Gradient Boosting                      |
| -------------------------------- | -------------------------------------- |
| Adjusts sample weights           | Uses loss gradients                    |
| Focuses on misclassified samples | Minimizes differentiable loss          |
| Sequential weak learners         | Sequential trees                       |
| Weight-based                     | Gradient-based                         |
| Often uses exponential loss      | Often uses log loss for classification |

---

# 20. What do the important parameters mean?

For your `GradientBoostingClassifier`, these are the ones you should understand:

### `n_estimators`

Number of boosting stages/trees.

```python
n_estimators=100
```

means approximately:

```text
100 sequential boosting stages
```

More trees → more capacity.

---

### `learning_rate`

How much each tree contributes:

[
F_m=F_{m-1}+\eta f_m
]

Small:

```python
learning_rate=0.05
```

Large:

```python
learning_rate=0.2
```

Usually:

> Smaller learning rate → need more trees.

---

### `max_depth`

Controls complexity of individual trees.

```python
max_depth=2
```

→ shallow trees

```python
max_depth=8
```

→ much more complex trees

Gradient Boosting commonly works well with **relatively shallow trees**.

---

### `loss`

For classification, the loss determines what the gradient is trying to minimize.

For example:

```python
loss='log_loss'
```

uses logarithmic loss.

---

# 21. The most important equation to remember

If you're preparing notes/exam theory, remember this chain:

[
\boxed{
p_i=\sigma(F_{m-1}(x_i))
}
]

then:

[
\boxed{
r_i=y_i-p_i
}
]

then train a tree:

[
\boxed{
f_m(x)\leftarrow r_i
}
]

then update:

[
\boxed{
F_m(x)=F_{m-1}(x)+\eta f_m(x)
}
]

and finally:

[
\boxed{
p(x)=\sigma(F_M(x))
}
]

That is essentially the **mathematical heart of GradientBoostingClassifier**.

One subtlety: the exact tree-fitting/update details in scikit-learn's implementation are a little more sophisticated than the simplified (y-p) explanation above, especially for multiclass classification. But for **binary classification with log loss**, this is the core mathematical intuition and derivation you want to understand.
