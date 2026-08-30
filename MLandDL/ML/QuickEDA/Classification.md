# Classification Problems — EDA & Preprocessing Handbook
*(Works for any classification dataset: binary or multi-class, linear or tree-based)*

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

---

## 📑 Table of Contents
1. [Initial Understanding](#1-initial-understanding)
2. [Data Quality Checks](#2-data-quality-checks)
3. [Target Variable Analysis](#3-target-variable-analysis)
4. [Univariate Analysis](#4-univariate-analysis)
5. [Bivariate Analysis](#5-bivariate-analysis-feature-vs-class)
6. [Multivariate Analysis](#6-multivariate-analysis)
7. [Statistical Threshold Reference](#7-statistical-threshold-reference-what-values-mean) ⭐
8. [Outlier Detection & Handling](#8-outlier-detection--handling)
9. [Missing Value Treatment](#9-missing-value-treatment)
10. [Class Imbalance Handling](#10-class-imbalance-handling) ⭐
11. [Feature Engineering](#11-feature-engineering)
12. [Encoding Categorical Variables](#12-encoding-categorical-variables)
13. [Feature Scaling](#13-feature-scaling)
14. [Train/Test Split](#14-traintest-split)
15. [Final Pre-Modeling Sanity Checks](#15-final-pre-modeling-sanity-checks)
16. [Quick Plot Reference](#16-quick-plot-reference)

---

## 1. Initial Understanding

**Theory:** Same discipline as regression — know your rows, columns, and types before plotting anything. The one classification-specific thing to nail down immediately: is this binary or multi-class, and how many classes?

- [ ] Shape
```python
df.shape
```
- [ ] Data types & non-null counts
```python
df.info()
```
- [ ] Summary statistics
```python
df.describe(include='all').T
```
- [ ] Identify target, confirm it's categorical/discrete
```python
target = "class"
df[target].dtype
df[target].nunique()
```
- [ ] Identify feature types
```python
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
if target in num_cols: num_cols.remove(target)
if target in cat_cols: cat_cols.remove(target)
```

---

## 2. Data Quality Checks

**Theory:** Same as regression — duplicates and constant columns are silent killers of validity. In classification specifically, duplicate rows across train/test splits are one of the most common causes of unrealistically high accuracy (this bit your heart disease notebook — same principle applies to any classifier).

- [ ] Missing values
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
pd.DataFrame({"missing": missing, "pct": missing_pct}).query("missing > 0")
```
- [ ] Duplicate rows
```python
df.duplicated().sum()
df = df.drop_duplicates()
```
- [ ] Constant / near-constant columns
```python
[col for col in df.columns if df[col].nunique() <= 1]
```
- [ ] High-cardinality / ID-like columns
```python
df.nunique().sort_values(ascending=False).head(10)
```
- [ ] Inconsistent category labels
```python
for col in cat_cols:
    print(col, df[col].unique())
```
- [ ] Fix incorrect data types
```python
df["some_numeric_col"] = pd.to_numeric(df["some_numeric_col"], errors='coerce')
```

---

## 3. Target Variable Analysis

**Theory:** The single most important classification-specific check. Class imbalance changes everything downstream: which metric you trust (accuracy becomes meaningless above ~80/20 imbalance), whether you need resampling, and how you split train/test (stratify, always).

- [ ] Class counts and proportions
```python
df[target].value_counts()
df[target].value_counts(normalize=True) * 100
```
- [ ] Class balance plot
```python
sns.countplot(x=target, data=df)
plt.title("Class Distribution")
plt.show()
```
- [ ] Imbalance ratio (see [Section 7](#7-statistical-threshold-reference-what-values-mean) for what the ratio means)
```python
counts = df[target].value_counts()
imbalance_ratio = counts.max() / counts.min()
imbalance_ratio
```

---

## 4. Univariate Analysis

**Theory:** Same as regression — look at every feature alone first. Watch for rare categories (a category with 2 rows can't help a classifier generalize) and heavily skewed numeric features that may need transforming before distance-based models.

- [ ] Numeric features
```python
for col in num_cols:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df[col], kde=True, ax=ax[0])
    sns.boxplot(x=df[col], ax=ax[1])
    plt.suptitle(col)
    plt.show()
```
- [ ] Categorical features
```python
for col in cat_cols:
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(col)
    plt.show()
```

---

## 5. Bivariate Analysis (feature vs class)

**Theory:** A feature is useful for classification if its distribution *differs* across classes — not if it correlates linearly with a number (there is no number; the target is a label). This is the key mental shift from regression: you're now looking for **separation**, not **correlation**.

- [ ] Numeric feature vs class (boxplot/violin — do distributions differ per class?)
```python
for col in num_cols:
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f"{col} by {target}")
    plt.show()
```
- [ ] Numeric feature vs class (density overlap — how separable are the classes?)
```python
for col in num_cols:
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(f"{col} density by {target}")
    plt.show()
```
- [ ] Categorical feature vs class (crosstab + grouped bar)
```python
for col in cat_cols:
    print(pd.crosstab(df[col], df[target], normalize='index'))
    sns.countplot(x=col, hue=target, data=df)
    plt.xticks(rotation=45)
    plt.show()
```
- [ ] Statistical significance — numeric feature vs binary/multi-class target (ANOVA F-test)
```python
from scipy.stats import f_oneway
groups = [df[df[target]==cls][col].dropna() for cls in df[target].unique()]
f_oneway(*groups)
```
- [ ] Statistical significance — categorical feature vs target (Chi-square)
```python
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df[col], df[target])
chi2, p, dof, expected = chi2_contingency(contingency)
chi2, p
```

---

## 6. Multivariate Analysis

**Theory:** Multicollinearity still hurts linear classifiers (Logistic Regression) the same way it hurts linear regression — unstable coefficients. Pairplots colored by class also give you a fast visual read on how separable your classes already are with just 2–3 features.

- [ ] Correlation heatmap (numeric features)
```python
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
```
- [ ] Multicollinearity flag
```python
corr_matrix = df[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col, row) for col in upper.columns for row in upper.index if upper.loc[row, col] > 0.8]
high_corr
```
- [ ] VIF check
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[num_cols].dropna()
vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif.sort_values("VIF", ascending=False)
```
- [ ] Pairplot colored by class
```python
sns.pairplot(df, vars=num_cols[:5], hue=target)
plt.show()
```

---

## 7. Statistical Threshold Reference — What Values Mean

### Class imbalance ratio

```python
counts = df[target].value_counts()
counts.max() / counts.min()
```

| Ratio | Interpretation | Action |
|---|---|---|
| `1 – 1.5` | Roughly balanced | No special handling needed |
| `1.5 – 3` | Mild imbalance | Use stratified split; accuracy still mostly OK but check F1 too |
| `3 – 10` | Moderate imbalance | Use F1/precision/recall/AUC, not accuracy; consider class weights |
| `> 10` | Severe imbalance | Resampling (SMOTE/undersampling) + class weights + threshold tuning required |

### Chi-square test (categorical feature vs categorical target)

```python
chi2, p, dof, expected = chi2_contingency(pd.crosstab(df[col], df[target]))
```

| p-value | Interpretation |
|---|---|
| `p < 0.05` | Statistically significant association — feature likely useful |
| `p ≥ 0.05` | No significant association — feature may be weak/noise |

**Effect size (Cramér's V)** — tells you *how strong* the association is, not just whether it exists:
```python
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

cramers_v(pd.crosstab(df[col], df[target]))
```
| Cramér's V | Strength |
|---|---|
| `0.0 – 0.1` | Negligible |
| `0.1 – 0.3` | Weak |
| `0.3 – 0.5` | Moderate |
| `> 0.5` | Strong |

### ANOVA F-test (numeric feature vs categorical target)

```python
f_oneway(*groups)
```
| p-value | Interpretation |
|---|---|
| `p < 0.05` | Class means differ significantly for this feature — good separator |
| `p ≥ 0.05` | Feature doesn't separate classes well |

### Skewness of numeric features (same rules as regression)

```python
df[col].skew()
```

| Skew value | Interpretation | Shape |
|---|---|---|
| `-0.5` to `0.5` | Approximately symmetric | No fix usually needed |
| `> 1.0` | **Right-skewed** | Long tail to the right, mean > median — most values low, few very high |
| `< -1.0` | **Left-skewed** | Long tail to the left, mean < median — most values high, few very low |

**Fixes (same as regression):**

| Skew type | Fix | Code |
|---|---|---|
| Right-skewed | Log transform | `np.log1p(df[col])` |
| Right-skewed (zeros/negatives) | Yeo-Johnson | `PowerTransformer(method='yeo-johnson')` |
| Left-skewed | Square/cube transform | `df[col]**2` |
| Left-skewed | Reflect + log | `np.log1p(df[col].max() - df[col])` |

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df[f"{col}_transformed"] = pt.fit_transform(df[[col]])
```

### VIF (Variance Inflation Factor) — same thresholds as regression

| VIF value | Interpretation |
|---|---|
| `1 – 5` | Acceptable |
| `5 – 10` | Investigate |
| `> 10` | Severe multicollinearity |

### Missing data severity — same as regression

| % Missing | Action |
|---|---|
| `< 5%` | Impute or drop rows |
| `5% – 30%` | Impute carefully + consider "was_missing" flag |
| `> 30%` | Consider dropping column |
| `> 60%` | Usually drop |

### Outlier fences — IQR method (same as regression)

| Multiplier | Meaning |
|---|---|
| `1.5 × IQR` | Standard fence |
| `3.0 × IQR` | Extreme fence |

---

## 8. Outlier Detection & Handling

**Theory:** Outliers matter less to tree-based classifiers (splits are rank-based) but can still distort distance-based models (KNN, SVM) and skew scalers. Always check whether an "outlier" is actually a distinct sub-population relevant to a class before removing it.

- [ ] IQR method
```python
def iqr_outliers(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return series[(series < lower) | (series > upper)]

for col in num_cols:
    print(col, "->", len(iqr_outliers(df[col])), "outliers")
```
- [ ] Z-score method
```python
from scipy.stats import zscore
z_scores = df[num_cols].apply(zscore)
(z_scores.abs() > 3).sum()
```
- [ ] Capping
```python
from scipy.stats.mstats import winsorize
df["col_capped"] = winsorize(df["col"], limits=[0.01, 0.01])
```

---

## 9. Missing Value Treatment

**Theory:** Same principles as regression. One extra classification-specific consideration: check whether missingness itself correlates with the target class (e.g. a lab test only run when a disease is suspected) — that's informative and worth keeping as a flag rather than imputing away.

- [ ] Check missingness vs target
```python
for col in df.columns[df.isnull().any()]:
    df[f"{col}_missing"] = df[col].isnull().astype(int)
    print(pd.crosstab(df[f"{col}_missing"], df[target], normalize='index'))
```
- [ ] Median imputation
```python
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
```
- [ ] Mode imputation
```python
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
```
- [ ] KNN imputation
```python
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
df[num_cols] = knn_imputer.fit_transform(df[num_cols])
```
⚠️ Fit imputer on train only, transform test — never fit on full data before splitting.

---

## 10. Class Imbalance Handling

**Theory:** This section has no regression equivalent — it's classification-specific. Imbalanced classes bias a model toward predicting the majority class since that minimizes overall error. Fixing this happens either at the data level (resampling) or the algorithm level (class weights).

- [ ] Check imbalance ratio (see Section 7)
```python
df[target].value_counts(normalize=True)
```
- [ ] Random oversampling (minority class)
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
```
- [ ] SMOTE (synthetic oversampling)
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```
- [ ] Random undersampling (majority class)
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```
- [ ] Class weights (algorithm-level, no resampling needed)
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```
⚠️ Always resample **after** the train/test split, and only on the training set — resampling before splitting leaks synthetic/duplicated rows into your test set.

---

## 11. Feature Engineering

**Theory:** Same intent as regression — surface signal the model can't derive on its own. In classification, binning continuous variables into meaningful buckets can sometimes make decision boundaries easier for simpler models to learn.

- [ ] Interaction terms
```python
df["feature_interaction"] = df["feature1"] * df["feature2"]
```
- [ ] Datetime extraction
```python
df["year"] = df["date_col"].dt.year
df["month"] = df["date_col"].dt.month
df["weekday"] = df["date_col"].dt.weekday
```
- [ ] Binning
```python
df["age_bin"] = pd.cut(df["age"], bins=[0,18,35,60,100], labels=["teen","young_adult","adult","senior"])
```
- [ ] Ratio features
```python
df["ratio_feature"] = df["feature1"] / df["feature2"]
```

---

## 12. Encoding Categorical Variables

**Theory:** Identical logic to regression. One classification nuance: if using target encoding, make sure it's computed per-fold in cross-validation, not once globally — otherwise it leaks target class information into features.

- [ ] One-Hot (nominal)
```python
df = pd.get_dummies(df, columns=["nominal_col"], drop_first=True)
```
- [ ] Ordinal (has order)
```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
df["ordinal_col_encoded"] = enc.fit_transform(df[["ordinal_col"]])
```
- [ ] Label encoding (for the target itself, if needed)
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
```
- [ ] Frequency/target encoding (high cardinality — fit on train only)
```python
freq_map = df_train["high_card_col"].value_counts(normalize=True)
df_train["high_card_encoded"] = df_train["high_card_col"].map(freq_map)
df_test["high_card_encoded"] = df_test["high_card_col"].map(freq_map)
```

---

## 13. Feature Scaling

**Theory:** Same rule as regression: distance/gradient-based models (KNN, SVM, Logistic Regression, Neural Nets) need scaling; tree-based models (Random Forest, XGBoost, Decision Trees) don't.

- [ ] Standardization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- [ ] Normalization
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 14. Train/Test Split

**Theory:** For classification, splitting **must** preserve class proportions — an unstratified split can accidentally starve your test set of a minority class entirely, making evaluation meaningless.

- [ ] Stratified split (always default to this for classification)
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- [ ] Verify class proportions preserved
```python
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
```

---

## 15. Final Pre-Modeling Sanity Checks

- [ ] No nulls remaining
```python
assert X_train.isnull().sum().sum() == 0
```
- [ ] All columns numeric post-encoding
```python
X_train.dtypes.value_counts()
```
- [ ] Shapes match
```python
X_train.shape[0] == y_train.shape[0]
```
- [ ] Baseline sanity fit + classification metrics
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
```

---

## 16. Quick Plot Reference

| Purpose | Plot | Code |
|---|---|---|
| Class balance | Countplot | `sns.countplot(x=target, data=df)` |
| Feature distribution | Histogram, Boxplot | `sns.histplot(df[col])` |
| Feature vs class (numeric) | Boxplot/Violin | `sns.boxplot(x=target, y=col, data=df)` |
| Feature vs class (density) | KDE split by hue | `sns.kdeplot(data=df, x=col, hue=target)` |
| Feature vs class (categorical) | Grouped countplot | `sns.countplot(x=col, hue=target, data=df)` |
| Feature correlations | Heatmap | `sns.heatmap(df.corr(), annot=True)` |
| Multivariate class separability | Pairplot (hue=class) | `sns.pairplot(df, hue=target)` |
| Model evaluation | Confusion matrix | `sns.heatmap(confusion_matrix(y_test, preds), annot=True)` |
| Model evaluation | ROC curve | `from sklearn.metrics import RocCurveDisplay; RocCurveDisplay.from_estimator(model, X_test, y_test)` |

[⬆ Back to top](#-table-of-contents)