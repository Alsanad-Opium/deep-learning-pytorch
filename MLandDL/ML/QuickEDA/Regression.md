# Regression Problems — EDA & Preprocessing Handbook
*(Works for any regression dataset: linear, tree-based, or otherwise)*

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("data.csv")
```

---

## 📑 Table of Contents
1. [Initial Understanding](#1-initial-understanding)
2. [Data Quality Checks](#2-data-quality-checks)
3. [Target Variable Analysis](#3-target-variable-analysis)
4. [Univariate Analysis](#4-univariate-analysis)
5. [Bivariate Analysis](#5-bivariate-analysis-feature-vs-target)
6. [Multivariate Analysis](#6-multivariate-analysis)
7. [Statistical Threshold Reference](#7-statistical-threshold-reference-what-values-mean) ⭐ new
8. [Outlier Detection & Handling](#8-outlier-detection--handling)
9. [Missing Value Treatment](#9-missing-value-treatment)
10. [Feature Engineering](#10-feature-engineering)
11. [Encoding Categorical Variables](#11-encoding-categorical-variables)
12. [Feature Scaling](#12-feature-scaling)
13. [Train/Test Split](#13-traintest-split)
14. [Final Pre-Modeling Sanity Checks](#14-final-pre-modeling-sanity-checks)
15. [Quick Plot Reference](#15-quick-plot-reference)

---

## 1. Initial Understanding

**Theory:** Before touching a single plot, you need to know what you're working with — how many rows/columns, what types (numeric/categorical/datetime), and what the target actually represents. Skipping this step is the #1 cause of wasted rework later.

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
- [ ] Identify target, confirm continuous
```python
target = "price"
df[target].dtype
```
- [ ] Identify feature types
```python
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
```

---

## 2. Data Quality Checks

**Theory:** Garbage in, garbage out. Duplicate rows are the single biggest cause of inflated model scores (you've seen this bite you already — duplicate rows caused leakage in your heart disease notebook). Constant columns add zero signal and just waste compute/interpretability.

- [ ] Missing values (count + %)
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

**Theory:** Most regression models (especially linear ones) assume errors are roughly normally distributed. A heavily skewed target often means heavily skewed *errors* too, which distorts loss functions like MSE. Fixing target skew before modeling frequently improves performance more than any other single step.

- [ ] Distribution plot
```python
sns.histplot(df[target], kde=True)
plt.title(f"Distribution of {target}")
plt.show()
```
- [ ] Skewness — see [Section 7](#7-statistical-threshold-reference-what-values-mean) for what the number means
```python
df[target].skew()
```
- [ ] Outliers in target
```python
sns.boxplot(x=df[target])
plt.show()
```
- [ ] Transform if skewed (see Section 7 for fix rules)
```python
if df[target].skew() > 1:
    df[f"{target}_log"] = np.log1p(df[target])
```

---

## 4. Univariate Analysis

**Theory:** Look at every column in isolation before comparing to anything else. This is where you catch skew, multi-modal distributions (hint of hidden subgroups), rare categories, and impossible values (e.g. negative age).

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
- [ ] Datetime range
```python
df["date_col"] = pd.to_datetime(df["date_col"])
df["date_col"].min(), df["date_col"].max()
```

---

## 5. Bivariate Analysis (feature vs target)

**Theory:** A feature is only useful if it moves *with* the target in some way — linear, monotonic, or at least category-driven. This step is where you rank features by likely predictive value before even building a model.

- [ ] Numeric feature vs target
```python
for col in num_cols:
    if col != target:
        sns.regplot(x=col, y=target, data=df, scatter_kws={"alpha":0.4})
        plt.title(f"{col} vs {target}")
        plt.show()
```
- [ ] Categorical feature vs target
```python
for col in cat_cols:
    sns.boxplot(x=col, y=target, data=df)
    plt.xticks(rotation=45)
    plt.show()
```
- [ ] Correlation of numeric features with target
```python
df[num_cols].corrwith(df[target]).sort_values(ascending=False)
```

---

## 6. Multivariate Analysis

**Theory:** Features rarely act alone. Two features can each look weakly correlated with the target individually but be highly correlated *with each other* (multicollinearity) — this destabilizes linear model coefficients even if predictions look fine.

- [ ] Correlation heatmap
```python
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
```
- [ ] Multicollinearity flag (see Section 7 for thresholds)
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
- [ ] Pairplot of top features
```python
top_features = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False).head(5).index.tolist()
sns.pairplot(df[top_features + [target]])
plt.show()
```

---

## 7. Statistical Threshold Reference — What Values Mean

**This is your lookup table** for "is this number good or bad, and what do I do about it."

### Skewness — shape of a distribution

Skewness measures asymmetry. A perfectly symmetric distribution (like a normal/bell curve) has skewness ≈ 0.

```python
df[col].skew()
```

| Skew value | Interpretation | Shape |
|---|---|---|
| `-0.5` to `0.5` | Approximately symmetric | Normal-ish, usually no fix needed |
| `0.5` to `1.0` (or `-0.5` to `-1.0`) | Moderately skewed | Consider a transform if modeling is skew-sensitive |
| `> 1.0` | **Right-skewed** (positive skew) | Long tail stretches to the **right** — most values are low/small, with a few very large values (e.g. income, house prices) |
| `< -1.0` | **Left-skewed** (negative skew) | Long tail stretches to the **left** — most values are high, with a few very small/low outlier values (e.g. age at retirement, exam scores near a ceiling) |

**How to tell right vs left skew visually:**
- Right-skewed (positive): the "tail" of the histogram points right, mean > median.
- Left-skewed (negative): the tail points left, mean < median.
```python
df[col].mean(), df[col].median()   # mean > median → right skew; mean < median → left skew
```

**How to fix each:**

| Skew type | Fix | Code |
|---|---|---|
| Right-skewed | Log transform | `np.log1p(df[col])` |
| Right-skewed (has zeros/negatives) | Square root, or Box-Cox/Yeo-Johnson | `np.sqrt(df[col])` or `PowerTransformer(method='yeo-johnson')` |
| Left-skewed | Square / cube transform | `df[col]**2` |
| Left-skewed | Reflect then log (`log1p(max - x)`) | `np.log1p(df[col].max() - df[col])` |
| Either, general purpose | Box-Cox (needs strictly positive values) | `stats.boxcox(df[col])` |
| Either, general purpose (handles negatives/zeros) | Yeo-Johnson | `from sklearn.preprocessing import PowerTransformer; PowerTransformer(method='yeo-johnson').fit_transform(df[[col]])` |

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df[f"{col}_transformed"] = pt.fit_transform(df[[col]])
```

### Correlation — strength of linear relationship

```python
df[col].corr(df[target])
```

| \|r\| value | Interpretation |
|---|---|
| `0.0 – 0.1` | Negligible / no linear relationship |
| `0.1 – 0.3` | Weak |
| `0.3 – 0.5` | Moderate |
| `0.5 – 0.7` | Strong |
| `0.7 – 1.0` | Very strong |
| `> 0.8` between two **features** (not target) | Multicollinearity risk — consider dropping one |

### VIF (Variance Inflation Factor) — multicollinearity severity

| VIF value | Interpretation |
|---|---|
| `1` | No correlation with other features |
| `1 – 5` | Moderate, generally acceptable |
| `5 – 10` | High — investigate, consider dropping/combining |
| `> 10` | Severe multicollinearity — should address before linear modeling |

### Kurtosis — "tailedness" / outlier-proneness of a distribution

```python
df[col].kurt()
```

| Kurtosis value | Interpretation |
|---|---|
| `≈ 0` (normal reference, using excess kurtosis) | Normal-tailed (mesokurtic) |
| `> 0` | Heavy tails, more outlier-prone (leptokurtic) |
| `< 0` | Light tails, fewer extreme values (platykurtic) |

### Missing data severity — what % missing means for action

| % Missing | Typical action |
|---|---|
| `< 5%` | Safe to impute (median/mode) or drop rows |
| `5% – 30%` | Impute carefully (median/KNN); consider a "was_missing" flag feature |
| `> 30%` | Consider dropping the column unless it's known to be highly predictive |
| `> 60%` | Usually drop, unless missingness itself is informative |

### Outlier severity — IQR method

```python
Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
```
| Multiplier | Meaning |
|---|---|
| `1.5 × IQR` | Standard outlier fence |
| `3.0 × IQR` | "Extreme" outlier fence |

---

## 8. Outlier Detection & Handling

**Theory:** Outliers in regression can drastically pull a fitted line since squared-error loss punishes large residuals heavily. Whether to remove, cap, or keep an outlier depends on whether it's a data error or a genuine rare event.

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
- [ ] Capping (winsorizing)
```python
from scipy.stats.mstats import winsorize
df["col_capped"] = winsorize(df["col"], limits=[0.01, 0.01])
```

---

## 9. Missing Value Treatment

**Theory:** How you impute affects the variance and correlation structure of your data. Mean imputation deflates variance; more advanced methods (KNN, iterative) preserve relationships better but cost more compute.

- [ ] Median imputation
```python
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
```
- [ ] Mode imputation (categorical)
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
⚠️ Always fit imputers on train only, then transform test — fitting on full data leaks test-set information into train.

---

## 10. Feature Engineering

**Theory:** Raw columns rarely capture everything useful. Ratios, differences, and date-parts often carry more signal than the originals, especially for tree-based models that can't learn interactions/ratios natively.

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
df["price_per_sqft"] = df["price"] / df["sqft"]
```

---

## 11. Encoding Categorical Variables

**Theory:** Models need numbers. One-hot works for unordered categories but explodes dimensionality with high cardinality; ordinal encoding preserves order but assumes equal spacing between categories, which may be wrong.

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
- [ ] Frequency/target encoding (high cardinality — fit on train only)
```python
freq_map = df_train["high_card_col"].value_counts(normalize=True)
df_train["high_card_encoded"] = df_train["high_card_col"].map(freq_map)
df_test["high_card_encoded"] = df_test["high_card_col"].map(freq_map)
```

---

## 12. Feature Scaling

**Theory:** Distance-based and gradient-based models (KNN, SVR, linear regression with regularization, neural nets) are sensitive to feature scale; tree-based models (Random Forest, XGBoost) are not.

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

## 13. Train/Test Split

**Theory:** Any preprocessing step that "learns" something from data (scaler mean/std, imputer median, encoder frequencies) must be fit only on train data. Fitting on the full dataset before splitting is data leakage — the most common silent bug in ML pipelines.

- [ ] Basic split
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- [ ] Stratified split by binned target (skewed target)
```python
df["target_bin"] = pd.qcut(df[target], q=5, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df["target_bin"]
)
```

---

## 14. Final Pre-Modeling Sanity Checks

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
- [ ] Baseline sanity fit
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

model = LinearRegression()
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
mean_absolute_error(y_test, preds)
```

---

## 15. Quick Plot Reference

| Purpose | Plot | Code |
|---|---|---|
| Target distribution | Histogram, KDE | `sns.histplot(df[target], kde=True)` |
| Target outliers | Boxplot | `sns.boxplot(x=df[target])` |
| Feature distribution | Histogram, Boxplot | `sns.histplot(df[col])` |
| Feature vs target (numeric) | Scatter/Regplot | `sns.regplot(x=col, y=target, data=df)` |
| Feature vs target (categorical) | Boxplot | `sns.boxplot(x=col, y=target, data=df)` |
| Feature correlations | Heatmap | `sns.heatmap(df.corr(), annot=True)` |
| Multivariate relationships | Pairplot | `sns.pairplot(df[top_features])` |
| Residuals (post-model) | Residual/Q-Q plot | `sns.residplot(x=preds, y=y_test-preds)` |

[⬆ Back to top](#-table-of-contents)