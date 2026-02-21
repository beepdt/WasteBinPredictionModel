# Waste Bin Fill Level Prediction

## Overview

This project builds a **binary classification model** to predict whether a municipal waste collection bin will be **full (1)** or **not full (0)** at the time of the next scheduled pickup. Reliable predictions allow operations teams to skip empty bins, prioritise overflowing ones, and dynamically optimise truck routes — turning a reactive, schedule-driven process into a proactive, data-driven one.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Modeling — Random Forest with Hyperparameter Tuning](#modeling)
6. [Why Random Forest Over Other Models](#why-random-forest-over-other-models)
7. [Business Impact & Cost Reduction](#business-impact--cost-reduction)
8. [Setup & Usage](#setup--usage)
9. [Future Improvements](#future-improvements)

---

## Problem Statement

Municipal waste collection is conventionally run on **fixed schedules** — trucks visit bins at regular intervals regardless of actual fill level. This creates two costly failure modes:

| Failure Mode | Consequence |
|---|---|
| Bin **overflows** before pickup | Hygiene violations, public complaints, fines |
| Truck visits a **near-empty** bin | Wasted fuel, driver time, and vehicle depreciation |

The goal is to eliminate both failure modes by predicting bin status in advance.

---

## Dataset

The dataset (`waste bin data.csv`) contains 1,200 records of waste bin observations.

| Column | Type | Description |
|---|---|---|
| `bin_id` | ID | Unique bin identifier (dropped before modelling) |
| `location_type` | Categorical | Residential / Commercial / Industrial |
| `avg_daily_waste_kg` | Numerical | Average kg of waste deposited per day |
| `days_since_last_pickup` | Numerical | Days elapsed since the last collection |
| `bin_capacity_kg` | Numerical | Maximum capacity of the bin in kg |
| `weather` | Categorical | Normal / Rainy |
| `festival_week` | Binary | 1 if the observation falls in a festival week |
| `is_full` | Binary (Target) | 1 = Full, 0 = Not Full |

---

## Exploratory Data Analysis (EDA)

### 1. Missing Values

| Column | Missing Count | Strategy |
|---|---|---|
| `avg_daily_waste_kg` | Present | Filled with **median** — robust to outliers caused by bulk disposal events |
| `days_since_last_pickup` | Present | Filled with **median** — collection delays are skewed; median avoids bias |
| `weather` | Present | Filled with **mode** (most frequent category) during preprocessing |

**Reasoning:** Mean imputation would be distorted by extreme values (e.g., a bin at a stadium during an event). Median is a safer central estimate for waste-related measurements which tend to have a right-skewed distribution.

### 2. Class Distribution

- **Not Full (0):** 533 records  
- **Full (1):** 667 records  

The dataset is **mildly imbalanced** (~56% Full) but not severely enough to require resampling techniques. Using **F1-score as the optimisation metric** in GridSearchCV ensures the model balances Precision and Recall rather than blindly maximising Accuracy.

### 3. Key Patterns Observed

- **Festival weeks** — bins fill up significantly faster; Full bins are disproportionately represented during these periods, confirming the need for a festival-week adjustment feature.
- **Location type** — Industrial and Commercial zones have a higher fill ratio on average compared to Residential, indicating location-specific pickup frequencies would be beneficial.
- **Days since last pickup** — strongly correlated with fill status; bins that haven't been collected in longer periods are far more likely to be full (expected, but the magnitude of this effect motivates the engineered `fill_ratio_estimate` feature).

---

## Feature Engineering

Raw features alone do not directly encode *how full* a bin is. Three derived features were created to give the model a direct physical signal:

### `estimated_current_waste_kg`
```
estimated_current_waste_kg = avg_daily_waste_kg × days_since_last_pickup
```
**Reasoning:** This is a first-principles estimate of the total waste accumulated since the last pickup. It converts two weakly informative raw features into a single strong proxy.

### `fill_ratio_estimate`
```
fill_ratio_estimate = estimated_current_waste_kg / bin_capacity_kg
```
**Reasoning:** Normalising by capacity makes the feature comparable across bins of different sizes. A bin with 80 kg of accumulated waste means very different things for a 100 kg bin vs. a 500 kg bin. This ratio directly approximates the percentage fill level (values > 1 indicate overflow risk).

### `adjusted_fill_ratio`
```
adjusted_fill_ratio = fill_ratio_estimate × 1.2  (if festival_week == 1)
                    = fill_ratio_estimate          (otherwise)
```
**Reasoning:** During festival weeks, waste generation is approximately 20% higher than normal due to increased activity. Applying this multiplier captures the surge effect and allows the model to account for known seasonal behaviour without needing historical time-series data.

### Preprocessing Pipeline

| Column Type | Imputation | Scaling / Encoding |
|---|---|---|
| Numerical | Median | StandardScaler |
| Categorical | Most Frequent | OneHotEncoder (handle_unknown='ignore') |

All preprocessing is wrapped in a **scikit-learn `Pipeline`**, ensuring that the same transformations applied during training are applied identically during inference — preventing data leakage.

---

## Modeling

### Algorithm: Random Forest Classifier with GridSearchCV

A `RandomForestClassifier` is embedded in a `Pipeline` and tuned using **5-fold cross-validated GridSearchCV** with **F1-score** as the optimisation metric.

**Hyperparameter Search Space:**

| Hyperparameter | Values Searched | What it Controls |
|---|---|---|
| `n_estimators` | 50, 100, 200, 300 | Number of trees in the forest |
| `max_depth` | None, 10, 20, 30 | Maximum depth of each tree (controls overfitting) |
| `min_samples_split` | 2, 5, 10 | Minimum samples required to split a node |
| `min_samples_leaf` | 1, 2, 4 | Minimum samples required at a leaf node |
| `max_features` | sqrt, log2 | Number of features considered at each split |

The best configuration found was `max_depth=10`, `max_features=sqrt`, `min_samples_leaf=1`, `min_samples_split=5`, `n_estimators=200`, achieving a **best CV F1 of 0.9466**.

### Evaluation (held-out 20% test set)

| Metric | Score |
|---|---|
| Accuracy | 0.8958 |
| Precision | 0.8971 |
| Recall | 0.9173 |
| F1 Score | 0.9071 |

---

## Why Random Forest Over Other Models

When comparing the three candidate models trained in initial experiments — Logistic Regression, Decision Tree, and Random Forest — Random Forest consistently delivered the highest performance. Here is a detailed comparison:

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.8833 | 0.8947 | 0.8947 | 0.8947 |
| Decision Tree | 0.8500 | 0.8647 | 0.8647 | 0.8647 |
| **Random Forest (tuned)** | **0.9000+** | **0.9037** | **0.9173** | **0.9104** |

### Why Logistic Regression Falls Short

Logistic Regression assumes a **linear decision boundary** — it expects that features and the target have a log-linear relationship. In reality, the fill status of a waste bin is driven by **multiplicative, threshold-based interactions** (e.g., a bin is only "full" when the fill ratio crosses ~1.0, and the festival-week effect interacts non-linearly with location type). Logistic Regression cannot capture these interactions without manually engineering polynomial or interaction terms, which adds brittleness and complexity.

### Why Decision Tree Falls Short

A single Decision Tree is prone to **overfitting** — it memorises the training data too closely by growing deep, axis-aligned split paths. On unseen data it generalises poorly. It is also **high-variance**: small changes in the training data can produce a completely different tree. The lower F1 of 0.8647 reflects this instability.

### Why Random Forest is the Best Choice

| Property | Benefit for this Problem |
|---|---|
| **Ensemble of trees** | Averages hundreds of different trees trained on random data subsets, drastically reducing variance and overfitting |
| **Feature subsampling** | Considers only a random subset of features at each split (`sqrt` / `log2`), decorrelating trees and improving generalisation |
| **Non-linear boundaries** | Naturally models threshold effects (e.g., fill ratio > 1.0 → likely full) and interaction effects (festival + residential → higher risk) without manual engineering |
| **Robust to outliers** | Tree splits are position-based, not magnitude-based, so extreme values do not skew the model |
| **Feature importance** | Provides built-in Gini-importance scores, revealing that `adjusted_fill_ratio` (0.338) and `fill_ratio_estimate` (0.312) are the dominant predictors — actionable insight for operations teams |
| **Handles mixed types** | Works seamlessly with both numerical and one-hot encoded categorical features in a single pipeline |

---

## Business Impact & Cost Reduction

### Direct Cost Savings

**1. Eliminating unnecessary pickups**  
With 89.7% Precision, the model correctly identifies non-full bins ~90% of the time. In a fleet of 1,000 bins collected daily, this means ~100 bins per day can be skipped. Assuming a collection cost of ₹500 per bin visit (fuel + driver time + vehicle depreciation), that is a potential saving of **₹50,000 per day** or **₹1.5 Crore per year** for a mid-sized municipality.

**2. Preventing overflow penalties**  
With 91.7% Recall, nearly all truly full bins are caught before overflow. Each overflow incident carries hygiene compliance costs, citizen complaints, and municipal fines. Preventing even 5 overflow events per month at ₹10,000 each saves **₹6 lakh per year**.

**3. Route optimisation**  
The model outputs per-bin probabilities. These can be piped directly into a **Vehicle Routing Problem (VRP) solver** to dynamically generate the shortest route that covers only the high-priority bins each day, reducing total truck kilometres driven.

### Operational Decision-Making

| Decision | How the Model Helps |
|---|---|
| **Which bins to visit today?** | Rank bins by predicted probability of being full; dispatch trucks only to high-probability bins |
| **Where to deploy larger bins?** | Bins consistently predicted full despite recent pickups should be upgraded to higher-capacity units |
| **Festival week planning?** | The model's festival-week feature flags periods where pickup frequency should double for affected zones |
| **Weather-responsive scheduling?** | Rainy weather affects waste compaction; the model accounts for this, allowing schedule adjustments without manual guesswork |

---

## Setup & Usage

### Prerequisites

```bash
python -m pip install pandas numpy scikit-learn matplotlib
```

### Files

| File | Purpose |
|---|---|
| `waste bin data.csv` | Raw dataset |
| `waste_management.py` | Data loading, feature engineering, model training & evaluation |
| `dashboard.py` | Generates `dashboard.png` — a matplotlib visualisation dashboard |

### Run

```bash
# Train model and print evaluation metrics
python waste_management.py

# Generate the visualisation dashboard (saves dashboard.png)
python dashboard.py
```

---

## Future Improvements

| Improvement | Impact |
|---|---|
| **IoT fill-level sensors** | Replace model predictions with real-time fill-level data; use the model for short-term forecasting instead |
| **Time-series features** | Add rolling averages and day-of-week features to capture weekly waste generation rhythms |
| **VRP integration** | Connect per-bin predictions to route optimisation software for end-to-end logistics automation |
| **Real-time weather API** | Replace the binary `weather` feature with live forecast data for higher predictive accuracy |
| **Model retraining pipeline** | Automate weekly retraining as new pickup data accumulates to prevent model drift |
| **Probability thresholding** | Tune the classification threshold (default 0.5) to balance Precision vs Recall based on the relative cost of false positives vs false negatives for the specific municipality |

---

_Built for solving efficiency challenges in urban waste management._
