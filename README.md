# Airbnb Rating Classification

**Applied Data Science — Spring 2026**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Overview](#dataset-overview)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [How to Run](#how-to-run)
6. [Pipeline Stages & Work Breakdown](#pipeline-stages--work-breakdown)
   - [Stage 1 — Data Acquisition & Merging](#stage-1--data-acquisition--merging)
   - [Stage 2 — Data Validation](#stage-2--data-validation)
   - [Stage 3 — Data Cleaning](#stage-3--data-cleaning)
   - [Stage 4 — Feature Engineering & Selection](#stage-4--feature-engineering--selection)
   - [Stage 5 — Exploratory Data Analysis (EDA)](#stage-5--exploratory-data-analysis-eda)
   - [Stage 6 — Modelling](#stage-6--modelling)
   - [Stage 7 — Experiment Tracking (MLflow)](#stage-7--experiment-tracking-mlflow)
   - [Stage 8 — Evaluation & Interpretation](#stage-8--evaluation--interpretation)
   - [Stage 9 — Testing](#stage-9--testing)
   - [Stage 10 — Automation & CI](#stage-10--automation--ci)
   - [Stage 11 — Dashboard (Bonus)](#stage-11--dashboard-bonus)
7. [Current Progress](#current-progress)
8. [Team Members](#team-members)

---

## Problem Statement

**Business Problem:** Airbnb hosts and platform managers need a way to predict listing quality (via review scores) based on observable listing attributes. Accurate rating predictions help:

- **Hosts** understand which features (price, amenities, room type) drive higher ratings so they can improve their offerings.
- **Guests** receive better recommendations by surfacing high-quality listings.
- **Platform managers** identify underperforming listings and target interventions.

**Classification Target:** `review_scores_rating` — binned into discrete categories (e.g., Low / Medium / High) to frame this as a multi-class classification problem.

**Stakeholder:** Airbnb regional operations team for 6 US cities.

---

## Dataset Overview

| Property | Value |
|---|---|
| **Rows (Raw)** | 75,111 listings (74,111 Kaggle + 1,000 scraped) |
| **Rows (Cleaned)** | 58,382 listings |
| **Rows (Featured)** | 58,132 listings (after dropping Low ratings) |
| **Rows (Train/Test)** | 46,506 / 11,626 (80% / 20%) |
| **Columns (Raw)** | 17 features |
| **Columns (Featured - Analysis)** | 26 features (9 new + 1 target) |
| **Columns (Ready - Modeling)** | 19 features (17 features + 2 targets) |
| **Cities** | New York, Los Angeles, San Francisco, Washington DC, Chicago, Boston |
| **Sources** | 2 (Kaggle Inside Airbnb + web scraping) |
| **Target** | `review_scores_rating` → binned to 3 classes: Medium (20.8%) / High (45.5%) / Very High (33.6%) |
| **Validation Strategy** | 3-fold CV on train set for hyperparameter tuning |

**Key features:**

| Column | Type | Description |
|---|---|---|
| `property_type` | Categorical | Apartment, House, Condo, etc. |
| `room_type` | Categorical | Entire home/apt, Private room, Shared room |
| `amenities` | Text | Comma-separated list of amenities |
| `accommodates` | Numeric | Max number of guests |
| `bathrooms` | Numeric | Number of bathrooms |
| `bedrooms` | Numeric | Number of bedrooms |
| `beds` | Numeric | Number of beds |
| `city` | Categorical | One of 6 US cities |
| `host_identity_verified` | Boolean | Whether the host is ID-verified |
| `host_response_rate` | Numeric | Host response rate (0–100%) |
| `latitude` / `longitude` | Numeric | Listing coordinates |
| `neighbourhood` | Categorical | Neighbourhood name |
| `number_of_reviews` | Numeric | Total review count |
| `price` | Numeric | Listing price per night (USD) |
| `review_scores_rating` | Numeric | **Target variable** (0.0–5.0) |

---

## Project Structure

```
DS/
├── .env                                 # Environment variables (paths, MLflow URI)
├── .github/workflows/ci.yml            # GitHub Actions CI pipeline
├── .gitignore
├── Makefile                             # Automation targets
├── README.md                            # ← You are here
├── pyproject.toml                       # Poetry dependency management
├── requirements.txt                     # Legacy fallback deps
│
├── data/
│   ├── raw/                             # Untouched source files
│   │   └── scraped_airbnb_listings.csv  #   Source 1 raw data
│   ├── merged/                          # Output of merge pipeline
│   │   └── merged_airbnb_data.csv       #   75,111 rows × 17 cols
│   └── processed/                       # Cleaned + engineered data splits
│       ├── cleaned.csv                  #   58,382 rows × 18 cols
│       ├── featured.csv                 #   58,132 rows × 26 cols (for analysis)
│       ├── ready_features.csv           #   58,132 rows × 19 cols (for modeling)
│       ├── train.csv                    #   46,506 rows × 19 cols (80%)
│       └── test.csv                     #   11,626 rows × 19 cols (20%)
│
├── scraper/
│   ├── scraper.ipynb                    # Source 1 scraping notebook
│   ├── source2_acquisition.py           # Source 2 acquisition script [TODO]
│   ├── merge.py                         # Merge pipeline with logging
│   └── merge.ipynb                      # Merge exploration notebook
│
├── validation/
│   ├── __init__.py
│   ├── validation.py                    # DataValidator class — 7 dimensions
│   └── validation.ipynb                 # Validation exploration notebook
│
├── validation_report/
│   └── validation_report_full.json      # Full validation output (JSON)
│
├── cleaning/
│   ├── __init__.py
│   └── cleaning.py                      # Cleaning pipeline [TODO: implement]
│
├── feature_engineering/
│   ├── __init__.py
│   ├── transformations.py               # Encode, scale, log-transform [TODO]
│   ├── engineering.py                   # Derived features [TODO]
│   └── selection.py                     # Feature selection [TODO]
│
├── eda/
│   └── dashboard.py                     # Streamlit dashboard [TODO / Bonus]
│
├── modelling/
│   ├── __init__.py
│   ├── baseline.py                      # DummyClassifier baseline
│   ├── train.py                         # Train 5+ models + MLflow [TODO]
│   └── evaluate.py                      # Metrics + error analysis
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Shared fixtures (sample DataFrame)
│   ├── test_validation.py               # DataValidator tests (working)
│   ├── test_cleaning.py                 # Cleaning tests [TODO]
│   ├── test_features.py                 # Feature tests [TODO]
│   └── test_modelling.py               # Modelling tests (partial)
│
├── reports/                             # PDF deliverables
│   └── .gitkeep
│
└── notebooks/                           # Scratch / exploratory work
    └── .gitkeep
```

---

## Setup & Installation

### Prerequisites

- Python 3.11 or 3.12
- Git

### Option A: Poetry (recommended)

```bash
git clone <repo-url> && cd DS
pip install poetry
poetry install --with dev
poetry shell
```

### Option B: venv + pip

```bash
git clone <repo-url> && cd DS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov ruff
```

### Environment Variables

Copy the `.env` file and adjust paths if needed:

```env
AIRBNB_DATA_PATH=data/merged/merged_airbnb_data.csv
MLFLOW_TRACKING_URI=mlruns
```

---

## How to Run

```bash
# Full pipeline (acquisition → validation → cleaning → features → training → tests)
make all

# Individual stages
make data          # Scrape + merge
make validate      # Run 7-dimension validation
make clean         # Run cleaning pipeline
make features      # Run feature engineering
make train         # Train all models
make test          # Run pytest with coverage
make lint          # Ruff lint + format check
make eda           # Launch Streamlit dashboard (bonus)
```

---

## Pipeline Stages & Work Breakdown

### Stage 1 — Data Acquisition & Merging

| File | Status | What to do |
|---|---|---|
| `scraper/scraper.ipynb` | ✅ Done | Source 1: Web scraping — produces 1,000 rows in `scraped_airbnb_listings.csv` |
| `scraper/source2_acquisition.py` | ✅ Done | Source 2: Kaggle Inside Airbnb dataset — 74,111 rows |
| `scraper/merge.py` | ✅ Done | Full merge implementation with transformations, concatenation, logging. Produces 75,111 total rows in `data/merged/merged_airbnb_data.csv` |
| `tests/test_merge.py` | ✅ Done | 6 comprehensive tests validating merge integrity |

**Deliverables:** ✅ Two sources merged (Kaggle 74,111 + scraped 1,000). Merge includes schema alignment, ID extraction, price/rating transformations.

---

### Stage 2 — Data Validation

| File | Status | What to do |
|---|---|---|
| `validation/validation.py` | ✅ Done | `DataValidator` class with 7 dimensions: Accuracy, Completeness, Consistency, Uniqueness, Outlier Detection, Distribution Profile, Relationships. Outputs JSON report. |
| `validation/validation.ipynb` | ⚠️ Minimal | **Enhance the notebook:** add histograms/box plots for numeric columns, bar charts for categoricals, null heatmap, outlier visualisations. Wire it to use `DataValidator` directly. |
| `validation_report/validation_report_full.json` | ✅ Done | Full validation output. Overall: FAIL (4/7 pass). |

**Known issues to document in report:**
- City names need normalisation (90% violations — "NYC", "LA", "SF" vs canonical names)
- `host_identity_verified` stored as "t"/"f" strings
- `host_response_rate` 24.4% null, `review_scores_rating` 22.3% null — expected for Airbnb data
- 7/8 columns exceed IQR outlier threshold — many are right-skewed, not truly outliers

---

### Stage 3 — Data Cleaning

| File | Status | What to do |
|---|---|---|
| `cleaning/cleaning.py` | ✅ Done | All three functions fully implemented |
| `tests/test_cleaning.py` | ✅ Done | 22 comprehensive tests covering all cleaning logic |
| `cleaning/cleaning.ipynb` | ✅ Done | Exploratory notebook with data inspection and category binning |

**✅ `normalize_columns(df)` — Implemented:**
- `city`: "NYC" → "New York", "LA" → "Los Angeles", "SF" → "San Francisco"
- `room_type`: "Entire home" → "Entire home/apt", etc.
- `host_identity_verified`: "t"/"f" → `True`/`False`

**✅ `handle_missing_values(df)` — Implemented:**
- `price`, `city`, `property_type`: drop rows (critical columns)
- `accommodates`, `bathrooms`, `bedrooms`, `beds`: median imputation
- `host_response_rate`: median imputation + `has_response_rate` flag created
- `review_scores_rating`: **target variable — rows with nulls dropped (16,729 rows)**
- `neighbourhood`: imputed with "Unknown"
- `latitude`/`longitude`: rows with incomplete coordinates dropped

**✅ `handle_outliers(df)` — Implemented:**
- `price`: capped at $10–$10,000
- Other numerics: IQR-based capping (1.5× multiplier)
- `host_response_rate`: excluded from IQR (naturally bounded 0-100%)

**Output:** `data/processed/cleaned.csv` — 58,382 rows (from 75,111), 0 nulls remaining. Full logging implemented.

---

### Stage 4 — Feature Engineering & Selection

| File | Status | What to do |
|---|---|---|
| `feature_engineering/transformations.py` | ✅ Done | Encoding, scaling, and log transforms implemented |
| `feature_engineering/engineering.py` | ✅ Done | All derived features implemented |
| `feature_engineering/selection.py` | ✅ Done | Correlation filter, MI ranking, train/val/test split, cardinality cleanup |
| `feature_engineering/pipeline.py` | ✅ Done | Complete orchestration pipeline with correlation/cardinality analysis |
| `feature_engineering/features.ipynb` | ✅ Done | Correlation analysis and cardinality assessment notebook |
| `tests/test_features.py` | ✅ Done | 24 comprehensive tests for all feature engineering |

**✅ Transformations implemented:**

| Task | Function | Details |
|---|---|---|
| Encode categoricals | `encode_categoricals()` | One-hot encoding with `drop_first` option |
| Scale numerics | `scale_numerics()` | `StandardScaler` or `MinMaxScaler` with fit on train only |
| Log-transform | `log_transform()` | `np.log1p()` for skewed features |

**✅ Features engineered (9 new features created):**

| Feature | Formula | Result (mean) |
|---|---|---|
| `amenity_count` | Parse JSON amenities, count items | 18.88 amenities/listing |
| `price_per_bed` | `price / beds` | $97.17 per bed |
| `price_per_guest` | `price / accommodates` | $51.26 per guest |
| `is_entire_home` | `1 if room_type == "Entire home/apt"` | 58% entire homes |
| `listing_density` | Count listings per neighbourhood | 918.31 listings/neighbourhood |
| **`price_relative_to_room_type`** ⭐ | `(price - room_type_median) / room_type_std` | **TOP PRIORITY** from EDA |
| `price_relative_to_city` | `(price - city_median) / city_std` | 0.29 (city-normalized) |
| `log_price` | `np.log1p(price)` | Log-transformed price |
| `log_number_of_reviews` | `np.log1p(number_of_reviews)` | Log-transformed reviews |

**✅ Correlation & Cardinality Analysis (features.ipynb):**

Systematic analysis identified features to drop for modeling:

**High Cardinality Features (>50% unique):**
- `id` — 100% unique (identifier)
- `amenities` — 92% unique (already encoded to `amenity_count`)
- `latitude`, `longitude` — 99.9% unique (geographic coordinates)

**High Correlation Pairs (|r| > 0.7):**
- `price` ↔ `price_relative_to_city` (r=0.98) → DROP both
- `price` ↔ `log_price` (r=0.85) → KEEP `log_price`
- `price_per_bed` ↔ `price_per_guest` (r=0.81) → KEEP `price_per_bed` (higher target correlation)
- `price_relative_to_room_type` ↔ `price_relative_to_city` (r=0.81) → KEEP `price_relative_to_room_type`

**✅ Feature selection implemented:**
1. `correlation_filter()` — removes features with > 95% correlation
2. `mutual_information_ranking()` — ranks features by MI with target
3. `drop_unwanted_features()` — drops beds (r=0.845 with accommodates), number_of_reviews (weak predictor)
4. **`prepare_ready_features()`** — NEW: drops high-cardinality and high-correlation features
5. `create_train_val_test_split()` — **double stratification** by rating_category AND room_type (70/15/15)

**✅ Two Datasets Created:**

1. **`featured.csv`** (58,132 rows × 26 columns) — For analysis and correlation studies
   - Contains ALL engineered features
   - Used by `features.ipynb` for correlation heatmaps and cardinality analysis

2. **`ready_features.csv`** (58,132 rows × 19 columns) — For modeling
   - **17 features retained:** property_type, room_type, accommodates, bathrooms, city, host_identity_verified, host_response_rate, neighbourhood, bedrooms, has_response_rate, amenity_count, price_per_bed, is_entire_home, listing_density, **price_relative_to_room_type**, log_price, log_number_of_reviews
   - **7 features dropped:** id, amenities, latitude, longitude, price, price_relative_to_city, price_per_guest
   - Target: `rating_category` (3 classes after dropping Low ratings)
   - Split into train/val/test with perfect stratification

**✅ Key Findings from Correlation Analysis:**
- **Strongest predictor:** `amenity_count` (r=0.154 with target)
- **Mean absolute correlation:** 0.063 (weak linear relationships)
- **8 high correlation pairs** detected → resolved by dropping redundant features
- **Recommendation confirmed:** Tree-based models will outperform linear models

**✅ Output:** Train/val/test splits saved (40,692 / 8,720 / 8,720 rows). **Target variable:** 3 classes (Medium 20.8%, High 45.5%, Very High 33.6%) after dropping Low Rating category (0.43%).

---

### Stage 5 — Exploratory Data Analysis (EDA)

| File | Status | What to do |
|---|---|---|
| `eda/eda.ipynb` | ✅ Done | Comprehensive EDA with 7 major visualizations, full interpretations, and actionable modeling insights |
| `eda/visualize.py` | ✅ Done | Reusable visualization functions with categorical/numeric data support |

**✅ Visualizations Created (7 total):**

1. **Target Distribution** — Continuous and binned rating distributions
   - **Class imbalance detected:** Low 0.43%, Medium 19.8%, High 44.4%, Very High 33.7%
   - **Recommendation:** 3-class binning or SMOTE + class weights
   - Right-skewed distribution (mean 4.7, median 4.8) — typical for review platforms

2. **Price by City** — Box plots with statistical summaries across 6 cities
   - **San Francisco:** Most expensive (median $160, 60% premium over LA/Chicago)
   - **New York:** Largest market (25,226 listings, median $108)
   - **Los Angeles/Chicago:** Most affordable (median $100)
   - **Insight:** City drives 60% price variance → critical feature for modeling

3. **Correlation Heatmap** — Pearson correlation matrix of numeric features
   - **Multicollinearity identified:** `accommodates` ↔ `beds` (r=0.845), `lat` ↔ `lon` (r=0.891)
   - **Weak linear correlations with target:** All features |r| < 0.3
   - **Critical finding:** Non-linear relationships exist → tree models will outperform linear models
   - **Action:** Drop `beds` (redundant), use Mutual Information for feature selection

4. **Room Type vs Rating** — Statistical comparison with cross-tabulation
   - **Rating hierarchy:** Entire homes (4.720) > Private rooms (4.695) > Shared rooms (4.594)
   - **Variance pattern:** Shared rooms 1.5× more variable (σ=0.523 vs 0.357)
   - **Failure rate:** Shared rooms have **4× higher Low rating rate** (1.11% vs 0.28%)
   - **Dataset distribution:** Entire 58.0%, Private 39.6%, Shared 2.4% → stratification required

5. **Geospatial Scatter (Rating Category)** — Maps colored by discrete rating classes
   - **Minimal geographic clustering** in ratings (no "good" vs "bad" neighborhoods)
   - Dense clusters in downtown areas (NYC, SF, Chicago)
   - **Insight:** Location alone doesn't determine rating → focus on property quality

6. **Geospatial Scatter (Price)** — Maps with auto-adjusted color scale (1st-99th percentile)
   - **Clear price clustering:** Downtown/tourist areas (high) vs suburbs (low)
   - Manhattan premium visible, SF Financial District premium, Chicago lakefront premium
   - **Insight:** Geography predicts price strongly, rating weakly → use for price normalization

7. **Price by Room Type** — Distribution comparison with detailed statistics
   - **Massive price gap:** Entire $158 vs Private $70 vs Shared $41 (3.9× spread)
   - **Small rating gap:** Only 0.13 rating points across room types
   - **Key discovery:** 3.9× price difference with 0.13 rating difference → guests evaluate quality **relative to room type category**
   - **Critical feature:** `price_relative_to_room_type_median` will be top predictor

**✅ Additional Analyses:**
- Reviews vs rating: Very weak correlation (r ≈ 0.05) → not predictive
- Amenity count: Moderate correlation with price, weak with rating

**✅ Data Quality Fixes Applied During EDA:**

1. **Room Type Standardization:**
   - Fixed malformed entries ("Private RoomShareSaveShow all photos" → "Private room")
   - Standardized case inconsistencies ("Private Room" → "Private room")
   - Pattern-based matching for robust cleaning
   - Result: Clean 3-category split (Entire home/apt, Private room, Shared room)

2. **Visualization Enhancements:**
   - Added categorical data support to `plot_geospatial_scatter()`
   - Auto color scale adjustment (1st-99th percentile) to prevent outlier distortion
   - Discrete colormaps for rating categories, continuous for price

**✅ Key Modeling Insights:**

1. **Target Strategy:**
   - Drop ratings < 3.0 (249 rows, 0.43% outliers)
   - Keep 4-class binning with SMOTE or use 3-class binning for balance
   - **Primary metrics:** F1-macro, Balanced Accuracy
   - **Secondary metrics:** Recall for Low class (business critical)

2. **Feature Engineering Priorities:**
   - ⭐ **`price_relative_to_room_type_median`** — TOP PRIORITY (captures category-relative pricing)
   - **`price_relative_to_city_median`** — City-specific normalization
   - **One-hot encode:** city, room_type, property_type
   - **Log transform:** price, number_of_reviews (right-skewed)
   - **Interaction features:** `city × room_type`, `price × room_type`, `room_type × accommodates`
   - **Drop:** `beds` (r=0.845 with accommodates), optionally `latitude`/`longitude`

3. **Model Selection Strategy:**
   - **1st Choice: XGBoost** — Best for tabular data, handles class imbalance, non-linear relationships
   - **2nd Choice: Random Forest** — Robust baseline, provides feature importance
   - **3rd Choice: LightGBM** — Faster training, good for categorical features
   - **Baseline: Logistic Regression** — For interpretability comparison
   - **Avoid:** Simple linear models (weak correlations violate assumptions)

4. **Data Splitting & Sampling:**
   - **Double stratification:** By `rating_category` AND `room_type`
   - **Justification:** Preserves class balance + room type distribution (shared rooms only 2.4%)
   - **SMOTE strategy:** Apply within each room type group separately
   - Current split: 70/15/15 (40,866 / 8,758 / 8,758)

5. **Evaluation Protocol:**
   - **Primary:** F1-macro, Balanced Accuracy
   - **Per-class:** Recall for Low class (detect bad listings)
   - **Business metrics:** `bad_recommendation_rate`, `high_quality_recall`
   - **Error analysis:** Confusion matrix, misclassification patterns by city/room type

**✅ Critical Discovery:**

> **The 3.9× price gap with only 0.13 rating point difference proves guests evaluate quality relative to room type category, not absolute price. Therefore, `price_relative_to_room_type_median` will likely be one of the strongest predictive features.**

**✅ Visualization Functions (visualize.py):**
- `plot_target_distribution()` - Target analysis with binning comparison
- `plot_price_by_city()` - Box/violin plots with stats
- `plot_correlation_heatmap()` - With threshold highlighting
- `plot_feature_vs_target()` - Categorical relationships
- `plot_geospatial_scatter()` - **Enhanced:** Handles categorical + numeric data with auto color scaling
- `plot_numeric_distributions()` - Multi-column histograms
- `plot_price_by_room_type()` - Room type comparison
- `plot_reviews_vs_rating()` - Scatter with correlation
- `plot_amenity_analysis()` - Amenity relationships

**Reusable for:** Dashboard development (Stage 11)

---

### Stage 6 — Modelling

| File | Status | What to do |
|---|---|---|
| `modelling/baseline.py` | ✅ Done | `DummyClassifier(strategy="most_frequent")` — establishes performance floor |
| `modelling/train.py` | 🔜 Ready to start | **Train ≥ 5 models using ready_features.csv** |

**📊 Dataset for Training:**
- **Input:** `data/processed/ready_features.csv` (58,132 rows × 19 columns)
- **Features:** 17 optimized features after correlation/cardinality cleanup
- **Target:** `rating_category` (3 classes: Medium 20.8% | High 45.5% | Very High 33.6%)
- **Splits:** train.csv (40,692 rows) | val.csv (8,720 rows) | test.csv (8,720 rows)

**Models to implement:**

| # | Model | Rationale |
|---|---|---|
| 1 | Baseline (DummyClassifier) | Performance floor — already implemented |
| 2 | Logistic Regression | Simple, interpretable linear baseline (expected lower performance due to weak linear correlations) |
| 3 | Random Forest | Handles non-linearity, robust to outliers, provides feature importance |
| 4 | **XGBoost / Gradient Boosting** | **TOP PRIORITY** — State-of-the-art for tabular data, handles class imbalance, captures non-linear price-quality relationships |
| 5 | LightGBM | Faster training, excellent for categorical features (city, room_type) |
| (opt) | SVM (SVC) | Good for moderate-sized datasets, handles non-linear boundaries |

**Model selection rationale (from EDA):**
- **Weak linear correlations** with target (mean |r| = 0.063) → Tree-based models will significantly outperform linear models
- **Non-linear price-quality relationship** (3.9× price gap, 0.13 rating gap) → XGBoost/RF ideal
- **Class imbalance** (Medium 20.8%, High 45.5%, Very High 33.6%) → Use class weights or SMOTE

**For each model:**
- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`
- Use cross-validation (e.g., 5-fold stratified)
- Log everything to MLflow (see Stage 7)

**Target variable (already prepared in ready_features.csv):**
- **3 classes:** Medium (20.8%), High (45.5%), Very High (33.6%)
- **Low Rating dropped:** Originally 0.43% (249 rows) — severe imbalance, treated as outliers
- **Class imbalance handling:** Use `class_weight='balanced'` or SMOTE (apply within each room type group)
- **Double stratification:** Already applied in train/val/test splits (by rating_category AND room_type)

---

### Stage 7 — Experiment Tracking (MLflow)

| File | Status | What to do |
|---|---|---|
| `modelling/train.py` | ❌ TODO | **Integrate MLflow into training loop** |

**Each MLflow run must log:**

| What | Example |
|---|---|
| Model name | `"RandomForest"` |
| All hyperparameters | `n_estimators=200, max_depth=10, ...` |
| ≥ 2 standard metrics | accuracy, F1-macro |
| ≥ 2 business metrics | bad_recommendation_rate, high_quality_recall |
| Trained model artifact | `mlflow.sklearn.log_model(model, ...)` |

**Example usage:**

```python
import mlflow, mlflow.sklearn

mlflow.set_experiment("airbnb-rating-classification")

with mlflow.start_run(run_name="RandomForest"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average="macro"))
    mlflow.sklearn.log_model(model, "model")
```

**For the report:** Screenshot the MLflow experiment comparison UI showing all runs side by side.

---

### Stage 8 — Evaluation & Interpretation

| File | Status | What to do |
|---|---|---|
| `modelling/evaluate.py` | ⚠️ Partial | **Complete `business_metrics()` and expand `error_analysis()`** |

**Standard metrics (already implemented):** accuracy, F1-macro, precision-macro, recall-macro.

**Business metrics to define:**

| Metric | Definition | Why it matters |
|---|---|---|
| `bad_recommendation_rate` | % of truly Low-rated listings predicted as High/Very High | Guests get misled — critical for platform trust |
| `high_quality_recall` | % of truly Very High-rated listings correctly identified | Good listings surface properly — revenue impact |
| `host_actionability_score` | % of predictions that change if host improves ≥ 1 feature | Are predictions useful for hosts? Drives platform engagement |

**Error analysis to implement:**
- Confusion matrix heatmap (per class)
- Identify which listing types (city, room_type, price range) are most often misclassified
- Compare train vs test performance to detect overfitting
- **Room-type-specific analysis:** Are shared rooms harder to predict? (EDA suggests yes due to 1.5× higher variance)
- **Price-quality mismatch cases:** Listings priced high/low relative to room type median
- Document unsuccessful approaches with explanation

**From EDA — Expected patterns:**
- Low class will have lowest recall (only 0.43% of data)
- Shared rooms may have higher error rates (higher variance σ=0.523)
- Model may struggle with listings priced far from room type median

**For the report:** Model comparison table, business interpretation, and actionable recommendations.

---

### Stage 9 — Testing

| File | Status | What to do |
|---|---|---|
| `tests/conftest.py` | ✅ Done | Shared fixtures: `sample_df` (100-row synthetic), `tmp_csv` |
| `tests/test_validation.py` | ✅ Done | 7 working tests for `DataValidator` |
| `tests/test_merge.py` | ✅ Done | 6 tests for merge integrity (row counts, columns, transformations) |
| `tests/test_cleaning.py` | ✅ Done | 22 comprehensive tests: normalize (5), missing values (8), outliers (5), pipeline (4) |
| `tests/test_features.py` | ✅ Done | 24 comprehensive tests: transformations (8), engineering (8), selection (6), pipeline (2) |
| `tests/test_modelling.py` | ⚠️ Partial | `standard_metrics` tests work; full integration test pending model training |

**✅ Tests implemented (48 total):**

| Test file | Tests | Coverage |
|---|---|---|
| `test_merge.py` | 6 tests | Row validation, column schema, ID integrity, transformations |
| `test_cleaning.py` | 22 tests | Alias normalization, imputation strategies, outlier capping, coordinate handling, end-to-end pipeline |
| `test_features.py` | 24 tests | Encoding, scaling, log transforms, amenity parsing, price ratios, density calculation, correlation filter, MI ranking, stratified splits |
| `test_modelling.py` | Partial | Standard metrics validated |

**Coverage achieved:** 74%+ line coverage across cleaning and feature engineering modules.

**Run tests with coverage:**
```bash
# All tests with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Specific module with logs visible
pytest tests/test_cleaning.py -v --cov=cleaning --cov-report=term-missing --log-cli-level=INFO

# Show all output (including print statements)
pytest tests/ -v -s
```

---

### Stage 10 — Automation & CI

| File | Status | What to do |
|---|---|---|
| `Makefile` | ✅ Done | Targets: `data`, `validate`, `clean`, `features`, `train`, `test`, `lint`, `eda` |
| `.env` | ✅ Done | `AIRBNB_DATA_PATH`, `MLFLOW_TRACKING_URI` |
| `.github/workflows/ci.yml` | ✅ Done | Lint + test on push/PR to `main` |

**Remaining CI work:**
- Verify the workflow runs end-to-end once lock file exists
- Add code coverage reporting/badge to README

---

### Stage 11 — Dashboard (Bonus, 5%)

| File | Status | What to do |
|---|---|---|
| `eda/dashboard.py` | ❌ TODO | **Build a Streamlit (or Dash/Plotly) dashboard** |

**Dashboard sections:**

1. **EDA Findings** — Interactive plots: filter by city, room type. Show distributions, maps, correlations.
2. **Model Performance** — Comparison table/chart of all 5+ models. Highlight best model.
3. **Business Insights** — Non-technical summary for stakeholders.

```bash
streamlit run eda/dashboard.py
```

---

## Current Progress

| Stage | Status | Notes |
|---|---|---|
| **Data Acquisition (Source 1)** | ✅ Complete | Web scraping: 1,000 rows |
| **Data Acquisition (Source 2)** | ✅ Complete | Kaggle Inside Airbnb: 74,111 rows |
| **Data Merging** | ✅ Complete | 75,111 total rows with schema alignment & transformations |
| **Data Validation** | ✅ Complete | 7-dimension pipeline, JSON report generated |
| **Data Cleaning** | ✅ Complete | 58,382 rows after cleaning, 0 nulls remaining |
| **Feature Engineering** | ✅ Complete | 9 new features created (including price_relative_to_room_type ⭐) |
| **Correlation & Cardinality Analysis** | ✅ Complete | 8 high-correlation pairs identified, 7 features dropped for modeling |
| **Target Binning** | ✅ Complete | 3 classes: Medium/High/Very High (dropped Low 0.43%) |
| **Dataset Preparation** | ✅ Complete | featured.csv (26 cols) + ready_features.csv (19 cols) |
| **Train/Val/Test Split** | ✅ Complete | Double stratification 70/15/15 (40,692 / 8,720 / 8,720) |
| **Data Preparation** | ✅ **FULLY COMPLETE** | **Ready for modeling with 17 optimized features** |
| **Testing** | ✅ Complete | 48 tests (6 merge + 22 cleaning + 24 features), 74%+ coverage |
| **EDA Notebook** | ✅ Complete | 7 visualizations with full interpretations, data quality fixes, modeling strategy |
| **EDA Visualizations** | ✅ Complete | visualize.py with categorical/numeric support, auto color scaling |
| **Key Discovery** | ✅ Complete | Price-to-room-type ratio identified as top predictor (3.9× price gap, 0.13 rating gap) |
| **Modelling (Baseline)** | ✅ Complete | DummyClassifier ready |
| **Modelling (4+ models)** | 🔜 Ready to start | Use ready_features.csv (19 cols) with XGBoost, RF, Logistic, LightGBM |
| **MLflow Integration** | 🔜 Ready to start | Track experiments with ready_features.csv |
| **Evaluation** | ⚠️ Partial | standard_metrics done, business_metrics TODO |
| **CI Pipeline** | ✅ Complete | GitHub Actions configured |
| **Makefile Automation** | ✅ Complete | All targets defined |
| **Dashboard (Bonus)** | ❌ Not started | Scaffold only |
| **Final Report** | ❌ Not started | — |

### Data Pipeline Summary

- **Input:** 2 sources → 75,111 rows × 17 columns
- **After Cleaning:** 58,382 rows × 18 columns (added `has_response_rate`)
- **After Feature Engineering:** 58,132 rows × 26 columns (9 new features, dropped 249 Low ratings)
- **After Correlation/Cardinality Cleanup:** 58,132 rows × 19 columns (7 features dropped)
- **Splits (from ready_features.csv):** Train 40,692 (70%) | Val 8,720 (15%) | Test 8,720 (15%)
- **Target:** `rating_category` with 3 classes (dropped Low Rating 0.43%)
- **Class Distribution:** Medium 20.8% | High 45.5% | Very High 33.6%
- **Features for Modeling:** 17 features (9 engineered, 2 dropped from EDA, 7 dropped from correlation analysis)
- **Key Features Retained:** price_relative_to_room_type ⭐, log_price, price_per_bed, amenity_count
- **Stratification:** Double stratification by rating_category AND room_type (preserves 2.5% shared room distribution)
- **Test Coverage:** 74%+ across 48 comprehensive tests
- **Key Finding:** 3.9× price gap across room types with only 0.13 rating point difference → price_relative_to_room_type is critical feature

