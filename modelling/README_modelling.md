# Model Training & Evaluation

This directory contains the complete training pipeline for Airbnb rating classification.

## Quick Start

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Quick training with reduced hyperparameter grids (recommended for testing)
make train-quick

# Full training with complete hyperparameter search
make train

# View MLflow UI to compare models
make mlflow-ui
# Then open: http://localhost:5000

# Compare models in terminal
make compare-models
```

## Files

### Core Training

- **`config.py`** - Model configurations and hyperparameter grids
  - `MODEL_CONFIGS`: Full hyperparameter grids for production
  - `QUICK_CONFIGS`: Reduced grids for faster testing
  - Feature definitions (categorical, numeric, target)

- **`train.py`** - Main training pipeline with MLflow tracking
  - Loads train/val/test splits from `data/processed/`
  - Preprocesses data (one-hot encoding + scaling)
  - Trains 6 models with hyperparameter tuning
  - Logs all metrics and artifacts to MLflow

- **`evaluate.py`** - Evaluation metrics (standard + business)
  - Standard: accuracy, F1, precision, recall
  - Business: over_promise_rate, undersell_rate, high_quality_recall, etc.

- **`baseline.py`** - DummyClassifier baseline

### Helper Scripts

- **`quick_train.py`** - Fast training with reduced grids
- **`compare_models.py`** - Compare all MLflow runs in a table

## Models Trained

Based on EDA findings (weak linear correlations, non-linear relationships):

1. **Baseline** (DummyClassifier) - Performance floor
2. **Logistic Regression** - Interpretable linear baseline
3. **Random Forest** - Robust to non-linearity
4. **XGBoost** ⭐ - TOP PRIORITY from EDA (best for tabular data)
5. **LightGBM** - Fast training, good for categorical features
6. **Gradient Boosting** - Alternative ensemble method

## Training Details

### Data
- **Input**: `data/processed/ready_features.csv` (58,132 rows × 19 cols)
- **Features**: 17 features after correlation/cardinality cleanup
- **Target**: `rating_category` with 3 classes
  - Medium Rating: 20.8%
  - High Rating: 45.5%
  - Very High Rating: 33.6%
- **Splits**: train (40,692) | val (8,720) | test (8,720)

### Preprocessing
- **Categorical features** (5): property_type, room_type, city, neighbourhood, host_identity_verified
  - One-hot encoded with `drop='first'`
- **Numeric features** (12): accommodates, bathrooms, bedrooms, etc.
  - Standardized with `StandardScaler`

### Hyperparameter Tuning
- **GridSearchCV** for small parameter grids (<= 3 params)
- **RandomizedSearchCV** for large grids
- **Cross-validation**: 3-fold stratified (default)
- **Scoring metric**: F1 macro

### Class Imbalance Handling
- `class_weight='balanced'` for all models
- Double stratification in splits (by rating + room type)

## MLflow Tracking

Each model run logs:

### Parameters
- Model name
- All hyperparameters
- Number of training samples
- Number of features

### Standard Metrics (Test Set)
- Accuracy
- Balanced accuracy
- F1 macro, F1 weighted
- Precision macro, recall macro
- Per-class F1, precision, recall

### Business Metrics (Test Set)
- **high_quality_recall**: % of "Very High" listings correctly identified
- **over_promise_rate**: % predicted higher quality than actual
- **undersell_rate**: % predicted lower quality than actual
- **severe_misclassification_rate**: % off by 2 levels (Medium ↔ Very High)
- **host_confidence_score**: 1 - undersell_rate

### Artifacts
- Trained model pipeline (preprocessor + classifier)
- Confusion matrix (text file)
- Cross-validation score

### Additional Metrics
- Training accuracy/F1 (for overfitting detection)
- Validation metrics
- Training time

## Usage Examples

### Train Specific Models

```bash
# Train only XGBoost and Random Forest
python modelling/train.py --models xgboost random_forest

# Quick mode for specific models
python modelling/train.py --models xgboost --quick
```

### Custom Configuration

```bash
# Change number of CV folds
python modelling/train.py --cv-folds 5

# Use different data directory
python modelling/train.py --data-dir /path/to/data/
```

### View Results

```bash
# Start MLflow UI
mlflow ui

# Export comparison table to CSV
python modelling/compare_models.py --export

# Compare specific experiment
python modelling/compare_models.py --experiment my-experiment
```

## Expected Results

Based on EDA findings:

- **Tree-based models** (XGBoost, RF, LightGBM) will significantly outperform linear models
- **XGBoost** expected to be best performer (non-linear relationships)
- **Baseline** accuracy ~45.5% (most frequent class)
- **Top models** F1 macro expected ~0.60-0.70
- **High quality recall** should be >0.70 (business priority)

## Business Interpretation

### Metrics Priority

1. **High Quality Recall** - Platform wants to surface best listings
2. **Over-Promise Rate** (minimize) - Don't mislead guests
3. **F1 Macro** - Overall balanced performance
4. **Undersell Rate** (minimize) - Don't hide quality from hosts

### Model Selection Criteria

Choose model with:
- Highest F1 macro (primary)
- High quality recall >0.70
- Over-promise rate <0.20
- Reasonable training time (<10 min)

## Troubleshooting

### Memory Issues
- Use `--quick` flag for reduced grids
- Train models individually: `--models xgboost`
- Reduce CV folds: `--cv-folds 2`

### MLflow Connection Issues
```bash
# Check MLflow tracking URI
echo $MLFLOW_TRACKING_URI

# Use local mlruns directory
export MLFLOW_TRACKING_URI=./mlruns
```

### Import Errors
```bash
# Install missing dependencies
pip install mlflow xgboost lightgbm scikit-learn
```

## Next Steps

After training:

1. **View MLflow UI** - Compare all models visually
2. **Export comparison** - Create table for report
3. **Select best model** - Based on business metrics
4. **Error analysis** - Investigate misclassifications
5. **Feature importance** - Understand what drives predictions
6. **Deploy** - Save best model for production use
