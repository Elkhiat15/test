# 🚀 Performance Improvement Strategies

## Current Performance Analysis

**Your Results (Quick Training):**
- Baseline: 45.5% accuracy (F1 macro: 0.21)
- Logistic Regression: 56.4% accuracy (F1 macro: 0.54)
- Random Forest: 61.6% accuracy (F1 macro: 0.56)
- XGBoost: **63.3% accuracy (F1 macro: 0.52)**

**Main Problem:** Medium Rating class severely underperforming
- Medium Rating F1: 0.20 (XGBoost), 0.34 (RF) ❌
- High Rating F1: 0.74 (XGBoost), 0.71 (RF) ✅
- Very High Rating F1: 0.63 (XGBoost), 0.62 (RF) ✅

**Root Cause:** Class imbalance (Medium: 20.8%, High: 45.5%, Very High: 33.6%)

---

## ✅ Implemented Solutions

### 1. **New Fast, High-Performance Models**

Added to `config.py`:

#### **CatBoost** ⭐ TOP RECOMMENDATION
- **Speed:** 2-3x faster than LightGBM
- **Accuracy:** Often beats XGBoost/LightGBM
- **Features:**
  - Native categorical feature support (no encoding needed)
  - Built-in class balancing (`auto_class_weights='Balanced'`)
  - Ordered target encoding (reduces overfitting)
  - Early stopping built-in

#### **HistGradientBoostingClassifier** ⭐ FAST
- Scikit-learn's optimized implementation
- Often matches XGBoost speed with similar accuracy
- Native missing value support
- 2-3x faster than standard GradientBoosting

#### **ExtraTrees** ⭐ FASTER THAN RANDOM FOREST
- More randomization → less overfitting
- No bootstrapping → faster training
- Often better generalization than Random Forest

**Install:**
```bash
pip install catboost imbalanced-learn
```

### 2. **Class Balancing Module** (`class_balancing.py`)

#### **SMOTE (Synthetic Minority Over-sampling)**
- Creates synthetic samples for minority class (Medium Rating)
- ✅ Recommended: `SMOTE + Tomek Links` (hybrid approach)
  - SMOTE oversamples minority
  - Tomek removes noisy majority samples near boundary
  - Cleaner decision boundaries

#### **When to Use SMOTE:**
- ✅ **YES** if Medium Rating F1 < 0.30 (currently 0.20-0.34)
- ✅ **YES** for tree-based models (XGBoost, CatBoost, RF)
- ⚠️ **CAUTION** with kNN, SVM (distance-based)
- ❌ **NO** if it causes overfitting (check train vs val F1)

#### **Alternatives to SMOTE:**
1. **Class weights** (already implemented - `class_weight='balanced'`)
2. **Sample weights** - per-instance weighting
3. **Focal Loss** - focus on hard examples
4. **Cost-sensitive learning** - asymmetric misclassification costs

### 3. **Enhanced Training Script** (`train_enhanced.py`)

**Usage:**
```bash
# Quick test with recommended models
python modelling/train_enhanced.py --balance smote_tomek

# Compare strategies
python modelling/train_enhanced.py --balance none --models catboost
python modelling/train_enhanced.py --balance smote_tomek --models catboost

# Fast models only
python modelling/train_enhanced.py --models catboost hist_gradient_boosting extra_trees
```

---

## 📊 Expected Performance Improvements

### **Conservative Estimates:**
| Model | Current F1 Macro | Expected F1 Macro | Speedup vs LightGBM |
|-------|------------------|-------------------|---------------------|
| XGBoost (tuned) | 0.52 | 0.55-0.58 | — |
| CatBoost | — | **0.58-0.62** ⭐ | 2-3x faster |
| HistGradientBoosting | — | 0.55-0.60 | 2-3x faster |
| ExtraTrees | — | 0.57-0.60 | 1.5x faster |
| XGBoost + SMOTE | — | **0.56-0.60** | — |
| CatBoost + SMOTE | — | **0.60-0.65** ⭐ | 2-3x faster |

### **Target Improvements:**
- Overall accuracy: 63% → **68-72%**
- F1 macro: 0.52 → **0.58-0.65**
- Medium Rating F1: 0.20 → **0.35-0.45** (critical improvement)
- High quality recall: 0.62 → **0.70-0.75** (business goal)

---

## 🎯 Recommended Action Plan

### **Phase 1: Quick Wins (30 min)**
1. Install dependencies:
   ```bash
   pip install catboost imbalanced-learn
   ```

2. Test CatBoost (fastest, often best):
   ```bash
   python modelling/train_enhanced.py --models catboost --quick
   ```

3. Compare with SMOTE:
   ```bash
   python modelling/train_enhanced.py --models catboost --balance smote_tomek --quick
   ```

### **Phase 2: Full Comparison (2-3 hours)**
4. Train all fast models:
   ```bash
   python modelling/train_enhanced.py --models catboost hist_gradient_boosting extra_trees xgboost --quick
   ```

5. Best model + SMOTE:
   ```bash
   # After identifying best model from step 4
   python modelling/train_enhanced.py --models <best_model> --balance smote_tomek --no-quick
   ```

### **Phase 3: Advanced Tuning (optional)**
6. Hyperparameter optimization with Optuna (future enhancement)
7. Ensemble methods (voting, stacking)
8. Feature engineering improvements

---

## 🚫 What NOT to Do

### ❌ Don't Use SMOTE Incorrectly
```python
# WRONG - applies to test set too
X_all, y_all = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all)

# CORRECT - only on training set
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_balanced, y_train_balanced = SMOTE().fit_resample(X_train, y_train)
```

### ❌ Don't Oversample Too Much
```python
# WRONG - creates too many synthetic samples
SMOTE(sampling_strategy={0: 50000, 1: 50000, 2: 50000})  # 96% synthetic!

# CORRECT - moderate balancing
SMOTE(sampling_strategy={1: 12000})  # Medium: 8,482 → 12,000 (reasonable)
```

### ❌ Don't Apply SMOTE Before Preprocessing
```python
# WRONG - SMOTE needs numeric features
X_raw = ['Apartment', 'NYC', ...]  # Categorical!
SMOTE().fit_resample(X_raw, y)  # ERROR

# CORRECT - preprocess first
X_encoded = preprocessor.fit_transform(X_raw)
SMOTE().fit_resample(X_encoded, y)
```

---

## 🔬 Why These Models are Better

### **CatBoost vs LightGBM:**
| Feature | CatBoost | LightGBM |
|---------|----------|----------|
| Speed | ⚡ Faster (2-3x) | Slow |
| Categorical handling | Native (no encoding) | Needs encoding |
| Overfitting | Less prone | More prone |
| Hyperparameter tuning | Easier | More sensitive |
| Default performance | Better out-of-box | Needs tuning |

### **HistGradientBoosting vs XGBoost:**
| Feature | HistGradientBoosting | XGBoost |
|---------|---------------------|---------|
| Speed | ⚡ Similar/faster | Fast |
| Scikit-learn integration | Native | Wrapper |
| Missing value handling | Native | Native |
| Early stopping | Built-in | Needs setup |
| Dependencies | None (sklearn) | Requires install |

---

## 📈 Monitoring Improvements

After implementing changes, track:

1. **Per-class F1 scores:**
   - Medium Rating F1 > 0.35 (currently 0.20)
   - High Rating F1 > 0.70 (currently 0.74 ✅)
   - Very High Rating F1 > 0.65 (currently 0.63)

2. **Business metrics:**
   - High quality recall > 0.70 (currently 0.62)
   - Over-promise rate < 0.20 (currently 0.23)
   - Severe misclassification < 0.08 (currently 0.09)

3. **Overfitting check:**
   - Train F1 - Val F1 < 0.05 (acceptable gap)
   - If gap > 0.10, reduce model complexity or add regularization

---

## 🛠️ Quick Reference

### **Install Dependencies**
```bash
pip install catboost imbalanced-learn
```

### **Quick Test**
```bash
# Test CatBoost (fastest, recommended)
python modelling/train_enhanced.py --models catboost --quick

# Test with SMOTE
python modelling/train_enhanced.py --models catboost --balance smote_tomek --quick
```

### **Full Training**
```bash
# All fast models
python modelling/train_enhanced.py --models catboost hist_gradient_boosting extra_trees --quick

# Best model with full grid search
python modelling/train_enhanced.py --models catboost --no-quick
```

### **Compare Results**
```bash
python modelling/compare_models.py --export
```

---

## 🎓 Key Takeaways

1. **CatBoost is your best bet** - fastest and often most accurate
2. **SMOTE helps minority class** - but test with/without to confirm
3. **Use SMOTE + Tomek** - cleaner than pure SMOTE
4. **Drop LightGBM** - too slow for your use case
5. **Monitor business metrics** - not just accuracy
6. **Expect 65-70% accuracy** - realistic for 3-class imbalanced problem

---

## ⏭️ Next Steps (Future Enhancements)

### **Feature Engineering:**
- Interaction features: `price × room_type`, `amenities × neighborhood`
- Geographic clustering: neighborhood quality tiers
- TF-IDF on amenities (instead of count)
- Host-level aggregations (if multiple listings per host)

### **Advanced Tuning:**
- Optuna/Hyperopt for hyperparameter search
- Focal loss for class imbalance
- Calibrated classifiers for probability estimates

### **Ensemble Methods:**
- Voting classifier (CatBoost + XGBoost + ExtraTrees)
- Stacking with Logistic Regression meta-learner
- Weighted averaging based on per-class performance

### **Model Explainability:**
- SHAP values for feature importance
- Partial dependence plots
- Error analysis by city/property_type
