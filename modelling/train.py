"""
Model Training
──────────────
Train ≥ 5 classification models and log each run to MLflow.

Models:
  1. Baseline (DummyClassifier)
  2. Logistic Regression
  3. Random Forest
  4. XGBoost (TOP PRIORITY from EDA)
  5. KNN (K-Nearest Neighbors)
  6. Gradient Boosting
  7. HistGradientBoosting
  8. CatBoost (if installed)

Each MLflow run logs:
  - Model name & hyperparameters
  - ≥ 2 standard metrics (accuracy, F1, etc.)
  - ≥ 2 business metrics (over_promise_rate, high_quality_recall, etc.)
  - Trained model artifact
  - Confusion matrix

Based on EDA findings:
- Use ready_features.csv (17 features, 3 target classes)
- Weak linear correlations → tree models will outperform
- Class imbalance → use balanced class weights
"""

import argparse
import logging
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress OneHotEncoder warnings about unknown categories during CV
# These are expected when rare categories appear in validation but not training folds
# The encoder handles this correctly by encoding unknowns as all zeros
warnings.filterwarnings('ignore', message='Found unknown categories')
# Suppress MLflow model serialization warnings (pickle format is standard for sklearn)
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')

from modelling.config import (
    MODEL_CONFIGS, 
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURES
)
from modelling.evaluate import (
    standard_metrics,
    business_metrics,
    error_analysis,
    print_evaluation_summary
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_splits(data_dir: str = "data/processed/"):
    """Load train / test CSVs from ready_features splits.
    
    Note: No separate validation set - use k-fold CV on train for tuning.
    
    Args:
        data_dir: Directory containing train.csv, test.csv
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Loading data splits from {data_dir}")
    
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    
    logger.info(f"  Train: {train_df.shape}")
    logger.info(f"  Test:  {test_df.shape}")
    logger.info(f"  Note: k-fold CV on train set for hyperparameter tuning")
    
    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare data for training: separate X/y and create preprocessing pipeline.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor, label_encoder)
    """
    logger.info("Preparing data for training...")
    
    # Separate features and target
    feature_cols = [col for col in train_df.columns if col not in TARGET_FEATURES]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    y_train_raw = train_df['rating_category']
    y_test_raw = test_df['rating_category']
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Target classes: {sorted(y_train_raw.unique())}")
    logger.info(f"  Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    logger.info(f"  Class distribution (train):")
    for cls, count in y_train_raw.value_counts().items():
        logger.info(f"    {cls}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # Create preprocessing pipeline
    # Categorical: One-hot encoding
    # Numeric: Standard scaling
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in feature_cols]
    numeric_features = [col for col in NUMERIC_FEATURES if col in feature_cols]
    
    logger.info(f"  Categorical features: {len(categorical_features)}")
    logger.info(f"  Numeric features: {len(numeric_features)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features),
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, label_encoder


def train_and_log(model_name: str, model, param_grid: dict, 
                 X_train, y_train, X_test, y_test,
                 preprocessor, label_encoder, use_grid_search: bool = True,
                 cv_folds: int = 3, n_iter: int = 10):
    """Train a model with hyperparameter tuning using k-fold CV and evaluate on test.
    
    Args:
        model_name: Name for MLflow run
        model: Scikit-learn compatible model
        param_grid: Hyperparameter grid for search
        X_train, y_train: Training data for CV (y_train is numeric encoded)
        X_test, y_test: Test data for final evaluation (y_test is numeric encoded)
        preprocessor: Preprocessing pipeline
        label_encoder: LabelEncoder for target variable
        use_grid_search: If True use GridSearchCV, else RandomizedSearchCV
        cv_folds: Number of cross-validation folds (default 3)
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        Trained pipeline with best parameters
    """
    logger.info("\n" + "="*70)
    logger.info(f"TRAINING: {model_name}")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Create pipeline: preprocessor + model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Adjust param_grid keys for pipeline
    pipeline_param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        
        # Hyperparameter tuning
        if param_grid:
            logger.info(f"  Hyperparameter tuning with {len(param_grid)} parameters...")
            
            if use_grid_search or len(param_grid) <= 3:
                # Use GridSearchCV for small grids or if specified
                search = GridSearchCV(
                    pipeline,
                    param_grid=pipeline_param_grid,
                    cv=cv_folds,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=1
                )
            else:
                # Use RandomizedSearchCV for large grids
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=pipeline_param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )
            
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
            
            logger.info(f"  Best parameters: {best_params}")
            logger.info(f"  Best CV score (F1 macro): {search.best_score_:.4f}")
            
            # Log best hyperparameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            
            mlflow.log_metric("cv_f1_macro", search.best_score_)
            
        else:
            # No hyperparameter tuning (e.g., baseline)
            logger.info(f"  Training without hyperparameter tuning...")
            best_pipeline = pipeline.fit(X_train, y_train)
        
        # Log model metadata
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("cv_folds", cv_folds)
        
        # Predictions on train and test sets (numeric encoded)
        y_train_pred_encoded = best_pipeline.predict(X_train)
        y_test_pred_encoded = best_pipeline.predict(X_test)
        
        # Convert predictions back to original labels for metrics
        y_train_true = label_encoder.inverse_transform(y_train)
        y_train_pred = label_encoder.inverse_transform(y_train_pred_encoded)
        
        y_test_true = label_encoder.inverse_transform(y_test)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
        
        # Compute metrics (using string labels)
        train_std_metrics = standard_metrics(y_train_true, y_train_pred)
        test_std_metrics = standard_metrics(y_test_true, y_test_pred)
        
        train_bus_metrics = business_metrics(y_train_true, y_train_pred)
        test_bus_metrics = business_metrics(y_test_true, y_test_pred)
        
        # Log standard metrics (test set)
        logger.info("\n  Standard Metrics (Test Set):")
        for metric_name, metric_value in test_std_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
            if metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']:
                logger.info(f"    {metric_name}: {metric_value:.4f}")
        
        # Log business metrics (test set)
        logger.info("\n  Business Metrics (Test Set):")
        for metric_name, metric_value in test_bus_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
            logger.info(f"    {metric_name}: {metric_value:.4f}")
        
        # Log training metrics to detect overfitting
        mlflow.log_metric("train_accuracy", train_std_metrics['accuracy'])
        mlflow.log_metric("train_f1_macro", train_std_metrics['f1_macro'])
        logger.info(f"\n  Overfitting check:")
        logger.info(f"    Train accuracy: {train_std_metrics['accuracy']:.4f}")
        logger.info(f"    Test accuracy:  {test_std_metrics['accuracy']:.4f}")
        logger.info(f"    Train F1:       {train_std_metrics['f1_macro']:.4f}")
        logger.info(f"    Test F1:        {test_std_metrics['f1_macro']:.4f}")
        
        # Error analysis (using string labels)
        error_info = error_analysis(y_test_true, y_test_pred)
        
        # Log confusion matrix as text
        cm = error_info['confusion_matrix']
        cm_text = np.array2string(cm, separator=', ')
        mlflow.log_text(cm_text, "confusion_matrix.txt")
        
        # Save trained model as MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model"
        )
        logger.info(f"\n  Model saved as MLflow artifact: {model_name}")
        
        # Log training time
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        logger.info(f"  Training time: {training_time:.2f} seconds")
        
        # Print summary
        all_metrics = {**test_std_metrics, **test_bus_metrics}
        print_evaluation_summary(all_metrics, model_name)
    
    return best_pipeline


def train_all_models(data_dir: str = "data/processed/",
                    models_to_train: list = None,
                    cv_folds: int = 3):
    """Train all models and log to MLflow.
    
    Args:
        data_dir: Directory containing train/val/test splits
        models_to_train: List of model names to train (None = all)
        cv_folds: Number of cross-validation folds
    """
    logger.info("="*70)
    logger.info("AIRBNB RATING CLASSIFICATION - MODEL TRAINING")
    logger.info("="*70)
    
    # Set MLflow tracking URI to use MLflow server
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    
    # Set MLflow experiment
    mlflow.set_experiment("airbnb-rating-classification")
    logger.info("MLflow experiment: airbnb-rating-classification")
    logger.info("MLflow tracking URI: http://127.0.0.1:5000")
    
    # Load data
    train_df, test_df = load_splits(data_dir)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = prepare_data(
        train_df, test_df
    )
    
    # Select configuration
    configs = MODEL_CONFIGS
    
    # Filter models if specified
    if models_to_train:
        configs = {k: v for k, v in configs.items() if k in models_to_train}
    
    logger.info(f"\nTraining {len(configs)} models: {list(configs.keys())}")
    
    # Train each model
    trained_models = {}
    
    for model_name, config in configs.items():
        try:
            trained_pipeline = train_and_log(
                model_name=model_name,
                model=config['model'],
                param_grid=config['params'],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                preprocessor=preprocessor,
                label_encoder=label_encoder,
                cv_folds=cv_folds
            )
            
            trained_models[model_name] = trained_pipeline
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Successfully trained {len(trained_models)} models")
    logger.info(f"View results: mlflow ui")
    
    return trained_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification models with MLflow tracking")
    parser.add_argument("--data-dir", default="data/processed/", 
                       help="Directory containing train/val/test splits")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to train (e.g., xgboost random_forest)")
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Number of cross-validation folds (default: 3)")
    
    args = parser.parse_args()
    
    train_all_models(
        data_dir=args.data_dir,
        models_to_train=args.models,
        cv_folds=args.cv_folds
    )
