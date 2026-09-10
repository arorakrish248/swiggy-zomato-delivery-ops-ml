"""
Model Training, Cross-Validation, and Benchmark Engine (Regression ETA)
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import get_feature_groups, build_preprocessor

def get_candidate_models() -> Dict[str, Any]:
    """Returns candidate regression models for delivery ETA prediction."""
    models = {
        'Ridge Regression (L2)': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=80, 
            max_depth=10, 
            min_samples_leaf=5, 
            random_state=42, 
            n_jobs=-1
        ),
        'XGBoost Regressor (Champion)': XGBRegressor(
            n_estimators=120, 
            max_depth=5, 
            learning_rate=0.08, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            n_jobs=-1
        )
    }
    return models

def benchmark_models_cv(X_train: pd.DataFrame, 
                        y_train: pd.Series, 
                        preprocessor: Any, 
                        cv_splits: int = 5) -> pd.DataFrame:
    """
    Performs 5-Fold Cross Validation across candidate models.
    """
    models = get_candidate_models()
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    records = []
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }
    
    for name, reg in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])
        
        cv_res = cross_validate(
            pipeline, X_train, y_train, 
            cv=kf, scoring=scoring, 
            return_train_score=False
        )
        
        records.append({
            'Model': name,
            'CV MAE (Mins ± Std)': f"{-np.mean(cv_res['test_mae']):.2f} ± {np.std(cv_res['test_mae']):.2f}",
            'CV RMSE (Mins ± Std)': f"{-np.mean(cv_res['test_rmse']):.2f} ± {np.std(cv_res['test_rmse']):.2f}",
            'CV R² (Mean ± Std)': f"{np.mean(cv_res['test_r2']):.3f} ± {np.std(cv_res['test_r2']):.3f}",
            'Raw_MAE': -np.mean(cv_res['test_mae']),
            'Raw_RMSE': -np.mean(cv_res['test_rmse']),
            'Raw_R2': np.mean(cv_res['test_r2'])
        })
        
    return pd.DataFrame(records)

def train_and_evaluate_champion(X_train: pd.DataFrame, 
                                y_train: pd.Series, 
                                X_test: pd.DataFrame, 
                                y_test: pd.Series, 
                                preprocessor: Any) -> Tuple[Pipeline, Dict[str, Any], np.ndarray]:
    """
    Trains champion XGBoost pipeline on entire train set and evaluates on holdout test set.
    """
    champion_reg = XGBRegressor(
        n_estimators=150, 
        max_depth=5, 
        learning_rate=0.08, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', champion_reg)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model_name': 'XGBoost Regressor',
        'test_mae_minutes': round(mae, 2),
        'test_rmse_minutes': round(rmse, 2),
        'test_r2_score': round(r2, 4)
    }
    
    return pipeline, metrics, y_pred
