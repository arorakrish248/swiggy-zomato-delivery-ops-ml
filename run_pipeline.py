"""
Master Execution Pipeline for Swiggy / Zomato Delivery Operations & Dynamic Surge ML
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline import load_raw_data, clean_data, split_data
from feature_engineering import engineer_features, get_feature_groups, build_preprocessor
from model_engine import benchmark_models_cv, train_and_evaluate_champion
from explainability import DeliveryDelayExplainer
from surge_and_sla_optimizer import (
    simulate_sla_buffer_economics, assign_dynamic_surge_policy, evaluate_order_batching_rules
)

def run_delivery_pipeline():
    print("=" * 80)
    print("SWIGGY / ZOMATO DELIVERY OPERATIONS & DYNAMIC SURGE ML SYSTEM")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'swiggy_delivery_data.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    reports_dir = os.path.join(base_dir, 'reports')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Ingest and Clean
    print("\n[Step 1/6] Ingesting and Auditing 45,500+ Indian Delivery Records...")
    raw_df = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)
    print(f"Audited records: {len(clean_df):,}. Average delivery time: {clean_df['time_taken'].mean():.1f} mins.")
    
    # 2. Leak-Free Train/Test Split
    print("\n[Step 2/6] Partitioning 80/20 Train/Test Split...")
    train_df, test_df = split_data(clean_df, test_size=0.2, random_state=42)
    
    # 3. Domain Feature Engineering
    print("\n[Step 3/6] Engineering Domain Friction Features (Traffic, Monsoon, Batching)...")
    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)
    
    train_eng.to_csv(os.path.join(processed_dir, 'train_engineered.csv'), index=False)
    test_eng.to_csv(os.path.join(processed_dir, 'test_engineered.csv'), index=False)
    
    num_cols, cat_cols, bin_cols = get_feature_groups()
    preprocessor = build_preprocessor(num_cols, cat_cols, bin_cols)
    
    X_train = train_eng.drop(columns=['rider_id', 'order_date', 'time_taken'])
    y_train = train_eng['time_taken']
    X_test = test_eng.drop(columns=['rider_id', 'order_date', 'time_taken'])
    y_test = test_eng['time_taken']
    
    # 4. Benchmark 3 Models (5-Fold CV)
    print("\n[Step 4/6] Running 5-Fold Cross-Validation on 3 Candidate Models...")
    cv_results = benchmark_models_cv(X_train, y_train, preprocessor, cv_splits=5)
    print(cv_results[['Model', 'CV MAE (Mins ± Std)', 'CV RMSE (Mins ± Std)', 'CV R² (Mean ± Std)']].to_string(index=False))
    cv_results.to_csv(os.path.join(reports_dir, 'cv_model_comparison.csv'), index=False)
    
    # Train Champion Model
    print("\n[Step 5/6] Training Champion XGBoost Regressor on Full Train Set...")
    fitted_pipeline, test_metrics, y_pred_test = train_and_evaluate_champion(
        X_train, y_train, X_test, y_test, preprocessor
    )
    print(f"Held-Out Test Set (N = {len(y_test):,} orders):")
    print(f"  Test MAE:  {test_metrics['test_mae_minutes']} minutes")
    print(f"  Test RMSE: {test_metrics['test_rmse_minutes']} minutes")
    print(f"  Test R²:   {test_metrics['test_r2_score']}")
    
    # 5. Prescriptive BizOps Optimization
    print("\n[Step 6/6] Solving Dynamic Surge Pricing & Guaranteed SLA Buffers...")
    sla_summary, sla_curve_df = simulate_sla_buffer_economics(
        y_true=y_test, 
        y_pred=y_pred_test, 
        refund_cost_inr=120.0, 
        conversion_loss_per_min_inr=3.5
    )
    sla_curve_df.to_csv(os.path.join(reports_dir, 'sla_buffer_simulation.csv'), index=False)
    
    print(f"  Optimal SLA Buffer: +{sla_summary['optimal_buffer_minutes']} mins buffer")
    print(f"  Late Delivery Rate: cut from 50.0% down to {sla_summary['optimal_late_rate_pct']}%")
    print(f"  Net Refund Cost Saved: INR {sla_summary['net_sla_savings_inr']:,.0f} per test cohort")
    
    # Dynamic surge policy assignment
    surge_policy_df = assign_dynamic_surge_policy(test_eng, y_pred_test)
    batching_df = evaluate_order_batching_rules(surge_policy_df)
    batching_df.to_csv(os.path.join(reports_dir, 'prescriptive_delivery_actions.csv'), index=False)
    
    surge_summary = batching_df.groupby('surge_tier').agg(
        order_count=('rider_id', 'count'),
        avg_actual_mins=('time_taken', 'mean'),
        avg_predicted_mins=('predicted_eta_minutes', 'mean'),
        customer_surge_fee=('customer_surge_fee_inr', 'first'),
        rider_bonus=('rider_incentive_bonus_inr', 'first')
    ).reset_index()
    surge_summary.to_csv(os.path.join(reports_dir, 'surge_policy_summary.csv'), index=False)
    print("\nDynamic Surge & Rider Bonus Breakdown:")
    print(surge_summary.to_string(index=False))
    
    # SHAP Feature Attribution
    print("\nComputing TreeSHAP Delay Attributions...")
    X_test_trans = fitted_pipeline.named_steps['preprocessor'].transform(X_test)
    feature_names = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()
    xgb_reg = fitted_pipeline.named_steps['regressor']
    
    explainer = DeliveryDelayExplainer(xgb_reg, preprocessor, feature_names)
    shap_vals = explainer.compute_shap_values(X_test_trans)
    delay_drivers = explainer.get_global_delay_drivers(shap_vals)
    delay_drivers.to_csv(os.path.join(reports_dir, 'shap_delay_importance.csv'), index=False)
    print("\nTop 5 Delay Bottlenecks (Mean Impact in Minutes):")
    print(delay_drivers.head(5).to_string(index=False))
    
    # Serialize champion package
    package = {
        'pipeline': fitted_pipeline,
        'feature_names': list(feature_names),
        'metrics': test_metrics,
        'sla_summary': sla_summary
    }
    joblib.dump(package, os.path.join(models_dir, 'delivery_xgb_pipeline.joblib'))
    
    print("\n[SUCCESS] Entire pipeline executed cleanly. Models and reports exported.")
    return test_metrics, sla_summary

if __name__ == "__main__":
    run_delivery_pipeline()
