import os
import pytest
import pandas as pd
import numpy as np
from src.surge_and_sla_optimizer import (
    simulate_sla_buffer_economics, assign_dynamic_surge_policy, evaluate_order_batching_rules
)

def test_simulate_sla_buffer_economics():
    y_true = pd.Series([25.0, 30.0, 40.0, 50.0])
    y_pred = np.array([20.0, 32.0, 38.0, 42.0])
    
    summary, curve_df = simulate_sla_buffer_economics(
        y_true=y_true, 
        y_pred=y_pred, 
        refund_cost_inr=100.0, 
        conversion_loss_per_min_inr=2.0
    )
    
    assert 'optimal_buffer_minutes' in summary
    assert summary['optimal_buffer_minutes'] >= 0
    assert len(curve_df) == 16

def test_assign_dynamic_surge_policy():
    df = pd.DataFrame({
        'traffic': ['jam', 'low'],
        'weather': ['stormy', 'sunny'],
        'distance': [5.0, 2.0]
    })
    y_pred = np.array([42.0, 18.0])
    
    res = assign_dynamic_surge_policy(df, y_pred)
    assert res.loc[0, 'surge_tier'] == 'CRITICAL_SURGE'
    assert res.loc[0, 'customer_surge_fee_inr'] == 45
    assert res.loc[1, 'surge_tier'] == 'STANDARD'
    assert res.loc[1, 'customer_surge_fee_inr'] == 0

def test_evaluate_order_batching_rules():
    df = pd.DataFrame({
        'traffic': ['jam', 'low'],
        'weather': ['sunny', 'sunny'],
        'distance': [8.0, 2.5]
    })
    res = evaluate_order_batching_rules(df)
    assert res.loc[0, 'batching_policy'] == 'PROHIBIT_BATCHING'
    assert res.loc[1, 'batching_policy'] == 'AUTO_BATCH_APPROVED'
