"""
Prescriptive BizOps Engine: Dynamic Surge Pricing, Guaranteed SLA Buffers, and Batching Policy
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def simulate_sla_buffer_economics(y_true: pd.Series, 
                                  y_pred: np.ndarray, 
                                  refund_cost_inr: float = 120.0, 
                                  conversion_loss_per_min_inr: float = 3.5) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Simulates operational trade-off between customer late refund compensation
    versus customer cart abandonment from inflated promised ETAs.
    """
    buffers = np.linspace(0, 15, 16)
    records = []
    
    for buf in buffers:
        promised_eta = y_pred + buf
        late_deliveries = int((y_true > promised_eta).sum())
        late_rate = late_deliveries / len(y_true)
        
        # Financial impacts (in INR)
        total_refund_payout = float(late_deliveries * refund_cost_inr)
        total_conversion_drag = float(len(y_true) * (buf * conversion_loss_per_min_inr))
        total_sla_cost = float(total_refund_payout + total_conversion_drag)
        
        records.append({
            'Buffer_Minutes': int(buf),
            'Late_Deliveries_Count': late_deliveries,
            'Late_Delivery_Rate_Pct': round(late_rate * 100, 2),
            'Refund_Payout_INR': round(total_refund_payout, 0),
            'Conversion_Drag_INR': round(total_conversion_drag, 0),
            'Total_SLA_Cost_INR': round(total_sla_cost, 0)
        })
        
    res_df = pd.DataFrame(records)
    best_row = res_df.loc[res_df['Total_SLA_Cost_INR'].idxmin()].to_dict()
    zero_buf_row = res_df.loc[res_df['Buffer_Minutes'] == 0].iloc[0].to_dict()
    
    summary = {
        'optimal_buffer_minutes': int(best_row['Buffer_Minutes']),
        'optimal_late_rate_pct': float(best_row['Late_Delivery_Rate_Pct']),
        'optimal_total_cost_inr': float(best_row['Total_SLA_Cost_INR']),
        'baseline_zero_buffer_cost_inr': float(zero_buf_row['Total_SLA_Cost_INR']),
        'net_sla_savings_inr': round(float(zero_buf_row['Total_SLA_Cost_INR']) - float(best_row['Total_SLA_Cost_INR']), 0)
    }
    
    return summary, res_df

def assign_dynamic_surge_policy(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Assigns dynamic surge fee and rider incentive bonuses based on predicted friction:
    - High Jam / Rain (ETA > 38 mins): ₹40 Surge Fee + ₹30 Rider Monsoon Bonus
    - Medium Friction (ETA 28-38 mins): ₹20 Surge Fee + ₹15 Rider Traffic Bonus
    - Normal (ETA < 28 mins): Standard base fee (₹0 Surge)
    """
    res = df.copy()
    res['predicted_eta_minutes'] = np.round(y_pred, 1)
    
    def calculate_surge(row):
        eta = row['predicted_eta_minutes']
        traffic = str(row['traffic']).lower()
        weather = str(row['weather']).lower()
        
        if (traffic in ['jam', 'high']) and (weather in ['stormy', 'rainy', 'fog']):
            return 'CRITICAL_SURGE', 45, 35, 'Severe Weather + Traffic Jam'
        elif (traffic in ['jam', 'high']) or (eta >= 35):
            return 'TRAFFIC_SURGE', 25, 20, 'Peak Traffic Bottleneck'
        elif (weather in ['stormy', 'rainy']):
            return 'RAIN_SURGE', 25, 20, 'Adverse Weather Protection'
        else:
            return 'STANDARD', 0, 0, 'Normal Operating Conditions'
            
    policy_results = res.apply(calculate_surge, axis=1)
    res['surge_tier'] = [p[0] for p in policy_results]
    res['customer_surge_fee_inr'] = [p[1] for p in policy_results]
    res['rider_incentive_bonus_inr'] = [p[2] for p in policy_results]
    res['surge_reason'] = [p[3] for p in policy_results]
    
    return res

def evaluate_order_batching_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates order stacking (multiple_deliveries) safety rules:
    - Safe to Batch: Distance <= 4.0 km and Traffic not Jam
    - High Risk of Breach: Distance > 5.0 km or Traffic Jam
    """
    res = df.copy()
    
    def get_batching_recommendation(row):
        dist = float(row['distance'])
        is_jam = (str(row['traffic']).lower() == 'jam')
        is_rain = (str(row['weather']).lower() in ['stormy', 'rainy'])
        
        if is_jam or is_rain or dist > 6.0:
            return 'PROHIBIT_BATCHING', 'High SLA Breach Risk (>45m) - Dedicate 1 Rider per Order'
        elif dist <= 3.5:
            return 'AUTO_BATCH_APPROVED', 'Safe to Stack 2 Orders - Saves 35% Rider Delivery Cost'
        else:
            return 'CONDITIONAL_BATCH', 'Allow Stacking only if Second Restaurant is within 500m'
            
    batch_res = res.apply(get_batching_recommendation, axis=1)
    res['batching_policy'] = [b[0] for b in batch_res]
    res['batching_guideline'] = [b[1] for b in batch_res]
    
    return res
