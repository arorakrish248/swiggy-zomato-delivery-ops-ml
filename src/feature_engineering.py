"""
Domain Feature Engineering for Indian Food Delivery & Quick Commerce
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs operational domain features capturing Indian logistics friction:
    - Peak traffic jam flags
    - Monsoon / severe weather indicators
    - Multi-order batching / stacking flags
    - Kitchen prep delay vs trip distance velocity
    - Lunch/dinner rush hour cohorts
    """
    df = df.copy()
    
    # 1. Traffic Jam Bottleneck
    df['is_traffic_jam'] = df['traffic'].isin(['jam', 'high']).astype(int)
    
    # 2. Monsoon / Adverse Weather Bottleneck
    severe_weathers = ['stormy', 'rainy', 'fog', 'sandstorms']
    df['is_severe_weather'] = df['weather'].isin(severe_weathers).astype(int)
    
    # 3. Order Stacking / Multi-Drop Flag
    df['is_order_stacked'] = (pd.to_numeric(df['multiple_deliveries'], errors='coerce').fillna(0) > 0).astype(int)
    
    # 4. Distance & pickup time clean
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(3.5).clip(0.5, 25.0)
    df['pickup_time_minutes'] = pd.to_numeric(df['pickup_time_minutes'], errors='coerce').fillna(10.0)
    df['pickup_to_distance_ratio'] = df['pickup_time_minutes'] / (df['distance'] + 0.5)
    
    # 5. Peak Rush Hour Windows (Indian Lunch: 12-14h, Dinner: 19-22h)
    peak_hours = [12, 13, 14, 19, 20, 21, 22]
    df['is_peak_rush_hour'] = df['order_time_hour'].isin(peak_hours).astype(int)
    
    # 6. Metropolitan Density
    df['is_metro'] = (df['city_type'] == 'metropolitian').astype(int)
    
    # 7. Delivery Partner Efficiency Index
    ratings = pd.to_numeric(df['ratings'], errors='coerce').fillna(4.6)
    age = pd.to_numeric(df['age'], errors='coerce').fillna(29.0)
    df['rider_experience_index'] = (ratings * 10.0) / (age + 1.0)
    
    # 8. Festival Surge
    df['is_festival'] = (df['festival'] == 'yes').astype(int)
    
    # 9. Weekend Flag
    df['is_weekend'] = pd.to_numeric(df['is_weekend'], errors='coerce').fillna(0).astype(int)
    
    return df

def get_feature_groups() -> Tuple[List[str], List[str], List[str]]:
    """Returns numerical, categorical, and binary feature column names."""
    numerical_cols = [
        'distance', 'age', 'ratings', 'pickup_time_minutes', 
        'order_time_hour', 'pickup_to_distance_ratio', 'rider_experience_index'
    ]
    
    categorical_cols = [
        'weather', 'traffic', 'type_of_vehicle', 'type_of_order', 
        'city_type', 'city_name', 'order_day_of_week'
    ]
    
    binary_cols = [
        'is_traffic_jam', 'is_severe_weather', 'is_order_stacked', 
        'is_peak_rush_hour', 'is_metro', 'is_festival', 'is_weekend'
    ]
    
    return numerical_cols, categorical_cols, binary_cols

def build_preprocessor(numerical_cols: List[str], 
                       categorical_cols: List[str], 
                       binary_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a leak-free scikit-learn ColumnTransformer with median imputation.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    bin_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols),
            ('bin', bin_pipeline, binary_cols)
        ],
        remainder='drop'
    )
    return preprocessor
