import os
import pytest
import pandas as pd
import numpy as np
from src.data_pipeline import load_raw_data, clean_data
from src.feature_engineering import (
    engineer_features, get_feature_groups, build_preprocessor
)

@pytest.fixture
def sample_cleaned():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'raw', 'swiggy_delivery_data.csv')
    return clean_data(load_raw_data(path).head(100))

def test_engineer_features(sample_cleaned):
    eng = engineer_features(sample_cleaned)
    expected_cols = [
        'is_traffic_jam', 'is_severe_weather', 'is_order_stacked', 
        'pickup_to_distance_ratio', 'is_peak_rush_hour', 'is_metro', 'rider_experience_index'
    ]
    for c in expected_cols:
        assert c in eng.columns
        
    assert set(eng['is_traffic_jam'].unique()).issubset({0, 1})
    assert set(eng['is_order_stacked'].unique()).issubset({0, 1})

def test_preprocessor(sample_cleaned):
    eng = engineer_features(sample_cleaned)
    num_cols, cat_cols, bin_cols = get_feature_groups()
    preprocessor = build_preprocessor(num_cols, cat_cols, bin_cols)
    
    X = eng.drop(columns=['rider_id', 'order_date', 'time_taken'])
    transformed = preprocessor.fit_transform(X)
    
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_cleaned)
    assert not np.isnan(transformed).any()
