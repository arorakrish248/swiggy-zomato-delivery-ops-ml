import os
import pytest
import pandas as pd
from src.data_pipeline import load_raw_data, clean_data, split_data

@pytest.fixture
def raw_csv_path():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'data', 'raw', 'swiggy_delivery_data.csv')

def test_load_raw_data(raw_csv_path):
    df = load_raw_data(raw_csv_path)
    assert len(df) > 40000
    assert 'time_taken' in df.columns
    assert 'rider_id' in df.columns

def test_clean_data(raw_csv_path):
    df = load_raw_data(raw_csv_path)
    cleaned = clean_data(df.head(500))
    
    # Check no nulls in target
    assert cleaned['time_taken'].isna().sum() == 0
    # Check distance range [0.5, 25.0]
    assert cleaned['distance'].min() >= 0.5
    assert cleaned['distance'].max() <= 25.0

def test_split_data(raw_csv_path):
    df = clean_data(load_raw_data(raw_csv_path).head(1000))
    train, test = split_data(df, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(df)
    assert abs(len(test) - int(len(df) * 0.2)) <= 1
