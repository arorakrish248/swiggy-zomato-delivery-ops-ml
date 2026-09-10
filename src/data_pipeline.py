"""
Data Ingestion, Auditing, and Leak-Free Splitting Pipeline
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw Swiggy/Zomato delivery records."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw delivery records:
    - Strips whitespace from strings
    - Imputes missing rider ratings and ages using median values
    - Imputes multiple_deliveries (default 0 for unbatched)
    - Clips extreme GPS distance anomalies to realistic urban ranges (0.5 to 30 km)
    - Standardizes target variable (time_taken in minutes)
    """
    df = df.copy()
    
    # Strip string columns
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        
    # Impute rider demographics
    df['ratings'] = df['ratings'].fillna(df['ratings'].median())
    df['age'] = df['age'].fillna(df['age'].median())
    df['multiple_deliveries'] = df['multiple_deliveries'].fillna(0.0).astype(int)
    
    # Clean distance: replace NaN/0 with median, clip extreme GPS errors
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['distance'] = df['distance'].fillna(df['distance'].median())
    df['distance'] = df['distance'].clip(lower=0.5, upper=25.0)
    
    # Target delivery time
    df['time_taken'] = pd.to_numeric(df['time_taken'], errors='coerce')
    df = df.dropna(subset=['time_taken'])
    df['time_taken'] = df['time_taken'].clip(lower=8.0, upper=90.0)
    
    # Fill festival and weather nulls
    df['festival'] = df['festival'].replace({'nan': 'no', 'none': 'no'}).fillna('no')
    df['weather'] = df['weather'].replace({'nan': 'sunny'}).fillna('sunny')
    df['traffic'] = df['traffic'].replace({'nan': 'medium'}).fillna('medium')
    df['city_type'] = df['city_type'].replace({'nan': 'metropolitian'}).fillna('metropolitian')
    
    return df.reset_index(drop=True)

def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs train-test split (80/20) before feature transformations to prevent leakage.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "swiggy_delivery_data.csv")
    df = clean_data(load_raw_data(raw_path))
    train_df, test_df = split_data(df)
    print(f"Data cleaned successfully. Train: {train_df.shape}, Test: {test_df.shape}")
