"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, duplicates, and data types
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    print("🧹 Starting data cleaning...")
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"   Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values in Delivery_Days with median
    if df_clean["Delivery_Days"].isnull().sum() > 0:
        missing_count = df_clean["Delivery_Days"].isnull().sum()
        median_delivery = df_clean["Delivery_Days"].median()
        df_clean["Delivery_Days"].fillna(median_delivery, inplace=True)
        print(f"   Filled {missing_count} missing Delivery_Days with median: {median_delivery}")
    
    # Convert Order_Date to datetime
    df_clean["Order_Date"] = pd.to_datetime(df_clean["Order_Date"])
    print(f"   Converted Order_Date to datetime")
    
    # Convert Returned to binary (0/1)
    df_clean["Returned_Binary"] = (df_clean["Returned"] == "Yes").astype(int)
    print(f"   Created Returned_Binary column (0/1)")
    
    # Ensure numeric columns are correct type
    df_clean["Price"] = pd.to_numeric(df_clean["Price"], errors="coerce")
    df_clean["Quantity"] = pd.to_numeric(df_clean["Quantity"], errors="coerce")
    df_clean["Delivery_Days"] = pd.to_numeric(df_clean["Delivery_Days"], errors="coerce")
    
    print(f"✅ Data cleaning completed!")
    print(f"   Final shape: {df_clean.shape}")
    
    return df_clean


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for analysis and modeling
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df_features = df.copy()
    
    print("🔧 Creating additional features...")
    
    # Extract month and year from Order_Date
    df_features["Order_Month"] = df_features["Order_Date"].dt.to_period("M")
    df_features["Order_Year"] = df_features["Order_Date"].dt.year
    df_features["Order_Month_Name"] = df_features["Order_Date"].dt.strftime("%B")
    
    # Calculate total order value
    df_features["Total_Value"] = df_features["Price"] * df_features["Quantity"]
    
    # Create delivery speed category
    df_features["Delivery_Speed"] = pd.cut(
        df_features["Delivery_Days"],
        bins=[0, 3, 7, 15],
        labels=["Fast", "Normal", "Slow"]
    )
    
    # Create price category
    df_features["Price_Category"] = pd.cut(
        df_features["Price"],
        bins=[0, 500, 1500, 3000, 10000],
        labels=["Low", "Medium", "High", "Premium"]
    )
    
    print(f"✅ Feature engineering completed!")
    
    return df_features


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for machine learning model
    
    Args:
        df: DataFrame with all features
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target variable
    """
    print("🤖 Preparing features for ML...")
    
    df_ml = df.copy()
    
    # Select features for modeling
    feature_cols = ["Price", "Quantity", "Delivery_Days", "Category", "Region"]
    
    # Create feature DataFrame
    X = df_ml[feature_cols].copy()
    
    # Encode categorical variables
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    
    X["Category_Encoded"] = le_category.fit_transform(X["Category"])
    X["Region_Encoded"] = le_region.fit_transform(X["Region"])
    
    # Drop original categorical columns
    X = X.drop(["Category", "Region"], axis=1)
    
    # Target variable
    y = df_ml["Returned_Binary"]
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Features: {list(X.columns)}")
    print(f"✅ ML features prepared!")
    
    return X, y, le_category, le_region


def get_preprocessing_pipeline():
    """
    Get scikit-learn preprocessing pipeline for production use
    
    Returns:
        Preprocessing pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    
    # Numerical features
    numeric_features = ["Price", "Quantity", "Delivery_Days"]
    numeric_transformer = StandardScaler()
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="passthrough"
    )
    
    return preprocessor
