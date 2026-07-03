"""
Data Loading Module
Handles loading and initial validation of the dataset
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from config import DATASET_PATH


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load product return dataset from CSV file
    
    Args:
        file_path: Path to CSV file. If None, uses default path from config
        
    Returns:
        DataFrame containing the raw dataset
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    if file_path is None:
        file_path = DATASET_PATH
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "returned_count": (df["Returned"] == "Yes").sum() if "Returned" in df.columns else 0,
        "not_returned_count": (df["Returned"] == "No").sum() if "Returned" in df.columns else 0
    }
    
    return summary


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset has all required columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = [
        "Order_ID", "Product_ID", "Category", "Price", "Quantity",
        "Order_Date", "Delivery_Days", "Region", "Returned", "Return_Reason"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    print("✅ All required columns present")
    return True
