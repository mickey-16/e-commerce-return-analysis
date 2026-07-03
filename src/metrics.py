"""
Metrics Calculation Module
Calculates key business metrics for return analysis
"""

import pandas as pd
from typing import Dict, Tuple


def calculate_return_rate(df: pd.DataFrame) -> float:
    """
    Calculate overall return rate
    
    Args:
        df: DataFrame with Returned column
        
    Returns:
        Return rate as percentage
    """
    total_orders = len(df)
    returned_orders = (df["Returned"] == "Yes").sum()
    return (returned_orders / total_orders) * 100


def calculate_category_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate return rate by product category
    
    Args:
        df: DataFrame with Category and Returned columns
        
    Returns:
        DataFrame with category-wise return rates
    """
    category_stats = df.groupby("Category").agg({
        "Returned": lambda x: ((x == "Yes").sum() / len(x)) * 100
    }).round(2)
    
    category_stats.columns = ["Return_Rate_%"]
    category_stats = category_stats.sort_values("Return_Rate_%", ascending=False)
    
    return category_stats


def calculate_region_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate return rate by region
    
    Args:
        df: DataFrame with Region and Returned columns
        
    Returns:
        DataFrame with region-wise return rates
    """
    region_stats = df.groupby("Region").agg({
        "Returned": lambda x: ((x == "Yes").sum() / len(x)) * 100
    }).round(2)
    
    region_stats.columns = ["Return_Rate_%"]
    region_stats = region_stats.sort_values("Return_Rate_%", ascending=False)
    
    return region_stats


def calculate_return_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distribution of return reasons
    
    Args:
        df: DataFrame with Return_Reason column
        
    Returns:
        DataFrame with return reason counts and percentages
    """
    # Filter only returned items
    returned_df = df[df["Returned"] == "Yes"]
    
    reason_stats = returned_df["Return_Reason"].value_counts()
    reason_pct = (reason_stats / reason_stats.sum() * 100).round(2)
    
    result = pd.DataFrame({
        "Count": reason_stats,
        "Percentage": reason_pct
    })
    
    return result


def calculate_financial_loss(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Calculate total financial loss from returns
    
    Args:
        df: DataFrame with Price, Quantity, and Returned columns
        
    Returns:
        Tuple of (total_loss, category_wise_loss)
    """
    # Filter returned items
    returned_df = df[df["Returned"] == "Yes"].copy()
    
    # Calculate loss per order
    returned_df["Loss"] = returned_df["Price"] * returned_df["Quantity"]
    
    # Total loss
    total_loss = returned_df["Loss"].sum()
    
    # Category-wise loss
    category_loss = returned_df.groupby("Category")["Loss"].sum().sort_values(ascending=False)
    category_loss = pd.DataFrame(category_loss).round(2)
    
    return total_loss, category_loss


def calculate_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly return trend
    
    Args:
        df: DataFrame with Order_Date and Returned columns
        
    Returns:
        DataFrame with monthly return counts
    """
    df_copy = df.copy()
    
    # Ensure Order_Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy["Order_Date"]):
        df_copy["Order_Date"] = pd.to_datetime(df_copy["Order_Date"])
    
    df_copy["Month"] = df_copy["Order_Date"].dt.to_period("M")
    
    # Calculate monthly returns
    monthly_returns = df_copy[df_copy["Returned"] == "Yes"].groupby("Month").size()
    monthly_orders = df_copy.groupby("Month").size()
    
    monthly_stats = pd.DataFrame({
        "Total_Orders": monthly_orders,
        "Returns": monthly_returns,
        "Return_Rate_%": (monthly_returns / monthly_orders * 100).round(2)
    }).fillna(0)
    
    return monthly_stats


def get_all_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate all key metrics in one function
    
    Args:
        df: Complete DataFrame
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        "overall_return_rate": calculate_return_rate(df),
        "category_return_rate": calculate_category_return_rate(df),
        "region_return_rate": calculate_region_return_rate(df),
        "return_reasons": calculate_return_reasons(df),
        "monthly_returns": calculate_monthly_returns(df)
    }
    
    total_loss, category_loss = calculate_financial_loss(df)
    metrics["total_financial_loss"] = total_loss
    metrics["category_financial_loss"] = category_loss
    
    return metrics
