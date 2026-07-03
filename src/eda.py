"""
Exploratory Data Analysis Module
Provides visualization and analysis functions
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import sys


def plot_return_by_category(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create bar chart showing return rate by category
    
    Args:
        df: DataFrame with Category and Returned columns
        save_path: Optional path to save the figure
    """
    # Calculate return rates
    category_stats = df.groupby("Category").agg({
        "Returned": lambda x: ((x == "Yes").sum() / len(x)) * 100
    }).reset_index()
    category_stats.columns = ["Category", "Return_Rate"]
    category_stats = category_stats.sort_values("Return_Rate", ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(category_stats["Category"], category_stats["Return_Rate"])
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Return Rate (%)", fontsize=12)
    plt.title("Return Rate by Product Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_return_reasons(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create pie chart showing distribution of return reasons
    
    Args:
        df: DataFrame with Return_Reason column
        save_path: Optional path to save the figure
    """
    # Filter only returned items
    returned_df = df[df["Returned"] == "Yes"]
    reason_counts = returned_df["Return_Reason"].value_counts()
    
    # Create plot
    plt.figure(figsize=(10, 7))
    plt.pie(reason_counts, labels=reason_counts.index, autopct="%1.1f%%", 
            startangle=90)
    plt.title("Distribution of Return Reasons", fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_monthly_return_trend(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create line chart showing monthly return trend
    
    Args:
        df: DataFrame with Order_Date and Returned columns
        save_path: Optional path to save the figure
    """
    df_copy = df.copy()
    
    # Ensure Order_Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy["Order_Date"]):
        df_copy["Order_Date"] = pd.to_datetime(df_copy["Order_Date"])
    
    df_copy["Month"] = df_copy["Order_Date"].dt.to_period("M")
    
    # Calculate monthly returns
    monthly_returns = df_copy[df_copy["Returned"] == "Yes"].groupby("Month").size()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    months_str = [str(m) for m in monthly_returns.index]
    plt.plot(months_str, monthly_returns.values, marker="o", linewidth=2)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Number of Returns", fontsize=12)
    plt.title("Monthly Return Trend", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_return_by_region(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create bar chart showing return rate by region
    
    Args:
        df: DataFrame with Region and Returned columns
        save_path: Optional path to save the figure
    """
    # Calculate return rates
    region_stats = df.groupby("Region").agg({
        "Returned": lambda x: ((x == "Yes").sum() / len(x)) * 100
    }).reset_index()
    region_stats.columns = ["Region", "Return_Rate"]
    region_stats = region_stats.sort_values("Return_Rate", ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(region_stats["Region"], region_stats["Return_Rate"])
    plt.xlabel("Region", fontsize=12)
    plt.ylabel("Return Rate (%)", fontsize=12)
    plt.title("Return Rate by Region", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_price_vs_returns(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create scatter plot showing relationship between price and returns
    
    Args:
        df: DataFrame with Price and Returned columns
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    returned_yes = df[df["Returned"] == "Yes"]
    returned_no = df[df["Returned"] == "No"]
    
    plt.scatter(returned_no["Price"], returned_no.index, alpha=0.3, label="Not Returned")
    plt.scatter(returned_yes["Price"], returned_yes.index, alpha=0.3, label="Returned")
    
    plt.xlabel("Price", fontsize=12)
    plt.ylabel("Order Index", fontsize=12)
    plt.title("Price Distribution: Returned vs Not Returned", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def create_all_visualizations(df: pd.DataFrame):
    """
    Generate all EDA visualizations
    
    Args:
        df: Complete DataFrame
    """
    print("📊 Generating visualizations...")
    
    print("\n1️⃣ Return Rate by Category")
    plot_return_by_category(df)
    
    print("\n2️⃣ Return Reasons Distribution")
    plot_return_reasons(df)
    
    print("\n3️⃣ Monthly Return Trend")
    plot_monthly_return_trend(df)
    
    print("\n4️⃣ Return Rate by Region")
    plot_return_by_region(df)
    
    print("\n✅ All visualizations generated!")


def print_eda_summary(df: pd.DataFrame):
    """
    Print comprehensive EDA summary
    
    Args:
        df: DataFrame to analyze
    """
    from metrics import get_all_metrics
    
    print("=" * 70)
    print("📊 EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Calculate all metrics
    metrics = get_all_metrics(df)
    
    # Overall return rate
    print(f"\n🔢 OVERALL RETURN RATE: {metrics['overall_return_rate']:.2f}%")
    
    # Financial loss
    print(f"\n💰 TOTAL FINANCIAL LOSS: ${metrics['total_financial_loss']:,.2f}")
    
    # Category-wise return rate
    print("\n📦 RETURN RATE BY CATEGORY:")
    print(metrics['category_return_rate'])
    
    # Region-wise return rate
    print("\n🌍 RETURN RATE BY REGION:")
    print(metrics['region_return_rate'])
    
    # Return reasons
    print("\n🔍 RETURN REASONS:")
    print(metrics['return_reasons'])
    
    # Category-wise financial loss
    print("\n💸 FINANCIAL LOSS BY CATEGORY:")
    print(metrics['category_financial_loss'])
    
    # Monthly trend summary
    print("\n📅 MONTHLY RETURN TREND (Last 5 months):")
    print(metrics['monthly_returns'].tail())
    
    print("\n" + "=" * 70)
