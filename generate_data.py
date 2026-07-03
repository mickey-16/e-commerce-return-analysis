"""
Synthetic Dataset Generator for Product Return Analysis
Generates realistic e-commerce return data with logical relationships
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from config import *

np.random.seed(RANDOM_STATE)


def generate_returns_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate realistic synthetic product return dataset
    
    Args:
        n_samples: Number of records to generate
        
    Returns:
        DataFrame with product return data
    """
    data = []
    
    # Generate date range (last 12 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for i in range(n_samples):
        order_id = f"ORD{10000 + i}"
        product_id = f"PRD{np.random.randint(1000, 9999)}"
        
        # Select category
        category = np.random.choice(CATEGORIES)
        
        # Generate price based on category
        price_min, price_max = PRICE_RANGES[category]
        price = np.round(np.random.uniform(price_min, price_max), 2)
        
        # Generate quantity (most orders are 1-3 items)
        quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
        
        # Generate order date
        random_days = np.random.randint(0, 365)
        order_date = start_date + timedelta(days=random_days)
        
        # Generate delivery days (2-15 days)
        delivery_days = np.random.choice(range(2, 16), p=[0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005])
        
        # Select region
        region = np.random.choice(REGIONS)
        
        # Calculate return probability based on multiple factors
        base_return_prob = RETURN_PROB_BY_CATEGORY[category]
        
        # Adjust probability based on delivery days (longer = higher return rate)
        delivery_factor = 1.0 + (delivery_days - 7) * 0.02
        
        # Adjust probability based on price (higher price = slightly lower return for electronics)
        if category == "Electronics" and price > 3000:
            price_factor = 0.85
        elif category == "Fashion":
            price_factor = 1.1  # Fashion has higher returns
        else:
            price_factor = 1.0
        
        # Final return probability
        return_prob = base_return_prob * delivery_factor * price_factor
        return_prob = np.clip(return_prob, 0.05, 0.45)  # Keep realistic bounds
        
        # Determine if returned
        returned = np.random.random() < return_prob
        
        # Assign return reason based on category and return status
        if returned:
            if category == "Fashion":
                return_reason = np.random.choice(
                    ["Size Issue", "Quality Issue", "Damaged", "Wrong Item"],
                    p=[0.5, 0.25, 0.15, 0.1]
                )
            elif category == "Electronics":
                return_reason = np.random.choice(
                    ["Damaged", "Quality Issue", "Wrong Item", "Size Issue"],
                    p=[0.4, 0.35, 0.2, 0.05]
                )
            elif category == "Home":
                return_reason = np.random.choice(
                    ["Damaged", "Wrong Item", "Quality Issue", "Size Issue"],
                    p=[0.4, 0.3, 0.25, 0.05]
                )
            else:  # Sports
                return_reason = np.random.choice(
                    ["Size Issue", "Quality Issue", "Damaged", "Wrong Item"],
                    p=[0.4, 0.3, 0.2, 0.1]
                )
        else:
            return_reason = "None"
        
        # Append record
        data.append({
            "Order_ID": order_id,
            "Product_ID": product_id,
            "Category": category,
            "Price": price,
            "Quantity": quantity,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Delivery_Days": delivery_days,
            "Region": region,
            "Returned": "Yes" if returned else "No",
            "Return_Reason": return_reason
        })
    
    df = pd.DataFrame(data)
    
    # Introduce some realistic missing values (2-3%)
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, "Delivery_Days"] = np.nan
    
    # Introduce a few duplicates (to test cleaning)
    duplicates = df.sample(n=5, random_state=RANDOM_STATE)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df


if __name__ == "__main__":
    print("🔄 Generating synthetic product return dataset...")
    
    # Generate dataset
    df = generate_returns_dataset()
    
    # Save to CSV
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Total records: {len(df)}")
    print(f"📁 Saved to: {DATASET_PATH}")
    print(f"\n📈 Dataset Preview:")
    print(df.head(10))
    print(f"\n📋 Dataset Info:")
    print(df.info())
