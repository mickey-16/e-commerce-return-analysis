"""
📦 Product Return Analysis & Prediction Dashboard
Interactive Streamlit application for analyzing product returns and making predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src and model directories to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "model"))

from src.data_loader import load_data
from src.preprocessing import clean_data, create_features
from src.metrics import (
    calculate_return_rate,
    calculate_category_return_rate,
    calculate_region_return_rate,
    calculate_return_reasons,
    calculate_financial_loss,
    calculate_monthly_returns
)
from src.config import CATEGORIES, REGIONS, MODEL_PATH
from model.utils import load_model, get_prediction_interpretation

# Page configuration
st.set_page_config(
    page_title="Product Return Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Load and prepare data with caching"""
    df = load_data()
    df_clean = clean_data(df)
    df_features = create_features(df_clean)
    return df_features


@st.cache_resource
def load_trained_model():
    """Load trained model with caching"""
    try:
        model, encoders = load_model(MODEL_PATH)
        return model, encoders, True
    except FileNotFoundError:
        return None, None, False


def add_prediction_to_dataset(price, quantity, delivery_days, category, region, prediction, probability):
    """
    Add new prediction to the dataset CSV file
    
    Args:
        price: Product price
        quantity: Order quantity
        delivery_days: Delivery time
        category: Product category
        region: Customer region
        prediction: Model prediction (0 or 1)
        probability: Prediction probability array
        
    Returns:
        Tuple of (success, order_id)
    """
    try:
        from datetime import datetime
        import os
        
        csv_path = DATASET_PATH
        
        # Load existing data
        df = pd.read_csv(csv_path)
        
        # Generate new IDs
        last_order_num = int(df['Order_ID'].str.replace('ORD', '').max())
        new_order_id = f"ORD{last_order_num + 1}"
        new_product_id = f"PRD{np.random.randint(1000, 9999)}"
        
        # Current date
        order_date = datetime.now().strftime("%Y-%m-%d")
        
        # Determine return reason based on prediction and category
        if prediction == 1:
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
        
        # Create new row
        new_row = pd.DataFrame([{
            "Order_ID": new_order_id,
            "Product_ID": new_product_id,
            "Category": category,
            "Price": price,
            "Quantity": quantity,
            "Order_Date": order_date,
            "Delivery_Days": delivery_days,
            "Region": region,
            "Returned": "Yes" if prediction == 1 else "No",
            "Return_Reason": return_reason
        }])
        
        # Append to CSV
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
        
        return True, new_order_id
        
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        return False, None


def display_kpi_metrics(df):
    """Display KPI metrics in cards"""
    total_orders = len(df)
    returned_orders = (df["Returned"] == "Yes").sum()
    return_rate = (returned_orders / total_orders) * 100
    total_loss, _ = calculate_financial_loss(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Orders", f"{total_orders:,}")
    
    with col2:
        st.metric("↩️ Returned Orders", f"{returned_orders:,}")
    
    with col3:
        st.metric("📈 Return Rate", f"{return_rate:.2f}%")
    
    with col4:
        st.metric("💰 Financial Loss", f"${total_loss:,.2f}")


def plot_category_returns(df):
    """Plot return rate by category"""
    category_stats = calculate_category_return_rate(df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(category_stats.index, category_stats["Return_Rate_%"])
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Return Rate (%)", fontsize=12)
    ax.set_title("Return Rate by Product Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig


def plot_return_reasons(df):
    """Plot return reasons pie chart"""
    returned_df = df[df["Returned"] == "Yes"]
    reason_counts = returned_df["Return_Reason"].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(reason_counts, labels=reason_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Distribution of Return Reasons", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    return fig


def plot_monthly_trend(df):
    """Plot monthly return trend"""
    monthly_stats = calculate_monthly_returns(df)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    months_str = [str(m) for m in monthly_stats.index]
    ax.plot(months_str, monthly_stats["Returns"].values, marker="o", linewidth=2)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Number of Returns", fontsize=12)
    ax.set_title("Monthly Return Trend", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_region_returns(df):
    """Plot return rate by region"""
    region_stats = calculate_region_return_rate(df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(region_stats.index, region_stats["Return_Rate_%"])
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Return Rate (%)", fontsize=12)
    ax.set_title("Return Rate by Region", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig


def main():
    """Main application function"""
    
    # Initialize session state for predictions
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    
    # Header
    st.title("📦 Product Return Analysis & Prediction System")
    st.markdown("### Comprehensive analytics and ML-powered return prediction")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_and_prepare_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Data", help="Reload dataset to see new predictions"):
        st.cache_data.clear()
        st.rerun()
    
    # Show notification if data was updated
    if st.session_state.prediction_made and 'last_order_id' in st.session_state:
        st.sidebar.success(f"✅ Dataset updated!\nNew Order: {st.session_state.last_order_id}")
    
    st.sidebar.markdown("---")
    
    # Category filter
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=CATEGORIES,
        default=CATEGORIES
    )
    
    # Region filter
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=REGIONS,
        default=REGIONS
    )
    
    # Apply filters
    df_filtered = df[
        (df["Category"].isin(selected_categories)) &
        (df["Region"].isin(selected_regions))
    ]
    
    # Check if data is available after filtering
    if len(df_filtered) == 0:
        st.warning("⚠️ No data available for selected filters. Please adjust your selection.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview & Analytics", "📈 Visualizations", "🤖 Return Prediction"])
    
    # Tab 1: Overview & Analytics
    with tab1:
        st.header("📊 Key Performance Indicators")
        display_kpi_metrics(df_filtered)
        
        st.markdown("---")
        
        # Two columns for detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 Return Rate by Category")
            category_stats = calculate_category_return_rate(df_filtered)
            st.dataframe(category_stats, use_container_width=True)
            
            st.subheader("💸 Financial Loss by Category")
            _, category_loss = calculate_financial_loss(df_filtered)
            st.dataframe(category_loss, use_container_width=True)
        
        with col2:
            st.subheader("🌍 Return Rate by Region")
            region_stats = calculate_region_return_rate(df_filtered)
            st.dataframe(region_stats, use_container_width=True)
            
            st.subheader("🔍 Return Reasons")
            return_reasons = calculate_return_reasons(df_filtered)
            st.dataframe(return_reasons, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📅 Monthly Return Trend")
        monthly_stats = calculate_monthly_returns(df_filtered)
        st.dataframe(monthly_stats.tail(10), use_container_width=True)
    
    # Tab 2: Visualizations
    with tab2:
        st.header("📈 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Analysis")
            fig1 = plot_category_returns(df_filtered)
            st.pyplot(fig1)
            
            st.subheader("Region Analysis")
            fig3 = plot_region_returns(df_filtered)
            st.pyplot(fig3)
        
        with col2:
            st.subheader("Return Reasons")
            fig2 = plot_return_reasons(df_filtered)
            st.pyplot(fig2)
            
            st.subheader("Monthly Trend")
            fig4 = plot_monthly_trend(df_filtered)
            st.pyplot(fig4)
    
    # Tab 3: Prediction
    with tab3:
        st.header("🤖 Return Probability Prediction")
        
        # Load model
        model, encoders, model_loaded = load_trained_model()
        
        if not model_loaded:
            st.error("❌ Model not found! Please train the model first by running `python model/train_model.py`")
            return
        
        st.success("✅ Model loaded successfully!")
        
        st.markdown("### Enter Order Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price = st.number_input("Price ($)", min_value=100.0, max_value=10000.0, value=1500.0, step=100.0)
            quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
        
        with col2:
            delivery_days = st.slider("Delivery Days", min_value=2, max_value=15, value=7)
            category = st.selectbox("Category", options=CATEGORIES)
        
        with col3:
            region = st.selectbox("Region", options=REGIONS)
        
        # Prediction button
        if st.button("🔮 Predict Return Probability", type="primary"):
            # Store prediction inputs in session state
            with st.spinner("Making prediction..."):
                try:
                    # Encode features
                    cat_encoded = encoders["category"].transform([category])[0]
                    reg_encoded = encoders["region"].transform([region])[0]
                    
                    # Create feature array
                    features = np.array([[price, quantity, delivery_days, cat_encoded, reg_encoded]])
                    
                    # Scale features
                    scaler = encoders["scaler"]
                    features_scaled = scaler.transform(features)
                    
                    # Make prediction
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                    
                    # Save prediction to dataset
                    success, order_id = add_prediction_to_dataset(
                        price, quantity, delivery_days, category, region, 
                        prediction, probability
                    )
                    
                    # Store results in session state
                    st.session_state.prediction_made = True
                    st.session_state.prediction_results = {
                        'prediction': prediction,
                        'probability': probability,
                        'inputs': {
                            'price': price,
                            'quantity': quantity,
                            'delivery_days': delivery_days,
                            'category': category,
                            'region': region
                        },
                        'saved': success,
                        'order_id': order_id
                    }
                    
                    if success:
                        st.session_state.last_order_id = order_id
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.session_state.prediction_made = False
        
        # Display results if prediction was made
        if st.session_state.prediction_made and st.session_state.prediction_results:
            results = st.session_state.prediction_results
            prediction = results['prediction']
            probability = results['probability']
            inputs = results['inputs']
            
            # Show save status
            if results.get('saved', False):
                st.success(f"✅ **Prediction saved to dataset!** Order ID: `{results['order_id']}`")
                st.info("💡 Click **🔄 Refresh Data** in the sidebar to see updated analytics and visualizations!")
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error(f"**Prediction:** WILL RETURN")
                else:
                    st.success(f"**Prediction:** WILL NOT RETURN")
            
            with col2:
                st.metric("Return Probability", f"{probability[1]*100:.2f}%")
            
            with col3:
                st.metric("No-Return Probability", f"{probability[0]*100:.2f}%")
            
            # Interpretation
            interpretation = get_prediction_interpretation(prediction, probability)
            
            if prediction == 1:
                st.warning(f"### {interpretation}")
                st.markdown("""
                **Recommendations:**
                - 📞 Contact customer for delivery preferences
                - 📦 Ensure proper packaging
                - 📸 Include detailed product images
                - ✅ Verify order details before shipment
                """)
            else:
                st.success(f"### {interpretation}")
                st.markdown("""
                **Recommendations:**
                - ✅ Standard processing
                - 📦 Regular packaging
                - 🚚 Normal delivery schedule
                """)
            
            # Show input summary
            st.markdown("---")
            st.markdown("### 📋 Input Summary")
            input_df = pd.DataFrame({
                "Parameter": ["Price", "Quantity", "Delivery Days", "Category", "Region"],
                "Value": [f"${inputs['price']}", inputs['quantity'], inputs['delivery_days'], 
                         inputs['category'], inputs['region']]
            })
            st.table(input_df)
            
            # Clear prediction button
            if st.button("🔄 Make Another Prediction"):
                st.session_state.prediction_made = False
                st.session_state.prediction_results = None
                st.rerun()
            
            # Clear data cache to show new prediction in analytics
            if results.get('saved', False):
                st.markdown("---")
                if st.button("📊 View Updated Analytics", type="secondary"):
                    st.cache_data.clear()
                    st.success("✅ Data refreshed! Switch to Overview or Visualizations tab to see changes.")
                    st.rerun()
        
        # Add some information
        st.markdown("---")
        st.info("""
        **ℹ️ How it works:**
        
        This prediction system uses a Logistic Regression model trained on historical return data.
        The model considers:
        - Product price and quantity
        - Delivery time
        - Product category and region
        
        Higher delivery times and fashion products typically have higher return probabilities.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>📦 Product Return Analysis System | Built with Streamlit & Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
