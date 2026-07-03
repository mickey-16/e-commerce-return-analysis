"""
Machine Learning Model Training Script
Trains a Logistic Regression model to predict product returns
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from preprocessing import clean_data, prepare_ml_features
from config import TEST_SIZE, ML_RANDOM_STATE, MODEL_PATH
import utils as model_utils


def train_return_prediction_model():
    """
    Complete pipeline for training return prediction model
    """
    print("\n" + "=" * 70)
    print("🚀 STARTING MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    df = load_data()
    
    # Step 2: Clean data
    print("\n🧹 Step 2: Cleaning data...")
    df_clean = clean_data(df)
    
    # Step 3: Prepare ML features
    print("\n🔧 Step 3: Preparing ML features...")
    X, y, le_category, le_region = prepare_ml_features(df_clean)
    
    # Step 4: Split data
    print("\n✂️ Step 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Step 5: Scale features
    print("\n📊 Step 5: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   Features scaled using StandardScaler")
    
    # Step 6: Train model
    print("\n🤖 Step 6: Training Logistic Regression model...")
    model = LogisticRegression(random_state=ML_RANDOM_STATE, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print(f"   ✅ Model training completed!")
    
    # Step 7: Make predictions
    print("\n🔮 Step 7: Making predictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Step 8: Evaluate model
    print("\n📈 Step 8: Evaluating model...")
    results = model_utils.evaluate_model(y_test, y_pred)
    model_utils.print_model_evaluation(results)
    
    # Step 9: Feature importance
    print("\n🔍 Step 9: Analyzing feature importance...")
    feature_names = list(X.columns)
    feature_importance = model_utils.get_feature_importance(model, feature_names)
    print("\n📊 FEATURE IMPORTANCE (by absolute coefficient):")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["Feature"], feature_importance["Coefficient"])
    plt.xlabel("Coefficient Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Feature Importance (Logistic Regression Coefficients)", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "feature_importance.png", dpi=300, bbox_inches="tight")
    print("   📊 Feature importance plot saved to model/feature_importance.png")
    plt.close()
    
    # Step 10: Save model
    print("\n💾 Step 10: Saving model...")
    encoders = {
        "category": le_category,
        "region": le_region,
        "scaler": scaler
    }
    model_utils.save_model(model, encoders, MODEL_PATH)
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return model, encoders, results


def test_single_prediction(model, encoders):
    """
    Test the model with a sample prediction
    
    Args:
        model: Trained model
        encoders: Dictionary of encoders
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING SINGLE PREDICTION")
    print("=" * 70)
    
    # Sample test case
    test_cases = [
        {
            "price": 1500.0,
            "quantity": 1,
            "delivery_days": 12,
            "category": "Fashion",
            "region": "East"
        },
        {
            "price": 3500.0,
            "quantity": 1,
            "delivery_days": 3,
            "category": "Electronics",
            "region": "West"
        },
        {
            "price": 800.0,
            "quantity": 2,
            "delivery_days": 5,
            "category": "Home",
            "region": "North"
        }
    ]
    
    scaler = encoders["scaler"]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"   Price: ${test['price']}")
        print(f"   Quantity: {test['quantity']}")
        print(f"   Delivery Days: {test['delivery_days']}")
        print(f"   Category: {test['category']}")
        print(f"   Region: {test['region']}")
        
        # Encode features
        cat_encoded = encoders["category"].transform([test["category"]])[0]
        reg_encoded = encoders["region"].transform([test["region"]])[0]
        
        # Create feature array
        features = np.array([[
            test["price"],
            test["quantity"],
            test["delivery_days"],
            cat_encoded,
            reg_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        interpretation = model_utils.get_prediction_interpretation(prediction, probability)
        print(f"\n   {interpretation}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Train the model
    model, encoders, results = train_return_prediction_model()
    
    # Test with sample predictions
    test_single_prediction(model, encoders)
