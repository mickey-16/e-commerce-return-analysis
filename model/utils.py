"""
Model Utilities Module
Helper functions for model training, evaluation, and prediction
"""

import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import MODEL_PATH


def save_model(model: Any, encoders: Dict, filepath: Path = MODEL_PATH):
    """
    Save trained model and encoders to disk
    
    Args:
        model: Trained scikit-learn model
        encoders: Dictionary of label encoders
        filepath: Path to save the model
    """
    model_package = {
        "model": model,
        "encoders": encoders
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(model_package, f)
    
    print(f"✅ Model saved to: {filepath}")


def load_model(filepath: Path = MODEL_PATH) -> Tuple[Any, Dict]:
    """
    Load trained model and encoders from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Tuple of (model, encoders)
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model not found at: {filepath}")
    
    with open(filepath, "rb") as f:
        model_package = pickle.load(f)
    
    return model_package["model"], model_package["encoders"]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=["Not Returned", "Returned"])
    
    results = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }
    
    return results


def print_model_evaluation(results: Dict):
    """
    Print model evaluation results in a formatted way
    
    Args:
        results: Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 70)
    print("🤖 MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 ACCURACY: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    print("\n📉 CONFUSION MATRIX:")
    print(results['confusion_matrix'])
    print("\n   [[TN  FP]")
    print("    [FN  TP]]")
    
    print("\n📋 CLASSIFICATION REPORT:")
    print(results['classification_report'])
    
    print("=" * 70)


def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from the trained model
    
    Args:
        model: Trained model with coefficients
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, "coef_"):
        importance = model.coef_[0]
        
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": importance,
            "Abs_Coefficient": np.abs(importance)
        }).sort_values("Abs_Coefficient", ascending=False)
        
        return feature_importance
    else:
        print("⚠️ Model does not have coefficients")
        return pd.DataFrame()


def predict_return_probability(
    model: Any,
    encoders: Dict,
    price: float,
    quantity: int,
    delivery_days: int,
    category: str,
    region: str
) -> Tuple[int, float]:
    """
    Predict return probability for a single order
    
    Args:
        model: Trained model
        encoders: Dictionary of label encoders
        price: Product price
        quantity: Order quantity
        delivery_days: Delivery time in days
        category: Product category
        region: Customer region
        
    Returns:
        Tuple of (prediction, probability)
    """
    # Encode categorical features
    category_encoded = encoders["category"].transform([category])[0]
    region_encoded = encoders["region"].transform([region])[0]
    
    # Create feature array
    features = np.array([[price, quantity, delivery_days, category_encoded, region_encoded]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return prediction, probability


def get_prediction_interpretation(prediction: int, probability: np.ndarray) -> str:
    """
    Generate human-readable interpretation of prediction
    
    Args:
        prediction: Predicted class (0 or 1)
        probability: Probability array [prob_no_return, prob_return]
        
    Returns:
        Interpretation string
    """
    if prediction == 1:
        confidence = probability[1] * 100
        return f"⚠️ HIGH RISK: This order is likely to be returned ({confidence:.1f}% probability)"
    else:
        confidence = probability[0] * 100
        return f"✅ LOW RISK: This order is unlikely to be returned ({confidence:.1f}% confidence)"
