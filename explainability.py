import shap
import numpy as np

def explain_predictions(model, X_test):
    """
    Generate SHAP values to explain model predictions.
    
    Args:
        model: Trained TensorFlow model
        X_test: Test data to explain predictions for
        
    Returns:
        shap_values: SHAP values for model predictions
    """
    # Create a background dataset for SHAP
    background = X_test[:100]  # Using first 100 samples as background
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test[:10])  # Explain first 10 predictions
    return shap_values
