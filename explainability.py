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
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    return shap_values
