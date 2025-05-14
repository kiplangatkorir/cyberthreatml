"""
Module for model explainability using SHAP.
""" 

import numpy as np
import matplotlib.pyplot as plt
from .logger import logger

# Make SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available. Model explainability features will be limited.")
    SHAP_AVAILABLE = False

def explain_prediction(model, X_sample, feature_names=None):
    """
    Explain a single prediction using SHAP.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_sample (numpy.ndarray): Sample input for explanation (should be 2D).
        feature_names (list, optional): List of feature names.
        
    Returns:
        dict: Dictionary with SHAP values and base value.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Returning basic prediction without explanation.")
        # Ensure X_sample is 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        prediction = float(model.predict_proba(X_sample)[0])
        return {
            'prediction': prediction,
            'feature_names': feature_names if feature_names is not None else [],
            'shap_values': [],
            'base_value': 0.5
        }
    
    # Ensure X_sample is 2D
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model.predict_proba, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values.values[0],
            'base_value': shap_values.base_values[0],
            'prediction': float(model.predict_proba(X_sample)[0])
        }
        
        if feature_names is not None:
            explanation['feature_names'] = feature_names
        
        logger.info(f"Generated SHAP explanation for sample, prediction={explanation['prediction']:.4f}")
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error during SHAP explanation: {str(e)}")
        raise

def explain_model(model, X_background, X_explain=None, feature_names=None, max_display=10):
    """
    Create a SHAP explainer for the model and generate summary plots.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_background (numpy.ndarray): Background data for SHAP explainer.
        X_explain (numpy.ndarray, optional): Data to explain (if None, uses X_background).
        feature_names (list, optional): List of feature names.
        max_display (int): Maximum number of features to display in summary plots.
        
    Returns:
        tuple: (SHAP explainer, SHAP values) or (None, None) if SHAP is not available
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Cannot create model explanation.")
        return None, None
        
    try:
        # Create a function that returns the model's predicted probabilities
        predict_fn = model.predict_proba
        
        # Create SHAP explainer
        logger.info(f"Creating SHAP explainer with {len(X_background)} background samples")
        explainer = shap.Explainer(predict_fn, X_background)
        
        # Data to explain
        X_to_explain = X_explain if X_explain is not None else X_background
        
        # Calculate SHAP values
        logger.info(f"Calculating SHAP values for {len(X_to_explain)} samples")
        shap_values = explainer(X_to_explain)
        
        return explainer, shap_values
    
    except Exception as e:
        logger.error(f"Error during model explanation: {str(e)}")
        raise

def plot_shap_summary(shap_values, feature_names=None, max_display=10):
    """
    Plot a SHAP summary plot.
    
    Args:
        shap_values (shap.Explanation): SHAP values.
        feature_names (list, optional): List of feature names.
        max_display (int): Maximum number of features to display.
        
    Returns:
        matplotlib.figure.Figure: The figure object or None if SHAP is not available.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Cannot create summary plot.")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "SHAP not available", horizontalalignment='center', 
                 verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()
        
    try:
        plt.figure(figsize=(10, 8))
        
        # Set feature names if provided
        if feature_names is not None and len(feature_names) == shap_values.values.shape[1]:
            shap_values.feature_names = feature_names
        
        # Create summary plot
        shap.summary_plot(shap_values, max_display=max_display, show=False)
        
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', 
                 verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

def plot_shap_waterfall(shap_values, sample_idx=0, feature_names=None, max_display=10):
    """
    Plot a SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values (shap.Explanation): SHAP values.
        sample_idx (int): Index of the sample to explain.
        feature_names (list, optional): List of feature names.
        max_display (int): Maximum number of features to display.
        
    Returns:
        matplotlib.figure.Figure: The figure object or None if SHAP is not available.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Cannot create waterfall plot.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "SHAP not available", horizontalalignment='center', 
                 verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()
        
    try:
        plt.figure(figsize=(10, 6))
        
        # Set feature names if provided
        if feature_names is not None and len(feature_names) == shap_values.values.shape[1]:
            shap_values.feature_names = feature_names
        
        # Create waterfall plot
        shap.plots.waterfall(shap_values[sample_idx], max_display=max_display, show=False)
        
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error creating SHAP waterfall plot: {str(e)}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', 
                 verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

def plot_shap_force(shap_values, sample_idx=0, feature_names=None):
    """
    Plot a SHAP force plot for a single prediction.
    
    Args:
        shap_values (shap.Explanation): SHAP values.
        sample_idx (int): Index of the sample to explain.
        feature_names (list, optional): List of feature names.
        
    Returns:
        shap.plots._force.AdditiveForceVisualizer or None if SHAP is not available.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Cannot create force plot.")
        return None
        
    try:
        # Set feature names if provided
        if feature_names is not None and len(feature_names) == shap_values.values.shape[1]:
            shap_values.feature_names = feature_names
        
        # Create force plot
        return shap.plots.force(shap_values[sample_idx])
    except Exception as e:
        logger.error(f"Error creating SHAP force plot: {str(e)}")
        return None

def get_top_features(shap_values, feature_names=None, top_n=10):
    """
    Get the top N most important features based on SHAP values.
    
    Args:
        shap_values (shap.Explanation): SHAP values.
        feature_names (list, optional): List of feature names.
        top_n (int): Number of top features to return.
        
    Returns:
        list: List of tuples (feature_name, importance_value).
    """
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Use provided feature names or generate default ones
    if feature_names is not None and len(feature_names) == len(mean_abs_shap):
        names = feature_names
    else:
        names = [f"feature_{i}" for i in range(len(mean_abs_shap))]
    
    # Create feature importance pairs
    feature_importance = list(zip(names, mean_abs_shap))
    
    # Sort by importance and return top N
    sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    logger.info(f"Top {min(top_n, len(sorted_importance))} important features identified")
    return sorted_importance[:top_n]
