"""
Module for model evaluation and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, precision_recall_curve,
                            roc_curve, classification_report as sk_classification_report)
from .logger import logger

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained model on test data.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        threshold (float): Decision threshold for binary classification.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)
    
    # Get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    logger.info(f"Model evaluation results: Accuracy={metrics['accuracy']:.4f}, "
               f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
               f"F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")
    
    return metrics

def classification_report(model, X_test, y_test, threshold=0.5):
    """
    Generate a classification report for the model.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        threshold (float): Decision threshold for binary classification.
        
    Returns:
        str: Classification report as a string.
    """
    # Get binary predictions
    y_pred = model.predict(X_test, threshold=threshold)
    
    # Generate report
    report = sk_classification_report(y_test, y_pred)
    logger.info(f"Classification report:\n{report}")
    
    return report

def plot_confusion_matrix(model, X_test, y_test, threshold=0.5, normalize=True, 
                         cmap='Blues', figsize=(8, 6)):
    """
    Plot the confusion matrix for the model.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        threshold (float): Decision threshold for binary classification.
        normalize (bool): Whether to normalize the confusion matrix.
        cmap (str): Colormap for the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Get binary predictions
    y_pred = model.predict(X_test, threshold=threshold)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure and plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
               xticklabels=['Normal', 'Threat'], yticklabels=['Normal', 'Threat'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    return plt.gcf()

def plot_roc_curve(model, X_test, y_test, figsize=(8, 6)):
    """
    Plot the ROC curve for the model.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create figure and plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    return plt.gcf()

def plot_precision_recall_curve(model, X_test, y_test, figsize=(8, 6)):
    """
    Plot the Precision-Recall curve for the model.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Create figure and plot
    plt.figure(figsize=figsize)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Add average precision score
    ap = np.mean(precision)
    plt.annotate(f'Average Precision = {ap:.4f}', xy=(0.5, 0.5), xycoords='axes fraction')
    
    return plt.gcf()

def find_optimal_threshold(model, X_val, y_val, metric='f1'):
    """
    Find the optimal threshold for binary classification based on a validation set.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        metric (str): Metric to optimize ('f1', 'precision', 'recall', or 'accuracy').
        
    Returns:
        float: Optimal threshold.
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_val)
    
    # Try different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        scores.append(score)
    
    # Find threshold with the highest score
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    
    logger.info(f"Optimal threshold (optimizing {metric}): {optimal_threshold:.2f} "
               f"with {metric} = {scores[optimal_idx]:.4f}")
    
    return optimal_threshold

def plot_threshold_scores(model, X_val, y_val, metrics=None, figsize=(10, 6)):
    """
    Plot scores for different metrics across various thresholds.
    
    Args:
        model (ThreatDetectionModel): Trained model.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        metrics (list): List of metrics to plot ('f1', 'precision', 'recall', 'accuracy').
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_val)
    
    # Try different thresholds
    thresholds = np.arange(0.05, 1.0, 0.05)
    results = {metric: [] for metric in metrics}
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if 'accuracy' in metrics:
            results['accuracy'].append(accuracy_score(y_val, y_pred))
        if 'precision' in metrics:
            results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        if 'recall' in metrics:
            results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        if 'f1' in metrics:
            results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
    # Create figure and plot
    plt.figure(figsize=figsize)
    for metric in metrics:
        plt.plot(thresholds, results[metric], label=metric)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()
