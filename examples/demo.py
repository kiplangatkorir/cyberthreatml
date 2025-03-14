"""
Simple demonstration of the CyberThreat-ML model.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.evaluation import evaluate_model, plot_confusion_matrix

def create_synthetic_data(n_samples=1000, n_features=10):
    """Create synthetic network traffic data."""
    np.random.seed(42)
    
    # Generate normal traffic (mean around 0.3)
    normal = np.random.normal(loc=0.3, scale=0.2, size=(n_samples // 2, n_features))
    
    # Generate threat traffic (mean around 0.7)
    threats = np.random.normal(loc=0.7, scale=0.2, size=(n_samples // 2, n_features))
    
    # Combine data and labels
    X = np.vstack([normal, threats])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    return X, y

def main():
    print("CyberThreat-ML Demo")
    print("-----------------")
    
    # Create synthetic data
    print("\nGenerating synthetic network traffic data...")
    X, y = create_synthetic_data()
    print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Split data
    print("\nSplitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    print("\nTraining threat detection model...")
    model = ThreatDetectionModel(
        input_shape=(X.shape[1],),
        model_config={
            'hidden_layers': [32, 16],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam'
        }
    )
    
    history = model.train(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nTest Results:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to 'confusion_matrix.png'")
    
    # Make some example predictions
    print("\nExample Predictions:")
    n_examples = 5
    example_data = X_test[:n_examples]
    true_labels = y_test[:n_examples]
    predictions = model.predict_proba(example_data)
    
    print("\nSample  True Label  Predicted Probability")
    print("-------  ----------  --------------------")
    for i in range(n_examples):
        print(f"{i+1:^7d}  {int(true_labels[i]):^10d}  {predictions[i]:^20.4f}")

if __name__ == "__main__":
    main()
