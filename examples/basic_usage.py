"""
Basic usage example for the CyberThreat-ML library.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the cyberthreat_ml package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.evaluation import evaluate_model, classification_report, plot_confusion_matrix
from cyberthreat_ml.explain import explain_model, plot_shap_summary
from cyberthreat_ml.utils import load_dataset, save_dataset, split_data

def main():
    """
    Example of basic usage of the CyberThreat-ML library.
    """
    print("CyberThreat-ML Basic Usage Example")
    print("------------------------------------")
    
    # Step 1: Create or load a dataset
    # For demonstration, we'll create a synthetic dataset
    print("\nStep 1: Creating a synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=1000, n_features=20)
    print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Step 2: Split the dataset
    print("\nStep 2: Splitting dataset into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.25, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Create and train the model
    print("\nStep 3: Creating and training the model...")
    model = ThreatDetectionModel(
        input_shape=(X_train.shape[1],),
        model_config={
            'hidden_layers': [64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy', 'AUC', 'Precision', 'Recall'],
            'optimizer': 'adam'
        }
    )
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=10,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=3
    )
    
    # Step 4: Evaluate the model
    print("\nStep 4: Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test F1 score: {metrics['f1_score']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(model, X_test, y_test)
    print(report)
    
    # Step 5: Visualize the confusion matrix
    print("\nStep 5: Creating visualizations...")
    plt.figure()
    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    
    # Step 6: Model explainability
    print("\nStep 6: Generating model explanations...")
    explainer, shap_values = explain_model(model, X_train[:100], X_test[:10])
    
    plt.figure()
    plot_shap_summary(shap_values)
    plt.savefig('shap_summary.png')
    print("SHAP summary plot saved to 'shap_summary.png'")
    
    # Step 7: Save the model
    print("\nStep 7: Saving the model...")
    # Create model directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    model.save_model(os.path.join('models', 'threat_detection_model'), 
                    os.path.join('models', 'threat_detection_metadata.json'))
    print("Model saved to 'models/threat_detection_model'")
    print("Model metadata saved to 'models/threat_detection_metadata.json'")
    
    print("\nBasic usage example completed successfully!")


def create_synthetic_dataset(n_samples=1000, n_features=20):
    """
    Create a synthetic dataset for cybersecurity threat detection.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        
    Returns:
        tuple: (X, y) - feature matrix and labels.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels based on a synthetic rule
    # Here, we're creating a simple rule where samples with high values in 
    # certain features are more likely to be threats
    
    # Create base probabilities
    probs = np.zeros(n_samples)
    
    # Add influence from specific features (simulating indicators of compromise)
    for i in range(3):  # Use first 3 features as stronger indicators
        feature_idx = i
        # Scale feature to 0-1 range and add to probability
        probs += 0.2 * (X[:, feature_idx] - X[:, feature_idx].min()) / (X[:, feature_idx].max() - X[:, feature_idx].min())
    
    # Create interactions between features (simulating complex patterns)
    for i in range(2):
        for j in range(i+1, 4):
            interaction = X[:, i] * X[:, j]
            interaction = (interaction - interaction.min()) / (interaction.max() - interaction.min())
            probs += 0.1 * interaction
    
    # Add some noise and normalize to 0-1
    probs += 0.05 * np.random.randn(n_samples)
    probs = (probs - probs.min()) / (probs.max() - probs.min())
    
    # Convert to binary labels with some class imbalance (20% threats)
    threshold = np.percentile(probs, 80)
    y = (probs >= threshold).astype(int)
    
    return X, y


if __name__ == "__main__":
    main()
