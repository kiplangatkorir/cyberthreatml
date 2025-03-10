"""
Example of multi-class cyber threat classification with CyberThreat-ML.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the cyberthreat_ml package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.utils import split_data

def main():
    """
    Example of multi-class threat classification using CyberThreat-ML.
    """
    print("CyberThreat-ML Multi-Class Threat Classification Example")
    print("------------------------------------------------------")
    
    # Define the threat classes
    threat_classes = [
        "Normal Traffic",
        "Port Scan",
        "DDoS",
        "Brute Force",
        "Data Exfiltration",
        "Command & Control"
    ]
    num_classes = len(threat_classes)
    
    # Step 1: Create a synthetic dataset
    print("\nStep 1: Creating a synthetic multi-class dataset...")
    X, y = create_synthetic_multiclass_dataset(
        n_samples=2000, 
        n_features=25,
        n_classes=num_classes
    )
    print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution:")
    for i, name in enumerate(threat_classes):
        count = np.sum(y == i)
        print(f"  Class {i} ({name}): {count} samples ({count/len(y)*100:.1f}%)")
    
    # Step 2: Split the dataset
    print("\nStep 2: Splitting dataset into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.25, random_state=42
    )
    
    # Step 3: Normalize features
    print("\nStep 3: Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Step 4: Create and train the model
    print("\nStep 4: Creating and training the multi-class model...")
    model = ThreatDetectionModel(
        input_shape=(X_train.shape[1],),
        num_classes=num_classes,
        model_config={
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'softmax',
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam',
            'class_names': threat_classes
        }
    )
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=15,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=3
    )
    
    # Step 5: Evaluate the model
    print("\nStep 5: Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=threat_classes)
    print(report)
    
    # Step 6: Visualize the confusion matrix
    print("\nStep 6: Creating visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(threat_classes))
    plt.xticks(tick_marks, threat_classes, rotation=45)
    plt.yticks(tick_marks, threat_classes)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('multiclass_confusion_matrix.png')
    print("Confusion matrix saved to 'multiclass_confusion_matrix.png'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('multiclass_training_history.png')
    print("Training history plot saved to 'multiclass_training_history.png'")
    
    # Step 7: Show detailed prediction examples
    print("\nStep 7: Demonstrating detailed predictions...")
    detailed_results = model.predict_with_explanations(X_test[:5])
    
    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"  True class: {threat_classes[y_test[i]]}")
        pred_class = detailed_results['predictions'][i]
        print(f"  Predicted class: {threat_classes[pred_class]}")
        
        # Get probabilities for all classes
        probs = detailed_results['probabilities'][i]
        print("  Class probabilities:")
        for j, class_name in enumerate(threat_classes):
            print(f"    {class_name}: {probs[j]:.4f}")
    
    # Step 8: Save the model
    print("\nStep 8: Saving the model...")
    # Create model directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    model.save_model(
        os.path.join('models', 'multiclass_threat_model'), 
        os.path.join('models', 'multiclass_threat_metadata.json')
    )
    print("Model saved to 'models/multiclass_threat_model'")
    print("Model metadata saved to 'models/multiclass_threat_metadata.json'")
    
    print("\nMulti-class classification example completed successfully!")


def create_synthetic_multiclass_dataset(n_samples=2000, n_features=25, n_classes=6):
    """
    Create a synthetic multi-class dataset for cyber threat detection.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_classes (int): Number of threat classes (including normal traffic).
        
    Returns:
        tuple: (X, y) - features and class labels.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate feature matrix with slight correlations between features
    X = np.random.randn(n_samples, n_features)
    
    # Create base vectors for each class with some distinguishing patterns
    class_patterns = []
    for i in range(n_classes):
        # Create a class pattern - each class will have certain feature ranges
        pattern = np.random.uniform(-0.5, 0.5, n_features)
        
        # Emphasize specific features for each class to make them more distinguishable
        strong_features = np.random.choice(n_features, size=int(n_features * 0.3), replace=False)
        pattern[strong_features] = np.random.uniform(1.5, 2.5, len(strong_features)) * np.sign(pattern[strong_features])
        
        class_patterns.append(pattern)
    
    # Generate labels with class imbalance (normal traffic should be more common)
    class_weights = np.array([0.4, 0.15, 0.15, 0.1, 0.1, 0.1])  # Adjust to have n_classes elements
    class_weights = class_weights[:n_classes] / np.sum(class_weights[:n_classes])  # Normalize
    
    y = np.random.choice(n_classes, size=n_samples, p=class_weights)
    
    # Modify features based on class patterns
    for i in range(n_samples):
        class_idx = y[i]
        # Add class-specific pattern with some noise
        X[i] += class_patterns[class_idx] + np.random.normal(0, 0.3, n_features)
    
    # Add some feature interdependencies for specific threat types
    
    # Port Scan features (class 1)
    port_scan_samples = np.where(y == 1)[0]  # Get indices where y == 1
    if len(port_scan_samples) > 0:
        # Increase variability in port-related features (e.g., 0, 1, 2)
        port_features = np.array([0, 1, 2])
        for i in port_scan_samples:
            X[i, port_features] *= 1.5
        X[port_scan_samples[:, np.newaxis], port_features] += np.random.uniform(1.0, 2.0, (len(port_scan_samples), len(port_features)))
    
    # DDoS features (class 2)
    ddos_samples = np.where(y == 2)[0]  # Get indices where y == 2
    if len(ddos_samples) > 0:
        # Increase in traffic volume and packet size features (e.g., 3, 4, 5)
        traffic_features = np.array([3, 4, 5])
        for i in ddos_samples:
            X[i, traffic_features] *= 2.0
        X[ddos_samples[:, np.newaxis], traffic_features] += np.random.uniform(1.5, 3.0, (len(ddos_samples), len(traffic_features)))
    
    # Brute Force features (class 3)
    brute_force_samples = np.where(y == 3)[0]  # Get indices where y == 3
    if len(brute_force_samples) > 0:
        # Repeated authentication attempts (e.g., 6, 7, 8)
        auth_features = np.array([6, 7, 8])
        for i in brute_force_samples:
            X[i, auth_features] *= 1.8
        X[brute_force_samples[:, np.newaxis], auth_features] += np.random.uniform(1.0, 2.0, (len(brute_force_samples), len(auth_features)))
    
    # Data Exfiltration features (class 4)
    exfil_samples = np.where(y == 4)[0]  # Get indices where y == 4
    if len(exfil_samples) > 0:
        # Large outbound data transfers (e.g., 9, 10, 11)
        data_features = np.array([9, 10, 11])
        for i in exfil_samples:
            X[i, data_features] *= 1.7
        X[exfil_samples[:, np.newaxis], data_features] += np.random.uniform(1.2, 2.5, (len(exfil_samples), len(data_features)))
    
    # Command & Control features (class 5)
    cnc_samples = np.where(y == 5)[0]  # Get indices where y == 5
    if len(cnc_samples) > 0:
        # Periodic communication patterns (e.g., 12, 13, 14)
        cnc_features = np.array([12, 13, 14])
        for i in cnc_samples:
            X[i, cnc_features] *= 1.6
        X[cnc_samples[:, np.newaxis], cnc_features] += np.random.uniform(0.8, 1.8, (len(cnc_samples), len(cnc_features)))
    
    return X, y


if __name__ == "__main__":
    main()