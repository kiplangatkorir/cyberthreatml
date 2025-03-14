#!/usr/bin/env python3
"""
Basic Real-World Testing for CyberThreat-ML

This script demonstrates a basic version of real-world testing
without requiring external libraries.

It creates simple simulated data and evaluates basic threat detection capabilities.
"""

import os
import sys
import random
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import library modules
try:
    from cyberthreat_ml.model import ThreatDetectionModel
    HAS_MODEL = True
except ImportError:
    print("Failed to import CyberThreat-ML model module. Testing without it.")
    HAS_MODEL = False
    
    # Define a placeholder ThreatDetectionModel to avoid unbound variable error
    class ThreatDetectionModel:
        def __init__(self, input_shape=None, num_classes=None, model_config=None):
            pass
        def train(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            pass

# Directory for storing results
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define attack types
ATTACK_TYPES = [
    'Port Scan',
    'Normal',
    'DoS/DDoS',
    'Data Exfiltration',
    'Web Attack',
    'Brute Force'
]

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def create_simple_dataset(n_samples=100, n_features=10, class_ratio=0.7):
    """
    Create a simple dataset that simulates network traffic.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features per sample
        class_ratio (float): Ratio of normal traffic to total traffic
        
    Returns:
        tuple: (X, y, class_counts)
    """
    print_section("Creating Simple Dataset")
    print(f"Generating {n_samples} samples with {n_features} features")
    
    # Initialize lists for features and labels
    X = []
    y = []
    class_counts = {label: 0 for label in range(len(ATTACK_TYPES))}
    
    # Calculate number of samples for each class
    n_normal = int(n_samples * class_ratio)
    n_attacks = n_samples - n_normal
    
    # Generate normal traffic
    for _ in range(n_normal):
        # Normal traffic has moderate values
        features = np.random.normal(0.3, 0.1, n_features)
        X.append(features)
        y.append(1)  # Normal traffic class
        class_counts[1] += 1
    
    # Generate attack traffic
    attack_types = [0, 2, 3, 4, 5]  # All types except normal (1)
    for _ in range(n_attacks):
        attack_type = random.choice(attack_types)
        
        # Different attack types have different patterns
        if attack_type == 0:  # Port Scan
            features = np.random.normal(0.8, 0.1, n_features)
            features[0] = np.random.normal(0.9, 0.05)  # High port activity
        elif attack_type == 2:  # DoS/DDoS
            features = np.random.normal(0.7, 0.1, n_features)
            features[1] = np.random.normal(0.95, 0.05)  # High traffic volume
        elif attack_type == 3:  # Data Exfiltration
            features = np.random.normal(0.6, 0.1, n_features)
            features[2] = np.random.normal(0.85, 0.05)  # High data transfer
        elif attack_type == 4:  # Web Attack
            features = np.random.normal(0.65, 0.1, n_features)
            features[3] = np.random.normal(0.8, 0.05)  # Suspicious HTTP patterns
        else:  # Brute Force
            features = np.random.normal(0.75, 0.1, n_features)
            features[4] = np.random.normal(0.9, 0.05)  # High auth failures
        
        X.append(features)
        y.append(attack_type)
        class_counts[attack_type] += 1
    
    # Convert to numpy arrays and shuffle
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into training and test sets
    train_size = int(0.7 * n_samples)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X, y, class_counts, X_train, X_test, y_train, y_test

class SimpleModel:
    """A simple model that predicts based on feature thresholds."""
    
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.feature_thresholds = {}
        self.class_weights = {}
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the simple model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training simple model...")
        
        # Find average value for each feature for each class
        class_features = {}
        for i in range(len(X_train)):
            label = y_train[i]
            features = X_train[i]
            
            if label not in class_features:
                class_features[label] = []
                
            class_features[label].append(features)
        
        # Calculate average for each feature for each class
        class_averages = {}
        for label, feature_list in class_features.items():
            feature_count = len(feature_list[0])
            averages = [0] * feature_count
            
            for features in feature_list:
                for j in range(feature_count):
                    averages[j] += features[j] / len(feature_list)
            
            class_averages[label] = averages
        
        # Calculate thresholds for each feature
        self.feature_thresholds = {}
        for j in range(len(X_train[0])):
            self.feature_thresholds[j] = {}
            
            for label, averages in class_averages.items():
                self.feature_thresholds[j][label] = averages[j]
        
        # Calculate class weights based on class frequency
        class_counts = {}
        for label in y_train:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        total_samples = len(y_train)
        for label, count in class_counts.items():
            self.class_weights[label] = 1 - (count / total_samples)
        
        print("Model training completed")
        
        # Return mock history object for compatibility
        return type('obj', (object,), {
            'history': {
                'accuracy': [0.7, 0.75, 0.8],
                'loss': [0.6, 0.5, 0.4]
            }
        })
    
    def predict(self, X_test, threshold=0.5):
        """
        Make predictions with the simple model.
        
        Args:
            X_test: Test features
            
        Returns:
            list: Predicted labels
        """
        y_pred = []
        
        for features in X_test:
            # Calculate score for each class
            class_scores = {}
            for label in range(self.num_classes):
                score = 0
                feature_count = len(features)
                
                for j in range(feature_count):
                    if j in self.feature_thresholds and label in self.feature_thresholds[j]:
                        threshold = self.feature_thresholds[j][label]
                        # Score based on how close the feature is to the threshold
                        distance = abs(features[j] - threshold)
                        feature_score = 1 - min(1, distance * 2)
                        score += feature_score
                
                # Normalize score
                score = score / feature_count
                
                # Apply class weight
                if label in self.class_weights:
                    score *= (1 + self.class_weights[label])
                
                class_scores[label] = score
            
            # Get label with highest score
            predicted_label = None
            highest_score = -1
            for label, score in class_scores.items():
                if score > highest_score:
                    highest_score = score
                    predicted_label = label
            y_pred.append(predicted_label)
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Make probability predictions with the simple model.
        
        Args:
            X_test: Test features
            
        Returns:
            list: Probability predictions
        """
        y_proba = []
        
        for features in X_test:
            # Calculate score for each class
            class_scores = {}
            for label in range(self.num_classes):
                score = 0
                feature_count = len(features)
                
                for j in range(feature_count):
                    if j in self.feature_thresholds and label in self.feature_thresholds[j]:
                        threshold = self.feature_thresholds[j][label]
                        # Score based on how close the feature is to the threshold
                        distance = abs(features[j] - threshold)
                        feature_score = 1 - min(1, distance * 2)
                        score += feature_score
                
                # Normalize score
                score = score / feature_count
                
                # Apply class weight
                if label in self.class_weights:
                    score *= (1 + self.class_weights[label])
                
                class_scores[label] = score
            
            # Normalize scores to probabilities
            total_score = sum(class_scores.values())
            if total_score > 0:
                proba = [class_scores.get(i, 0) / total_score for i in range(self.num_classes)]
            else:
                proba = [1/self.num_classes] * self.num_classes
            
            y_proba.append(proba)
        
        return y_proba

def evaluate_model_performance(y_true, y_pred):
    """
    Calculate performance metrics for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Performance metrics
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'class_metrics': {}
    }
    
    # Calculate per-class metrics
    for i in range(len(ATTACK_TYPES)):
        # Convert to binary classification for each class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Skip if no samples for this class
        if sum(y_true_binary) == 0:
            continue
        
        # Calculate metrics
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        support = sum(y_true_binary)
        
        results['class_metrics'][i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return results

def evaluate_signature_based(X_train, y_train, X_test, y_test):
    """
    Evaluate signature-based detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Evaluation results
    """
    print_section("Signature-Based Detection Evaluation")
    
    # Create and train the model
    print("Creating and training signature-based model...")
    
    # Convert labels to one-hot encoding for multi-class
    y_train_cat = to_categorical(y_train, num_classes=len(ATTACK_TYPES))
    y_test_cat = to_categorical(y_test, num_classes=len(ATTACK_TYPES))
    
    # Use real model if available, otherwise use simple model
    if HAS_MODEL:
        model = ThreatDetectionModel(
            input_shape=(X_train.shape[1],),
            num_classes=len(ATTACK_TYPES),
            model_config={
                'hidden_layers': [128, 64],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'output_activation': 'softmax',  # For multi-class
                'loss': 'categorical_crossentropy',  # For multi-class
                'optimizer': 'adam',
                'metrics': ['accuracy', 'AUC', 'Precision', 'Recall']
            }
        )
    else:
        model = SimpleModel(num_classes=len(ATTACK_TYPES))
    
    # Train the model
    start_time = time.time()
    history = model.train(
        X_train, y_train_cat,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
    # Calculate metrics
    results = evaluate_model_performance(y_test, y_pred)
    
    # Print results
    print(f"\nOverall accuracy: {results['accuracy']:.4f}")
    print("\nClass metrics:")
    for label, metrics in results['class_metrics'].items():
        print(f"  - {ATTACK_TYPES[label]}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")
    
    return results

def run_simple_anomaly_detection(X_train, y_train, X_test, y_test):
    """
    Run a simple anomaly detection evaluation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Evaluation results
    """
    print_section("Simple Anomaly-Based Detection Evaluation")
    
    # Extract normal samples for training
    normal_X_train = []
    for i, label in enumerate(y_train):
        if label == 1:  # 1 = Normal
            normal_X_train.append(X_train[i])
    
    print(f"Training anomaly detector with {len(normal_X_train)} normal samples")
    
    # Calculate average feature values for normal traffic
    feature_count = len(normal_X_train[0])
    normal_averages = [0] * feature_count
    
    for features in normal_X_train:
        for j in range(feature_count):
            normal_averages[j] += features[j] / len(normal_X_train)
    
    # Calculate standard deviations
    normal_stds = [0] * feature_count
    for features in normal_X_train:
        for j in range(feature_count):
            normal_stds[j] += (features[j] - normal_averages[j]) ** 2 / len(normal_X_train)
    
    normal_stds = [std ** 0.5 for std in normal_stds]
    
    # Set anomaly threshold (number of standard deviations from mean)
    threshold_multiplier = 1.5
    
    # Evaluate on test set
    print("\nEvaluating anomaly detector on test set...")
    anomaly_scores = []
    
    for features in X_test:
        # Calculate z-score for each feature
        z_scores = []
        for j in range(feature_count):
            if normal_stds[j] > 0:
                z_score = abs(features[j] - normal_averages[j]) / normal_stds[j]
            else:
                z_score = abs(features[j] - normal_averages[j])
            z_scores.append(z_score)
        
        # Use average z-score as anomaly score
        anomaly_scores.append(sum(z_scores) / len(z_scores))
    
    # Convert to binary predictions (1 for anomaly, 0 for normal)
    threshold = threshold_multiplier
    y_pred_binary = [1 if score > threshold else 0 for score in anomaly_scores]
    
    # Convert ground truth to binary (1 for attack, 0 for normal)
    y_test_binary = [1 if label > 1 else 0 for label in y_test]
    
    # Calculate metrics
    tp = sum(1 for true, pred in zip(y_test_binary, y_pred_binary) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_test_binary, y_pred_binary) if true == 0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_test_binary, y_pred_binary) if true == 0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_test_binary, y_pred_binary) if true == 1 and pred == 0)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAnomaly detection results (binary classification):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Negatives: {fn}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def compare_models(signature_results, anomaly_results):
    """
    Compare signature-based and anomaly-based models.
    
    Args:
        signature_results: Results from signature-based evaluation
        anomaly_results: Results from anomaly-based evaluation
    """
    print_section("Model Comparison")
    
    print("Signature-based vs. Anomaly-based detection:")
    print(f"  Signature accuracy: {signature_results['accuracy']:.4f}")
    print(f"  Anomaly accuracy: {anomaly_results['accuracy']:.4f}")
    
    print("\nStrengths and weaknesses:")
    
    print("\nSignature-based detection:")
    print("  Strengths:")
    print("  - Can classify specific attack types")
    print("  - Higher precision for known attacks")
    print("  - More detailed threat information")
    print("  Weaknesses:")
    print("  - Cannot detect novel attacks")
    print("  - Requires labeled training data")
    print("  - Updates needed for new attack patterns")
    
    print("\nAnomaly-based detection:")
    print("  Strengths:")
    print("  - Can detect novel/zero-day attacks")
    print("  - Only requires normal traffic for training")
    print("  - Adapts to network environment")
    print("  Weaknesses:")
    print("  - Higher false positive rate")
    print("  - Cannot classify attack types")
    print("  - Less specific threat information")
    
    print("\nRecommended usage:")
    print("  - Use signature-based detection for known threat classification")
    print("  - Use anomaly-based detection for novel threat detection")
    print("  - Combine both approaches for comprehensive security")

def main():
    """
    Main function for basic real-world testing.
    """
    print_section("CyberThreat-ML Basic Real-World Testing")
    print("This script demonstrates CyberThreat-ML capabilities using basic simulated data")
    
    # Create simple dataset
    _, _, _, X_train, X_test, y_train, y_test = create_simple_dataset(
        n_samples=100,
        n_features=10,
        class_ratio=0.7
    )
    
    # Evaluate signature-based detection
    signature_results = evaluate_signature_based(X_train, y_train, X_test, y_test)
    
    # Evaluate simple anomaly detection
    anomaly_results = run_simple_anomaly_detection(X_train, y_train, X_test, y_test)
    
    # Compare models
    compare_models(signature_results, anomaly_results)
    
    print_section("Testing Completed")
    print("Basic real-world testing completed successfully!")

if __name__ == "__main__":
    main()