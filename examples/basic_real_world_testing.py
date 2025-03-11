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
    'Normal',
    'Brute Force',
    'DoS/DDoS',
    'Web Attack',
    'Port Scan',
    'Data Exfiltration'
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
    
    # Create feature matrix with random values
    X = []
    for _ in range(n_samples):
        # Generate random features between 0 and 1
        features = [random.random() for _ in range(n_features)]
        X.append(features)
    
    # Create labels (0 = Normal, 1-5 = Attack types)
    y = []
    normal_samples = int(n_samples * class_ratio)
    attack_samples = n_samples - normal_samples
    
    # Create normal samples
    for _ in range(normal_samples):
        y.append(0)  # 0 = Normal
    
    # Create attack samples
    for _ in range(attack_samples):
        # Random attack type (1-5)
        attack_type = random.randint(1, len(ATTACK_TYPES) - 1)
        y.append(attack_type)
    
    # Shuffle the dataset
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    
    # Count samples per class
    class_counts = {}
    for label in y:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    # Print class distribution
    print("\nClass distribution:")
    for label, count in class_counts.items():
        percentage = (count / n_samples) * 100
        print(f"  - {ATTACK_TYPES[label]}: {count} samples ({percentage:.2f}%)")
    
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
    # Convert tuples to lists if needed
    if isinstance(y_true, tuple):
        y_true = list(y_true)
    if isinstance(y_pred, tuple):
        y_pred = list(y_pred)
        
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    
    # Calculate per-class metrics
    class_metrics = {}
    for label in set(y_true + y_pred):
        # True positives
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        
        # False positives
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        
        # False negatives
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Support (number of true instances)
        support = sum(1 for true in y_true if true == label)
        
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return {
        'accuracy': accuracy,
        'class_metrics': class_metrics
    }

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
    
    # Use real model if available, otherwise use simple model
    if HAS_MODEL:
        model = ThreatDetectionModel(
            input_shape=(len(X_train[0]),),
            num_classes=len(ATTACK_TYPES),
            model_config={
                'hidden_layers': [64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
        )
    else:
        model = SimpleModel(num_classes=len(ATTACK_TYPES))
    
    # Train the model
    start_time = time.time()
    history = model.train(X_train, y_train, epochs=5, batch_size=32)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    
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
        if label == 0:  # 0 = Normal
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
    y_test_binary = [1 if label > 0 else 0 for label in y_test]
    
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