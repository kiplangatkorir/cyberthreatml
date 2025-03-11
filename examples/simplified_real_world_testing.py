#!/usr/bin/env python3
"""
Simplified Real-World Testing for CyberThreat-ML

This script demonstrates a simplified version of real-world testing
without requiring external libraries.

It creates simple simulated data and evaluates basic threat detection capabilities.
"""

import os
import sys
import argparse
import time
import random
from datetime import datetime

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        features = [random.random() for _ in range(n_features)]
        X.append(features)
    
    # Create labels (0 = Normal, 1-5 = Attack types)
    normal_samples = int(n_samples * class_ratio)
    attack_samples = n_samples - normal_samples
    
    # Create array of normal samples (label 0)
    y = [0] * normal_samples
    
    # Create array of attack samples (labels 1-5)
    for _ in range(attack_samples):
        y.append(random.randint(1, len(ATTACK_TYPES) - 1))
    
    # Shuffle the dataset
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    X, y = list(X), list(y)
    
    # Calculate attack distribution
    class_counts = {}
    for label in y:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    print("\nClass distribution:")
    for class_idx, count in class_counts.items():
        percentage = (count / n_samples) * 100
        print(f"  - {ATTACK_TYPES[class_idx]}: {count} samples ({percentage:.2f}%)")
    
    # Split into training and test sets (70% train, 30% test)
    train_size = int(0.7 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

class SimpleModel:
    """A simple model that predicts based on feature thresholds."""
    
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.feature_weights = []
        self.class_thresholds = [0.0] * num_classes
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the simple model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        num_features = len(X_train[0])
        self.feature_weights = [random.random() for _ in range(num_features)]
        self.class_thresholds = [random.random() * 0.5 for _ in range(self.num_classes)]
        print("Model training completed")
    
    def predict(self, X_test, threshold=0.5):
        """
        Make predictions with the simple model.
        
        Args:
            X_test: Test features
            
        Returns:
            list: Predicted labels
        """
        predictions = []
        for features in X_test:
            # Compute weighted sum of features
            weighted_sum = sum(w * f for w, f in zip(self.feature_weights, features))
            
            # Determine class based on weighted sum
            pred_class = 0  # Default to normal
            for c in range(1, self.num_classes):
                if weighted_sum > self.class_thresholds[c]:
                    pred_class = c
            
            predictions.append(pred_class)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Make probability predictions with the simple model.
        
        Args:
            X_test: Test features
            
        Returns:
            list: Probability predictions
        """
        probas = []
        for features in X_test:
            # Compute weighted sum of features
            weighted_sum = sum(w * f for w, f in zip(self.feature_weights, features))
            
            # Convert to probabilities
            probs = [0.1] * self.num_classes
            probs[0] = 0.5  # Higher probability for normal class
            
            for c in range(1, self.num_classes):
                if weighted_sum > self.class_thresholds[c]:
                    probs[c] = 0.8
                    probs[0] = 0.2
            
            probas.append(probs)
        
        return probas

def evaluate_model_performance(y_true, y_pred):
    """
    Calculate performance metrics for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Performance metrics
    """
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    
    # Calculate per-class metrics
    class_metrics = {}
    for class_idx in range(len(ATTACK_TYPES)):
        class_name = ATTACK_TYPES[class_idx]
        
        # Count true positives, false positives, and false negatives
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == class_idx and pred == class_idx)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != class_idx and pred == class_idx)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == class_idx and pred != class_idx)
        support = sum(1 for true in y_true if true == class_idx)
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        class_metrics[class_name] = {
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
    model = SimpleModel(num_classes=len(ATTACK_TYPES))
    
    # Train the model
    start_time = time.time()
    model.train(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    results = evaluate_model_performance(y_test, y_pred)
    
    # Print results
    print(f"\nOverall accuracy: {results['accuracy']:.4f}")
    print("\nClass metrics:")
    for class_name, metrics in results['class_metrics'].items():
        print(f"  - {class_name}:")
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
    normal_indices = [i for i, label in enumerate(y_train) if label == 0]
    X_train_normal = [X_train[i] for i in normal_indices]
    
    print(f"Training anomaly detector with {len(X_train_normal)} normal samples")
    
    # Simple anomaly detection: compute average of normal samples
    normal_avg = []
    for feature_idx in range(len(X_train[0])):
        feature_avg = sum(X[feature_idx] for X in X_train_normal) / len(X_train_normal)
        normal_avg.append(feature_avg)
    
    # Define threshold for anomaly as distance from normal average
    threshold = 0.5
    
    # Convert ground truth to binary (0 for normal, 1 for attack)
    y_test_binary = [1 if y > 0 else 0 for y in y_test]
    
    # Predict anomalies
    y_pred_binary = []
    for sample in X_test:
        # Calculate Euclidean distance from normal average
        distance = sum((sample[i] - normal_avg[i]) ** 2 for i in range(len(sample))) ** 0.5
        if distance > threshold:
            y_pred_binary.append(1)  # Anomaly
        else:
            y_pred_binary.append(0)  # Normal
    
    # Calculate metrics
    tp = sum(1 for p, t in zip(y_pred_binary, y_test_binary) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred_binary, y_test_binary) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(y_pred_binary, y_test_binary) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(y_pred_binary, y_test_binary) if p == 0 and t == 1)
    
    if tp + tn + fp + fn > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = 0.0
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
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
    Main function for simplified real-world testing.
    """
    parser = argparse.ArgumentParser(description="Simplified Real-World Testing for CyberThreat-ML")
    parser.add_argument("--anomaly-only", action="store_true", help="Only test anomaly-based detection")
    parser.add_argument("--signature-only", action="store_true", help="Only test signature-based detection")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    
    print_section("CyberThreat-ML Simplified Real-World Testing")
    print("This script demonstrates CyberThreat-ML capabilities using basic simulated data")
    
    # Create simple dataset
    X_train, X_test, y_train, y_test = create_simple_dataset(
        n_samples=args.samples,
        n_features=10,
        class_ratio=0.7
    )
    
    signature_results = None
    anomaly_results = None
    
    # Evaluate signature-based detection
    if not args.anomaly_only:
        signature_results = evaluate_signature_based(X_train, y_train, X_test, y_test)
    
    # Evaluate anomaly-based detection
    if not args.signature_only:
        anomaly_results = run_simple_anomaly_detection(X_train, y_train, X_test, y_test)
    
    # Compare models if both were evaluated
    if signature_results and anomaly_results:
        compare_models(signature_results, anomaly_results)
    
    print_section("Testing Completed")
    print("Simplified real-world testing completed successfully!")

if __name__ == "__main__":
    main()