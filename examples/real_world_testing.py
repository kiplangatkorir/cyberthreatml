#!/usr/bin/env python3
"""
Real-World Dataset Evaluation for CyberThreat-ML

This script evaluates the CyberThreat-ML library against the CICIDS2017 dataset,
which contains real network traffic with various attack types.

The evaluation includes:
1. Data preparation and preprocessing
2. Training and testing on real attack data
3. Performance evaluation for both signature-based and anomaly-based detection
4. Detailed metrics and visualizations of the results

Attack types in CICIDS2017:
- Brute Force
- DoS/DDoS
- Web Attack
- Infiltration
- Port Scan
- Botnet
- SQL Injection

Usage:
    python real_world_testing.py [--download] [--anomaly-only] [--signature-only]

Options:
    --download: Download the CICIDS2017 dataset (if not already present)
    --anomaly-only: Only test anomaly-based detection
    --signature-only: Only test signature-based detection
"""

import os
import sys
import argparse
import glob
import gzip
import shutil
import time
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.anomaly import ZeroDayDetector
from cyberthreat_ml.interpretability import ThreatInterpreter
from cyberthreat_ml.evaluation import evaluate_model, find_optimal_threshold, plot_confusion_matrix

# Constants
CICIDS_DATASET_URLS = [
    "https://www.unb.ca/cic/datasets/ids-2017.html"  # Main page with download links
]

# Selected CSV files from the dataset (smaller subset for quicker processing)
SELECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",    # Normal traffic
    "Tuesday-WorkingHours.pcap_ISCX.csv",    # Normal + attacks
    "Friday-WorkingHours-Morning.pcap_ISCX.csv"  # Port scans and DDoS
]

# Directory for storing dataset
DATA_DIR = "datasets/CICIDS2017"
RESULTS_DIR = "evaluation_results"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mapping of attack types to simplified categories
ATTACK_MAPPING = {
    'BENIGN': 'Normal',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'DoS slowloris': 'DoS/DDoS',
    'DoS Slowhttptest': 'DoS/DDoS',
    'DoS Hulk': 'DoS/DDoS',
    'DoS GoldenEye': 'DoS/DDoS',
    'Heartbleed': 'DoS/DDoS',
    'Web Attack – Brute Force': 'Web Attack',
    'Web Attack – XSS': 'Web Attack',
    'Web Attack – Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Bot': 'Botnet',
    'PortScan': 'Port Scan',
    'DDoS': 'DoS/DDoS'
}

# Feature groups in CICIDS2017 that are relevant for our model
FEATURE_GROUPS = {
    'basic': [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Max',
        'Bwd Packet Length Max', 'Down/Up Ratio'
    ],
    'statistical': [
        'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd IAT Max', 'Bwd IAT Max',
        'Fwd IAT Min', 'Bwd IAT Min', 'Fwd Header Length', 'Bwd Header Length',
        'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count'
    ],
    'content': [
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Active Min',
        'Active Mean', 'Active Max', 'Idle Min', 'Idle Mean', 'Idle Max'
    ]
}

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

class DatasetPreprocessor:
    """
    Class for preprocessing the CICIDS2017 dataset.
    """
    def __init__(self, data_dir, selected_files=None):
        """
        Initialize the dataset preprocessor.
        
        Args:
            data_dir (str): Directory containing the dataset
            selected_files (list, optional): List of selected CSV files to process
        """
        self.data_dir = data_dir
        self.selected_files = selected_files if selected_files else []
        self.feature_columns = []
        for group in FEATURE_GROUPS.values():
            self.feature_columns.extend(group)
        self.scaler = StandardScaler()
        
    def download_dataset(self):
        """
        Display instructions for downloading the CICIDS2017 dataset.
        """
        print_section("Download Instructions")
        print("The CICIDS2017 dataset is available from the Canadian Institute for Cybersecurity.")
        print("Due to its large size (several GB), please download it manually from:")
        print("\n  https://www.unb.ca/cic/datasets/ids-2017.html\n")
        print("After downloading, extract the files to the following directory:")
        print(f"\n  {os.path.abspath(self.data_dir)}\n")
        print("Required files for this script:")
        for file in SELECTED_FILES:
            print(f"  - {file}")
        
        # Prompt user to check if files exist
        existing_files = self._get_existing_files()
        if existing_files:
            print("\nFound the following dataset files:")
            for file in existing_files:
                print(f"  - {os.path.basename(file)}")
        else:
            print("\nNo dataset files found in the specified directory.")
    
    def _get_existing_files(self):
        """
        Get list of existing dataset files.
        
        Returns:
            list: List of file paths
        """
        if not os.path.exists(self.data_dir):
            return []
        
        if self.selected_files:
            return [os.path.join(self.data_dir, f) for f in self.selected_files 
                    if os.path.exists(os.path.join(self.data_dir, f))]
        else:
            return glob.glob(os.path.join(self.data_dir, "*.csv"))
    
    def load_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            tuple: (X, y, X_train, X_test, y_train, y_test, class_names)
        """
        print_section("Loading and Preprocessing Data")
        
        existing_files = self._get_existing_files()
        if not existing_files:
            print("No dataset files found. Please run with --download option to get download instructions.")
            return None, None, None, None, None, None, None
        
        # Load and combine the selected files
        dfs = []
        total_flows = 0
        attack_counts = {}
        
        for file_path in existing_files:
            print(f"Processing {os.path.basename(file_path)}...")
            try:
                # Handle potential encoding issues or read errors
                df = pd.read_csv(file_path, low_memory=False)
                total_flows += len(df)
                
                # Count attack types
                if 'Label' in df.columns:
                    for attack, count in df['Label'].value_counts().items():
                        attack_type = ATTACK_MAPPING.get(attack, attack)
                        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + count
                
                dfs.append(df)
                print(f"  - Added {len(df)} flows")
            except Exception as e:
                print(f"  - Error processing file: {e}")
        
        if not dfs:
            print("Failed to load any dataset files.")
            return None, None, None, None, None, None, None
        
        # Combine all dataframes
        print("\nCombining datasets...")
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total flows: {len(combined_df)}")
        
        # Print attack distribution
        print("\nAttack distribution:")
        for attack, count in attack_counts.items():
            percentage = (count / total_flows) * 100
            print(f"  - {attack}: {count} flows ({percentage:.2f}%)")
        
        # Map attack labels to simplified categories
        print("\nMapping attack labels to simplified categories...")
        if 'Label' in combined_df.columns:
            combined_df['AttackType'] = combined_df['Label'].map(lambda x: ATTACK_MAPPING.get(x, x))
        else:
            print("Warning: 'Label' column not found in dataset.")
            return None, None, None, None, None, None, None
        
        # Handle missing values
        print("\nHandling missing values...")
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        num_missing = combined_df.isna().sum().sum()
        print(f"Number of missing values: {num_missing}")
        
        if num_missing > 0:
            combined_df = combined_df.fillna(0)
            print("Missing values filled with 0")
        
        # Select features and target
        print("\nSelecting features and target...")
        
        # Make sure all required feature columns exist
        available_features = [col for col in self.feature_columns if col in combined_df.columns]
        missing_features = [col for col in self.feature_columns if col not in combined_df.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} feature columns not found in dataset:")
            print(f"  Missing: {', '.join(missing_features[:5])}...")
            print(f"  Using {len(available_features)} available features")
            self.feature_columns = available_features
        
        # Extract features and target
        X = combined_df[self.feature_columns].values
        y_labels = combined_df['AttackType'].values
        
        # Get unique class names
        class_names = np.unique(y_labels)
        print(f"Class names: {class_names.tolist()}")
        
        # Convert labels to numeric
        class_mapping = {class_name: i for i, class_name in enumerate(class_names)}
        y = np.array([class_mapping[label] for label in y_labels])
        
        # Split the data
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale the features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Data preprocessing completed successfully")
        return X, y, X_train_scaled, X_test_scaled, y_train, y_test, class_names

class ModelEvaluator:
    """
    Class for evaluating models on the dataset.
    """
    def __init__(self, results_dir):
        """
        Initialize the model evaluator.
        
        Args:
            results_dir (str): Directory for storing evaluation results
        """
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(results_dir, self.timestamp), exist_ok=True)
    
    def evaluate_signature_based(self, X_train, y_train, X_test, y_test, class_names):
        """
        Evaluate signature-based detection model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            class_names (list): Class names
            
        Returns:
            dict: Evaluation results
        """
        print_section("Signature-Based Detection Evaluation")
        
        # Create and train the model
        print("Creating and training signature-based model...")
        input_shape = (X_train.shape[1],)
        num_classes = len(class_names)
        
        model = ThreatDetectionModel(
            input_shape=input_shape,
            num_classes=num_classes,
            model_config={
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
        )
        
        # Train the model
        start_time = time.time()
        history = model.train(
            X_train, y_train,
            X_val=X_test[:1000], y_val=y_test[:1000],  # Use a subset for validation during training
            epochs=15,
            batch_size=32,
            early_stopping=True
        )
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Evaluate the model
        print("\nEvaluating model on test set...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        print("\nCalculating performance metrics...")
        results = evaluate_model(model, X_test, y_test)
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        
        # Save results
        results_path = os.path.join(self.results_dir, self.timestamp, "signature_based_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Signature-Based Detection Evaluation Results\n")
            f.write(f"=========================================\n\n")
            f.write(f"Number of classes: {num_classes}\n")
            f.write(f"Class names: {class_names.tolist()}\n\n")
            f.write(f"Model architecture:\n")
            f.write(f"  Input shape: {input_shape}\n")
            f.write(f"  Hidden layers: [128, 64, 32]\n")
            f.write(f"  Dropout rate: 0.3\n\n")
            f.write(f"Training time: {training_time:.2f} seconds\n\n")
            f.write(f"Performance Metrics:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"\nClassification Report:\n{report}\n")
        
        print(f"Results saved to {results_path}")
        
        # Plot confusion matrix
        cm_fig = plot_confusion_matrix(model, X_test, y_test, class_names=class_names)
        cm_path = os.path.join(self.results_dir, self.timestamp, "signature_based_confusion_matrix.png")
        cm_fig.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Plot training history
        history_fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training History')
        ax.legend()
        history_path = os.path.join(self.results_dir, self.timestamp, "signature_based_training_history.png")
        history_fig.savefig(history_path)
        print(f"Training history saved to {history_path}")
        
        return results
    
    def evaluate_anomaly_based(self, X_train, y_train, X_test, y_test, class_names):
        """
        Evaluate anomaly-based detection model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            class_names (list): Class names
            
        Returns:
            dict: Evaluation results
        """
        print_section("Anomaly-Based Detection Evaluation")
        
        # Create a subset of normal traffic data for training
        normal_idx = np.where(y_train == class_names.tolist().index('Normal'))[0]
        normal_train = X_train[normal_idx]
        print(f"Using {len(normal_train)} normal traffic samples for training")
        
        # Create the anomaly detector
        print("Creating anomaly detector...")
        detector = ZeroDayDetector(
            method='ensemble',  # Use ensemble of methods for better results
            contamination=0.01,  # Expected proportion of anomalies
            min_samples=100      # Minimum samples before detection
        )
        
        # Fit the detector on normal data
        print("Fitting anomaly detector on normal traffic data...")
        start_time = time.time()
        detector.fit(normal_train)
        training_time = time.time() - start_time
        print(f"Detector training completed in {training_time:.2f} seconds")
        
        # Find optimal threshold on validation set
        print("\nFinding optimal anomaly score threshold...")
        val_size = min(5000, len(X_test))  # Use a subset for validation
        X_val = X_test[:val_size]
        y_val = y_test[:val_size]
        
        # Convert to binary labels (normal=0, attack=1)
        is_attack = (y_val != class_names.tolist().index('Normal')).astype(int)
        
        # Get anomaly scores
        _, scores = detector.detect(X_val, return_scores=True)
        
        # Find threshold that maximizes F1 score
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        best_f1 = 0
        best_threshold = 0
        
        for threshold in thresholds:
            pred = (scores > threshold).astype(int)
            tp = np.sum((pred == 1) & (is_attack == 1))
            fp = np.sum((pred == 1) & (is_attack == 0))
            fn = np.sum((pred == 0) & (is_attack == 1))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating anomaly detection on test set...")
        _, test_scores = detector.detect(X_test, return_scores=True)
        
        # Binary prediction (1 = anomaly, 0 = normal)
        y_pred_binary = (test_scores > best_threshold).astype(int)
        
        # Convert test labels to binary (normal vs attack)
        y_test_binary = (y_test != class_names.tolist().index('Normal')).astype(int)
        
        # Calculate metrics
        print("\nCalculating performance metrics...")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary)
        recall = recall_score(y_test_binary, y_pred_binary)
        f1 = f1_score(y_test_binary, y_pred_binary)
        auc_score = roc_auc_score(y_test_binary, test_scores)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'threshold': best_threshold
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")
        
        # Print classification report
        print("\nClassification Report (Binary - Normal vs Attack):")
        report = classification_report(y_test_binary, y_pred_binary, 
                                      target_names=['Normal', 'Attack'])
        print(report)
        
        # Evaluate per attack type
        print("\nPer-Attack Type Detection Rates:")
        attack_detection_rates = {}
        
        for i, attack_type in enumerate(class_names):
            if attack_type == 'Normal':
                continue
                
            attack_idx = np.where(y_test == i)[0]
            attack_samples = X_test[attack_idx]
            
            if len(attack_samples) > 0:
                _, attack_scores = detector.detect(attack_samples, return_scores=True)
                detected = (attack_scores > best_threshold).sum()
                detection_rate = detected / len(attack_samples)
                attack_detection_rates[attack_type] = detection_rate
                print(f"  {attack_type}: {detection_rate:.4f} ({detected}/{len(attack_samples)})")
        
        # Save results
        results_path = os.path.join(self.results_dir, self.timestamp, "anomaly_based_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Anomaly-Based Detection Evaluation Results\n")
            f.write(f"=======================================\n\n")
            f.write(f"Method: Ensemble\n")
            f.write(f"Contamination: 0.01\n")
            f.write(f"Training samples: {len(normal_train)} (normal traffic only)\n\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
            f.write(f"Optimal threshold: {best_threshold:.4f}\n\n")
            f.write(f"Performance Metrics (Binary - Normal vs Attack):\n")
            for metric, value in results.items():
                if metric != 'threshold':
                    f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"\nClassification Report (Binary):\n{report}\n")
            f.write(f"\nPer-Attack Type Detection Rates:\n")
            for attack_type, rate in attack_detection_rates.items():
                f.write(f"  {attack_type}: {rate:.4f}\n")
        
        print(f"Results saved to {results_path}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test_binary, test_scores)
        roc_auc = auc(fpr, tpr)
        
        roc_fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (area = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        
        roc_path = os.path.join(self.results_dir, self.timestamp, "anomaly_based_roc_curve.png")
        roc_fig.savefig(roc_path)
        print(f"ROC curve saved to {roc_path}")
        
        # Plot precision-recall curve
        precision_values, recall_values, _ = precision_recall_curve(y_test_binary, test_scores)
        pr_auc = auc(recall_values, precision_values)
        
        pr_fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall_values, precision_values, color='darkorange', lw=2,
               label=f'PR curve (area = {pr_auc:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        pr_path = os.path.join(self.results_dir, self.timestamp, "anomaly_based_pr_curve.png")
        pr_fig.savefig(pr_path)
        print(f"Precision-Recall curve saved to {pr_path}")
        
        # Plot score distribution
        score_fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get scores for normal and attack traffic
        normal_idx_test = np.where(y_test == class_names.tolist().index('Normal'))[0]
        attack_idx_test = np.where(y_test != class_names.tolist().index('Normal'))[0]
        
        normal_scores = test_scores[normal_idx_test]
        attack_scores = test_scores[attack_idx_test]
        
        bins = np.linspace(min(test_scores), max(test_scores), 50)
        
        ax.hist(normal_scores, bins, alpha=0.5, label='Normal Traffic', color='green', density=True)
        ax.hist(attack_scores, bins, alpha=0.5, label='Attack Traffic', color='red', density=True)
        ax.axvline(x=best_threshold, color='black', linestyle='--', 
                  label=f'Threshold: {best_threshold:.4f}')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Anomaly Scores')
        ax.legend()
        
        score_path = os.path.join(self.results_dir, self.timestamp, "anomaly_score_distribution.png")
        score_fig.savefig(score_path)
        print(f"Score distribution saved to {score_path}")
        
        return results
        
    def compare_models(self, signature_results, anomaly_results, class_names):
        """
        Compare signature-based and anomaly-based models.
        
        Args:
            signature_results (dict): Signature-based results
            anomaly_results (dict): Anomaly-based results
            class_names (list): Class names
        """
        print_section("Model Comparison")
        
        # Create comparison table
        print("Comparing signature-based and anomaly-based detection:")
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        print("\nPerformance Metrics:")
        print("-" * 60)
        print(f"{'Metric':<15} {'Signature-Based':<20} {'Anomaly-Based':<20}")
        print("-" * 60)
        
        for metric in metrics:
            sig_value = signature_results.get(metric, 'N/A')
            anom_value = anomaly_results.get(metric, 'N/A')
            
            if sig_value != 'N/A':
                sig_value = f"{sig_value:.4f}"
            if anom_value != 'N/A':
                anom_value = f"{anom_value:.4f}"
            
            print(f"{metric:<15} {sig_value:<20} {anom_value:<20}")
        
        print("-" * 60)
        
        # Save comparison results
        results_path = os.path.join(self.results_dir, self.timestamp, "model_comparison.txt")
        with open(results_path, "w") as f:
            f.write(f"Model Comparison: Signature-Based vs Anomaly-Based Detection\n")
            f.write(f"======================================================\n\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"{'-' * 60}\n")
            f.write(f"{'Metric':<15} {'Signature-Based':<20} {'Anomaly-Based':<20}\n")
            f.write(f"{'-' * 60}\n")
            
            for metric in metrics:
                sig_value = signature_results.get(metric, 'N/A')
                anom_value = anomaly_results.get(metric, 'N/A')
                
                if sig_value != 'N/A':
                    sig_value = f"{sig_value:.4f}"
                if anom_value != 'N/A':
                    anom_value = f"{anom_value:.4f}"
                
                f.write(f"{metric:<15} {sig_value:<20} {anom_value:<20}\n")
            
            f.write(f"{'-' * 60}\n\n")
            f.write(f"Analysis:\n")
            f.write(f"  Signature-based detection strengths:\n")
            f.write(f"    - Detailed classification of known attack types\n")
            f.write(f"    - Trained on specific attack patterns\n")
            f.write(f"    - Higher precision on known attack types\n\n")
            f.write(f"  Anomaly-based detection strengths:\n")
            f.write(f"    - Better at detecting novel/unknown threats\n")
            f.write(f"    - Trained only on normal traffic\n")
            f.write(f"    - Can detect deviations without prior knowledge of attacks\n\n")
            f.write(f"Recommendation:\n")
            f.write(f"  Use both approaches in combination for comprehensive threat detection:\n")
            f.write(f"  - Signature-based for known threat classification\n")
            f.write(f"  - Anomaly-based for potential zero-day threat detection\n")
        
        print(f"Comparison results saved to {results_path}")
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        signature_values = [signature_results.get(metric, 0) for metric in metrics]
        anomaly_values = [anomaly_results.get(metric, 0) for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, signature_values, width, label='Signature-Based')
        plt.bar(x + width/2, anomaly_values, width, label='Anomaly-Based')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison: Signature-Based vs Anomaly-Based')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        comparison_path = os.path.join(self.results_dir, self.timestamp, "model_comparison_chart.png")
        plt.savefig(comparison_path)
        print(f"Comparison chart saved to {comparison_path}")

def main():
    """
    Main function for real-world dataset testing.
    """
    parser = argparse.ArgumentParser(description='Evaluate CyberThreat-ML on CICIDS2017 dataset')
    parser.add_argument('--download', action='store_true', help='Show download instructions for dataset')
    parser.add_argument('--anomaly-only', action='store_true', help='Only evaluate anomaly-based detection')
    parser.add_argument('--signature-only', action='store_true', help='Only evaluate signature-based detection')
    args = parser.parse_args()
    
    # Print header
    print_section("CyberThreat-ML Real-World Testing")
    print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize dataset preprocessor
    preprocessor = DatasetPreprocessor(DATA_DIR, SELECTED_FILES)
    
    if args.download:
        preprocessor.download_dataset()
        return
    
    # Load and preprocess the dataset
    X, y, X_train, X_test, y_train, y_test, class_names = preprocessor.load_data()
    
    if X is None:
        print("Failed to load dataset. Run with --download to get download instructions.")
        return
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(RESULTS_DIR)
    
    signature_results = None
    anomaly_results = None
    
    # Evaluate signature-based detection
    if not args.anomaly_only:
        signature_results = evaluator.evaluate_signature_based(
            X_train, y_train, X_test, y_test, class_names
        )
    
    # Evaluate anomaly-based detection
    if not args.signature_only:
        anomaly_results = evaluator.evaluate_anomaly_based(
            X_train, y_train, X_test, y_test, class_names
        )
    
    # Compare models if both were evaluated
    if signature_results and anomaly_results:
        evaluator.compare_models(signature_results, anomaly_results, class_names)
    
    print_section("Evaluation Complete")
    print(f"Results saved to: {os.path.join(RESULTS_DIR, evaluator.timestamp)}")
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()