#!/usr/bin/env python3
"""
Basic Zero-Day Threat Detection Example for CyberThreat-ML

This script demonstrates a simplified version of zero-day threat detection
without requiring external libraries.
"""

import os
import sys
import random
import time
from datetime import datetime

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def create_datasets():
    """
    Create synthetic datasets for normal traffic, known attacks, and zero-day attacks.
    
    Returns:
        tuple: (normal_data, known_attacks, zero_day_attacks)
    """
    print_section("Creating Synthetic Datasets")
    
    # Create normal network traffic
    print("Generating normal network traffic dataset...")
    normal_data = []
    for _ in range(100):
        # Features represent various network characteristics
        # Lower values for certain features indicate normal behavior
        features = [
            random.random() * 0.5,              # Low packet size
            random.random() * 0.3,              # Low destination port entropy
            random.random() * 0.4,              # Low source port entropy
            random.random() * 0.3,              # Low packet rate
            random.random() * 0.2,              # Low data transfer rate
            random.random() * 0.3 + 0.1,        # Medium connection duration
            random.random() * 0.2,              # Low payload entropy
            random.random() * 0.3,              # Low protocol anomaly score
            random.random() * 0.2,              # Low header anomaly score
            random.random() * 0.3               # Low behavioral anomaly score
        ]
        normal_data.append(features)
    
    # Create known attack patterns
    print("Generating known attack patterns dataset...")
    known_attacks = []
    known_attack_labels = []
    
    for _ in range(60):
        attack_type = random.randint(1, len(ATTACK_TYPES) - 1)  # 1-5 (skip 0 which is Normal)
        
        # Different attack types have different characteristic feature patterns
        if attack_type == 1:  # Brute Force
            features = [
                random.random() * 0.3 + 0.1,              # Low packet size
                random.random() * 0.2,                    # Low destination port entropy
                random.random() * 0.3,                    # Low source port entropy
                random.random() * 0.6 + 0.3,              # High packet rate
                random.random() * 0.3,                    # Low data transfer rate
                random.random() * 0.2 + 0.7,              # High connection duration
                random.random() * 0.3,                    # Low payload entropy
                random.random() * 0.3,                    # Low protocol anomaly score
                random.random() * 0.3,                    # Low header anomaly score
                random.random() * 0.6 + 0.3               # High behavioral anomaly score
            ]
        elif attack_type == 2:  # DoS/DDoS
            features = [
                random.random() * 0.5 + 0.5,              # High packet size
                random.random() * 0.2,                    # Low destination port entropy
                random.random() * 0.2 + 0.4,              # Medium source port entropy
                random.random() * 0.4 + 0.6,              # Very high packet rate
                random.random() * 0.5 + 0.4,              # High data transfer rate
                random.random() * 0.3,                    # Low connection duration
                random.random() * 0.3,                    # Low payload entropy
                random.random() * 0.3 + 0.2,              # Medium protocol anomaly score
                random.random() * 0.3,                    # Low header anomaly score
                random.random() * 0.6 + 0.4               # High behavioral anomaly score
            ]
        elif attack_type == 3:  # Web Attack
            features = [
                random.random() * 0.4 + 0.2,              # Medium packet size
                random.random() * 0.2,                    # Low destination port entropy
                random.random() * 0.3,                    # Low source port entropy
                random.random() * 0.3 + 0.2,              # Medium packet rate
                random.random() * 0.4 + 0.1,              # Medium data transfer rate
                random.random() * 0.4 + 0.2,              # Medium connection duration
                random.random() * 0.5 + 0.4,              # High payload entropy
                random.random() * 0.5 + 0.4,              # High protocol anomaly score
                random.random() * 0.4 + 0.3,              # Medium-high header anomaly score
                random.random() * 0.5 + 0.3               # High behavioral anomaly score
            ]
        elif attack_type == 4:  # Port Scan
            features = [
                random.random() * 0.2,                    # Very low packet size
                random.random() * 0.6 + 0.4,              # High destination port entropy
                random.random() * 0.2 + 0.2,              # Medium-low source port entropy
                random.random() * 0.5 + 0.5,              # High packet rate
                random.random() * 0.2,                    # Low data transfer rate
                random.random() * 0.2,                    # Low connection duration
                random.random() * 0.2,                    # Low payload entropy
                random.random() * 0.4 + 0.2,              # Medium protocol anomaly score
                random.random() * 0.3 + 0.2,              # Medium header anomaly score
                random.random() * 0.5 + 0.4               # High behavioral anomaly score
            ]
        else:  # Data Exfiltration
            features = [
                random.random() * 0.5 + 0.3,              # Medium-high packet size
                random.random() * 0.3,                    # Low destination port entropy
                random.random() * 0.2,                    # Low source port entropy
                random.random() * 0.3 + 0.1,              # Medium-low packet rate
                random.random() * 0.6 + 0.4,              # High data transfer rate
                random.random() * 0.4 + 0.3,              # Medium-high connection duration
                random.random() * 0.5 + 0.5,              # Very high payload entropy
                random.random() * 0.3 + 0.1,              # Medium protocol anomaly score
                random.random() * 0.3,                    # Low header anomaly score
                random.random() * 0.6 + 0.3               # High behavioral anomaly score
            ]
        
        known_attacks.append(features)
        known_attack_labels.append(attack_type)
    
    # Create zero-day attack patterns (unique combinations not in training data)
    print("Generating zero-day attack patterns dataset...")
    zero_day_attacks = []
    for _ in range(20):
        # Zero-day attacks have characteristics that don't match known patterns
        # but still have anomalous values in certain features
        features = [
            random.random() * 0.7 + 0.3,                # High packet size
            random.random() * 0.7 + 0.3,                # High destination port entropy
            random.random() * 0.7 + 0.3,                # High source port entropy
            random.random() * 0.4,                      # Low packet rate
            random.random() * 0.8 + 0.2,                # High data transfer rate
            random.random() * 0.2 + 0.7,                # High connection duration
            random.random() * 0.8 + 0.2,                # High payload entropy
            random.random() * 0.7 + 0.3,                # High protocol anomaly score
            random.random() * 0.7 + 0.3,                # High header anomaly score
            random.random() * 0.7 + 0.3                 # High behavioral anomaly score
        ]
        zero_day_attacks.append(features)
    
    # Print dataset sizes
    print(f"Normal traffic: {len(normal_data)} samples")
    print(f"Known attacks: {len(known_attacks)} samples")
    print(f"Zero-day attacks: {len(zero_day_attacks)} samples")
    
    # Count attack types
    attack_counts = {}
    for label in known_attack_labels:
        attack_type = ATTACK_TYPES[label]
        if attack_type in attack_counts:
            attack_counts[attack_type] += 1
        else:
            attack_counts[attack_type] = 1
    
    print("\nKnown attack distribution:")
    for attack_type, count in attack_counts.items():
        print(f"  - {attack_type}: {count} samples")
    
    return normal_data, known_attacks, known_attack_labels, zero_day_attacks

class SignatureDetector:
    """Simple signature-based detector for known attack patterns."""
    
    def __init__(self):
        self.feature_thresholds = {}
        self.class_weights = {}
    
    def train(self, normal_data, attack_data, attack_labels):
        """
        Train the detector on normal and attack data.
        
        Args:
            normal_data: Normal traffic data
            attack_data: Known attack data
            attack_labels: Labels for known attacks
        """
        print("\nTraining signature-based detector...")
        
        # Combine data and labels
        all_data = normal_data + attack_data
        all_labels = [0] * len(normal_data) + attack_labels
        
        # Calculate average feature values for each class
        class_features = {}
        for i, features in enumerate(all_data):
            label = all_labels[i]
            
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
        for j in range(len(all_data[0])):
            self.feature_thresholds[j] = {}
            
            for label, averages in class_averages.items():
                self.feature_thresholds[j][label] = averages[j]
        
        # Calculate class weights based on class frequency
        class_counts = {}
        for label in all_labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        total_samples = len(all_labels)
        for label, count in class_counts.items():
            self.class_weights[label] = 1 - (count / total_samples)
    
    def detect(self, data):
        """
        Detect attacks in the given data.
        
        Args:
            data: Data to analyze
            
        Returns:
            list: Predictions for each sample
        """
        predictions = []
        confidences = []
        
        for features in data:
            # Calculate score for each class
            class_scores = {}
            for label in range(len(ATTACK_TYPES)):
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
            predicted_label = 0
            max_score = -1
            for label, score in class_scores.items():
                if score > max_score:
                    max_score = score
                    predicted_label = label
            
            predictions.append(predicted_label)
            confidences.append(max_score)
        
        return predictions, confidences

class AnomalyDetector:
    """Simple anomaly-based detector for zero-day threats."""
    
    def __init__(self, threshold_multiplier=2.0):
        self.feature_means = []
        self.feature_stds = []
        self.threshold_multiplier = threshold_multiplier
    
    def train(self, normal_data):
        """
        Train the detector on normal data.
        
        Args:
            normal_data: Normal traffic data
        """
        print("\nTraining anomaly-based detector...")
        
        # Calculate mean and standard deviation for each feature
        feature_count = len(normal_data[0])
        self.feature_means = [0] * feature_count
        
        # Calculate means
        for features in normal_data:
            for j in range(feature_count):
                self.feature_means[j] += features[j] / len(normal_data)
        
        # Calculate standard deviations
        self.feature_stds = [0] * feature_count
        for features in normal_data:
            for j in range(feature_count):
                self.feature_stds[j] += (features[j] - self.feature_means[j]) ** 2 / len(normal_data)
        
        self.feature_stds = [std ** 0.5 for std in self.feature_stds]
    
    def detect(self, data):
        """
        Detect anomalies in the given data.
        
        Args:
            data: Data to analyze
            
        Returns:
            list: Anomaly scores for each sample
        """
        anomaly_scores = []
        predictions = []
        
        for features in data:
            # Calculate z-score for each feature
            z_scores = []
            for j in range(len(features)):
                if self.feature_stds[j] > 0:
                    z_score = abs(features[j] - self.feature_means[j]) / self.feature_stds[j]
                else:
                    z_score = abs(features[j] - self.feature_means[j])
                z_scores.append(z_score)
            
            # Use maximum z-score as anomaly score
            score = max(z_scores)
            anomaly_scores.append(score)
            
            # Classify as anomaly if score exceeds threshold
            prediction = 1 if score > self.threshold_multiplier else 0
            predictions.append(prediction)
        
        return predictions, anomaly_scores

class HybridDetector:
    """Hybrid detector that combines signature and anomaly detection."""
    
    def __init__(self, signature_detector, anomaly_detector):
        self.signature_detector = signature_detector
        self.anomaly_detector = anomaly_detector
    
    def detect(self, data):
        """
        Detect threats using both signature and anomaly detection.
        
        Args:
            data: Data to analyze
            
        Returns:
            tuple: (predictions, signature_preds, anomaly_preds, confidences)
        """
        # Run signature detection
        signature_preds, sig_confidences = self.signature_detector.detect(data)
        
        # Run anomaly detection
        anomaly_preds, anomaly_scores = self.anomaly_detector.detect(data)
        
        # Combine predictions
        predictions = []
        confidences = []
        
        for i in range(len(data)):
            sig_pred = signature_preds[i]
            sig_conf = sig_confidences[i]
            anomaly_pred = anomaly_preds[i]
            anomaly_score = anomaly_scores[i]
            
            # If both detectors agree it's normal, it's normal
            if sig_pred == 0 and anomaly_pred == 0:
                predictions.append(0)
                confidences.append(1 - anomaly_score / 4)  # High confidence it's normal
            
            # If signature detector identifies a specific attack
            elif sig_pred > 0:
                predictions.append(sig_pred)
                confidences.append(sig_conf)
            
            # If anomaly detector finds an anomaly but signature doesn't identify it
            elif anomaly_pred == 1:
                predictions.append(-1)  # -1 indicates unknown attack type
                confidences.append(anomaly_score / 4)  # Convert anomaly score to confidence
            
            # Fallback
            else:
                predictions.append(0)
                confidences.append(0.5)
        
        return predictions, signature_preds, anomaly_preds, confidences

def evaluate_detectors(normal_data, known_attacks, known_attack_labels, zero_day_attacks):
    """
    Evaluate signature, anomaly, and hybrid detection approaches.
    
    Args:
        normal_data: Normal traffic data
        known_attacks: Known attack data
        known_attack_labels: Labels for known attacks
        zero_day_attacks: Zero-day attack data
    """
    print_section("Evaluating Detection Approaches")
    
    # Split data for training and testing
    train_ratio = 0.7
    normal_train_count = int(len(normal_data) * train_ratio)
    attack_train_count = int(len(known_attacks) * train_ratio)
    
    # Training data
    normal_train = normal_data[:normal_train_count]
    attack_train = known_attacks[:attack_train_count]
    attack_label_train = known_attack_labels[:attack_train_count]
    
    # Testing data
    normal_test = normal_data[normal_train_count:]
    attack_test = known_attacks[attack_train_count:]
    attack_label_test = known_attack_labels[attack_train_count:]
    
    # Create test set with normal, known, and zero-day attacks
    test_data = normal_test + attack_test + zero_day_attacks
    test_labels = [0] * len(normal_test) + attack_label_test + [-1] * len(zero_day_attacks)
    
    # Train signature detector
    signature_detector = SignatureDetector()
    signature_detector.train(normal_train, attack_train, attack_label_train)
    
    # Train anomaly detector
    anomaly_detector = AnomalyDetector(threshold_multiplier=2.0)
    anomaly_detector.train(normal_train)
    
    # Create hybrid detector
    hybrid_detector = HybridDetector(signature_detector, anomaly_detector)
    
    # Test detectors
    print("\nTesting detectors on normal, known, and zero-day attacks...")
    hybrid_preds, signature_preds, anomaly_preds, confidences = hybrid_detector.detect(test_data)
    
    # Create arrays indicating which samples are zero-day attacks
    is_zero_day = [label == -1 for label in test_labels]
    is_known_attack = [label > 0 for label in test_labels]
    is_normal = [label == 0 for label in test_labels]
    
    # Evaluate signature detector on known attacks
    signature_accuracy = sum(1 for i, pred in enumerate(signature_preds) if pred == test_labels[i]) / len(test_data)
    signature_zero_day_detection = sum(1 for i, pred in enumerate(signature_preds) if is_zero_day[i] and pred > 0) / sum(is_zero_day)
    
    # Evaluate anomaly detector
    anomaly_accuracy = sum(1 for i, pred in enumerate(anomaly_preds) if (pred == 1 and test_labels[i] != 0) or (pred == 0 and test_labels[i] == 0)) / len(test_data)
    anomaly_zero_day_detection = sum(1 for i, pred in enumerate(anomaly_preds) if is_zero_day[i] and pred == 1) / sum(is_zero_day)
    
    # Evaluate hybrid detector
    hybrid_accuracy = sum(1 for i, pred in enumerate(hybrid_preds) if 
        (pred > 0 and test_labels[i] > 0) or  # Correctly identified known attack
        (pred == -1 and is_zero_day[i]) or    # Correctly identified zero-day attack as unknown
        (pred == 0 and test_labels[i] == 0)   # Correctly identified normal traffic
    ) / len(test_data)
    hybrid_zero_day_detection = sum(1 for i, pred in enumerate(hybrid_preds) if is_zero_day[i] and pred == -1) / sum(is_zero_day)
    
    # Calculate additional metrics
    signature_false_positives = sum(1 for i, pred in enumerate(signature_preds) if is_normal[i] and pred > 0) / sum(is_normal)
    anomaly_false_positives = sum(1 for i, pred in enumerate(anomaly_preds) if is_normal[i] and pred == 1) / sum(is_normal)
    hybrid_false_positives = sum(1 for i, pred in enumerate(hybrid_preds) if is_normal[i] and pred != 0) / sum(is_normal)
    
    # Print results
    print("\nEvaluation Results:")
    print("\nSignature-based detection:")
    print(f"  Overall accuracy: {signature_accuracy:.4f}")
    print(f"  Zero-day detection rate: {signature_zero_day_detection:.4f}")
    print(f"  False positive rate: {signature_false_positives:.4f}")
    
    print("\nAnomaly-based detection:")
    print(f"  Overall accuracy: {anomaly_accuracy:.4f}")
    print(f"  Zero-day detection rate: {anomaly_zero_day_detection:.4f}")
    print(f"  False positive rate: {anomaly_false_positives:.4f}")
    
    print("\nHybrid detection:")
    print(f"  Overall accuracy: {hybrid_accuracy:.4f}")
    print(f"  Zero-day detection rate: {hybrid_zero_day_detection:.4f}")
    print(f"  False positive rate: {hybrid_false_positives:.4f}")
    
    # Print detection results for each approach
    print_section("Detection Results Comparison")
    
    # Count correct detections for each category
    sig_correct_normal = sum(1 for i, pred in enumerate(signature_preds) if is_normal[i] and pred == 0)
    sig_correct_known = sum(1 for i, pred in enumerate(signature_preds) if is_known_attack[i] and pred == test_labels[i])
    sig_correct_zero_day = sum(1 for i, pred in enumerate(signature_preds) if is_zero_day[i] and pred > 0)
    
    anom_correct_normal = sum(1 for i, pred in enumerate(anomaly_preds) if is_normal[i] and pred == 0)
    anom_correct_known = sum(1 for i, pred in enumerate(anomaly_preds) if is_known_attack[i] and pred == 1)
    anom_correct_zero_day = sum(1 for i, pred in enumerate(anomaly_preds) if is_zero_day[i] and pred == 1)
    
    hybrid_correct_normal = sum(1 for i, pred in enumerate(hybrid_preds) if is_normal[i] and pred == 0)
    hybrid_correct_known = sum(1 for i, pred in enumerate(hybrid_preds) if is_known_attack[i] and pred > 0)
    hybrid_correct_zero_day = sum(1 for i, pred in enumerate(hybrid_preds) if is_zero_day[i] and pred == -1)
    
    normal_count = sum(is_normal)
    known_count = sum(is_known_attack)
    zero_day_count = sum(is_zero_day)
    
    print(f"{'Traffic Type':<15} {'Count':<8} {'Signature':<15} {'Anomaly':<15} {'Hybrid':<15}")
    print(f"{'-'*15:<15} {'-'*8:<8} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}")
    print(f"{'Normal':<15} {normal_count:<8} {sig_correct_normal}/{normal_count} ({sig_correct_normal/normal_count:.2f}) {anom_correct_normal}/{normal_count} ({anom_correct_normal/normal_count:.2f}) {hybrid_correct_normal}/{normal_count} ({hybrid_correct_normal/normal_count:.2f})")
    print(f"{'Known Attacks':<15} {known_count:<8} {sig_correct_known}/{known_count} ({sig_correct_known/known_count:.2f}) {anom_correct_known}/{known_count} ({anom_correct_known/known_count:.2f}) {hybrid_correct_known}/{known_count} ({hybrid_correct_known/known_count:.2f})")
    print(f"{'Zero-Day':<15} {zero_day_count:<8} {sig_correct_zero_day}/{zero_day_count} ({sig_correct_zero_day/zero_day_count:.2f}) {anom_correct_zero_day}/{zero_day_count} ({anom_correct_zero_day/zero_day_count:.2f}) {hybrid_correct_zero_day}/{zero_day_count} ({hybrid_correct_zero_day/zero_day_count:.2f})")

def analyze_zero_day_attacks(zero_day_attacks, anomaly_detector):
    """
    Analyze zero-day attacks and explain why they were detected.
    
    Args:
        zero_day_attacks: Zero-day attack data
        anomaly_detector: Trained anomaly detector
    """
    print_section("Zero-Day Attack Analysis")
    
    # Detect zero-day attacks
    _, anomaly_scores = anomaly_detector.detect(zero_day_attacks)
    
    # Select a few attacks to analyze
    num_to_analyze = min(3, len(zero_day_attacks))
    
    for i in range(num_to_analyze):
        features = zero_day_attacks[i]
        score = anomaly_scores[i]
        
        print(f"\nZero-Day Attack #{i+1}")
        print(f"Anomaly Score: {score:.4f}")
        
        # Calculate z-scores for each feature
        z_scores = []
        for j in range(len(features)):
            if anomaly_detector.feature_stds[j] > 0:
                z_score = abs(features[j] - anomaly_detector.feature_means[j]) / anomaly_detector.feature_stds[j]
            else:
                z_score = abs(features[j] - anomaly_detector.feature_means[j])
            z_scores.append(z_score)
        
        # Find the top 3 most anomalous features
        feature_names = [
            "Packet Size",
            "Destination Port Entropy",
            "Source Port Entropy",
            "Packet Rate",
            "Data Transfer Rate",
            "Connection Duration",
            "Payload Entropy",
            "Protocol Anomaly Score",
            "Header Anomaly Score",
            "Behavioral Anomaly Score"
        ]
        
        # Sort features by z-score
        sorted_features = sorted(zip(feature_names, features, z_scores), key=lambda x: x[2], reverse=True)
        
        print("Most anomalous features:")
        for name, value, z_score in sorted_features[:3]:
            print(f"  - {name}: {value:.4f} (z-score: {z_score:.4f})")
        
        # Generate explanation
        print("Explanation:")
        if sorted_features[0][2] > 4.0:
            print(f"  This traffic is highly anomalous due to extreme values in {sorted_features[0][0]}.")
        elif sum(z > 2.0 for _, _, z in sorted_features[:3]) >= 2:
            print(f"  Multiple unusual features detected, indicating a potential complex attack.")
        else:
            print(f"  Subtle deviations across multiple features suggest a sophisticated attack attempt.")
        
        print("Recommendation:")
        if "Payload" in sorted_features[0][0] or "Protocol" in sorted_features[0][0]:
            print("  Inspect packet payloads and protocol conformance for malicious content.")
        elif "Port" in sorted_features[0][0]:
            print("  Investigate unusual port usage and consider additional firewall rules.")
        elif "Rate" in sorted_features[0][0]:
            print("  Implement rate limiting and monitor for resource exhaustion attacks.")
        else:
            print("  Deploy additional monitoring and consider behavioral analysis.")

def main():
    """Main function for zero-day detection demo."""
    print_section("CyberThreat-ML Zero-Day Detection Demo")
    print("This script demonstrates how CyberThreat-ML can detect zero-day threats\n"
          "by combining signature-based and anomaly-based detection techniques.")
    
    # Create datasets
    normal_data, known_attacks, known_attack_labels, zero_day_attacks = create_datasets()
    
    # Evaluate detectors
    evaluate_detectors(normal_data, known_attacks, known_attack_labels, zero_day_attacks)
    
    # Train anomaly detector for analysis
    anomaly_detector = AnomalyDetector()
    anomaly_detector.train(normal_data)
    
    # Analyze zero-day attacks
    analyze_zero_day_attacks(zero_day_attacks, anomaly_detector)
    
    print_section("Conclusion")
    print("The hybrid approach achieves the best overall results by:")
    print("1. Using signature detection to identify known attack patterns")
    print("2. Using anomaly detection to identify deviations from normal behavior")
    print("3. Combining both techniques to minimize false positives while maximizing detection capabilities")
    print("\nThis approach is effective for detecting both known and zero-day threats,")
    print("making it suitable for comprehensive cybersecurity protection.")

if __name__ == "__main__":
    main()