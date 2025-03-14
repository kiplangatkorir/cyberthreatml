"""
Example demonstrating the zero-day threat detection capabilities of CyberThreat-ML.

This example shows how to:
1. Use anomaly detection to detect zero-day threats
2. Analyze and explain detected anomalies
3. Combine signature-based and anomaly-based detection
4. Train adaptive detection models using recent traffic
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import random
import tensorflow as tf

# Import from cyberthreat_ml library
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.anomaly import ZeroDayDetector, RealTimeZeroDayDetector, get_anomaly_description, recommend_action
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.logger import CyberThreatLogger

# Configure logging
logger = CyberThreatLogger("zero_day_detection", logging.INFO).get_logger()

# Create output directories
os.makedirs("security_output/zero_day", exist_ok=True)

def main():
    """
    Example of zero-day threat detection using CyberThreat-ML.
    """
    print("\nCyberThreat-ML Zero-Day Threat Detection Example")
    print("=" * 50)

    # Step 1: Create synthetic dataset with normal, known attacks, and zero-day attacks
    print("\nStep 1: Creating synthetic datasets...")
    data_normal, labels_normal = create_normal_dataset(n_samples=1000)
    print(f"Created {len(data_normal)} normal traffic samples")

    data_known, labels_known = create_known_attacks_dataset(n_samples=200)
    print(f"Created {len(data_known)} known attack samples")

    data_zero_day, labels_zero_day = create_zero_day_attacks_dataset(n_samples=50)
    print(f"Created {len(data_zero_day)} zero-day attack samples")

    # Combine all datasets for testing
    combined_data = np.vstack([data_normal, data_known, data_zero_day])
    combined_labels = np.concatenate([labels_normal, labels_known, labels_zero_day])
    
    # Keep track of which samples are zero-day attacks
    is_zero_day = np.zeros(len(combined_labels), dtype=bool)
    is_zero_day[len(data_normal) + len(data_known):] = True

    # Step 2: Train a traditional classifier on normal and known attack data
    print("\nStep 2: Training signature-based classifier...")
    # Only use normal and known attack data for training
    X_train = np.vstack([data_normal, data_known])
    y_train = np.concatenate([labels_normal, labels_known])
    
    # Binary classification: 0 = normal, 1 = attack
    y_train_binary = (y_train > 0).astype(int)
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_binary, test_size=0.2, random_state=42)
    
    # Create and train signature-based model
    input_dim = X_train.shape[1]  # Get actual number of features
    signature_model = ThreatDetectionModel(
        input_shape=(input_dim,),  # Use actual feature dimension
        num_classes=1,      # Binary classification: single output node
        model_config={
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'sigmoid',  # Binary classification
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy', 'AUC'],  # Added AUC for better anomaly detection evaluation
            'optimizer': 'adam',
            'class_names': ['Normal', 'Attack']
        }
    )

    # Train with proper validation split and early stopping
    print("\nTraining signature-based model...")
    print(f"Input shape: {input_dim} features")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    history = signature_model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    print("Signature-based classifier trained successfully")
    
    # Step 3: Train a zero-day detector on normal data only
    print("\nStep 3: Training zero-day detector...")
    # Use only normal data for training the anomaly detector
    zero_day_detector = ZeroDayDetector(
        method='ensemble',
        contamination=0.01  # Expect 1% false positives
    )
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(data_normal.shape[1])]
    
    # Fit on normal data only
    zero_day_detector.fit(data_normal, feature_names)
    print("Zero-day detector trained successfully")
    
    # Step 4: Evaluate detection performance
    print("\nStep 4: Evaluating detection performance...")
    
    # Use signature-based model to detect known attacks
    signature_predictions = signature_model.predict(combined_data)
    
    # Use zero-day detector to detect anomalies
    anomaly_predictions, anomaly_scores = zero_day_detector.detect(combined_data, return_scores=True)
    # Convert predictions: 1 = normal, -1 = anomaly -> 0 = normal, 1 = anomaly
    anomaly_predictions = (anomaly_predictions == -1).astype(int)
    
    # Hybrid detection system combines both
    hybrid_predictions = np.logical_or(signature_predictions, anomaly_predictions).astype(int)
    
    # Calculate metrics
    true_labels = (combined_labels > 0).astype(int)  # Convert to binary: 0 = normal, 1 = attack
    
    # Calculate detection rates
    signature_accuracy = np.mean(signature_predictions == true_labels)
    anomaly_accuracy = np.mean(anomaly_predictions == true_labels)
    hybrid_accuracy = np.mean(hybrid_predictions == true_labels)
    
    # Calculate detection rates specifically for zero-day attacks
    zero_day_signature_detection = np.mean(signature_predictions[is_zero_day] == true_labels[is_zero_day])
    zero_day_anomaly_detection = np.mean(anomaly_predictions[is_zero_day] == true_labels[is_zero_day])
    zero_day_hybrid_detection = np.mean(hybrid_predictions[is_zero_day] == true_labels[is_zero_day])
    
    print(f"Signature-based detection accuracy: {signature_accuracy:.4f}")
    print(f"Anomaly-based detection accuracy: {anomaly_accuracy:.4f}")
    print(f"Hybrid detection accuracy: {hybrid_accuracy:.4f}")
    print(f"Zero-day detection rate (signature): {zero_day_signature_detection:.4f}")
    print(f"Zero-day detection rate (anomaly): {zero_day_anomaly_detection:.4f}")
    print(f"Zero-day detection rate (hybrid): {zero_day_hybrid_detection:.4f}")
    
    # Create a detailed comparison table
    results_table = create_results_table(
        signature_predictions, anomaly_predictions, hybrid_predictions,
        true_labels, is_zero_day
    )
    
    print("\nDetection Performance Summary:")
    print(results_table)
    
    # Save the results table
    results_file = "security_output/zero_day/detection_results.txt"
    with open(results_file, 'w') as f:
        f.write("CyberThreat-ML Zero-Day Detection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(results_table) + "\n\n")
        f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Detection results saved to {results_file}")
    
    # Step 5: Analyze anomalies
    print("\nStep 5: Analyzing detected anomalies...")
    
    # Find the indices of samples detected as anomalies
    anomaly_indices = np.where(anomaly_predictions == 1)[0]
    print(f"Found {len(anomaly_indices)} anomalies to analyze")
    
    # Analyze a few of the detected anomalies
    num_to_analyze = min(5, len(anomaly_indices))
    
    # Create a summary of anomaly analyses
    analyses = []
    for i in range(num_to_analyze):
        idx = anomaly_indices[i]
        sample = combined_data[idx]
        score = anomaly_scores[idx]
        true_class = "Zero-Day Attack" if is_zero_day[idx] else "Known Attack" if combined_labels[idx] > 0 else "Normal"
        
        # Analyze the anomaly
        analysis = zero_day_detector.analyze_anomaly(sample, score)
        analyses.append({
            "sample_index": idx,
            "anomaly_score": score,
            "true_class": true_class,
            "severity": analysis["severity"],
            "severity_level": analysis["severity_level"],
            "description": get_anomaly_description(analysis),
            "recommended_action": recommend_action(analysis)["priority"]
        })
        
        # Print the analysis
        print(f"\nAnomaly {i+1}/{num_to_analyze}:")
        print(f"  True class: {true_class}")
        print(f"  Anomaly score: {score:.4f}")
        print(f"  Severity: {analysis['severity_level']} ({analysis['severity']:.4f})")
        print(f"  Description: {get_anomaly_description(analysis)}")
        print(f"  Recommended action priority: {recommend_action(analysis)['priority']}")
        
        # Save the full analysis
        analysis_file = f"security_output/zero_day/anomaly_{i+1}_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write(f"ANOMALY {i+1} ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Sample index: {idx}\n")
            f.write(f"True class: {true_class}\n")
            f.write(f"Anomaly score: {score:.4f}\n")
            f.write(f"Severity: {analysis['severity_level']} ({analysis['severity']:.4f})\n\n")
            f.write("FEATURE ANALYSIS:\n")
            for feature, details in analysis["feature_details"].items():
                f.write(f"  {feature}:\n")
                f.write(f"    Value: {details['value']:.4f}\n")
                f.write(f"    Z-score: {details['z_score']:.4f}\n")
                f.write(f"    Deviation: {details['deviation']:.4f}\n")
                f.write(f"    Baseline mean: {details['baseline_mean']:.4f}\n")
                f.write(f"    Baseline std: {details['baseline_std']:.4f}\n\n")
            
            f.write("RECOMMENDED ACTIONS:\n")
            actions = recommend_action(analysis)
            f.write(f"  Priority: {actions['priority']}\n")
            f.write("  Actions:\n")
            for action in actions["actions"]:
                f.write(f"    - {action}\n")
            f.write(f"  Investigation: {actions['investigation']}\n")
    
    # Create a summary table of anomalies
    anomaly_df = pd.DataFrame(analyses)
    print("\nAnomaly Analysis Summary:")
    print(anomaly_df[["true_class", "anomaly_score", "severity_level", "recommended_action"]])
    
    # Save the summary
    anomaly_df.to_csv("security_output/zero_day/anomaly_summary.csv", index=False)
    
    # Step 6: Demonstrate real-time zero-day detection
    print("\nStep 6: Demonstrating real-time zero-day detection...")
    
    # Create feature extractor for real-time detection
    class SimpleFeatureExtractor:
        def transform(self, data):
            # Simulates feature extraction from raw data
            if isinstance(data, dict):
                # If it's a packet-like dict, extract features
                return np.array([
                    data.get('size', 0) / 10000.0,
                    data.get('entropy', 0),
                    data.get('tcp_flags', 0) / 255.0,
                    data.get('src_port', 0) / 65535.0,
                    data.get('dst_port', 0) / 65535.0,
                    data.get('duration', 0) / 10.0,
                    data.get('packet_count', 0) / 100.0,
                    data.get('bytes_in', 0) / 10000.0,
                    data.get('bytes_out', 0) / 10000.0,
                    data.get('protocol', 0) / 255.0
                ]).reshape(1, -1)
            elif isinstance(data, np.ndarray):
                # If already a feature vector, return as is
                return data
            else:
                # Otherwise, return a random feature vector
                return np.random.rand(1, 10)
    
    feature_extractor = SimpleFeatureExtractor()
    feature_names = [
        "Size", "Entropy", "TCP Flags", "Source Port", "Destination Port",
        "Duration", "Packet Count", "Bytes In", "Bytes Out", "Protocol"
    ]
    
    # Create a real-time zero-day detector
    realtime_detector = RealTimeZeroDayDetector(
        feature_extractor=feature_extractor,
        baseline_data=data_normal,
        feature_names=feature_names,
        method='isolation_forest',
        contamination=0.05,
        time_window=60
    )
    
    # Create a traditional signature-based detector
    signature_detector = PacketStreamDetector(
        signature_model,
        feature_extractor,
        threshold=0.5,
        batch_size=5,
        processing_interval=1.0
    )
    
    # Process both normal and anomalous traffic
    print("Simulating network traffic with zero-day attacks...")
    detected_anomalies = []
    
    def on_signature_threat(result):
        print(f"  - Signature detected: {result['class_name']} (confidence: {result['confidence']:.4f})")
        
    # Register callback for signature detection
    signature_detector.register_threat_callback(on_signature_threat)
    signature_detector.start()
    
    # Simulate traffic for 5 seconds
    start_time = time.time()
    traffic_samples = 0
    signature_detections = 0
    anomaly_detections = 0
    combined_detections = 0
    
    while time.time() - start_time < 5:
        # Decide what type of traffic to generate
        traffic_type = np.random.choice(["normal", "known", "zero_day"], p=[0.7, 0.2, 0.1])
        
        if traffic_type == "normal":
            # Generate normal traffic
            sample = create_random_packet(normal=True)
            true_label = 0
        elif traffic_type == "known":
            # Generate known attack traffic
            sample = create_random_packet(attack_type="known")
            true_label = 1
        else:
            # Generate zero-day attack traffic
            sample = create_random_packet(attack_type="zero_day")
            true_label = 2
            
        # Process with signature detector
        signature_detector.process_packet(sample)
        
        # Process with zero-day detector
        anomaly_result = realtime_detector.add_sample(sample)
        
        # Check for detection
        signature_detected = false_positive = False
        
        # Track detections
        if anomaly_result:
            anomaly_detections += 1
            detected_anomalies.append(anomaly_result)
            print(f"\nZero-day detector found an anomaly (score: {anomaly_result['anomaly_score']:.4f}):")
            print(f"  - Severity: {anomaly_result['severity_level']} ({anomaly_result['severity']:.4f})")
            print(f"  - Description: {get_anomaly_description(anomaly_result['analysis'])}")
            
            # Check if it was detected by the signature detector
            if signature_detector.get_stats()["threats_detected"] > signature_detections:
                signature_detected = True
                combined_detections += 1
                print("  - Also detected by signature detector")
            else:
                combined_detections += 1
                print("  - Not detected by signature detector")
                
            signature_detections = signature_detector.get_stats()["threats_detected"]
        
        traffic_samples += 1
        time.sleep(0.1)  # 10 samples per second
    
    # Stop signature detector
    signature_detector.stop()
    
    # Print results
    print(f"\nProcessed {traffic_samples} traffic samples in 5 seconds")
    print(f"Anomaly detection found {anomaly_detections} anomalies")
    print(f"Signature detection found {signature_detections} threats")
    print(f"Combined detection found {combined_detections} total threats")
    
    # Step 7: Adaptive learning
    print("\nStep 7: Demonstrating adaptive learning...")
    stats_before = realtime_detector.get_stats()
    
    # Create new normal baseline
    new_normal_data = create_normal_dataset(n_samples=200)[0]
    
    # Add the new normal data as samples
    for i in range(len(new_normal_data)):
        sample = {
            'size': random.uniform(100, 1500),
            'entropy': random.uniform(0.1, 0.5),  # Lower entropy for normal traffic
            'tcp_flags': random.randint(0, 255),
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 22, 25, 53]),
            'duration': random.uniform(0.1, 2.0),
            'packet_count': random.randint(1, 10),
            'bytes_in': random.uniform(100, 2000),
            'bytes_out': random.uniform(100, 1000),
            'protocol': random.choice([6, 17])  # TCP or UDP
        }
        realtime_detector.add_sample(sample)
    
    # Train on recent normal data
    print("Training on recent normal traffic...")
    result = realtime_detector.train_on_recent_normal(min_samples=50)
    
    if result:
        print("Adaptive training successful")
        
        # Get new stats
        stats_after = realtime_detector.get_stats()
        
        print("\nDetector Statistics Before:")
        for key, value in stats_before.items():
            if key in ['samples_collected', 'anomalies_detected', 'is_fitted']:
                print(f"  {key}: {value}")
        
        print("\nDetector Statistics After:")
        for key, value in stats_after.items():
            if key in ['samples_collected', 'anomalies_detected', 'is_fitted']:
                print(f"  {key}: {value}")
    else:
        print("Not enough normal samples for adaptive training")
    
    print("\nZero-day detection example completed successfully!")

def create_normal_dataset(n_samples=1000):
    """
    Create a synthetic dataset of normal network traffic.
    
    Args:
        n_samples (int): Number of samples to generate.
        
    Returns:
        tuple: (features, labels)
    """
    # Generate feature matrix with 10 features representing normal traffic patterns
    X = np.random.rand(n_samples, 10)
    
    # Adjust some features to have a "normal" distribution
    # Size (feature 0) - Normal traffic is usually smaller
    X[:, 0] = np.random.normal(0.3, 0.1, n_samples)
    X[:, 0] = np.clip(X[:, 0], 0, 1)
    
    # Entropy (feature 1) - Normal traffic usually has lower entropy
    X[:, 1] = np.random.normal(0.4, 0.1, n_samples)
    X[:, 1] = np.clip(X[:, 1], 0, 1)
    
    # Duration (feature 5) - Normal connections have a more consistent duration
    X[:, 5] = np.random.normal(0.5, 0.15, n_samples)
    X[:, 5] = np.clip(X[:, 5], 0, 1)
    
    # All labels are 0 (normal)
    y = np.zeros(n_samples)
    
    return X, y

def create_known_attacks_dataset(n_samples=200):
    """
    Create a synthetic dataset of known attack patterns.
    
    Args:
        n_samples (int): Number of samples to generate.
        
    Returns:
        tuple: (features, labels)
    """
    # Generate base feature matrix
    X = np.random.rand(n_samples, 10)
    
    # Divide samples between different attack types
    samples_per_type = n_samples // 4
    
    # 1. Port Scan Attack (high number of destination ports)
    X[:samples_per_type, 4] = np.random.normal(0.9, 0.05, samples_per_type)  # Destination port feature
    X[:samples_per_type, 6] = np.random.normal(0.8, 0.1, samples_per_type)   # Packet count feature
    
    # 2. DDoS Attack (high packet count, high bytes out)
    X[samples_per_type:2*samples_per_type, 6] = np.random.normal(0.95, 0.05, samples_per_type)  # Packet count
    X[samples_per_type:2*samples_per_type, 8] = np.random.normal(0.9, 0.1, samples_per_type)    # Bytes out
    
    # 3. Brute Force (high connection attempts, specific ports)
    X[2*samples_per_type:3*samples_per_type, 5] = np.random.normal(0.2, 0.05, samples_per_type)  # Duration
    X[2*samples_per_type:3*samples_per_type, 6] = np.random.normal(0.7, 0.1, samples_per_type)   # Packet count
    
    # 4. Data Exfiltration (high bytes out, high entropy)
    X[3*samples_per_type:, 1] = np.random.normal(0.85, 0.1, n_samples - 3*samples_per_type)  # Entropy
    X[3*samples_per_type:, 8] = np.random.normal(0.9, 0.1, n_samples - 3*samples_per_type)   # Bytes out
    
    # Labels: 1=Port Scan, 2=DDoS, 3=Brute Force, 4=Data Exfiltration
    y = np.ones(n_samples)
    y[samples_per_type:2*samples_per_type] = 2
    y[2*samples_per_type:3*samples_per_type] = 3
    y[3*samples_per_type:] = 4
    
    return X, y

def create_zero_day_attacks_dataset(n_samples=50):
    """
    Create a synthetic dataset of zero-day attack patterns.
    These patterns combine features in ways not seen in training data.
    
    Args:
        n_samples (int): Number of samples to generate.
        
    Returns:
        tuple: (features, labels)
    """
    # Generate base feature matrix
    X = np.random.rand(n_samples, 10)
    
    # Divide samples between different zero-day attack types
    samples_per_type = n_samples // 2
    
    # 1. Advanced Polymorphic Malware (unusual entropy patterns, variable sizes)
    X[:samples_per_type, 1] = np.sin(np.linspace(0, 6*np.pi, samples_per_type)) * 0.4 + 0.5  # Oscillating entropy
    X[:samples_per_type, 0] = np.abs(np.sin(np.linspace(0, 4*np.pi, samples_per_type))) * 0.7 + 0.2  # Variable sizes
    X[:samples_per_type, 7] = np.random.normal(0.7, 0.2, samples_per_type)  # Bytes in
    
    # 2. Unknown protocol tunneling attack (unusual protocol patterns, port combinations)
    X[samples_per_type:, 9] = np.random.normal(0.95, 0.05, n_samples - samples_per_type)  # Protocol marker
    X[samples_per_type:, 3] = np.random.normal(0.85, 0.1, n_samples - samples_per_type)   # Source port pattern
    X[samples_per_type:, 4] = np.random.normal(0.7, 0.2, n_samples - samples_per_type)    # Destination port pattern
    X[samples_per_type:, 1] = np.random.normal(0.75, 0.15, n_samples - samples_per_type)  # Higher entropy
    
    # Labels: 5=Polymorphic Malware, 6=Protocol Tunneling (these are "unseen" in training)
    y = np.ones(n_samples) * 5
    y[samples_per_type:] = 6
    
    return X, y

def create_random_packet(normal=False, attack_type=None):
    """
    Create a random network packet for simulation.
    
    Args:
        normal (bool): Whether to create a normal packet.
        attack_type (str): Type of attack ('known' or 'zero_day').
        
    Returns:
        dict: Synthetic packet data.
    """
    if normal:
        # Create normal packet
        return {
            'size': random.uniform(100, 1500),
            'entropy': random.uniform(0.1, 0.5),  # Lower entropy for normal traffic
            'tcp_flags': random.randint(0, 255),
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 22, 25, 53]),
            'duration': random.uniform(0.1, 2.0),
            'packet_count': random.randint(1, 10),
            'bytes_in': random.uniform(100, 2000),
            'bytes_out': random.uniform(100, 1000),
            'protocol': random.choice([6, 17])  # TCP or UDP
        }
    elif attack_type == "known":
        # Create a known attack packet
        attack_subtype = random.choice(["port_scan", "ddos", "brute_force", "exfiltration"])
        
        if attack_subtype == "port_scan":
            return {
                'size': random.uniform(60, 200),  # Smaller packets
                'entropy': random.uniform(0.3, 0.6),
                'tcp_flags': random.randint(0, 255),
                'src_port': random.randint(1024, 65535),
                'dst_port': random.randint(1, 65535),  # Variable destination ports
                'duration': random.uniform(0.01, 0.2),  # Very short duration
                'packet_count': random.randint(20, 100),  # Many packets
                'bytes_in': random.uniform(50, 200),
                'bytes_out': random.uniform(50, 200),
                'protocol': 6  # TCP
            }
        elif attack_subtype == "ddos":
            return {
                'size': random.uniform(100, 1000),
                'entropy': random.uniform(0.3, 0.7),
                'tcp_flags': random.randint(0, 255),
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([80, 443]),  # Web ports
                'duration': random.uniform(0.05, 0.5),
                'packet_count': random.randint(50, 200),  # Very high packet count
                'bytes_in': random.uniform(100, 500),
                'bytes_out': random.uniform(5000, 10000),  # High outbound traffic
                'protocol': random.choice([6, 17])
            }
        elif attack_subtype == "brute_force":
            return {
                'size': random.uniform(200, 500),
                'entropy': random.uniform(0.4, 0.7),
                'tcp_flags': random.randint(0, 255),
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([22, 3389]),  # SSH or RDP
                'duration': random.uniform(0.05, 0.2),  # Short connections
                'packet_count': random.randint(10, 30),
                'bytes_in': random.uniform(200, 600),
                'bytes_out': random.uniform(100, 300),
                'protocol': 6  # TCP
            }
        else:  # exfiltration
            return {
                'size': random.uniform(1000, 5000),  # Larger packets
                'entropy': random.uniform(0.7, 0.9),  # High entropy (possible encryption)
                'tcp_flags': random.randint(0, 255),
                'src_port': random.randint(1024, 65535),
                'dst_port': random.randint(1024, 65535),
                'duration': random.uniform(1.0, 5.0),  # Longer duration
                'packet_count': random.randint(5, 15),
                'bytes_in': random.uniform(500, 1000),
                'bytes_out': random.uniform(5000, 20000),  # Very high outbound data
                'protocol': random.choice([6, 17])
            }
    else:  # zero_day
        # Create a zero-day attack packet (patterns not seen in training)
        attack_subtype = random.choice(["polymorphic", "tunneling"])
        
        if attack_subtype == "polymorphic":
            # Polymorphic malware with unusual patterns
            base_entropy = random.uniform(0.5, 0.8)
            entropy_variation = random.uniform(-0.2, 0.2)
            
            return {
                'size': random.uniform(300, 3000) * random.uniform(0.8, 1.2),  # Variable sizes
                'entropy': max(0, min(1, base_entropy + entropy_variation)),  # Unusual entropy
                'tcp_flags': random.randint(0, 255),
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([80, 443, 8080, 8443]),  # Web ports
                'duration': random.uniform(0.5, 3.0) * random.uniform(0.7, 1.3),  # Variable duration
                'packet_count': random.randint(5, 20),
                'bytes_in': random.uniform(1000, 5000) * random.uniform(0.9, 1.1),
                'bytes_out': random.uniform(500, 2000) * random.uniform(0.9, 1.1),
                'protocol': 6 if random.random() < 0.9 else random.randint(41, 200)  # Unusual protocols
            }
        else:  # tunneling
            # Protocol tunneling attack
            unusual_port = random.randint(10000, 65000)
            
            return {
                'size': random.uniform(800, 1600),
                'entropy': random.uniform(0.7, 0.95),  # Very high entropy
                'tcp_flags': random.randint(0, 255),
                'src_port': unusual_port,
                'dst_port': unusual_port,  # Same source and destination ports
                'duration': random.uniform(2.0, 10.0),  # Long duration
                'packet_count': random.randint(10, 30),
                'bytes_in': random.uniform(2000, 8000),
                'bytes_out': random.uniform(2000, 8000),  # Balanced in/out
                'protocol': random.choice([41, 50, 51, 47])  # Unusual protocols
            }

def create_results_table(signature_preds, anomaly_preds, hybrid_preds, true_labels, is_zero_day):
    """
    Create a pandas DataFrame with detection results.
    
    Args:
        signature_preds: Predictions from signature-based model.
        anomaly_preds: Predictions from anomaly-based model.
        hybrid_preds: Predictions from hybrid approach.
        true_labels: Ground truth labels.
        is_zero_day: Boolean array indicating which samples are zero-day attacks.
        
    Returns:
        pandas.DataFrame: Results table.
    """
    # Calculate overall metrics
    total_samples = len(true_labels)
    normal_samples = np.sum(true_labels == 0)
    attack_samples = np.sum(true_labels == 1)
    zero_day_samples = np.sum(is_zero_day)
    known_attack_samples = attack_samples - zero_day_samples
    
    # Calculate detection rates
    sig_normal_correct = np.sum((signature_preds == 0) & (true_labels == 0))
    sig_attack_correct = np.sum((signature_preds == 1) & (true_labels == 1))
    sig_known_correct = np.sum((signature_preds == 1) & (true_labels == 1) & ~is_zero_day)
    sig_zero_day_correct = np.sum((signature_preds == 1) & is_zero_day)
    
    anom_normal_correct = np.sum((anomaly_preds == 0) & (true_labels == 0))
    anom_attack_correct = np.sum((anomaly_preds == 1) & (true_labels == 1))
    anom_known_correct = np.sum((anomaly_preds == 1) & (true_labels == 1) & ~is_zero_day)
    anom_zero_day_correct = np.sum((anomaly_preds == 1) & is_zero_day)
    
    hybrid_normal_correct = np.sum((hybrid_preds == 0) & (true_labels == 0))
    hybrid_attack_correct = np.sum((hybrid_preds == 1) & (true_labels == 1))
    hybrid_known_correct = np.sum((hybrid_preds == 1) & (true_labels == 1) & ~is_zero_day)
    hybrid_zero_day_correct = np.sum((hybrid_preds == 1) & is_zero_day)
    
    # Calculate rates
    sig_normal_rate = sig_normal_correct / normal_samples if normal_samples > 0 else 0
    sig_attack_rate = sig_attack_correct / attack_samples if attack_samples > 0 else 0
    sig_known_rate = sig_known_correct / known_attack_samples if known_attack_samples > 0 else 0
    sig_zero_day_rate = sig_zero_day_correct / zero_day_samples if zero_day_samples > 0 else 0
    
    anom_normal_rate = anom_normal_correct / normal_samples if normal_samples > 0 else 0
    anom_attack_rate = anom_attack_correct / attack_samples if attack_samples > 0 else 0
    anom_known_rate = anom_known_correct / known_attack_samples if known_attack_samples > 0 else 0
    anom_zero_day_rate = anom_zero_day_correct / zero_day_samples if zero_day_samples > 0 else 0
    
    hybrid_normal_rate = hybrid_normal_correct / normal_samples if normal_samples > 0 else 0
    hybrid_attack_rate = hybrid_attack_correct / attack_samples if attack_samples > 0 else 0
    hybrid_known_rate = hybrid_known_correct / known_attack_samples if known_attack_samples > 0 else 0
    hybrid_zero_day_rate = hybrid_zero_day_correct / zero_day_samples if zero_day_samples > 0 else 0
    
    # Create results DataFrame
    data = {
        "Metric": [
            "Total Samples", 
            "Normal Samples", 
            "Attack Samples",
            "Known Attack Samples",
            "Zero-Day Attack Samples",
            "Normal Detection Rate",
            "Attack Detection Rate",
            "Known Attack Detection Rate",
            "Zero-Day Attack Detection Rate",
            "False Positive Rate",
            "False Negative Rate"
        ],
        "Signature-Based": [
            total_samples,
            normal_samples,
            attack_samples,
            known_attack_samples,
            zero_day_samples,
            f"{sig_normal_rate:.4f}",
            f"{sig_attack_rate:.4f}",
            f"{sig_known_rate:.4f}",
            f"{sig_zero_day_rate:.4f}",
            f"{(1-sig_normal_rate):.4f}",
            f"{(1-sig_attack_rate):.4f}"
        ],
        "Anomaly-Based": [
            total_samples,
            normal_samples,
            attack_samples,
            known_attack_samples,
            zero_day_samples,
            f"{anom_normal_rate:.4f}",
            f"{anom_attack_rate:.4f}",
            f"{anom_known_rate:.4f}",
            f"{anom_zero_day_rate:.4f}",
            f"{(1-anom_normal_rate):.4f}",
            f"{(1-anom_attack_rate):.4f}"
        ],
        "Hybrid": [
            total_samples,
            normal_samples,
            attack_samples,
            known_attack_samples,
            zero_day_samples,
            f"{hybrid_normal_rate:.4f}",
            f"{hybrid_attack_rate:.4f}",
            f"{hybrid_known_rate:.4f}",
            f"{hybrid_zero_day_rate:.4f}",
            f"{(1-hybrid_normal_rate):.4f}",
            f"{(1-hybrid_attack_rate):.4f}"
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    main()