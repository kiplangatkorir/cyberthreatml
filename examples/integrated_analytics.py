"""
Example demonstrating how to integrate visualization and interpretability features
for advanced cyber threat analytics.
"""

import sys
import os
import time
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter, get_threat_pattern_insights

def main():
    """
    Example of integrated visualization and interpretability for cyber threat analytics.
    """
    print("CyberThreat-ML Integrated Analytics Example")
    print("============================================")
    
    # Define threat classes
    threat_classes = [
        "Normal Traffic",
        "Port Scan",
        "DDoS",
        "Brute Force",
        "Data Exfiltration"
    ]
    
    # Step 1: Set up the model
    print("\nStep 1: Loading or creating a model...")
    model, feature_names = create_or_load_model(threat_classes)
    
    # Step 2: Initialize both visualization and interpretability components
    print("\nStep 2: Initializing analytics components...")
    dashboard = ThreatVisualizationDashboard(max_history=50, update_interval=1.0)
    interpreter = ThreatInterpreter(model, feature_names=feature_names, class_names=threat_classes)
    
    # Create sample data to initialize the interpreter
    X_background = create_synthetic_dataset(100, len(feature_names), len(threat_classes))[0]
    interpreter.initialize(X_background)
    
    # Create output directories
    os.makedirs('analytics_output', exist_ok=True)
    
    # Step 3: Run a simulated threat detection session
    print("\nStep 3: Running simulated threat detection (5 seconds)...")
    threat_data = run_simulated_detection(model, dashboard, threat_classes)
    
    # Step 4: Perform post-detection analytics
    print("\nStep 4: Performing post-detection analytics...")
    analyze_detected_threats(threat_data, interpreter, threat_classes)
    
    # Step 5: Generate threat pattern insights
    print("\nStep 5: Generating threat pattern insights...")
    generate_threat_pattern_insights(threat_data, interpreter, threat_classes)
    
    print("\nIntegrated analytics example completed successfully!")

def create_or_load_model(threat_classes):
    """
    Create a new model or load an existing model.
    
    Args:
        threat_classes (list): List of threat class names
        
    Returns:
        tuple: (model, feature_names) - The model and feature names
    """
    # Define feature names for interpretability
    feature_names = [
        "Source Port",
        "Destination Port",
        "Packet Size",
        "TCP Flag",
        "UDP Flag",
        "ICMP Flag",
        "TCP Flags Value",
        "TTL Value",
        "Payload Entropy",
        "Connection Duration",
        "Packet Count",
        "Bytes Transferred",
        "Packet Interval",
        "Flow Direction",
        "Protocol",
        "Window Size",
        "Header Length",
        "Urgent Pointer",
        "Payload Size",
        "Suspicious Domain",
        "Suspicious Port Combo",
        "Encrypted Payload",
        "Geographic Distance",
        "Time of Day",
        "Day of Week"
    ]
    
    # Try to load existing model
    model_path = os.path.join('models', 'multiclass_threat_model')
    
    if os.path.exists(model_path):
        try:
            print("Loading existing model from", model_path)
            model = load_model(
                model_path,
                os.path.join('models', 'multiclass_threat_metadata.json')
            )
            print("Model loaded successfully")
            return model, feature_names
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model instead")
            
    # Create synthetic data for training
    print("Creating synthetic dataset...")
    X_train, y_train = create_synthetic_dataset(2000, len(feature_names), len(threat_classes))
    
    # Split into train/validation sets
    n_val = int(len(X_train) * 0.2)
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]
    y_train = y_train[:-n_val]
    
    # Create and train model
    print("Creating and training model...")
    model = ThreatDetectionModel(
        input_shape=(X_train.shape[1],),
        num_classes=len(threat_classes),
        model_config={
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'softmax',
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy'],
            'class_names': threat_classes
        }
    )
    
    # Train the model with a small number of epochs for quick execution
    model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )
        ]
    )
    
    return model, feature_names

def run_simulated_detection(model, dashboard, class_names):
    """
    Run a simulated threat detection session.
    
    Args:
        model (ThreatDetectionModel): Trained model
        dashboard (ThreatVisualizationDashboard): Visualization dashboard
        class_names (list): List of class names
        
    Returns:
        dict: Dictionary of detected threats by type
    """
    # Initialize a detector with our feature extractor
    class SimpleFeatureExtractor:
        def transform(self, x):
            # Simply return x as features are already extracted
            return x
    
    detector = PacketStreamDetector(
        model=model,
        feature_extractor=SimpleFeatureExtractor(),
        threshold=0.5,
        batch_size=5,
        processing_interval=0.5
    )
    
    # Start the dashboard and detector
    dashboard.start()
    detector.start()
    
    # Set up storage for detected threats
    detected_threats = {class_name: [] for class_name in class_names}
    
    # Define the threat detection callback
    def on_threat_detected(result):
        # Add to dashboard
        dashboard.add_threat(result)
        
        # Record the threat for analysis
        class_idx = result.get('predicted_class', 0)
        
        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
            detected_threats[class_name].append(result)
            
            # Print threat information
            confidence = result.get('confidence', 0.0)
            print(f"Detected {class_name} threat (confidence: {confidence:.4f})")
    
    # Register the callback
    detector.register_threat_callback(on_threat_detected)
    
    # Generate and process synthetic packets for 5 seconds
    start_time = time.time()
    packet_count = 0
    threat_count = {class_name: 0 for class_name in class_names}
    
    try:
        # Bias the distribution to ensure we get some of each threat type
        while time.time() - start_time < 5:
            # Create synthetic packets with different threat classes
            for class_idx in range(len(class_names)):
                # Generate feature vector biased toward this class
                sample = create_biased_sample(class_idx)
                
                # Process the packet
                detector.process_packet(sample)
                packet_count += 1
                
                # Short delay
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    finally:
        # Stop the detector and dashboard
        detector.stop()
        dashboard.stop()
        
        # Generate statistics
        for class_name in class_names:
            threat_count[class_name] = len(detected_threats[class_name])
            
        # Print summary
        print(f"\nProcessed {packet_count} packets in {time.time() - start_time:.2f} seconds")
        print("Threat detection summary:")
        for class_name, count in threat_count.items():
            print(f"  {class_name}: {count} detections")
    
    return detected_threats

def analyze_detected_threats(threat_data, interpreter, class_names):
    """
    Analyze detected threats using the interpreter.
    
    Args:
        threat_data (dict): Dictionary of detected threats by type
        interpreter (ThreatInterpreter): Threat interpreter
        class_names (list): List of class names
    """
    # Create analysis output directory
    analysis_dir = os.path.join('analytics_output', 'threat_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze each threat type that was detected
    for class_idx, class_name in enumerate(class_names):
        threats = threat_data[class_name]
        
        if not threats:
            print(f"No {class_name} threats detected for analysis")
            continue
            
        print(f"\nAnalyzing {len(threats)} {class_name} threats:")
        
        # Get feature vectors from threats
        feature_vectors = np.array([threat['packet'] for threat in threats])
        
        # Select a sample threat for detailed analysis
        sample_idx = random.randint(0, len(threats) - 1)
        sample_threat = threats[sample_idx]
        sample_vector = sample_threat['packet'].reshape(1, -1)
        
        # Generate explanation for the sample threat
        explanation = interpreter.explain_prediction(
            sample_vector,
            method="auto",
            target_class=class_idx,
            top_features=7
        )
        
        # Create visualization of the explanation
        fig = interpreter.plot_explanation(
            explanation,
            plot_type="bar",
            save_path=os.path.join(analysis_dir, f"{class_name.lower().replace(' ', '_')}_explanation.png")
        )
        
        # Print top features
        print(f"Top features for {class_name} detection:")
        for name, value in explanation["top_features"][:5]:
            print(f"  {name}: {value:.4f}")
            
        # Create a report
        report = interpreter.create_feature_importance_report(
            explanation,
            os.path.join(analysis_dir, f"{class_name.lower().replace(' ', '_')}_report.txt")
        )
        
def generate_threat_pattern_insights(threat_data, interpreter, class_names):
    """
    Generate insights about threat patterns across multiple detections.
    
    Args:
        threat_data (dict): Dictionary of detected threats by type
        interpreter (ThreatInterpreter): Threat interpreter
        class_names (list): List of class names
    """
    # Create insights output directory
    insights_dir = os.path.join('analytics_output', 'threat_insights')
    os.makedirs(insights_dir, exist_ok=True)
    
    # Generate correlation statistics
    correlations = {}
    
    # For each threat type
    for class_idx, class_name in enumerate(class_names[1:], 1):  # Skip normal traffic
        threats = threat_data[class_name]
        
        if len(threats) < 2:
            continue
            
        # Get feature vectors from threats
        feature_vectors = np.array([threat['packet'] for threat in threats])
        
        # Get insights for this threat type
        insights = get_threat_pattern_insights(
            interpreter, 
            feature_vectors, 
            class_idx,
            top_features=5
        )
        
        if not insights:
            continue
            
        # Print insights
        print(f"\nPattern insights for {class_name} threats:")
        
        # Print the key features
        print("  Key predictive features:")
        for feature, importance in insights["key_features"]:
            print(f"    {feature}: {importance:.4f}")
            
        # Print consistency data if available
        if insights["consistency"]:
            print("  Feature consistency:")
            for feature, data in list(insights["consistency"].items())[:3]:  # Show top 3
                cv = data.get("coefficient_variation", float('inf'))
                reliability = "High" if cv < 0.3 else "Medium" if cv < 0.6 else "Low"
                print(f"    {feature}: {reliability} reliability (CV: {cv:.2f})")
                
        # Save insights to file
        with open(os.path.join(insights_dir, f"{class_name.lower().replace(' ', '_')}_insights.txt"), 'w') as f:
            f.write(f"THREAT PATTERN INSIGHTS: {class_name}\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("KEY PREDICTIVE FEATURES\n")
            for feature, importance in insights["key_features"]:
                f.write(f"  {feature}: {importance:.4f}\n")
                
            f.write("\nFEATURE RELIABILITY\n")
            for feature, data in insights["consistency"].items():
                cv = data.get("coefficient_variation", float('inf'))
                reliability = "High" if cv < 0.3 else "Medium" if cv < 0.6 else "Low"
                f.write(f"  {feature}: {reliability} reliability (CV: {cv:.2f})\n")

def create_biased_sample(class_idx, feature_names=None):
    """
    Create a sample that's biased toward a specific class.
    
    Args:
        class_idx (int): Class index to bias toward
        feature_names (list, optional): List of feature names
        
    Returns:
        numpy.ndarray: Biased sample
    """
    # Get the input size from feature names
    input_size = len(feature_names) if feature_names else 25
    
    # Create base sample
    sample = np.random.randn(input_size) * 0.1  # Lower variance for better separation
    
    # Ensure values are normalized
    sample = np.clip(sample, -1, 1)  # Normalize to [-1, 1] range
    
    # Apply class-specific biases
    if class_idx == 0:  # Normal Traffic
        # Normal traffic has no specific pattern, just random noise
        pass
        
    elif class_idx == 1:  # Port Scan
        # Higher variability in port-related features
        sample[0] = 0.9  # Source Port
        sample[1] = 0.8  # Destination Port
        sample[20] = 1.0  # Suspicious Port Combo
        sample[10] = 0.7  # Packet Count (many small packets)
        sample[12] = 0.9  # Packet Interval (rapid succession)
        
    elif class_idx == 2:  # DDoS
        # Increased traffic volume and packet size
        sample[2] = 0.95  # Packet Size
        sample[5] = 1.0  # ICMP Flag (often used in DDoS)
        sample[10] = 0.9  # Packet Count (high volume)
        sample[11] = 0.8  # Bytes Transferred (high volume)
        sample[18] = 0.7  # Payload Size (can be small but many)
        
    elif class_idx == 3:  # Brute Force
        # Repeated authentication attempts
        sample[0] = 0.3  # Source Port
        sample[1] = 0.99  # Destination Port (typically 22 for SSH or 3389 for RDP)
        sample[3] = 1.0  # TCP Flag
        sample[10] = 0.8  # Packet Count (many attempts)
        sample[12] = 0.5  # Packet Interval (consistent timing)
        
    elif class_idx == 4:  # Data Exfiltration
        # Large outbound data transfers
        sample[2] = 0.9  # Packet Size (large)
        sample[8] = 0.9  # Payload Entropy (high for encrypted/compressed data)
        sample[11] = 0.95  # Bytes Transferred (large amount)
        sample[13] = 0.8  # Flow Direction (outbound)
        sample[21] = 0.7  # Encrypted Payload
        
    # Add small random noise for variation between samples
    sample += np.random.randn(input_size) * 0.05
    sample = np.clip(sample, -1, 1)
    
    return sample

def create_synthetic_dataset(n_samples=1000, n_features=25, n_classes=5, normal_prob=0.4):
    """
    Create a synthetic multi-class dataset for cyber threat detection.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_classes (int): Number of threat classes (including normal traffic).
        normal_prob (float): Probability of normal traffic samples.
        
    Returns:
        tuple: (X, y) - features and class labels.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate feature matrix
    X = np.random.randn(n_samples, n_features) * 0.1
    
    # Calculate how many normal vs threat samples to create
    n_normal = int(n_samples * normal_prob)
    n_threats = n_samples - n_normal
    
    # Assign normal class (0) to normal_prob percentage of samples
    y = np.zeros(n_samples, dtype=int)
    
    # Distribute remaining samples among threat classes (1 to n_classes-1)
    if n_classes > 1 and n_threats > 0:
        threat_indices = np.random.choice(n_samples, size=n_threats, replace=False)
        threat_classes = np.random.randint(1, n_classes, size=n_threats)
        y[threat_indices] = threat_classes
    
    # Create class patterns to make data more separable
    class_patterns = []
    for i in range(n_classes):
        pattern = np.random.uniform(-0.5, 0.5, n_features)
        strong_features = np.random.choice(n_features, size=int(n_features * 0.3), replace=False)
        pattern[strong_features] = np.random.uniform(1.5, 2.5, len(strong_features)) * np.sign(pattern[strong_features])
        class_patterns.append(pattern)
    
    # Apply class patterns to make data more separable
    for i in range(n_samples):
        class_idx = y[i]
        X[i] += class_patterns[class_idx] + np.random.normal(0, 0.3, n_features)
    
    return X, y

if __name__ == "__main__":
    main()