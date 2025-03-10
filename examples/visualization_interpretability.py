"""
Example demonstrating visualization and interpretability features for cybersecurity threat detection.
"""

import sys
import os
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter

def main():
    """
    Example demonstrating visualization and interpretability features.
    """
    print("CyberThreat-ML Visualization and Interpretability Example")
    print("------------------------------------------------------")
    
    # Define threat classes
    threat_classes = [
        "Normal Traffic",
        "Port Scan",
        "DDoS",
        "Brute Force",
        "Data Exfiltration"
    ]
    
    # Step 1: Create/load a model
    print("\nStep 1: Loading or creating a model...")
    model = create_and_train_model(threat_classes)
    
    # Step 2: Initialize visualization dashboard
    print("\nStep 2: Initializing visualization dashboard...")
    dashboard = setup_visualization(model)
    
    # Step 3: Initialize interpretability tools
    print("\nStep 3: Setting up model interpretability...")
    interpreter = setup_interpretability(model, threat_classes)
    
    # Step 4: Generate sample threats and visualize
    print("\nStep 4: Generating sample threats and visualizing...")
    sample_data, sample_threats = generate_sample_threats(model, dashboard)
    
    # Step 5: Interpret specific threats
    print("\nStep 5: Interpreting detected threats...")
    interpret_threats(interpreter, sample_data, threat_classes)
    
    # Step 6: Demonstrate real-time visualization with threat callbacks
    print("\nStep 6: Demonstrating real-time visualization...")
    if dashboard is not None:
        run_real_time_demo(model, dashboard, threat_classes)
    
    print("\nVisualization and interpretability example completed!")


def create_and_train_model(threat_classes):
    """
    Create and train a threat detection model or load a saved model.
    
    Args:
        threat_classes (list): List of threat class names
        
    Returns:
        ThreatDetectionModel: Trained model
    """
    # For consistency with model dimensionality, we'll use 25 features (same as other examples)
    num_features = 25
    
    # Create synthetic data for training
    print("Creating synthetic dataset...")
    X_train, y_train = create_synthetic_dataset(2000, num_features, len(threat_classes))
    
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
    
    # Train the model
    print("Training model...")
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=5,  # Reduced for quicker execution
        batch_size=32,
        early_stopping=True
    )
    
    # Create directory for model if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    print("Saving model...")
    model.save_model(
        os.path.join('models', 'viz_interpretability_model'),
        os.path.join('models', 'viz_interpretability_metadata.json')
    )
    
    # Print training results
    train_acc = history.history.get('accuracy', [-1])[-1]
    val_acc = history.history.get('val_accuracy', [-1])[-1]
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    return model


def setup_visualization(model):
    """
    Set up the visualization dashboard.
    
    Args:
        model (ThreatDetectionModel): Trained model
        
    Returns:
        ThreatVisualizationDashboard: Configured dashboard
    """
    try:
        # Import and initialize dashboard
        from cyberthreat_ml.visualization import ThreatVisualizationDashboard
        dashboard = ThreatVisualizationDashboard(max_history=100, update_interval=1.0)
        print("Visualization dashboard created")
        
        # Create output directory for visualization artifacts
        os.makedirs('visualization_output', exist_ok=True)
        
        return dashboard
    except Exception as e:
        print(f"Error setting up visualization: {e}")
        print("Visualization features will be disabled")
        return None


def setup_interpretability(model, class_names):
    """
    Set up the threat interpreter.
    
    Args:
        model (ThreatDetectionModel): Trained model
        class_names (list): List of class names
        
    Returns:
        ThreatInterpreter: Configured interpreter
    """
    try:
        # Create feature names based on model input shape
        input_shape = model.model.input_shape[1]
        feature_names = []
        
        # Create interpretable feature names
        for i in range(input_shape):
            if i == 0:
                feature_names.append("Source Port")
            elif i == 1:
                feature_names.append("Destination Port")
            elif i == 2:
                feature_names.append("Packet Size")
            elif i == 3:
                feature_names.append("TCP Flag")
            elif i == 4:
                feature_names.append("UDP Flag")
            elif i == 5:
                feature_names.append("ICMP Flag")
            elif i == 6:
                feature_names.append("TCP Flags Value")
            elif i == 7:
                feature_names.append("TTL Value")
            elif i == 8:
                feature_names.append("Payload Entropy")
            elif i == 9:
                feature_names.append("Suspicious Port Combo")
            else:
                feature_names.append(f"Feature {i}")
        
        # Import and initialize interpreter
        from cyberthreat_ml.interpretability import ThreatInterpreter
        interpreter = ThreatInterpreter(model, feature_names=feature_names, class_names=class_names)
        
        # Create sample data to initialize the interpreter
        X_background = create_synthetic_dataset(100, input_shape, len(class_names))[0]
        interpreter.initialize(X_background)
        
        print("Threat interpreter initialized with feature names:")
        for i, name in enumerate(feature_names):
            print(f"  {i}: {name}")
            
        # Create output directory for interpretation artifacts
        os.makedirs('interpretation_output', exist_ok=True)
        
        return interpreter
    except Exception as e:
        print(f"Error setting up interpretability: {e}")
        print("Interpretability features will be limited")
        return None


def generate_sample_threats(model, dashboard):
    """
    Generate sample threat data and visualize it.
    
    Args:
        model (ThreatDetectionModel): Trained model
        dashboard (ThreatVisualizationDashboard): Visualization dashboard
        
    Returns:
        tuple: (sample_data, sample_threats) - Arrays of sample data and their predictions
    """
    # Create sample threat data
    n_samples = 50
    input_shape = model.model.input_shape[1]
    num_classes = model.num_classes
    
    # Generate synthetic data with bias toward threats
    X, y = create_synthetic_dataset(
        n_samples, 
        input_shape, 
        num_classes,
        normal_prob=0.2  # Lower probability of normal traffic
    )
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create threat data
    sample_threats = []
    for i in range(n_samples):
        # Create result dictionary similar to RealTimeDetector output
        timestamp = time.time() - random.uniform(0, 3600)  # Random time in the last hour
        
        result = {
            'id': i,
            'timestamp': timestamp,
            'packet': X[i],
            'predicted_class': int(predictions[i]),
            'class_probabilities': probabilities[i].tolist(),
            'is_threat': predictions[i] > 0,
            'confidence': float(probabilities[i][predictions[i]]),
            'threshold': 0.5,
            'is_binary': False,
            'class_names': model.model_config.get('class_names', None)
        }
        
        sample_threats.append(result)
        
        # Add to dashboard if available
        if dashboard is not None:
            dashboard.add_threat(result)
    
    # Save a dashboard snapshot if available
    if dashboard is not None:
        dashboard.save_snapshot('visualization_output/threat_dashboard_snapshot.png')
        print("Dashboard snapshot saved to 'visualization_output/threat_dashboard_snapshot.png'")
    
    return X, sample_threats


def interpret_threats(interpreter, sample_data, class_names):
    """
    Interpret specific threat examples using the interpreter.
    
    Args:
        interpreter (ThreatInterpreter): Configured interpreter
        sample_data (numpy.ndarray): Sample input data
        class_names (list): List of class names
    """
    if interpreter is None:
        print("Interpretability is not available")
        return
    
    # Select a sample for each class
    for class_idx in range(1, len(class_names)):  # Skip normal traffic
        class_name = class_names[class_idx]
        
        print(f"\nGenerating explanation for {class_name} threat:")
        
        # Create a biased sample for this class
        sample = create_biased_sample(class_idx, interpreter.feature_names)
        
        # Generate explanation
        explanation = interpreter.explain_prediction(
            sample.reshape(1, -1),
            method="auto",
            target_class=class_idx,
            top_features=5
        )
        
        # Create text report
        report = interpreter.create_feature_importance_report(
            explanation,
            f"interpretation_output/{class_name.lower().replace(' ', '_')}_report.txt"
        )
        
        # Print a summary of the report
        print(f"Top features for {class_name}:")
        for name, value in explanation["top_features"]:
            print(f"  {name}: {value:.4f}")
            
        # Create visualization
        fig = interpreter.plot_explanation(
            explanation,
            plot_type="bar",
            save_path=f"interpretation_output/{class_name.lower().replace(' ', '_')}_explanation.png"
        )
        
        print(f"Explanation saved to 'interpretation_output/{class_name.lower().replace(' ', '_')}_explanation.png'")


def create_biased_sample(class_idx, feature_names=None):
    """
    Create a sample that's biased toward a specific class.
    
    Args:
        class_idx (int): Class index to bias toward
        feature_names (list, optional): List of feature names
        
    Returns:
        numpy.ndarray: Biased sample
    """
    # Get the input size from the model
    input_size = len(feature_names) if feature_names else 25
    
    # Create base sample
    sample = np.random.randn(input_size)
    
    # Ensure values are normalized
    sample = np.clip(sample / 5, -1, 1)  # Normalize to [-1, 1] range
    
    # Apply class-specific biases to the first 10 features (or fewer if input_size < 10)
    max_idx = min(10, input_size)
    
    # Bias the sample based on class
    if class_idx == 1:  # Port Scan
        # Higher variability in port-related features
        if input_size > 0: sample[0] = 0.9  # Source port (normalized)
        if input_size > 1: sample[1] = 0.8  # Destination port (normalized)
        if input_size > 9: sample[9] = 1.0  # Suspicious port combination
        
    elif class_idx == 2:  # DDoS
        # Increased traffic volume and packet size
        if input_size > 2: sample[2] = 0.95  # Packet size
        if input_size > 5: sample[5] = 1.0   # ICMP flag (often used in DDoS)
        
    elif class_idx == 3:  # Brute Force
        # Repeated authentication patterns
        if input_size > 0: sample[0] = 0.3   # Source port (normalized)
        if input_size > 1: sample[1] = 0.99  # Destination port (normalized, e.g., targeting SSH/22)
        if input_size > 3: sample[3] = 1.0   # TCP flag (often used in brute force)
        
    elif class_idx == 4:  # Data Exfiltration
        # Large outbound data transfers
        if input_size > 2: sample[2] = 0.9   # Packet size (large)
        if input_size > 8: sample[8] = 0.9   # Payload entropy (high for encrypted/compressed data)
        
    return sample


def run_real_time_demo(model, dashboard, class_names):
    """
    Run a real-time demonstration with the dashboard.
    
    Args:
        model (ThreatDetectionModel): Trained model
        dashboard (ThreatVisualizationDashboard): Visualization dashboard
        class_names (list): List of class names
    """
    # Skip if dashboard is not available
    if dashboard is None:
        print("Dashboard is not available, skipping real-time demo")
        return
    
    print("Starting real-time visualization demo...")
    print("(This will run for 10 seconds)")
    
    # Start the dashboard
    dashboard.start()
    
    # Create a detector with proper feature extractor
    # This needs to be a class with a transform method, not just a lambda
    class SimpleFeatureExtractor:
        def transform(self, x):
            # Simply return x as features are already extracted
            return x
    
    feature_extractor = SimpleFeatureExtractor()
    detector = PacketStreamDetector(
        model=model,
        feature_extractor=feature_extractor,
        threshold=0.5,
        batch_size=5,
        processing_interval=1.0
    )
    
    # Register callback to update dashboard
    def on_threat_detected(result):
        # Add to dashboard
        dashboard.add_threat(result)
        
        # Log to console
        class_idx = result.get('predicted_class', 0)
        confidence = result.get('confidence', 0.0)
        
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        print(f"Detected {class_name} threat (confidence: {confidence:.4f})")
    
    # Register the callback
    detector.register_threat_callback(on_threat_detected)
    
    # Start the detector
    detector.start()
    
    # Generate and process packets for 10 seconds
    start_time = time.time()
    packet_count = 0
    
    try:
        while time.time() - start_time < 10:
            # Generate random feature vector (simulated packet)
            sample = create_random_threat_sample(class_names)
            
            # Process the sample
            detector.process_packet(sample)
            
            packet_count += 1
            
            # Short delay
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        
    finally:
        # Stop the detector
        detector.stop()
        
        # Stop the dashboard
        dashboard.stop()
        
        # Save final snapshot
        dashboard.save_snapshot('visualization_output/real_time_demo_final.png')
        print("Final dashboard snapshot saved to 'visualization_output/real_time_demo_final.png'")
        
        print(f"Processed {packet_count} simulated packets")


def create_random_threat_sample(class_names):
    """
    Create a random threat sample biased toward threats.
    
    Args:
        class_names (list): List of class names
        
    Returns:
        numpy.ndarray: Random feature vector
    """
    # Randomly choose a class with bias toward threats
    weights = [0.2] + [0.8 / (len(class_names) - 1)] * (len(class_names) - 1)
    class_idx = random.choices(range(len(class_names)), weights=weights)[0]
    
    # Create a biased sample for this class
    return create_biased_sample(class_idx)


def create_synthetic_dataset(n_samples=1000, n_features=10, n_classes=5, normal_prob=0.4):
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
    X = np.random.randn(n_samples, n_features)
    
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