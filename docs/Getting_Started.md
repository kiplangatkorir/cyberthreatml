# CyberThreat-ML: Getting Started Guide

This guide provides quick examples for common use cases with the CyberThreat-ML library.

## Table of Contents

1. [Installation](#installation)
2. [Basic Threat Detection](#basic-threat-detection)
3. [Multi-class Threat Classification](#multi-class-threat-classification)
4. [Real-time Threat Detection](#real-time-threat-detection)
5. [Visualizing Threats](#visualizing-threats)
6. [Explaining Model Decisions](#explaining-model-decisions)
7. [Integrated Analytics](#integrated-analytics)

## Installation

```bash
pip install cyberthreat-ml
```

Ensure you have the required dependencies:

```bash
pip install tensorflow numpy<2.0.0 pandas scikit-learn matplotlib
# Optional dependencies for advanced features
pip install shap seaborn
```

## Basic Threat Detection

Basic binary classification to detect potential threats.

```python
import numpy as np
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.utils import split_data

# Create a synthetic dataset for demonstration
def create_synthetic_dataset(n_samples=1000, n_features=20):
    # Generate random features
    X = np.random.rand(n_samples, n_features)
    
    # Generate binary labels (0: normal, 1: threat)
    # 30% of samples are threats in this synthetic data
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    return X, y

# Create dataset
X, y = create_synthetic_dataset()

# Split data into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Create and train the model for binary classification
model = ThreatDetectionModel(input_shape=(20,), num_classes=2)
model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=10)

# Evaluate the model
from cyberthreat_ml.evaluation import evaluate_model, classification_report

metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

report = classification_report(model, X_test, y_test)
print(report)

# Save the model
model.save_model('models/binary_threat_model')
```

## Multi-class Threat Classification

Classify threats into multiple categories.

```python
import numpy as np
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.utils import split_data
from sklearn.preprocessing import StandardScaler

# Create a synthetic multi-class dataset
def create_synthetic_multiclass_dataset(n_samples=2000, n_features=25, n_classes=6):
    # Generate random features
    X = np.random.rand(n_samples, n_features)
    
    # Generate multi-class labels
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Ensure class 0 (normal traffic) is more frequent
    normal_idx = np.random.choice(n_samples, size=int(n_samples * 0.4), replace=False)
    y[normal_idx] = 0
    
    return X, y

# Create dataset
X, y = create_synthetic_multiclass_dataset()

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define class names for better interpretability
class_names = [
    "Normal Traffic",
    "Port Scan",
    "DDoS",
    "Brute Force",
    "Data Exfiltration",
    "Command & Control"
]

# Create and train the multi-class model
model = ThreatDetectionModel(
    input_shape=(25,),
    num_classes=6,
    model_config={
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3
    }
)
model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=15)

# Evaluate and visualize
from cyberthreat_ml.evaluation import plot_confusion_matrix
import matplotlib.pyplot as plt

# Display classification report
report = classification_report(model, X_test, y_test)
print(report)

# Plot confusion matrix
fig = plot_confusion_matrix(model, X_test, y_test, normalize=True)
plt.savefig('multiclass_confusion_matrix.png')

# Demonstrate detailed predictions on a few samples
sample_indices = np.random.choice(len(X_test), 5)
for i, idx in enumerate(sample_indices):
    probs = model.predict_proba(X_test[idx:idx+1])[0]
    pred_class = np.argmax(probs)
    true_class = y_test[idx]
    
    print(f"Sample {i+1}:")
    print(f"  True class: {class_names[true_class]}")
    print(f"  Predicted class: {class_names[pred_class]}")
    print("  Class probabilities:")
    for j, class_name in enumerate(class_names):
        print(f"    {class_name}: {probs[j]:.4f}")
```

## Real-time Threat Detection

Process network packets in real-time to detect threats.

```python
import numpy as np
import time
import random
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.realtime import PacketStreamDetector

# Try to load an existing model or create a new one
try:
    model = load_model('models/multiclass_threat_model')
    print("Loaded existing model")
except:
    # Create a simple model for demonstration
    model = ThreatDetectionModel(input_shape=(25,), num_classes=6)
    print("Created new model")

# Define class names for better output
class_names = [
    "Normal Traffic",
    "Port Scan",
    "DDoS",
    "Brute Force",
    "Data Exfiltration",
    "Command & Control"
]

# Define action recommendations for each threat type
actions = {
    "Port Scan": "MONITOR source IP for additional scanning activity",
    "DDoS": "ACTIVATE ANTI-DDOS MEASURES and ALERT SECURITY TEAM",
    "Brute Force": "TEMPORARY ACCOUNT LOCKOUT and ENABLE 2FA",
    "Data Exfiltration": "ISOLATE affected systems and INVESTIGATE data access",
    "Command & Control": "QUARANTINE affected systems and BLOCK C2 servers"
}

# Simple feature extractor for demonstration
class SimpleFeatureExtractor:
    def transform(self, packet):
        # In a real system, this would extract meaningful features
        # Here we just create a random feature vector
        return np.random.rand(25)

# Set up the real-time detector
detector = PacketStreamDetector(model, SimpleFeatureExtractor())

# Define callback for when a threat is detected
def on_threat_detected(result):
    class_idx = result['class_idx']
    class_name = class_names[class_idx]
    
    # Only show threats (not normal traffic)
    if class_idx > 0:
        print(f"🚨 THREAT DETECTED! - {class_name} 🚨")
        print(f"  Timestamp: {time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  Feature vector: {result['features']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print("  Class probabilities:")
        for i, name in enumerate(class_names[1:], 1):  # Skip normal traffic
            if result['probabilities'][i] > 0.21:  # Only show significant probabilities
                print(f"    {name}: {result['probabilities'][i]:.4f}")
        print(f"  Suggested action: {actions.get(class_name, 'INVESTIGATE')}")

# Define callback for batch processing
def on_batch_processed(results):
    threats = sum(1 for r in results if r['class_idx'] > 0)
    print(f"Batch processed: {len(results)} packets, {threats} threats detected")

# Register callbacks
detector.register_threat_callback(on_threat_detected)
detector.register_processing_callback(on_batch_processed)

# Start the detector
detector.start()

# Generate some synthetic packets for demonstration
def generate_random_packet():
    packet = {
        'timestamp': time.time(),
        'source_ip': f"192.168.1.{random.randint(1, 254)}",
        'destination_ip': f"10.0.0.{random.randint(1, 254)}",
        'source_port': random.randint(1024, 65535),
        'destination_port': random.randint(1, 1024),
        'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
        'size': random.randint(64, 1500),
        'payload': np.random.bytes(random.randint(10, 100))
    }
    return packet

# Process some packets
for _ in range(10):
    packet = generate_random_packet()
    detector.process_packet(packet)
    time.sleep(0.1)  # Small delay to simulate packet arrival

# Print statistics
stats = detector.get_stats()
print(f"Statistics:")
print(f"  Packets processed: {stats['packets_processed']}")
print(f"  Threats detected: {stats['threats_detected']}")
print(f"  Queue size: {stats['queue_size']}")

# Stop the detector when done
detector.stop()
```

## Visualizing Threats

Create a visual dashboard for monitoring threats.

```python
import numpy as np
import time
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.realtime import PacketStreamDetector

# Create or load a model
model = ThreatDetectionModel(input_shape=(25,), num_classes=5)

# Define threat class names
class_names = [
    "Normal Traffic",
    "Port Scan",
    "DDoS",
    "Brute Force",
    "Data Exfiltration"
]

# Set up visualization dashboard
dashboard = ThreatVisualizationDashboard(max_history=100)
dashboard.start()

# Simple feature extractor for demonstration
class SimpleFeatureExtractor:
    def transform(self, x):
        # In a real system, this would extract meaningful features
        return np.random.rand(25)

# Set up real-time detector
detector = PacketStreamDetector(model, SimpleFeatureExtractor())

# Register callback to update dashboard when threats are detected
def on_threat_detected(result):
    if result['class_idx'] > 0:  # Skip normal traffic
        class_name = class_names[result['class_idx']]
        print(f"Detected {class_name} threat (confidence: {result['confidence']:.4f})")
        
        # Add the threat to the dashboard
        threat_data = {
            'timestamp': time.time(),
            'class_name': class_name,
            'class_idx': result['class_idx'],
            'confidence': result['confidence'],
            'source_ip': result.get('source_ip', '192.168.1.1'),
            'destination_ip': result.get('destination_ip', '10.0.0.1'),
            'features': result['features']
        }
        dashboard.add_threat(threat_data)

detector.register_threat_callback(on_threat_detected)

# Start the detector
detector.start()

# Simulate some network traffic
for _ in range(30):
    # Generate random data
    packet = {
        'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
        'destination_ip': f'10.0.0.{np.random.randint(1, 255)}',
        'payload': np.random.bytes(100)
    }
    detector.process_packet(packet)
    time.sleep(0.2)  # Small delay between packets

# Save a snapshot of the dashboard
dashboard.save_snapshot('dashboard_snapshot.png')

# Cleanup
detector.stop()
dashboard.stop()
```

## Explaining Model Decisions

Use interpretability features to explain why a threat was detected.

```python
import numpy as np
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.interpretability import ThreatInterpreter

# Create a model (or load an existing one)
model = ThreatDetectionModel(input_shape=(25,), num_classes=5)

# Feature names improve interpretability
feature_names = [
    "Source Port", "Destination Port", "Packet Size", "Flow Duration",
    "Bytes Transferred", "Packet Count", "TCP Flags", "Time-to-live",
    "Inter-arrival Time", "Flow Direction", "Protocol Type", "Window Size",
    "Payload Length", "Payload Entropy", "Encrypted Payload", "Header Length",
    "Source IP Entropy", "Dest IP Entropy", "Connection State", "Suspicious Port Combo",
    "Rate of SYN Packets", "Unique Destinations", "Bytes per Packet", "Fragment Bits",
    "Packet Sequence"
]

# Class names for threat types
class_names = [
    "Normal Traffic",
    "Port Scan",
    "DDoS",
    "Brute Force",
    "Data Exfiltration"
]

# Create interpreter
interpreter = ThreatInterpreter(model, feature_names, class_names)

# Create background data for initializing explainers
background_data = np.random.rand(100, 25)  # 100 random samples
interpreter.initialize(background_data)

# Create a sample to explain
# This could be from a real threat detection or a synthetic example
sample_data = np.random.rand(1, 25)

# Explain a prediction with SHAP
explanation = interpreter.explain_prediction(
    sample_data[0],
    method="shap",
    target_class=3,  # Explain why this was classified as "Brute Force"
    top_features=5
)

# Visualize the explanation
plot = interpreter.plot_explanation(
    explanation, 
    plot_type="bar",
    save_path="brute_force_explanation.png"
)

# Generate a text report
report = interpreter.create_feature_importance_report(
    explanation,
    output_path="brute_force_report.txt"
)
print(f"Explanation report saved to brute_force_report.txt")
print(f"Explanation plot saved to brute_force_explanation.png")

# Show the top features
print(f"Top features for {class_names[3]} detection:")
for feature, importance in explanation['top_features']:
    print(f"  {feature}: {importance:.4f}")
```

## Integrated Analytics

Combine visualization, interpretation, and analytics for comprehensive threat analysis.

```python
import numpy as np
import time
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter, get_threat_pattern_insights
from cyberthreat_ml.realtime import PacketStreamDetector

# Define threat class names
class_names = [
    "Normal Traffic",
    "Port Scan",
    "DDoS",
    "Brute Force",
    "Data Exfiltration"
]

# Try to load an existing model or create a new one
try:
    model = load_model('models/multiclass_threat_model')
    print("Loaded existing model")
except:
    print("Creating and training new model...")
    # For simplicity, we'll create a synthetic dataset and train a simple model
    X = np.random.rand(1000, 25)
    y = np.random.randint(0, 5, size=1000)
    model = ThreatDetectionModel(input_shape=(25,), num_classes=5)
    model.train(X, y, epochs=3, batch_size=32)

# Feature names for interpretability
feature_names = [
    "Source Port", "Destination Port", "Packet Size", "Flow Duration",
    "Bytes Transferred", "Packet Count", "TCP Flags", "Time-to-live",
    "Inter-arrival Time", "Flow Direction", "Protocol Type", "Window Size",
    "Payload Length", "Payload Entropy", "Encrypted Payload", "Header Length",
    "Source IP Entropy", "Dest IP Entropy", "Connection State", "Suspicious Port Combo",
    "Rate of SYN Packets", "Unique Destinations", "Bytes per Packet", "Fragment Bits",
    "Packet Sequence"
]

# Set up the visualization dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Set up the interpreter for explainability
interpreter = ThreatInterpreter(model, feature_names, class_names)
interpreter.initialize(np.random.rand(100, 25))  # Background data

# Create threat storage for analysis
threat_data = {class_idx: [] for class_idx in range(1, len(class_names))}  # Skip normal traffic

# Simple feature extractor for demonstration
class SimpleFeatureExtractor:
    def transform(self, x):
        return np.random.rand(25)

# Set up real-time detector
detector = PacketStreamDetector(model, SimpleFeatureExtractor())

# Register callback for threat detection
def on_threat_detected(result):
    if result['class_idx'] > 0:  # Skip normal traffic
        class_name = class_names[result['class_idx']]
        
        # Store the threat for later analysis
        threat_data[result['class_idx']].append(result['features'])
        
        # Add to dashboard
        dashboard_data = {
            'timestamp': time.time(),
            'class_name': class_name,
            'class_idx': result['class_idx'],
            'confidence': result['confidence'],
            'source_ip': '192.168.1.1',  # Placeholder
            'destination_ip': '10.0.0.1',  # Placeholder
            'features': result['features']
        }
        dashboard.add_threat(dashboard_data)

detector.register_threat_callback(on_threat_detected)

# Start the detector
detector.start()

# Simulate network traffic
print("Simulating network traffic and detecting threats...")
for _ in range(50):
    detector.process_packet({'payload': np.random.bytes(100)})
    time.sleep(0.1)

# Stop the detector
detector.stop()

# Analyze detected threats
print("Analyzing detected threats...")
for class_idx, features_list in threat_data.items():
    if features_list:
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Get insights about this threat class
        insights = get_threat_pattern_insights(
            interpreter, 
            features_array, 
            class_idx, 
            top_features=5
        )
        
        # Print insights
        print(f"Pattern insights for {class_names[class_idx]} threats:")
        print("  Key predictive features:")
        for feature, importance in insights['top_features']:
            print(f"    {feature}: {importance:.4f}")
        
        # Print feature consistency
        print("  Feature consistency:")
        for feature, cv in insights['feature_consistency'][:3]:
            reliability = "High" if cv < 0.1 else "Medium" if cv < 0.3 else "Low"
            print(f"    {feature}: {reliability} reliability (CV: {cv:.2f})")

# Save a final snapshot of the dashboard
dashboard.save_snapshot('visualization_output/integrated_analytics_final.png')
dashboard.stop()

print("Integrated analytics example completed successfully!")
```

These examples demonstrate the core functionalities of the CyberThreat-ML library. For more detailed information, please refer to the [Technical Documentation](TECHNICAL_DOCUMENTATION.md) and explore the example scripts in the `examples/` directory.