# CyberThreat-ML: Comprehensive Library Documentation

This document provides detailed documentation for the CyberThreat-ML library, explaining its architecture, modules, and usage patterns. It combines information from multiple documentation sources to provide a single, comprehensive reference.

> **Note**: This documentation consolidates information from the root directory's DOCUMENTATION.md and other related files to provide a complete reference guide for CyberThreat-ML.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Modules](#core-modules)
   - [Model Module](#model-module)
   - [Preprocessing Module](#preprocessing-module)
   - [Real-time Detection Module](#real-time-detection-module)
   - [Evaluation Module](#evaluation-module)
   - [Explainability Module](#explainability-module)
   - [Visualization Module](#visualization-module)
   - [Interpretability Module](#interpretability-module)
   - [Anomaly Detection Module](#anomaly-detection-module)
   - [Text Visualization Module](#text-visualization-module)
   - [Utilities Module](#utilities-module)
4. [Advanced Usage](#advanced-usage)
   - [Multi-class Classification](#multi-class-classification)
   - [Custom Models](#custom-models)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Feature Importance Analysis](#feature-importance-analysis)
   - [Zero-Day Threat Detection](#zero-day-threat-detection)
   - [Complex Pattern Detection](#complex-pattern-detection)
5. [Integration Examples](#integration-examples)
   - [Enterprise Security Integration](#enterprise-security-integration)
   - [IoT Security Monitoring](#iot-security-monitoring)
   - [Integrated Analytics](#integrated-analytics)
6. [Performance Optimization](#performance-optimization)
   - [Batch Processing](#batch-processing)
   - [Resource Management](#resource-management)
   - [Scaling Considerations](#scaling-considerations)
7. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
   - [Error Messages](#error-messages)
   - [Debug Mode](#debug-mode)
8. [API Reference](#api-reference)

## Introduction

CyberThreat-ML is a Python library designed for real-time cybersecurity threat detection using machine learning techniques. The library provides a comprehensive set of tools for building, training, evaluating, and deploying ML models for identifying various types of cyber threats in network traffic and system events.

### Key Features

- **Pre-built models** for common cybersecurity threats
- **Real-time detection** capabilities for continuous monitoring
- **Multi-class classification** to identify specific threat types
- **Anomaly detection** for identifying zero-day threats
- **Model explainability** tools to understand detection decisions
- **Visualization components** for security dashboards
- **Text-based visualization** for environments without graphical capabilities
- **Enterprise integration** examples and components

### Design Philosophy

CyberThreat-ML is built on the following principles:

1. **Accuracy**: Prioritizing high-precision threat detection with minimal false positives
2. **Explainability**: Making model decisions transparent and understandable
3. **Adaptability**: Supporting customization for different security environments
4. **Efficiency**: Optimizing for real-time performance with minimal resource usage
5. **Minimal Dependencies**: Core functionality works with limited external dependencies

## Architecture Overview

The CyberThreat-ML library is organized into modular components that can be used independently or combined for comprehensive security solutions:

```
cyberthreat_ml/
├── __init__.py         # Package initialization
├── model.py            # Core ML model definitions
├── preprocessing.py    # Data preprocessing tools
├── realtime.py         # Real-time detection components
├── evaluation.py       # Model evaluation utilities
├── explain.py          # Basic model explainability
├── interpretability.py # Advanced interpretability tools
├── visualization.py    # Visual components for dashboards
├── text_visualization.py # Text-based visualization tools
├── anomaly.py          # Zero-day threat detection
├── utils.py            # Utility functions
└── logger.py           # Logging functionality
```

The library follows a layered architecture:

1. **Data Layer**: Preprocessing and feature extraction
2. **Model Layer**: Machine learning models and training logic
3. **Detection Layer**: Real-time processing and threat identification
4. **Analysis Layer**: Evaluation, interpretation, and explainability
5. **Presentation Layer**: Visualization and reporting components

## Core Modules

### Model Module

The `model.py` module provides the core machine learning functionality for threat detection:

#### ThreatDetectionModel Class

This is the primary class for creating, training, and using threat detection models:

```python
from cyberthreat_ml.model import ThreatDetectionModel

# Create a binary classification model
model = ThreatDetectionModel(
    input_shape=(25,),  # 25 features
    num_classes=2,      # Binary classification
    model_config={
        'hidden_layers': [64, 32],
        'activation': 'relu',
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }
)

# Train the model
history = model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=20,
    batch_size=32,
    class_weight='balanced'
)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Save and load models
model.save_model('models/my_model')
from cyberthreat_ml.model import load_model
loaded_model = load_model('models/my_model')
```

#### MultiClassThreatModel Class

Extended version of the threat detection model specifically optimized for multi-class threat classification:

```python
from cyberthreat_ml.model import MultiClassThreatModel

model = MultiClassThreatModel(
    input_shape=(30,),
    num_classes=6,  # Normal + 5 threat types
    model_config={
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3
    }
)
```

#### Custom Model Support

The library allows you to use custom model architectures while maintaining the same interface:

```python
from cyberthreat_ml.model import CustomThreatModel
import tensorflow as tf

# Define a custom model architecture
def create_custom_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create a model with the custom architecture
model = CustomThreatModel(
    input_shape=(25,),
    num_classes=6,
    model_builder=create_custom_model
)
```

### Preprocessing Module

The `preprocessing.py` module provides tools for preparing data for threat detection models:

#### FeatureExtractor Class

This class handles feature extraction and normalization:

```python
from cyberthreat_ml.preprocessing import FeatureExtractor

# Create a feature extractor
extractor = FeatureExtractor(
    feature_columns=['src_port', 'dst_port', 'packet_size', 'protocol', 'flags'],
    numerical_columns=['src_port', 'dst_port', 'packet_size'],
    categorical_columns=['protocol', 'flags'],
    normalization='standard'  # 'standard', 'minmax', or 'robust'
)

# Fit the extractor on training data
extractor.fit(X_train)

# Transform training and test data
X_train_transformed = extractor.transform(X_train)
X_test_transformed = extractor.transform(X_test)

# Save the extractor for later use
extractor.save('models/feature_extractor')

# Load a saved extractor
from cyberthreat_ml.preprocessing import load_extractor
loaded_extractor = load_extractor('models/feature_extractor')
```

#### Feature Engineering Functions

The module also provides specialized functions for extracting features from network data:

```python
from cyberthreat_ml.preprocessing import (
    extract_packet_features,
    extract_flow_features,
    extract_connection_features
)

# Extract features from a single packet
packet_features = extract_packet_features(packet_data)

# Extract features from network flows
flow_features = extract_flow_features(flow_data)

# Extract features from connection records
connection_features = extract_connection_features(connection_data)
```

### Real-time Detection Module

The `realtime.py` module provides components for real-time threat detection:

#### RealTimeDetector Class

This is the base class for real-time detection:

```python
from cyberthreat_ml.realtime import RealTimeDetector

class MyDetector(RealTimeDetector):
    def __init__(self, model, feature_extractor):
        super().__init__(model, feature_extractor)
        
    def process_data(self, data):
        # Custom processing logic
        features = self.feature_extractor.transform(data)
        results = self.model.predict_proba(features)
        return self._format_results(data, features, results)

detector = MyDetector(model, feature_extractor)
detector.start()
results = detector.process_data(data_batch)
detector.stop()
```

#### PacketStreamDetector Class

This class is specialized for processing network packet streams:

```python
from cyberthreat_ml.realtime import PacketStreamDetector

# Create a detector with a model and feature extractor
detector = PacketStreamDetector(
    model,
    feature_extractor,
    threshold=0.7,         # Detection threshold
    batch_size=32,         # Process packets in batches
    processing_interval=1.0  # Process every 1 second
)

# Register callbacks
def on_threat_detected(result):
    print(f"Threat detected: {result}")

def on_batch_processed(results):
    print(f"Processed {len(results)} packets")

detector.register_threat_callback(on_threat_detected)
detector.register_processing_callback(on_batch_processed)

# Start the detector
detector.start()

# Process a packet
detector.process_packet(packet_data)

# Get statistics
stats = detector.get_stats()
print(stats)

# Stop the detector
detector.stop()
```

### Evaluation Module

The `evaluation.py` module provides tools for evaluating model performance:

```python
from cyberthreat_ml.evaluation import (
    evaluate_model,
    classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    find_optimal_threshold
)

# Evaluate a model
metrics = evaluate_model(model, X_test, y_test, threshold=0.5)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Generate a classification report
report = classification_report(model, X_test, y_test)
print(report)

# Plot a confusion matrix
import matplotlib.pyplot as plt
fig = plot_confusion_matrix(model, X_test, y_test, normalize=True)
plt.savefig('confusion_matrix.png')

# Plot ROC curve
fig = plot_roc_curve(model, X_test, y_test)
plt.savefig('roc_curve.png')

# Find the optimal threshold for a specific metric
optimal_threshold = find_optimal_threshold(model, X_val, y_val, metric='f1')
print(f"Optimal threshold: {optimal_threshold:.4f}")
```

### Explainability Module

The `explain.py` module provides basic explainability tools:

```python
from cyberthreat_ml.explain import (
    explain_prediction,
    explain_model,
    generate_explanation_report
)

# Explain a single prediction
explanation = explain_prediction(model, X_sample, feature_names)
print(f"Base value: {explanation['base_value']:.4f}")
print(f"Top features contributing to prediction:")
for feature, value in explanation['shap_values'].items():
    print(f"  {feature}: {value:.4f}")

# Generate global explanations for the model
shap_values, feature_importance = explain_model(model, X_background, X_explain)

# Generate a full explanation report
report_path = generate_explanation_report(
    model,
    X_sample,
    feature_names,
    class_names,
    output_path='explanation_report.html'
)
print(f"Report generated at {report_path}")
```

### Interpretability Module

The `interpretability.py` module provides more advanced model interpretation:

```python
from cyberthreat_ml.interpretability import ThreatInterpreter

# Create an interpreter
interpreter = ThreatInterpreter(
    model,
    feature_names=['src_port', 'dst_port', 'packet_size', ...],
    class_names=['Normal', 'Port Scan', 'DDoS', ...]
)

# Initialize with background data for SHAP
interpreter.initialize(X_background_data)

# Explain a prediction
explanation = interpreter.explain_prediction(
    X_sample[0],
    method='shap',
    target_class=2,  # Explain prediction for class 2 (e.g., DDoS)
    top_features=10
)

# Plot the explanation
interpreter.plot_explanation(
    explanation,
    plot_type='bar',
    save_path='ddos_explanation.png'
)

# Generate a text report
report = interpreter.create_feature_importance_report(
    explanation,
    output_path='ddos_report.txt'
)

# Compare multiple explanations
explanations = [
    interpreter.explain_prediction(X_sample[i]) for i in range(5)
]
interpreter.compare_explanations(
    explanations,
    plot_type='summary',
    save_path='explanation_comparison.png'
)
```

### Visualization Module

The `visualization.py` module provides components for creating security dashboards:

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Create a visualization dashboard
dashboard = ThreatVisualizationDashboard(
    max_history=100,
    update_interval=1.0,
    display_classes=[1, 2, 3, 4, 5]  # Skip normal traffic (class 0)
)

# Start the dashboard
dashboard.start()

# Add a detected threat
dashboard.add_threat({
    'timestamp': time.time(),
    'class_name': 'DDoS',
    'class_idx': 2,
    'confidence': 0.92,
    'source_ip': '192.168.1.100',
    'destination_ip': '10.0.0.5',
    'features': features
})

# Create a specific visualization
fig = dashboard.create_timeline_plot(
    start_time=time.time() - 3600,  # Last hour
    end_time=time.time(),
    title='Threat Timeline'
)
plt.savefig('threat_timeline.png')

# Save a dashboard snapshot
dashboard.save_snapshot('security_dashboard.png')

# Stop the dashboard
dashboard.stop()
```

### Anomaly Detection Module

The `anomaly.py` module provides tools for zero-day threat detection:

```python
from cyberthreat_ml.anomaly import ZeroDayDetector

# Create a zero-day detector
detector = ZeroDayDetector(
    method='isolation_forest',  # 'isolation_forest', 'local_outlier_factor', 'robust_covariance'
    contamination=0.1,          # Expected proportion of anomalies
    feature_columns=None,       # Use all features
    min_samples=100,            # Minimum samples before detection starts
    threshold=None              # Auto-calculate threshold
)

# Train the detector on normal traffic data
detector.train(normal_traffic_data)

# Detect anomalies
results = detector.detect(test_data)

# Get anomaly scores
scores = detector.get_anomaly_scores(test_data)

# Explain anomalies
explanation = detector.explain_anomaly(anomalous_sample)
print(f"Anomaly score: {explanation['anomaly_score']:.4f}")
print("Most anomalous features:")
for feature, score in explanation['feature_scores'].items():
    print(f"  {feature}: {score:.4f}")
```

### Text Visualization Module

The `text_visualization.py` module provides text-based visualization tools for environments without graphical capabilities:

```python
from cyberthreat_ml.text_visualization import (
    TextVisualizer,
    SecurityReportGenerator,
    TextBasedDashboard
)

# Create a text visualizer
visualizer = TextVisualizer(terminal_width=80)

# Visualize a timeline of security events
visualizer.visualize_timeline(events, title="Security Event Timeline")

# Visualize network connections
visualizer.visualize_connections(connections, title="Network Connection Map")

# Create a text-based dashboard
dashboard = TextBasedDashboard(update_interval=5.0)
dashboard.start()
dashboard.add_threat(threat_data)
dashboard.stop()

# Generate a security report
generator = SecurityReportGenerator()
report = generator.generate_report(
    threats=detected_threats,
    start_time=start_time,
    end_time=end_time,
    include_statistics=True,
    include_recommendations=True
)
print(report)
```

### Utilities Module

The `utils.py` module provides various utility functions:

```python
from cyberthreat_ml.utils import (
    split_data,
    balance_dataset,
    load_dataset,
    save_dataset,
    set_random_seed,
    calculate_statistics
)

# Split data into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, 
    test_size=0.2, 
    val_size=0.15
)

# Balance an imbalanced dataset
X_balanced, y_balanced = balance_dataset(
    X_train, y_train,
    method='smote',  # 'oversample', 'undersample', 'smote'
    random_state=42
)

# Load a dataset
X, y, metadata = load_dataset('datasets/my_dataset.npz')

# Save a dataset
save_dataset(X, y, 'datasets/my_dataset.npz', metadata=metadata)

# Set random seed for reproducibility
set_random_seed(42)

# Calculate dataset statistics
stats = calculate_statistics(X, y)
print(f"Class distribution: {stats['class_distribution']}")
print(f"Feature ranges: {stats['feature_ranges']}")
```

## Advanced Usage

### Multi-class Classification

Multi-class classification allows identifying specific types of threats:

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.evaluation import classification_report
import numpy as np

# Define class names
class_names = [
    "Normal Traffic",
    "Port Scan",
    "DDoS",
    "Brute Force",
    "Data Exfiltration",
    "Command & Control"
]

# Create and train a multi-class model
model = ThreatDetectionModel(
    input_shape=(25,),
    num_classes=len(class_names),
    model_config={
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3
    }
)

# Use class weights to handle imbalanced data
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {i: total_samples / (len(class_names) * count) for i, count in enumerate(class_counts)}

# Train with class weights
model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=20, class_weight=class_weights)

# Evaluate multi-class performance
report = classification_report(model, X_test, y_test, class_names=class_names)
print(report)
```

### Custom Models

You can create custom model architectures while maintaining the CyberThreat-ML interface:

```python
from cyberthreat_ml.model import CustomThreatModel
import tensorflow as tf

# Define a custom model architecture
def create_residual_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First block
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Residual block
    shortcut = x
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    
    # Output layer
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create the custom model
model = CustomThreatModel(
    input_shape=(25,),
    num_classes=6,
    model_builder=create_residual_model
)

# Use the model as usual
model.train(X_train, y_train, epochs=20)
predictions = model.predict(X_test)
```

### Hyperparameter Tuning

Optimize model performance with hyperparameter tuning:

```python
from cyberthreat_ml.model import tune_hyperparameters

# Define the hyperparameter search space
param_grid = {
    'hidden_layers': [[64, 32], [128, 64], [128, 64, 32]],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'activation': ['relu', 'elu']
}

# Perform hyperparameter tuning
best_params, best_model, results = tune_hyperparameters(
    X_train, y_train,
    X_val, y_val,
    param_grid,
    input_shape=(25,),
    num_classes=6,
    metric='val_accuracy',
    n_trials=30,
    epochs=10,
    random_state=42
)

print(f"Best parameters: {best_params}")
print(f"Best validation accuracy: {results['best_score']:.4f}")

# Use the best model
predictions = best_model.predict(X_test)
```

### Feature Importance Analysis

Understand which features contribute most to model decisions:

```python
from cyberthreat_ml.interpretability import analyze_feature_importance

# Analyze global feature importance
importance_scores = analyze_feature_importance(
    model,
    X_train,
    feature_names=['src_port', 'dst_port', 'packet_size', ...],
    method='permutation',  # 'permutation', 'shap', or 'integrated_gradients'
    n_repeats=10
)

# Print feature importance scores
for feature, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")

# Plot feature importance
from cyberthreat_ml.visualization import plot_feature_importance
import matplotlib.pyplot as plt

fig = plot_feature_importance(importance_scores, top_n=10)
plt.savefig('feature_importance.png')
```

### Zero-Day Threat Detection

Detect previously unknown threats using anomaly detection:

```python
from cyberthreat_ml.anomaly import ZeroDayDetector
from cyberthreat_ml.model import ThreatDetectionModel

# Create and train a signature-based detector for known threats
signature_model = ThreatDetectionModel(input_shape=(25,), num_classes=6)
signature_model.train(X_train, y_train, epochs=15)

# Create an anomaly detector for zero-day threats
zero_day_detector = ZeroDayDetector(
    method='isolation_forest',
    contamination=0.05,
    threshold=None  # Auto-calculate
)

# Extract normal traffic samples
normal_indices = [i for i, label in enumerate(y_train) if label == 0]
normal_data = X_train[normal_indices]

# Train the zero-day detector on normal traffic
zero_day_detector.train(normal_data)

# Combined detection function
def detect_threats(samples):
    # First, check for known threats
    signature_results = signature_model.predict_proba(samples)
    signature_preds = signature_model.predict(samples)
    
    # For samples classified as normal by the signature model,
    # check for anomalies (potential zero-day threats)
    anomaly_results = []
    for i, pred in enumerate(signature_preds):
        if pred == 0:  # If classified as normal
            # Check if it's actually an anomaly
            is_anomaly = zero_day_detector.detect([samples[i]])[0]
            if is_anomaly:
                # This could be a zero-day threat
                anomaly_score = zero_day_detector.get_anomaly_scores([samples[i]])[0]
                anomaly_results.append({
                    'sample_idx': i,
                    'signature_prediction': 'Normal',
                    'anomaly_detected': True,
                    'anomaly_score': anomaly_score,
                    'explanation': zero_day_detector.explain_anomaly(samples[i])
                })
    
    return {
        'signature_predictions': signature_preds,
        'signature_probabilities': signature_results,
        'zero_day_threats': anomaly_results
    }

# Test the combined detection
results = detect_threats(X_test)
print(f"Known threats: {sum(pred > 0 for pred in results['signature_predictions'])}")
print(f"Potential zero-day threats: {len(results['zero_day_threats'])}")
```

### Complex Pattern Detection

Detect multi-stage attacks and complex threat patterns:

```python
from cyberthreat_ml.patterns import TemporalPatternDetector, BehavioralCorrelationDetector

# Create a temporal pattern detector
temporal_detector = TemporalPatternDetector(
    time_window=24,  # 24-hour window
    min_pattern_length=3  # At least 3 steps in a pattern
)

# Define known attack patterns
patterns = [
    {
        'name': 'APT Campaign',
        'steps': ['Reconnaissance', 'Initial Access', 'Execution', 'Persistence'],
        'max_time_between_steps': 12  # hours
    },
    {
        'name': 'Ransomware Attack',
        'steps': ['Initial Access', 'Execution', 'Impact'],
        'max_time_between_steps': 4  # hours
    }
]

temporal_detector.add_patterns(patterns)

# Add security events
for event in security_events:
    temporal_detector.add_event(event)

# Detect temporal patterns
detected_patterns = temporal_detector.detect_patterns()
print(f"Detected {len(detected_patterns)} potential attack campaigns")

# Create a behavioral correlation detector
behavioral_detector = BehavioralCorrelationDetector(max_time_window=48)  # 48-hour window

# Detect correlated behaviors
correlated_behaviors = behavioral_detector.detect_correlated_behaviors(security_events)
print(f"Detected {len(correlated_behaviors)} suspicious behavior patterns")

# Combined detection
all_complex_patterns = detected_patterns + correlated_behaviors
print(f"Total complex patterns detected: {len(all_complex_patterns)}")
```

## Integration Examples

### Enterprise Security Integration

Integrate CyberThreat-ML into enterprise security infrastructure:

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.interpretability import ThreatInterpreter
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Load a pre-trained model
model = load_model('models/enterprise_threat_model')

# Create a feature extractor
class EnterpriseFeatureExtractor:
    def transform(self, packet):
        # Extract relevant features from enterprise network packets
        features = [
            packet.get('src_port', 0),
            packet.get('dst_port', 0),
            packet.get('packet_size', 0),
            # ... more feature extraction logic
        ]
        return np.array(features)

# Set up real-time detection
detector = PacketStreamDetector(
    model,
    EnterpriseFeatureExtractor(),
    threshold=0.8,
    batch_size=64,
    processing_interval=0.5
)

# Set up interpretation
interpreter = ThreatInterpreter(
    model,
    feature_names=[...],
    class_names=[...]
)
interpreter.initialize(background_data)

# Set up visualization dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Define SIEM integration callback
def send_to_siem(alert_data):
    # Format the alert for your SIEM system
    siem_alert = {
        'timestamp': alert_data['timestamp'],
        'severity': 'High' if alert_data['confidence'] > 0.9 else 'Medium',
        'source': alert_data['source_ip'],
        'destination': alert_data['destination_ip'],
        'alert_type': alert_data['class_name'],
        'confidence': alert_data['confidence'],
        'details': {
            'raw_features': alert_data['features'].tolist(),
            'model_name': 'CyberThreat-ML Enterprise Model',
            'model_version': '1.2.3'
        }
    }
    
    # Send to SIEM (implementation depends on your SIEM system)
    # siem_client.send_alert(siem_alert)
    print(f"Alert sent to SIEM: {siem_alert['alert_type']} with {siem_alert['confidence']:.2f} confidence")

# Define threat callback
def on_threat_detected(result):
    if result['class_idx'] > 0 and result['confidence'] > 0.7:
        # Update dashboard
        dashboard.add_threat(result)
        
        # Explain the threat
        explanation = interpreter.explain_prediction(
            result['features'],
            target_class=result['class_idx']
        )
        
        # Augment the result with explanation
        result['explanation'] = explanation
        
        # Send to SIEM
        send_to_siem(result)
        
        # Log the threat
        print(f"Enterprise threat detected: {result['class_name']} with {result['confidence']:.2f} confidence")

# Register the callback
detector.register_threat_callback(on_threat_detected)

# Start the detector
detector.start()
```

### IoT Security Monitoring

Specialized threat detection for IoT environments:

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.realtime import RealTimeDetector

# Load IoT-specific model
iot_model = load_model('models/iot_threat_model')

# IoT-specific feature extractor
class IoTFeatureExtractor:
    def transform(self, data):
        # Extract features relevant to IoT devices
        # (bandwidth usage, protocol conformance, connection frequency, etc.)
        return np.array([
            data.get('bandwidth', 0),
            data.get('packet_rate', 0),
            data.get('connection_count', 0),
            # ... more IoT-specific features
        ])

# Create IoT-specific detector class
class IoTDeviceDetector(RealTimeDetector):
    def __init__(self, model, feature_extractor, threshold=0.5, 
                batch_size=32, processing_interval=1.0):
        super().__init__(model, feature_extractor, threshold, 
                         batch_size, processing_interval)
        self.device_states = {}  # Track device state
    
    def process_device_reading(self, device_id, reading):
        # Process a reading from an IoT device
        features = self.feature_extractor.transform(reading)
        
        # Update device state
        if device_id not in self.device_states:
            self.device_states[device_id] = {
                'readings': [],
                'alerts': [],
                'last_seen': time.time()
            }
        
        # Add to device readings
        self.device_states[device_id]['readings'].append(reading)
        self.device_states[device_id]['last_seen'] = time.time()
        
        # Keep only recent readings
        max_readings = 100
        if len(self.device_states[device_id]['readings']) > max_readings:
            self.device_states[device_id]['readings'] = self.device_states[device_id]['readings'][-max_readings:]
        
        # Process through the model
        results = self.model.predict_proba(features.reshape(1, -1))[0]
        class_idx = np.argmax(results)
        confidence = results[class_idx]
        
        # If it's a threat, add to queue for callbacks
        if class_idx > 0 and confidence >= self.threshold:
            result = {
                'timestamp': time.time(),
                'device_id': device_id,
                'device_type': reading.get('device_type', 'unknown'),
                'class_idx': class_idx,
                'confidence': confidence,
                'probabilities': results,
                'features': features
            }
            
            # Add to device alerts
            self.device_states[device_id]['alerts'].append(result)
            
            # Add to processing queue
            self._add_to_queue(result)
            
        return {
            'device_id': device_id,
            'processed': True,
            'is_threat': class_idx > 0 and confidence >= self.threshold,
            'class_idx': class_idx,
            'confidence': confidence
        }
    
    def get_device_state(self, device_id):
        return self.device_states.get(device_id)
    
    def get_all_device_ids(self):
        return list(self.device_states.keys())

# Create and use the IoT detector
iot_detector = IoTDeviceDetector(
    iot_model,
    IoTFeatureExtractor(),
    threshold=0.7,
    processing_interval=5.0  # Check every 5 seconds
)

# Define callback for IoT threats
def on_iot_threat(result):
    device_id = result['device_id']
    device_type = result['device_type']
    threat_type = ["Normal", "Anomalous Behavior", "Data Exfiltration", 
                  "Command Injection", "Botnet Activity"][result['class_idx']]
    
    print(f"IoT threat detected on {device_type} (ID: {device_id})")
    print(f"Threat type: {threat_type}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Get device state for context
    device_state = iot_detector.get_device_state(device_id)
    if device_state:
        recent_readings = device_state['readings'][-5:]  # Last 5 readings
        print(f"Recent activity:")
        for i, reading in enumerate(recent_readings):
            print(f"  Reading {i+1}: {reading}")

# Register callback and start detector
iot_detector.register_threat_callback(on_iot_threat)
iot_detector.start()

# Process readings from various devices
for device_reading in device_readings:
    iot_detector.process_device_reading(
        device_reading['device_id'],
        device_reading
    )
```

### Integrated Analytics

Combine detection, visualization, and interpretation:

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter
from cyberthreat_ml.realtime import PacketStreamDetector

# Load model
model = load_model('models/threat_detection_model')

# Set up components
dashboard = ThreatVisualizationDashboard()
interpreter = ThreatInterpreter(model, feature_names=[...], class_names=[...])
detector = PacketStreamDetector(model, feature_extractor)

# Initialize interpreter
interpreter.initialize(background_data)

# Create threat storage for analysis
threat_data = {class_idx: [] for class_idx in range(1, 6)}  # Skip normal traffic

# Define threat callback
def on_threat_detected(result):
    if result['class_idx'] > 0:  # Skip normal traffic
        # Store the threat for later analysis
        threat_data[result['class_idx']].append(result)
        
        # Add to dashboard
        dashboard.add_threat(result)
        
        # Get explanation
        explanation = interpreter.explain_prediction(
            result['features'],
            target_class=result['class_idx']
        )
        
        # Print explanation
        print(f"Threat detected: {result['class_name']}")
        print(f"Explanation:")
        for feature, importance in explanation['top_features']:
            print(f"  {feature}: {importance:.4f}")

# Register callback
detector.register_threat_callback(on_threat_detected)

# Start components
dashboard.start()
detector.start()

# Process network traffic
for packet in network_traffic:
    detector.process_packet(packet)

# Analyze collected threats
for class_idx, threats in threat_data.items():
    if threats:
        # Analyze patterns for this threat type
        patterns = interpreter.analyze_threat_patterns(
            [t['features'] for t in threats],
            class_idx=class_idx
        )
        
        print(f"\nPatterns for {class_names[class_idx]}:")
        for pattern in patterns:
            print(f"  Pattern strength: {pattern['strength']:.4f}")
            print(f"  Common features:")
            for feature, value in pattern['features'].items():
                print(f"    {feature}: {value:.4f}")

# Generate comprehensive report
from cyberthreat_ml.reporting import generate_threat_report

report_path = generate_threat_report(
    threat_data=threat_data,
    class_names=class_names,
    interpreter=interpreter,
    output_path='threat_analysis_report.html'
)
print(f"Report generated at {report_path}")
```

## Performance Optimization

### Batch Processing

Improve throughput by processing data in batches:

```python
from cyberthreat_ml.realtime import BatchProcessor

# Create a batch processor
batch_processor = BatchProcessor(
    model,
    feature_extractor,
    batch_size=128,
    max_queue_size=1000,
    num_workers=4  # Parallel processing
)

# Start the processor
batch_processor.start()

# Add data to be processed
for packet in incoming_packets:
    batch_processor.add_data(packet)

# Get processed results
results = batch_processor.get_results(timeout=1.0)  # Wait up to 1 second

# Stop the processor
batch_processor.stop()
```

### Resource Management

Control resource usage:

```python
from cyberthreat_ml.utils import set_resource_limits

# Set resource limits
set_resource_limits(
    memory_limit=1024,  # MB
    cpu_limit=2,        # Cores
    gpu_memory=512      # MB
)

# Set model-specific optimization
from cyberthreat_ml.model import optimize_model

# Optimize the model for inference
optimized_model = optimize_model(
    model,
    optimization_level='balanced',  # 'speed', 'balanced', or 'memory'
    quantize=True,                 # Reduce precision for faster inference
    target_device='cpu'            # 'cpu', 'gpu', or 'tpu'
)
```

### Scaling Considerations

For high-throughput environments:

```python
from cyberthreat_ml.scaling import DistributedDetector

# Create a distributed detector
distributed_detector = DistributedDetector(
    model_path='models/threat_detection_model',
    feature_extractor_path='models/feature_extractor',
    num_workers=8,
    queue_size=10000,
    batch_size=256
)

# Start the distributed detector
distributed_detector.start()

# Process packets
for packet in packet_stream:
    distributed_detector.process(packet)

# Get statistics
stats = distributed_detector.get_stats()
print(f"Processed {stats['total_processed']} packets")
print(f"Average processing time: {stats['avg_processing_time']:.2f} ms")
print(f"Current throughput: {stats['current_throughput']:.2f} packets/sec")

# Stop the detector
distributed_detector.stop()
```

## Troubleshooting

### Common Issues

Here are solutions to common issues:

#### Model Performance Issues

- **Problem**: Model has low accuracy or high false positive rate
- **Solution**: Ensure proper feature normalization, use balanced datasets, and tune hyperparameters

```python
# Check class distribution
from collections import Counter
print(Counter(y_train))

# Apply class balancing
from cyberthreat_ml.utils import balance_dataset
X_balanced, y_balanced = balance_dataset(X_train, y_train, method='smote')

# Retrain with balanced data
model.train(X_balanced, y_balanced, epochs=20)
```

#### Memory Issues

- **Problem**: Out of memory errors during training or inference
- **Solution**: Reduce batch size, use memory-efficient model configuration, or implement data generators

```python
# Memory-efficient training
from cyberthreat_ml.utils import DataGenerator

# Create a data generator
generator = DataGenerator(
    X_train, y_train,
    batch_size=32,
    shuffle=True
)

# Train using the generator
model.train_with_generator(
    generator,
    steps_per_epoch=len(generator),
    epochs=20
)
```

#### Real-time Detection Delays

- **Problem**: Detection callbacks are delayed
- **Solution**: Adjust processing interval, increase batch size, or optimize feature extraction

```python
# Optimize detector settings
detector = PacketStreamDetector(
    model,
    feature_extractor,
    batch_size=64,           # Process more packets at once
    processing_interval=0.1, # Process more frequently
    max_queue_size=1000      # Allow more packets to be queued
)

# Use a more efficient feature extractor
class OptimizedFeatureExtractor:
    def transform(self, packet):
        # Implement faster feature extraction
        # Focus only on the most important features
        return np.array([
            packet.get('src_port', 0),
            packet.get('dst_port', 0),
            # Only extract critical features
        ])
```

### Error Messages

Common error messages and their solutions:

#### Model Loading Errors

- **Error**: "Failed to load model: No such file or directory"
- **Solution**: Ensure the model path is correct and the model was saved properly

```python
# Check if model file exists
import os
model_path = 'models/threat_detection_model'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    
    # Train and save a new model
    model = ThreatDetectionModel(input_shape=(25,), num_classes=6)
    model.train(X_train, y_train, epochs=10)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model.save_model(model_path)
    print(f"New model saved to {model_path}")
```

#### Feature Mismatch Errors

- **Error**: "Input shape mismatch: expected (25,) but got (30,)"
- **Solution**: Ensure the feature extractor produces the correct number of features

```python
# Check feature dimensions
print(f"Model input shape: {model.input_shape}")
print(f"Feature extractor output shape: {feature_extractor.transform(sample_data).shape}")

# Fix feature extractor if needed
class FixedFeatureExtractor:
    def transform(self, data):
        features = original_extractor.transform(data)
        # Ensure correct number of features
        if features.shape[0] != model.input_shape[0]:
            # Either trim or pad the features
            if features.shape[0] > model.input_shape[0]:
                return features[:model.input_shape[0]]
            else:
                padded = np.zeros(model.input_shape[0])
                padded[:features.shape[0]] = features
                return padded
        return features
```

#### Callback Errors

- **Error**: "Exception in threat callback: ..."
- **Solution**: Use try-except in callbacks and validate callback parameters

```python
# Robust callback implementation
def on_threat_detected(result):
    try:
        # Validate result structure
        if not isinstance(result, dict) or 'class_idx' not in result:
            print("Invalid result format")
            return
            
        # Process the result
        if result['class_idx'] > 0:
            print(f"Threat detected: {result.get('class_name', 'Unknown')}")
            # Further processing...
    except Exception as e:
        print(f"Error in threat callback: {e}")
        # Log the error, don't crash
```

### Debug Mode

Enable debug mode for more detailed logging:

```python
from cyberthreat_ml.logger import set_log_level, get_logger

# Set debug log level
set_log_level('DEBUG')

# Get a logger
logger = get_logger('cyberthreat_ml')

# Use the logger
logger.debug("Feature extractor initialized")
logger.info("Model loaded successfully")
logger.warning("Using default threshold")
logger.error("Failed to process packet")

# Test components in debug mode
from cyberthreat_ml.utils import debug_test_detector

# Run diagnostics on the detector
report = debug_test_detector(
    detector,
    test_data,
    test_labels,
    verbose=True
)

print(f"Debug test report:")
print(f"  Successful detections: {report['successful_detections']}")
print(f"  Failed detections: {report['failed_detections']}")
print(f"  Average processing time: {report['avg_processing_time']:.4f} ms")
print(f"  Memory usage: {report['memory_usage']:.2f} MB")
```

## API Reference

For complete API details, please refer to the individual module documentation:

- [model.py](../api/model.html) - ML model creation and training
- [preprocessing.py](../api/preprocessing.html) - Data preprocessing
- [realtime.py](../api/realtime.html) - Real-time detection
- [evaluation.py](../api/evaluation.html) - Model evaluation
- [explain.py](../api/explain.html) - Basic model explainability
- [interpretability.py](../api/interpretability.html) - Advanced interpretability
- [visualization.py](../api/visualization.html) - Visualization components
- [text_visualization.py](../api/text_visualization.html) - Text-based visualization
- [anomaly.py](../api/anomaly.html) - Zero-day threat detection
- [utils.py](../api/utils.html) - Utility functions
- [logger.py](../api/logger.html) - Logging functionality