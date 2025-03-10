# CyberThreat-ML

A Python library for real-time cybersecurity threat detection using TensorFlow.

## Overview

CyberThreat-ML is a comprehensive library for building, training, and deploying machine learning models to detect cyber threats in real-time. It provides tools for processing network traffic data, training models, evaluating performance, explaining predictions, visualizing threats, and deploying models in real-time detection systems.

## Features

- **Advanced Threat Detection Model**: Sophisticated neural network architecture for detecting various cyber threats
- **Multi-class Classification**: Support for detecting multiple threat categories:
  - Normal Traffic
  - Port Scan
  - DDoS (Distributed Denial of Service)
  - Brute Force Attacks
  - Data Exfiltration
  - Command & Control Communication
  - IoT-specific threats (Botnet Activity, Firmware Tampering, Replay Attacks)
- **Zero-Day Attack Detection**: Anomaly-based detection capable of identifying previously unknown threats
- **Hybrid Detection Model**: Combines signature-based and anomaly-based approaches for comprehensive protection
- **Real-time Processing**: Stream-based detection with support for batch processing
- **Explainable AI**: Integrated interpretability features using SHAP to explain model decisions
- **Interactive Visualization**: Dashboard for real-time threat visualization
- **Extensive Analytics**: Tools for analyzing threat patterns and extracting insights
- **Adaptive Learning**: Continuous model improvement based on recent traffic patterns
- **Easy Integration**: API for integration with existing security infrastructure
- **Persistence**: Model saving and loading functionality

## Installation

```bash
pip install cyberthreat-ml
```

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy <2.0.0
- Pandas
- Scikit-learn
- Matplotlib (for visualizations)
- SHAP (optional, for model explainability)
- Seaborn (optional, for enhanced visualizations)

## Quick Start

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.realtime import PacketStreamDetector

# Create or load a model
model = ThreatDetectionModel(input_shape=(25,), num_classes=6)
model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=10)

# Initialize a feature extractor
feature_extractor = FeatureExtractor()

# Set up real-time detection
detector = PacketStreamDetector(model, feature_extractor)

# Register callback for threat detection
def on_threat_detected(result):
    print(f"🚨 THREAT DETECTED! - {result['class_name']} 🚨")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Suggested action: {result['action']}")

detector.register_threat_callback(on_threat_detected)

# Start the detector
detector.start()

# Process a packet
detector.process_packet(packet_data)
```

## Module Documentation

### Model Module

The core of the library, responsible for creating, training, and using threat detection models.

```python
from cyberthreat_ml.model import ThreatDetectionModel, load_model

# Create a new model
model = ThreatDetectionModel(
    input_shape=(25,),  # Feature vector dimension
    num_classes=6,      # Number of threat classes
    model_config={      # Optional architecture configuration
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'activation': 'relu'
    }
)

# Train the model
history = model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=15,
    batch_size=32,
    early_stopping=True
)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Save the model
model.save_model('models/threat_model')

# Load a saved model
loaded_model = load_model('models/threat_model')
```

### Real-time Detection

Process data streams and detect threats in real-time.

```python
from cyberthreat_ml.realtime import RealTimeDetector, PacketStreamDetector

# Initialize the detector
detector = PacketStreamDetector(model, feature_extractor)

# Register callbacks
detector.register_threat_callback(on_threat_detected)
detector.register_processing_callback(on_batch_processed)

# Start the detection process
detector.start()

# Process network packets
detector.process_packet(packet_data)

# Get statistics
stats = detector.get_stats()

# Stop the detector when done
detector.stop()
```

### Visualization and Interpretability

Visualize threats and explain model decisions.

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter

# Set up a visualization dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Add detected threats to the dashboard
dashboard.add_threat(threat_data)

# Initialize a threat interpreter
interpreter = ThreatInterpreter(model, feature_names, class_names)
interpreter.initialize(background_data)

# Explain a prediction
explanation = interpreter.explain_prediction(
    input_data,
    method="shap",
    target_class=3,
    top_features=5
)

# Visualize the explanation
interpreter.plot_explanation(explanation, plot_type="bar")

# Generate a report
report = interpreter.create_feature_importance_report(explanation)
```

### Evaluation

Evaluate model performance with various metrics and visualizations.

```python
from cyberthreat_ml.evaluation import (
    evaluate_model, 
    classification_report, 
    plot_confusion_matrix,
    plot_roc_curve,
    find_optimal_threshold
)

# Evaluate the model
metrics = evaluate_model(model, X_test, y_test)

# Generate a classification report
report = classification_report(model, X_test, y_test)

# Plot a confusion matrix
fig = plot_confusion_matrix(model, X_test, y_test, normalize=True)

# Find the optimal decision threshold
threshold = find_optimal_threshold(model, X_val, y_val, metric='f1')
```

## Example Use Cases

- **Network Security Monitoring**: Detect anomalous patterns in network traffic
- **Intrusion Detection Systems**: Add ML capabilities to existing IDS solutions
- **Security Operations Centers**: Provide real-time threat detection and visualization
- **Threat Hunting**: Use explanations to understand attack patterns
- **Forensic Analysis**: Analyze past incidents with interpretability features

## Example Scripts

The library includes several example scripts demonstrating different capabilities:

- `examples/basic_usage.py`: Basic binary classification of threats
- `examples/multiclass_classification.py`: Multi-class threat classification
- `examples/realtime_detection.py`: Real-time detection of threats in a packet stream
- `examples/visualization_interpretability.py`: Visualization and explainability features
- `examples/integrated_analytics.py`: Integrated analytics for comprehensive threat analysis
- `examples/enterprise_security.py`: Enterprise security integration example
- `examples/iot_security.py`: IoT device security monitoring and threat detection
- `examples/zero_day_detection.py`: Zero-day attack detection using anomaly-based methods

For more information on IoT security capabilities, see the [IoT_SECURITY.md](IOT_SECURITY.md) documentation.  
For more information on zero-day detection capabilities, see the [ZERO_DAY_DETECTION.md](ZERO_DAY_DETECTION.md) documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.