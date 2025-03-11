# CyberML101: Getting Started with CyberThreat-ML

Welcome to CyberML101! This guide will introduce you to the CyberThreat-ML library, a powerful Python toolkit for real-time cybersecurity threat detection using machine learning.

## What is CyberThreat-ML?

CyberThreat-ML is a specialized library designed to help security professionals:
- Detect known cybersecurity threats using signature-based methods
- Identify unknown (zero-day) threats through anomaly detection
- Process and analyze network traffic in real-time
- Visualize and interpret detection results
- Generate actionable intelligence from threat detections

## Installation

```bash
pip install cyberthreat-ml
```

## Quick Start Example

Let's start with a simple example to detect threats in network traffic:

```python
import numpy as np
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.evaluation import evaluate_model
from cyberthreat_ml.realtime import PacketStreamDetector

# 1. Create and train a threat detection model
model = ThreatDetectionModel(
    input_shape=(10,),  # 10 features
    num_classes=2       # Binary classification (normal vs threat)
)

# Train with your data
X_train = your_feature_data  # Replace with your data
y_train = your_labels        # Replace with your labels
model.train(X_train, y_train, epochs=10)

# 2. Set up real-time detection
class SimpleFeatureExtractor:
    def transform(self, packet):
        # Extract features from network packet
        # This is a simplified example
        return np.array([
            packet.get('size', 0) / 10000.0,
            packet.get('src_port', 0) / 65535.0,
            packet.get('dst_port', 0) / 65535.0,
            # ... other features
        ]).reshape(1, -1)

# Create detector
detector = PacketStreamDetector(
    model=model,
    feature_extractor=SimpleFeatureExtractor(),
    threshold=0.7
)

# 3. Register callback for threats
def on_threat(result):
    print(f"Threat detected! Score: {result['threat_score']}")
    print(f"Threat details: {result['data']}")

detector.register_threat_callback(on_threat)

# 4. Start monitoring
detector.start()

# 5. Process packets as they arrive
sample_packet = {
    'size': 1240,
    'src_port': 55123,
    'dst_port': 443,
    'protocol': 6,  # TCP
    # ... other packet data
}
detector.process_packet(sample_packet)

# When finished
detector.stop()
```

## Core Components

### 1. Models (ThreatDetectionModel)

The `ThreatDetectionModel` class is the core ML engine that powers threat detection:

```python
from cyberthreat_ml.model import ThreatDetectionModel

# Create a multi-class model
model = ThreatDetectionModel(
    input_shape=(25,),       # 25 features
    num_classes=5,           # 5 different threat classes
    model_config={
        'hidden_layers': [64, 32, 16],  # Architecture
        'dropout_rate': 0.3,            # Regularization
        'activation': 'relu',           # Activation function
        'class_names': [                # Class names
            "Normal Traffic",
            "Port Scan",
            "DDoS",
            "Brute Force",
            "Data Exfiltration"
        ]
    }
)
```

### 2. Zero-Day Detection (ZeroDayDetector)

For detecting previously unknown threats:

```python
from cyberthreat_ml.anomaly import ZeroDayDetector, get_anomaly_description

# Create detector
detector = ZeroDayDetector(
    method='ensemble',       # Use multiple methods
    contamination=0.01       # Expected % of anomalies
)

# Train on normal data only
detector.fit(normal_traffic_data)

# Detect anomalies
predictions, scores = detector.detect(new_traffic, return_scores=True)

# Analyze an anomaly
for i, pred in enumerate(predictions):
    if pred == -1:  # Anomaly detected
        analysis = detector.analyze_anomaly(new_traffic[i], scores[i])
        print(get_anomaly_description(analysis))
        print(f"Severity: {analysis['severity_level']}")
```

### 3. Real-Time Detection (RealTimeDetector)

For processing live network traffic:

```python
from cyberthreat_ml.realtime import RealTimeDetector

# Create real-time detector
detector = RealTimeDetector(
    model=model,
    feature_extractor=your_extractor,
    threshold=0.8,
    batch_size=32,
    processing_interval=1.0  # Process every second
)

# Register callbacks
detector.register_threat_callback(on_threat_detected)
detector.register_processing_callback(on_batch_processed)

# Start and stop
detector.start()
# ... your application logic
detector.stop()
```

### 4. Visualization (ThreatVisualizationDashboard)

For visualizing detection results:

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Create dashboard
dashboard = ThreatVisualizationDashboard(max_history=500)

# Add threat when detected
def on_threat(result):
    dashboard.add_threat(result)
    
# Start dashboard
dashboard.start()
```

### 5. Interpretability (ThreatInterpreter)

For explaining model predictions:

```python
from cyberthreat_ml.interpretability import ThreatInterpreter

# Create interpreter
interpreter = ThreatInterpreter(
    model=model,
    feature_names=feature_names,
    class_names=class_names
)

# Initialize with background data
interpreter.initialize(background_data)

# Explain a prediction
explanation = interpreter.explain_prediction(
    input_data=sample,
    method="shap",
    top_features=5
)

# Visualize explanation
interpreter.plot_explanation(explanation, plot_type="waterfall")
```

## Common Use Cases

### 1. Network Security Monitoring

Implement real-time monitoring of network traffic:

```python
# Process packets as they arrive
def process_network_traffic(packet):
    detector.process_packet(packet)
    
# Set up packet capture (using your preferred packet capture library)
capture_packets(callback=process_network_traffic)
```

### 2. Threat Hunting

Analyze captured traffic to hunt for potential threats:

```python
# Load captured traffic
captured_traffic = load_pcap_file("captured_traffic.pcap")

# Process each packet
for packet in captured_traffic:
    features = feature_extractor.transform(packet)
    anomaly_predictions = zero_day_detector.detect(features)
    
    # Check for anomalies
    if anomaly_predictions[0] == -1:
        print(f"Potential threat found in packet {packet['id']}")
```

### 3. Security Operations Center (SOC) Dashboard

Implement a SOC dashboard for monitoring:

```python
# Create and start dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Register threat callback
def on_threat(result):
    # Add to dashboard
    dashboard.add_threat(result)
    
    # Alert SOC analyst
    send_alert(result)
    
detector.register_threat_callback(on_threat)
```

## Advanced Topics

### 1. Hybrid Detection (Combining Signature and Anomaly-Based Detection)

For comprehensive coverage, combine both detection methods:

```python
# Get predictions from signature-based model
signature_predictions = signature_model.predict(data)

# Get predictions from anomaly detection
anomaly_predictions, _ = zero_day_detector.detect(data, return_scores=True)
# Convert predictions: 1 = normal, -1 = anomaly -> 0 = normal, 1 = anomaly
anomaly_predictions = (anomaly_predictions == -1).astype(int)

# Combine detections (threat if either method detects it)
hybrid_predictions = np.logical_or(signature_predictions, anomaly_predictions).astype(int)
```

### 2. Adaptive Learning

Continuously improve detection by retraining on recent data:

```python
# With the real-time zero-day detector
realtime_detector.train_on_recent_normal(min_samples=100)

# Or manually with new normal data
new_normal_data = collect_recent_normal_data()
zero_day_detector.fit(new_normal_data)
```

### 3. Temporal Analysis

Analyze time-based patterns in network traffic:

```python
# Get temporal analysis from real-time detector
time_patterns = realtime_detector.analyze_time_patterns()

print(f"Sample rate: {time_patterns['sample_rate_per_second']} packets/sec")
print(f"Anomaly rate: {time_patterns['anomaly_rate']}")
```

## Best Practices

1. **Start with Quality Data**: Ensure your training data properly represents normal traffic and known threats

2. **Tune Contamination Rate**: Adjust the contamination parameter (expected % of anomalies) based on your environment:
   ```python
   # Higher value = more sensitivity but more false positives
   detector = ZeroDayDetector(contamination=0.05)  # 5% expected anomalies
   ```

3. **Periodically Retrain**: Network traffic patterns evolve - retrain your models regularly

4. **Verify Alerts**: Always have human verification of critical alerts

5. **Layered Approach**: Use multiple detection methods for comprehensive coverage

## Next Steps

After becoming familiar with the basics, explore the following:

1. **Custom Feature Extraction**: Create specialized feature extractors for your network traffic
2. **Model Tuning**: Optimize models for your specific environment
3. **Integration**: Integrate with your existing security infrastructure
4. **Deployment Automation**: Set up automated deployment and monitoring

## Resources

- Example Scripts: See the `examples/` directory
- API Documentation: Detailed docstrings in the source code
- Visualization Tools: Explore the visualization module for creating dashboards

Happy threat hunting!