# CyberThreat-ML Quick Reference

This quick reference provides the most common code patterns for using the CyberThreat-ML library.

## Core Components

| Component | Description | Key Classes |
|-----------|-------------|------------|
| **Models** | ML models for threat detection | `ThreatDetectionModel` |
| **Zero-Day Detection** | Anomaly-based detection | `ZeroDayDetector`, `RealTimeZeroDayDetector` |
| **Real-time Detection** | Processing live data | `RealTimeDetector`, `PacketStreamDetector` |
| **Visualization** | Interactive dashboards | `ThreatVisualizationDashboard` |
| **Interpretability** | Explanation of detections | `ThreatInterpreter` |
| **Evaluation** | Model performance metrics | Functions in `evaluation` module |

## Common Code Patterns

### 1. Training a Model

```python
from cyberthreat_ml.model import ThreatDetectionModel

# Create a model
model = ThreatDetectionModel(
    input_shape=(25,),       # Feature dimensions
    num_classes=5,           # Number of threat classes
    model_config={
        'hidden_layers': [64, 32, 16],
        'dropout_rate': 0.3
    }
)

# Train the model
history = model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=10, 
    batch_size=32
)

# Save model
model.save_model('my_model')
```

### 2. Zero-Day Detection

```python
from cyberthreat_ml.anomaly import ZeroDayDetector, get_anomaly_description

# Create detector
detector = ZeroDayDetector(
    method='ensemble',       # Use multiple algorithms
    contamination=0.01       # Expected anomaly rate
)

# Train on normal data only
detector.fit(normal_data, feature_names)

# Detect anomalies
predictions, scores = detector.detect(test_data, return_scores=True)

# Analyze anomalies
for i, pred in enumerate(predictions):
    if pred == -1:  # Anomaly
        analysis = detector.analyze_anomaly(test_data[i], scores[i])
        print(get_anomaly_description(analysis))
```

### 3. Real-time Detection

```python
from cyberthreat_ml.realtime import PacketStreamDetector

# Create detector
detector = PacketStreamDetector(
    model=model,
    feature_extractor=your_extractor,
    threshold=0.7,
    batch_size=32
)

# Register threat callback
def on_threat(result):
    print(f"Threat detected: {result['class_name']}")
    
detector.register_threat_callback(on_threat)

# Start monitoring
detector.start()

# Process packets
detector.process_packet(packet_data)

# Stop when done
detector.stop()
```

### 4. Visualization

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Create dashboard
dashboard = ThreatVisualizationDashboard()

# Start in background
dashboard.start()

# Add threats to display
dashboard.add_threat(threat_result)

# Save snapshot
dashboard.save_snapshot('dashboard.png')

# Stop when done
dashboard.stop()
```

### 5. Interpretability

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

# Visualize
interpreter.plot_explanation(explanation, plot_type="waterfall")
```

### 6. Model Evaluation

```python
from cyberthreat_ml.evaluation import evaluate_model, classification_report
from cyberthreat_ml.evaluation import plot_confusion_matrix, find_optimal_threshold

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Get classification report
report = classification_report(model, X_test, y_test)
print(report)

# Plot confusion matrix
plot_confusion_matrix(model, X_test, y_test)

# Find optimal threshold
optimal_threshold = find_optimal_threshold(model, X_val, y_val, metric='f1')
```

## Module Imports

```python
# Core model functionality
from cyberthreat_ml.model import ThreatDetectionModel, load_model

# Zero-day and anomaly detection
from cyberthreat_ml.anomaly import ZeroDayDetector, RealTimeZeroDayDetector
from cyberthreat_ml.anomaly import get_anomaly_description, recommend_action

# Real-time detection
from cyberthreat_ml.realtime import RealTimeDetector, PacketStreamDetector

# Visualization
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Interpretability
from cyberthreat_ml.interpretability import ThreatInterpreter

# Evaluation
from cyberthreat_ml.evaluation import evaluate_model, classification_report
from cyberthreat_ml.evaluation import plot_confusion_matrix, find_optimal_threshold

# Preprocessing
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.preprocessing import extract_packet_features, extract_flow_features

# Utilities
from cyberthreat_ml.utils import save_dataset, load_dataset, split_data
```

## Configuration Examples

### Model Configuration

```python
model_config = {
    'hidden_layers': [128, 64, 32],  # Architecture
    'dropout_rate': 0.3,             # Regularization
    'activation': 'relu',            # Activation function
    'output_activation': 'softmax',  # For multi-class (use 'sigmoid' for binary)
    'optimizer': 'adam',             # Optimizer
    'loss': 'sparse_categorical_crossentropy',  # Loss function
    'metrics': ['accuracy'],         # Metrics to track
    'class_names': ["Normal", "Attack"]  # Optional class names
}

model = ThreatDetectionModel(
    input_shape=(25,),
    num_classes=2,
    model_config=model_config
)
```

### Zero-Day Detector Configuration

```python
# For high-dimensional data with moderate false positive tolerance
detector = ZeroDayDetector(
    method='isolation_forest',
    contamination=0.05,
    min_samples=100
)

# For low false positive requirements
detector = ZeroDayDetector(
    method='ensemble',
    contamination=0.01,
    min_samples=500
)

# For fast detection with some performance tradeoff
detector = ZeroDayDetector(
    method='one_class_svm',
    contamination=0.03
)
```

### Real-time Detector Configuration

```python
# High-throughput configuration
detector = RealTimeDetector(
    model=model,
    feature_extractor=extractor,
    threshold=0.7,
    batch_size=64,
    processing_interval=0.5
)

# Low-latency configuration
detector = RealTimeDetector(
    model=model,
    feature_extractor=extractor,
    threshold=0.6,
    batch_size=1,
    processing_interval=0.1
)
```

## Common Workflows

### Hybrid Detection (Signature + Zero-Day)

```python
# First check with signature-based detection
result = signature_model.predict(data)

# If not detected, try zero-day detection
if result == 0:  # No known threat
    anomaly_result = zero_day_detector.detect(data)
    if anomaly_result[0] == -1:  # Anomaly found
        # Process zero-day threat
```

### Adaptive Learning

```python
# Periodically retrain on recent normal data
if time.time() - last_training > 86400:  # Daily retraining
    normal_samples = collect_recent_normal_data()
    if len(normal_samples) >= 1000:
        zero_day_detector.fit(normal_samples)
        last_training = time.time()
```