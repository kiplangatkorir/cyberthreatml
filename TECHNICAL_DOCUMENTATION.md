# CyberThreat-ML Technical Documentation

## Architecture Overview

CyberThreat-ML is built with a modular architecture that separates concerns into distinct components, allowing for flexible configuration and extension. The library consists of the following core modules:

1. **Model Module** (`cyberthreat_ml.model`): Defines the neural network architecture and provides training and inference capabilities.
2. **Preprocessing Module** (`cyberthreat_ml.preprocessing`): Handles data cleaning, normalization, and feature extraction.
3. **Evaluation Module** (`cyberthreat_ml.evaluation`): Provides metrics and visualizations for model evaluation.
4. **Explain Module** (`cyberthreat_ml.explain`): Implements model explainability using SHAP.
5. **Interpretability Module** (`cyberthreat_ml.interpretability`): Extends explainability with more advanced interpretation capabilities.
6. **Realtime Module** (`cyberthreat_ml.realtime`): Implements real-time threat detection capabilities.
7. **Visualization Module** (`cyberthreat_ml.visualization`): Provides dashboards and visualizations for threat monitoring.
8. **Utils Module** (`cyberthreat_ml.utils`): Contains utility functions used across the library.

## Model Architecture

The default model architecture is a multilayer perceptron (MLP) with configurable hidden layers, dropout for regularization, and either sigmoid (binary) or softmax (multi-class) output activation.

Default configurations:
- Binary classification: `[64, 32]` neurons in hidden layers
- Multi-class classification: `[128, 64, 32]` neurons in hidden layers
- Dropout rate: 0.3
- Hidden layer activation: ReLU
- Output activation: Sigmoid (binary) or Softmax (multi-class)
- Loss function: Binary cross-entropy (binary) or Categorical cross-entropy (multi-class)
- Optimizer: Adam with learning rate 0.001

### Sample Model Configuration

```python
model_config = {
    'hidden_layers': [128, 64, 32],  # Number of neurons in each hidden layer
    'dropout_rate': 0.3,             # Dropout rate for regularization
    'activation': 'relu',            # Activation function for hidden layers
    'learning_rate': 0.001,          # Learning rate for Adam optimizer
    'l2_regularization': 0.01        # L2 regularization factor
}

model = ThreatDetectionModel(input_shape=(25,), num_classes=6, model_config=model_config)
```

## Threat Categories

The library supports the following threat categories in multi-class mode:

1. **Normal Traffic (Class 0)**: Benign network traffic with no malicious intent.
2. **Port Scan (Class 1)**: Attempts to discover open ports or services on a network.
3. **DDoS (Class 2)**: Distributed Denial of Service attacks aimed at overwhelming services.
4. **Brute Force (Class 3)**: Repeated attempts to guess passwords or encryption keys.
5. **Data Exfiltration (Class 4)**: Unauthorized data transfer from a system.
6. **Command & Control (Class 5)**: Communication between compromised systems and control servers.

## Feature Engineering

### Network Traffic Features

For network traffic analysis, the library extracts and uses the following features:

1. **Header Features**:
   - Source/destination IP addresses
   - Source/destination ports
   - Protocol (TCP, UDP, ICMP, etc.)
   - Packet length
   - TCP flags
   - Time-to-live (TTL)

2. **Flow Features**:
   - Flow duration
   - Number of packets in flow
   - Bytes transferred
   - Packet rate
   - Byte rate
   - Flow direction

3. **Payload Features**:
   - Payload entropy (measure of randomness)
   - Byte frequency distribution
   - Presence of specific patterns or signatures
   - Encrypted payload detection

### Feature Normalization

Features are normalized using one of the following methods:

- **Standard Scaling**: Zero mean and unit variance
- **MinMax Scaling**: Scaled to range [0, 1]
- **Custom Normalization**: Domain-specific normalization functions for certain features

## Real-time Detection

The real-time detection system operates in the following way:

1. **Data Ingestion**: Raw data (e.g., network packets) are ingested into the system.
2. **Feature Extraction**: Features are extracted from the raw data.
3. **Preprocessing**: Features are normalized and prepared for model input.
4. **Prediction**: The model makes predictions on the processed features.
5. **Threat Detection**: Threats are identified based on prediction scores and thresholds.
6. **Notification**: Callbacks are triggered for detected threats.

### Detection Workflow

```
Raw Data → Feature Extraction → Preprocessing → Model Prediction → Threat Detection → Notification
```

### Thread Safety

The real-time components are designed to be thread-safe, using queue-based processing and thread synchronization to ensure reliable operation in multi-threaded environments.

## Explainability and Interpretability

The library provides multiple approaches to explain model predictions:

1. **SHAP (SHapley Additive exPlanations)**: Explains individual predictions by showing each feature's contribution.
2. **LIME (Local Interpretable Model-agnostic Explanations)**: Approximates the model locally with an interpretable model.
3. **Rules-based Explanations**: Simple statistical explanations based on feature values.

### Interpreting SHAP Values

- **Positive SHAP values**: Feature pushes the prediction higher
- **Negative SHAP values**: Feature pushes the prediction lower
- **Magnitude**: Indicates the strength of the feature's influence

## Visualization Capabilities

The visualization module offers:

1. **Real-time Dashboard**: Animated visualization of threats as they are detected.
2. **Threat Timeline**: Chronological view of detected threats.
3. **Distribution Charts**: Distribution of threat types and confidence scores.
4. **Heatmaps**: Visual representation of threat intensity.
5. **Explanation Plots**: Visualizations of feature importance and model explanations.

## Advanced Usage Examples

### Custom Feature Extraction

```python
from cyberthreat_ml.preprocessing import FeatureExtractor

# Define feature categories
categorical_features = ['protocol', 'port_category', 'direction']
numeric_features = ['packet_size', 'packet_count', 'duration', 'byte_rate']
ip_features = ['source_ip', 'destination_ip']

# Create a custom feature extractor
extractor = FeatureExtractor(
    categorical_features=categorical_features,
    numeric_features=numeric_features,
    ip_features=ip_features,
    scaling='standard',
    handle_missing=True
)

# Fit and transform data
X_transformed = extractor.fit_transform(raw_data)
```

### Custom Threat Categorization

```python
from cyberthreat_ml.model import ThreatDetectionModel

# Define your own threat classes
threat_classes = [
    'Benign',
    'Reconnaissance',
    'DoS',
    'Credential Stuffing',
    'Data Theft',
    'Botnet'
]

# Create a model with custom classes
model = ThreatDetectionModel(
    input_shape=(feature_count,),
    num_classes=len(threat_classes)
)

# Use class names in callbacks
def on_threat(result):
    class_idx = result['class_idx']
    class_name = threat_classes[class_idx]
    print(f"Detected {class_name} threat with confidence {result['confidence']}")
```

### Integration with Security Infrastructure

```python
from cyberthreat_ml.realtime import PacketStreamDetector
import syslog

# Initialize detector with your trained model
detector = PacketStreamDetector(model, feature_extractor)

# Set up integration with SIEM system
def alert_security_team(result):
    threat_info = f"SECURITY ALERT: {result['class_name']} detected from {result['source_ip']}"
    syslog.syslog(syslog.LOG_ALERT, threat_info)
    
    # Additional actions like sending to a SIEM system
    send_to_siem(threat_info, result)

# Register the callback
detector.register_threat_callback(alert_security_team)
```

## Performance Optimization

### Model Optimization

1. **Batch Processing**: Process data in batches for improved throughput.
2. **Early Stopping**: Avoid overfitting and reduce training time.
3. **Model Quantization**: Reduce model size and inference time (experimental).

### Memory Management

1. **Stream Processing**: Process data as it arrives to avoid storing large datasets.
2. **Data Pruning**: Maintain only necessary historical data for analysis.
3. **Efficient Feature Representation**: Use optimized data structures for feature storage.

## Error Handling and Logging

The library includes a comprehensive logging system that can be configured with different verbosity levels:

```python
from cyberthreat_ml.logger import CyberThreatLogger

# Create a logger with file output
logger = CyberThreatLogger(
    name="my_cyberthreat_app",
    log_level=logging.INFO,
    log_to_file=True
).get_logger()

# Use the logger
logger.info("Starting threat detection system")
logger.warning("Suspicious activity detected")
logger.error("Failed to process packet data")
```

## Extending the Library

### Creating Custom Models

You can create custom model architectures by subclassing the ThreatDetectionModel:

```python
class CustomThreatModel(ThreatDetectionModel):
    def _build_model(self):
        # Define your custom TensorFlow model architecture here
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
        model.add(tf.keras.layers.GlobalMaxPooling1D())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
```

### Custom Threat Actions

You can define custom actions for different threat types:

```python
def generate_threat_action(result):
    threat_class = result['class_name']
    confidence = result['confidence']
    source_ip = result['source_ip']
    
    actions = {
        'Port Scan': f"BLOCK {source_ip} and ADD TO WATCHLIST",
        'DDoS': "ACTIVATE TRAFFIC SCRUBBING and CONTACT UPSTREAM PROVIDER",
        'Brute Force': f"IMPLEMENT PROGRESSIVE RATE LIMITING for {source_ip}",
        'Data Exfiltration': "ISOLATE AFFECTED SERVER and CAPTURE FULL PACKET DATA",
        'Command & Control': "QUARANTINE INFECTED HOST and COLLECT MEMORY DUMP"
    }
    
    default_action = "ALERT SECURITY TEAM for further investigation"
    return actions.get(threat_class, default_action)
```

## Best Practices

1. **Model Monitoring**: Regularly evaluate model performance against new data.
2. **Retraining Schedule**: Plan periodic model retraining to adapt to evolving threats.
3. **Threat Intelligence Integration**: Incorporate external threat intelligence for enhanced detection.
4. **Data Quality Management**: Ensure high-quality data for training and detection.
5. **Scalability Planning**: Design deployment for horizontal scaling under increased load.
6. **Security Considerations**: Protect the ML system itself from attacks and manipulation.

## Troubleshooting

### Common Issues and Solutions

1. **High False Positive Rate**
   - Solution: Adjust detection thresholds
   - Solution: Retrain model with more balanced data
   - Solution: Add more features to improve discrimination

2. **Model Overfitting**
   - Solution: Increase dropout rate
   - Solution: Add L2 regularization
   - Solution: Reduce model complexity

3. **Slow Detection Speed**
   - Solution: Increase batch size
   - Solution: Reduce feature dimensionality
   - Solution: Optimize preprocessing pipeline

4. **Integration Issues**
   - Solution: Check API compatibility
   - Solution: Verify data format consistency
   - Solution: Ensure proper error handling

## Version Compatibility

| CyberThreat-ML | Python    | TensorFlow | NumPy      |
|----------------|-----------|------------|------------|
| 1.0.x          | 3.8 - 3.11 | 2.8 - 2.12 | <2.0.0     |

## Future Development

Planned future enhancements include:

- Advanced deep learning architectures (LSTM, transformer-based models)
- Unsupervised anomaly detection capabilities
- Federated learning for collaborative threat detection
- Integration with cloud security platforms
- Reinforcement learning for adaptive defense mechanisms

---

## Appendix: API Reference

Full method signatures for key library functions:

### ThreatDetectionModel

```python
def __init__(self, input_shape, num_classes=2, model_config=None)
def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, early_stopping=True, early_stopping_patience=3, checkpoint_path=None)
def predict(self, X, threshold=0.5)
def predict_proba(self, X)
def save_model(self, model_path, metadata_path=None)
```

### RealTimeDetector

```python
def __init__(self, model, feature_extractor=None, threshold=0.5, batch_size=32, processing_interval=1.0)
def start(self)
def stop(self)
def add_data(self, data)
def register_threat_callback(self, callback)
def register_processing_callback(self, callback)
```

### ThreatInterpreter

```python
def __init__(self, model, feature_names=None, class_names=None)
def initialize(self, background_data)
def explain_prediction(self, input_data, method="auto", target_class=None, top_features=5)
def plot_explanation(self, explanation, plot_type="bar", save_path=None)
def create_feature_importance_report(self, explanation, output_path=None)
```

### ThreatVisualizationDashboard

```python
def __init__(self, max_history=1000, update_interval=1.0)
def start(self)
def stop(self)
def add_threat(self, threat_data)
def save_snapshot(self, filename=None)
```