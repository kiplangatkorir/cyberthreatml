# CyberThreat-ML Documentation

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
   - [Utilities Module](#utilities-module)
4. [Advanced Usage](#advanced-usage)
   - [Multi-class Classification](#multi-class-classification)
   - [Custom Models](#custom-models)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Feature Importance Analysis](#feature-importance-analysis)
5. [Example Workflows](#example-workflows)
   - [Basic Threat Detection](#basic-threat-detection)
   - [Multi-class Threat Classification](#multi-class-threat-classification)
   - [Real-time Detection](#real-time-detection)
   - [Visualization and Interpretability](#visualization-and-interpretability)
   - [Integrated Analytics](#integrated-analytics)
6. [Best Practices](#best-practices)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## Introduction

CyberThreat-ML is a Python library designed to provide machine learning-based cybersecurity threat detection capabilities. Built on TensorFlow, it offers a comprehensive set of tools for creating, training, evaluating, and deploying threat detection models.

The library supports multiple threat categories including:
- Normal Traffic (benign network traffic)
- Port Scan (systematic probing of network ports)
- DDoS (Distributed Denial of Service attacks)
- Brute Force (password guessing attacks)
- Data Exfiltration (unauthorized data removal)
- Command & Control (communication with malicious command servers)

With integrated real-time detection, model explainability, and visualization capabilities, CyberThreat-ML provides security professionals with powerful tools to identify and understand cyber threats.

## Architecture Overview

CyberThreat-ML follows a modular architecture with the following major components:

1. **Core Model**: TensorFlow-based neural network for threat classification
2. **Feature Processing**: Tools for extracting and transforming features from network data
3. **Real-Time Detection**: Stream-based detection engine for immediate threat identification
4. **Explainability Layer**: SHAP-based interpretation of model decisions
5. **Visualization Components**: Dashboard for real-time monitoring and analysis
6. **Analytics Layer**: Tools for extracting insights from detected threats

The library is designed to be extensible, allowing users to customize each component according to their specific requirements.

## Core Modules

### Model Module

The `model` module provides the core functionality for creating, training, and using threat detection models.

**Key Classes:**

- `ThreatDetectionModel`: The main model class with methods for training and prediction
- `load_model`: Function to load a saved model from disk

**Example:**

```python
from cyberthreat_ml.model import ThreatDetectionModel, load_model

# Create a new model
model = ThreatDetectionModel(
    input_shape=(20,),
    num_classes=2,  # Binary classification
    model_config={
        'hidden_layers': [64, 32],
        'dropout_rate': 0.2
    }
)

# Train the model
model.train(X_train, y_train, epochs=10, batch_size=32)

# Save and load the model
model.save_model('models/my_model')
loaded_model = load_model('models/my_model')
```

### Preprocessing Module

The `preprocessing` module contains tools for extracting and transforming features from network data.

**Key Classes:**

- `FeatureExtractor`: Class for preprocessing network data features
- `IPAddressTransformer`: Transformer for IP address features

**Example:**

```python
from cyberthreat_ml.preprocessing import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor(
    categorical_features=['protocol', 'flag'],
    numeric_features=['packet_size', 'duration'],
    ip_features=['src_ip', 'dst_ip'],
    scaling='standard'
)

# Fit and transform data
X_transformed = extractor.fit_transform(raw_data)
```

### Real-time Detection Module

The `realtime` module provides tools for real-time threat detection.

**Key Classes:**

- `RealTimeDetector`: Base class for real-time detection
- `PacketStreamDetector`: Specialized detector for network packet streams

**Example:**

```python
from cyberthreat_ml.realtime import PacketStreamDetector

# Initialize detector
detector = PacketStreamDetector(
    model,
    feature_extractor,
    threshold=0.7,
    batch_size=32
)

# Register callbacks
detector.register_threat_callback(on_threat_detected)

# Start detection
detector.start()

# Process packets
detector.process_packet(packet_data)
```

### Evaluation Module

The `evaluation` module contains functions for evaluating model performance.

**Key Functions:**

- `evaluate_model`: Calculate performance metrics
- `classification_report`: Generate a classification report
- `plot_confusion_matrix`: Create a confusion matrix visualization
- `find_optimal_threshold`: Find the optimal decision threshold

**Example:**

```python
from cyberthreat_ml.evaluation import evaluate_model, plot_confusion_matrix

# Evaluate the model
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC AUC: {metrics['roc_auc']:.4f}")

# Plot confusion matrix
plot_confusion_matrix(model, X_test, y_test, normalize=True)
```

### Explainability Module

The `explain` module provides SHAP-based explainability tools.

**Key Functions:**

- `explain_prediction`: Explain a single prediction
- `explain_model`: Create a SHAP explainer for the model
- `plot_shap_summary`: Create a SHAP summary plot

**Example:**

```python
from cyberthreat_ml.explain import explain_model, plot_shap_summary

# Create explainer and calculate SHAP values
explainer, shap_values = explain_model(
    model,
    X_background=X_train[:100],  # Background data
    X_explain=X_test,            # Data to explain
    feature_names=feature_names
)

# Plot summary
plot_shap_summary(shap_values, feature_names)
```

### Visualization Module

The `visualization` module provides tools for visualizing threats.

**Key Classes:**

- `ThreatVisualizationDashboard`: Interactive dashboard for threat visualization

**Example:**

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Create dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Add threats to the dashboard
dashboard.add_threat(threat_data)

# Save a snapshot
dashboard.save_snapshot('dashboard_snapshot.png')
```

### Interpretability Module

The `interpretability` module provides advanced interpretability features.

**Key Classes:**

- `ThreatInterpreter`: Class for interpreting and explaining threat detections

**Example:**

```python
from cyberthreat_ml.interpretability import ThreatInterpreter

# Initialize interpreter
interpreter = ThreatInterpreter(
    model,
    feature_names=feature_names,
    class_names=class_names
)

# Initialize with background data
interpreter.initialize(X_train[:100])

# Explain a prediction
explanation = interpreter.explain_prediction(
    input_data,
    method="shap",
    target_class=2,
    top_features=5
)

# Plot the explanation
interpreter.plot_explanation(explanation, plot_type="bar")
```

### Utilities Module

The `utils` module provides utility functions for dataset management and feature normalization.

**Key Functions:**

- `save_dataset`: Save a dataset to disk
- `load_dataset`: Load a dataset from disk
- `split_data`: Split data into training, validation, and test sets

**Example:**

```python
from cyberthreat_ml.utils import split_data, save_dataset

# Split the data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y,
    test_size=0.2,
    val_size=0.25
)

# Save the dataset
save_dataset(X, y, 'data/my_dataset.npz', metadata={'description': 'My dataset'})
```

## Advanced Usage

### Multi-class Classification

CyberThreat-ML supports multi-class classification for identifying various types of threats:

```python
# Create a multi-class model
model = ThreatDetectionModel(
    input_shape=(25,),
    num_classes=6,  # Multiple threat classes
    model_config={
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3
    }
)

# Train with multi-class data
model.train(X_train, y_train_multiclass, epochs=15)

# Make multi-class predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Custom Models

You can customize the model architecture through the `model_config` parameter:

```python
model = ThreatDetectionModel(
    input_shape=(30,),
    num_classes=2,
    model_config={
        'hidden_layers': [256, 128, 64],  # Custom layer sizes
        'dropout_rate': 0.4,              # Custom dropout rate
        'activation': 'elu',              # Custom activation function
        'learning_rate': 0.001,           # Custom learning rate
        'l2_regularization': 0.01         # Add L2 regularization
    }
)
```

### Hyperparameter Tuning

While the library doesn't include built-in hyperparameter tuning, it can be combined with libraries like `keras-tuner`:

```python
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from cyberthreat_ml.preprocessing import FeatureExtractor

# Define the model-building function
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        hp.Int('units', min_value=32, max_value=256, step=32),
        activation=hp.Choice('activation', ['relu', 'tanh', 'elu']),
        input_shape=(20,)
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create a tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='hyperparameter_tuning',
    project_name='threat_detection'
)

# Search for optimal hyperparameters
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
```

### Feature Importance Analysis

Use the interpretability module to analyze feature importance across multiple threats:

```python
# Get insights for a specific threat class
insights = get_threat_pattern_insights(
    interpreter,
    samples=threat_samples,
    threat_class_id=2,  # DDoS
    top_features=10,
    method="shap"
)

# Print top features for this threat
print("Top features for DDoS detection:")
for feature, importance in insights['top_features']:
    print(f"  {feature}: {importance:.4f}")
```

## Example Workflows

### Basic Threat Detection

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.utils import split_data
from cyberthreat_ml.evaluation import evaluate_model, plot_confusion_matrix

# Create synthetic dataset or load your own data
# ...

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Create and train model
model = ThreatDetectionModel(input_shape=(X_train.shape[1],), num_classes=2)
model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=10)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")

# Create visualizations
plot_confusion_matrix(model, X_test, y_test)

# Save model
model.save_model('models/threat_detection_model')
```

### Multi-class Threat Classification

```python
from cyberthreat_ml.model import ThreatDetectionModel
import numpy as np
import matplotlib.pyplot as plt

# Create or load multi-class data
# ...

# Create and train multi-class model
model = ThreatDetectionModel(input_shape=(X_train.shape[1],), num_classes=6)
history = model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=15)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('training_history.png')

# Make detailed predictions
samples = X_test[:5]
true_classes = np.argmax(y_test[:5], axis=1)
predictions = model.predict(samples)
probabilities = model.predict_proba(samples)

class_names = ['Normal', 'Port Scan', 'DDoS', 'Brute Force', 'Data Exfiltration', 'C&C']

for i, (true, pred, probs) in enumerate(zip(true_classes, predictions, probabilities)):
    print(f"Sample {i+1}:")
    print(f"  True class: {class_names[true]}")
    print(f"  Predicted class: {class_names[pred]}")
    print("  Class probabilities:")
    for j, prob in enumerate(probs):
        print(f"    {class_names[j]}: {prob:.4f}")
    print()
```

### Real-time Detection

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.realtime import PacketStreamDetector
import time

# Load a trained model
model = load_model('models/threat_detection_model')

# Create feature extractor
feature_extractor = FeatureExtractor()

# Define callbacks
def on_threat_detected(result):
    print(f"🚨 THREAT DETECTED! - {result['class_name']} 🚨")
    print(f"  Timestamp: {result['timestamp']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Class probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"    {cls}: {prob:.4f}")
    print(f"  Suggested action: {result['action']}")

def on_batch_processed(results):
    print(f"Batch processed: {len(results)} packets, {sum(r['is_threat'] for r in results)} threats detected")

# Initialize detector
detector = PacketStreamDetector(
    model,
    feature_extractor,
    threshold=0.5,
    batch_size=32,
    processing_interval=1.0
)

# Register callbacks
detector.register_threat_callback(on_threat_detected)
detector.register_processing_callback(on_batch_processed)

# Start detector
detector.start()

# Generate or process packets
# ...

# Display statistics periodically
try:
    while True:
        time.sleep(10)
        stats = detector.get_stats()
        print(f"Statistics at {time.strftime('%H:%M:%S')}:")
        print(f"  Packets processed: {stats['packets_processed']}")
        print(f"  Threats detected: {stats['threats_detected']}")
        print(f"  Queue size: {stats['queue_size']}")
except KeyboardInterrupt:
    print("Stopping detector...")
    detector.stop()
```

### Visualization and Interpretability

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter
import matplotlib.pyplot as plt
import numpy as np

# Create or load a model
model = ThreatDetectionModel(input_shape=(25,), num_classes=5)
model.train(X_train, y_train, epochs=5)

# Define class names and feature names
class_names = ['Normal Traffic', 'Port Scan', 'DDoS', 'Brute Force', 'Data Exfiltration']
feature_names = ['Feature_' + str(i) for i in range(25)]

# Set up visualization dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Set up interpreter
interpreter = ThreatInterpreter(model, feature_names, class_names)
interpreter.initialize(X_train[:100])

# Generate sample threat data
for i in range(100):
    # Generate random threat data
    sample = np.random.rand(1, 25)
    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    
    # Determine class
    class_idx = prediction
    class_name = class_names[class_idx]
    confidence = proba[class_idx]
    
    # Create threat data
    threat_data = {
        'timestamp': time.time(),
        'features': sample[0],
        'prediction': class_idx,
        'class_name': class_name,
        'confidence': confidence,
        'probabilities': {class_names[i]: proba[i] for i in range(len(class_names))}
    }
    
    # Add to dashboard
    dashboard.add_threat(threat_data)
    
    # Generate explanation for specific class samples
    if class_name == 'Port Scan' and i % 10 == 0:
        print(f"Generating explanation for {class_name} threat:")
        explanation = interpreter.explain_prediction(
            sample[0],
            method="shap",
            target_class=class_idx,
            top_features=5
        )
        
        # Plot explanation
        fig = interpreter.plot_explanation(
            explanation,
            plot_type="bar",
            save_path=f"interpretation_output/{class_name.lower().replace(' ', '_')}_explanation.png"
        )
        
        # Create report
        report = interpreter.create_feature_importance_report(
            explanation,
            output_path=f"interpretation_output/{class_name.lower().replace(' ', '_')}_report.txt"
        )
        
        # Print top features
        print(f"Top features for {class_name}:")
        for feature, importance in explanation['feature_importances']:
            print(f"  {feature}: {importance:.4f}")
        
        print(f"Explanation saved to 'interpretation_output/{class_name.lower().replace(' ', '_')}_explanation.png'")

# Save final dashboard snapshot
dashboard.save_snapshot('visualization_output/final_dashboard.png')
dashboard.stop()

print("Visualization and interpretability example completed!")
```

### Integrated Analytics

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter, get_threat_pattern_insights
import os

# Create output directories
os.makedirs('analytics_output/threat_analysis', exist_ok=True)
os.makedirs('analytics_output/threat_insights', exist_ok=True)

# Load model or create a new one
model_path = 'models/multiclass_threat_model'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully")
else:
    # Create and train a new model
    model = ThreatDetectionModel(input_shape=(25,), num_classes=6)
    model.train(X_train, y_train, epochs=10)
    model.save_model(model_path)
    print("New model created and trained")

# Initialize analytics components
class_names = ['Normal Traffic', 'Port Scan', 'DDoS', 'Brute Force', 'Data Exfiltration', 'Command & Control']
feature_names = ['Feature_' + str(i) for i in range(25)]

# Initialize visualization dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Initialize threat interpreter
interpreter = ThreatInterpreter(model, feature_names, class_names)
interpreter.initialize(X_train[:100])

# Run simulated detection to generate threat data
threat_data = {}
for cls_idx in range(1, len(class_names)):  # Skip normal traffic
    threat_data[cls_idx] = []

# Run a simulated real-time detection scenario
detector = PacketStreamDetector(model, SimpleFeatureExtractor())
detector.register_threat_callback(on_threat_detected)
detector.start()

# Generate synthetic packets for detection
# ...

# Analyze detected threats
for cls_idx, threats in threat_data.items():
    if len(threats) > 0:
        print(f"Analyzing {len(threats)} {class_names[cls_idx]} threats...")
        
        # Stack features from all threats of this class
        features = np.vstack([t['features'] for t in threats])
        
        # Generate explanation
        explanation = interpreter.explain_prediction(
            features[0],  # Use the first threat as an example
            method="rules",
            target_class=cls_idx,
            top_features=5
        )
        
        # Plot and save explanation
        fig = interpreter.plot_explanation(
            explanation,
            plot_type="bar",
            save_path=f"analytics_output/threat_analysis/{class_names[cls_idx].lower().replace(' ', '_')}_explanation.png"
        )
        
        # Save explanation report
        report = interpreter.create_feature_importance_report(
            explanation,
            output_path=f"analytics_output/threat_analysis/{class_names[cls_idx].lower().replace(' ', '_')}_report.txt"
        )
        
        # Print top features
        print(f"Top features for {class_names[cls_idx]} detection:")
        for feature, importance in explanation['feature_importances']:
            print(f"  {feature}: {importance:.4f}")
    else:
        print(f"No {class_names[cls_idx]} threats detected for analysis")

# Generate threat pattern insights
print("Step 5: Generating threat pattern insights...")
for cls_idx, threats in threat_data.items():
    if len(threats) >= 5:  # Need multiple samples for meaningful patterns
        insights = get_threat_pattern_insights(
            interpreter,
            samples=np.vstack([t['features'] for t in threats]),
            threat_class_id=cls_idx,
            top_features=5,
            method="rules"
        )
        
        # Save insights
        with open(f"analytics_output/threat_insights/{class_names[cls_idx].lower().replace(' ', '_')}_patterns.txt", 'w') as f:
            f.write(f"Threat Pattern Insights: {class_names[cls_idx]}\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"Based on analysis of {len(threats)} samples\n\n")
            f.write("TOP CONTRIBUTING FEATURES:\n")
            for feature, importance in insights['top_features']:
                f.write(f"- {feature}: {importance:.4f}\n")
            f.write("\nCOMMON PATTERNS:\n")
            for pattern in insights['patterns']:
                f.write(f"- {pattern}\n")

print("Integrated analytics example completed successfully!")
```

## Best Practices

### Model Training

1. **Data Preprocessing**: Always normalize or standardize your features
2. **Class Imbalance**: Use class weights or resampling techniques for imbalanced datasets
3. **Validation**: Always use a validation set to monitor for overfitting
4. **Early Stopping**: Enable early stopping to prevent overfitting
5. **Model Architecture**: Start with simpler models and gradually increase complexity

### Real-time Detection

1. **Batch Size**: Adjust batch size based on your hardware capabilities
2. **Threshold Tuning**: Find the optimal threshold for your specific use case
3. **Performance Monitoring**: Regularly check detector statistics
4. **Error Handling**: Implement robust error handling for production environments
5. **Resource Management**: Be mindful of memory usage with large datasets

### Visualization and Interpretability

1. **Background Data**: Use a representative sample for background data in SHAP
2. **Dashboard Performance**: Limit history size for better performance
3. **Report Generation**: Save explanation reports for important detections
4. **Visualization Export**: Export visualizations for sharing and documentation
5. **Interpretation Context**: Consider the context when interpreting feature importance

## API Reference

For detailed API documentation, refer to the docstrings in each module. The major classes and functions are:

### Model Module

- `ThreatDetectionModel(input_shape, num_classes=2, model_config=None)`
- `load_model(model_path, metadata_path=None)`

### Preprocessing Module

- `FeatureExtractor(categorical_features=None, numeric_features=None, ip_features=None, scaling='standard', handle_missing=True)`
- `IPAddressTransformer()`
- `extract_packet_features(packet_data, include_headers=True, include_payload=True, max_payload_length=1024)`
- `extract_flow_features(flow_data)`

### Real-time Module

- `RealTimeDetector(model, feature_extractor=None, threshold=0.5, batch_size=32, processing_interval=1.0)`
- `PacketStreamDetector(model, feature_extractor, threshold=0.5, batch_size=32, processing_interval=1.0)`

### Evaluation Module

- `evaluate_model(model, X_test, y_test, threshold=0.5)`
- `classification_report(model, X_test, y_test, threshold=0.5)`
- `plot_confusion_matrix(model, X_test, y_test, threshold=0.5, normalize=True, cmap='Blues', figsize=(8, 6))`
- `plot_roc_curve(model, X_test, y_test, figsize=(8, 6))`
- `find_optimal_threshold(model, X_val, y_val, metric='f1')`

### Explain Module

- `explain_prediction(model, X_sample, feature_names=None)`
- `explain_model(model, X_background, X_explain=None, feature_names=None, max_display=10)`
- `plot_shap_summary(shap_values, feature_names=None, max_display=10)`
- `get_top_features(shap_values, feature_names=None, top_n=10)`

### Visualization Module

- `ThreatVisualizationDashboard(max_history=1000, update_interval=1.0)`
- `get_dashboard()`

### Interpretability Module

- `ThreatInterpreter(model, feature_names=None, class_names=None)`
- `get_threat_pattern_insights(interpreter, samples, threat_class_id, top_features=5, method="auto")`

### Utilities Module

- `save_dataset(X, y, dataset_path, metadata=None)`
- `load_dataset(dataset_path)`
- `split_data(X, y, test_size=0.2, val_size=0.25, random_state=None)`
- `normalize_packet_size(size, max_size=65535)`
- `normalize_port_number(port)`
- `calculate_entropy(data)`
- `plot_training_history(history, figsize=(12, 4))`

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cyberthreat_ml'**
   - Ensure the package is installed correctly
   - Check your Python path
   
2. **CUDA/CuDNN errors**
   - These are warnings from TensorFlow and can be safely ignored if you're using CPU
   - For GPU support, ensure compatible CUDA and CuDNN versions
   
3. **Memory errors during training**
   - Reduce batch size
   - Use a smaller dataset for training
   
4. **Slow real-time detection**
   - Increase batch size
   - Reduce model complexity
   - Increase processing interval
   
5. **Visualization not showing**
   - Check if matplotlib is installed
   - Ensure your environment supports GUI (or use non-GUI backends)
   - Use `save_snapshot()` to save visualizations
   
6. **SHAP dependency errors**
   - Install SHAP library (`pip install shap`)
   - The library will fall back to rules-based explanations if SHAP is not available

### Getting Help

If you encounter issues not covered in this documentation, please:

1. Check the GitHub repository issues
2. Look for examples in the `examples/` directory
3. Refer to the docstrings in the source code
4. Submit an issue on GitHub with a detailed description and steps to reproduce
