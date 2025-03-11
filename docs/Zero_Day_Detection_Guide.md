# Zero-Day Threat Detection with CyberThreat-ML

This document provides an overview of zero-day threat detection capabilities in CyberThreat-ML.

## Table of Contents

1. [Introduction](#introduction)
2. [Zero-Day Detection Approaches](#zero-day-detection-approaches)
3. [Implementation Guide](#implementation-guide)
4. [Integrating with Signature-Based Detection](#integrating-with-signature-based-detection)
5. [Analyzing Detected Anomalies](#analyzing-detected-anomalies)
6. [Adaptive Learning](#adaptive-learning)
7. [Best Practices](#best-practices)
8. [Example Code](#example-code)

## Introduction

Zero-day threats are cybersecurity vulnerabilities and attacks that have not been seen before, making them difficult to detect using signature-based methods. CyberThreat-ML provides advanced anomaly detection capabilities that can identify potentially malicious activities that don't match known threat patterns.

### Key Benefits

- **Beyond Signature-Based Detection**: Identify threats that evade traditional rule-based systems
- **Early Warning System**: Detect potential zero-day attacks before signatures are available
- **Anomaly Analysis**: Detailed analysis of why a sample was flagged as anomalous
- **Adaptive Learning**: Continuously update baseline models as new normal patterns emerge
- **Hybrid Detection**: Combine with signature-based methods for comprehensive security

## Zero-Day Detection Approaches

CyberThreat-ML implements several approaches for zero-day threat detection:

### Isolation Forest

Isolation Forest is particularly effective at detecting outliers in high-dimensional data. This algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

```python
from cyberthreat_ml.anomaly import ZeroDayDetector

# Create detector with Isolation Forest
detector = ZeroDayDetector(method='isolation_forest', contamination=0.01)
```

### Local Outlier Factor (LOF)

LOF measures the local deviation of a data point with respect to its neighbors. It works by comparing the local density of a point to the local densities of its neighbors.

```python
# Create detector with Local Outlier Factor
detector = ZeroDayDetector(method='local_outlier_factor', contamination=0.01)
```

### Robust Covariance (Elliptic Envelope)

This method assumes the normal data comes from a Gaussian distribution and fits an elliptic envelope to the central data points, considering points outside the ellipse as outliers.

```python
# Create detector with Robust Covariance
detector = ZeroDayDetector(method='robust_covariance', contamination=0.01)
```

### One-Class SVM

One-Class SVM learns a boundary that encloses the majority of the normal data points, treating points outside the boundary as outliers.

```python
# Create detector with One-Class SVM
detector = ZeroDayDetector(method='one_class_svm', contamination=0.01)
```

### Ensemble Methods

CyberThreat-ML can combine multiple anomaly detection methods to improve detection accuracy and reduce false positives.

```python
# Create an ensemble detector
detector = ZeroDayDetector(method='ensemble', contamination=0.01)
```

## Implementation Guide

### Step 1: Initialize the Zero-Day Detector

```python
from cyberthreat_ml.anomaly import ZeroDayDetector

# Create a detector
detector = ZeroDayDetector(
    method='ensemble',         # Use ensemble of methods
    contamination=0.01,        # Expected percentage of anomalies
    min_samples=100,           # Minimum samples before detection
    feature_columns=None       # Use all features (or specify subset)
)
```

### Step 2: Establish a Baseline of Normal Behavior

Train the detector using known normal traffic:

```python
# Assuming X_normal contains normal network traffic features
feature_names = ['Size', 'Entropy', 'TCP Flags', 'Source Port', ...]
detector.fit(X_normal, feature_names)
```

### Step 3: Detect Anomalies in New Data

```python
# Detect anomalies (returns 1 for normal, -1 for anomalies)
predictions, scores = detector.detect(X_new, return_scores=True)

# Find anomalous samples
anomaly_indices = np.where(predictions == -1)[0]

# Print information about detected anomalies
for idx in anomaly_indices:
    print(f"Anomaly detected at index {idx} with score {scores[idx]:.4f}")
```

### Step 4: Analyze Detected Anomalies

```python
# Analyze why a sample was flagged as anomalous
sample = X_new[anomaly_indices[0]]
analysis = detector.analyze_anomaly(sample)

# Get a human-readable description
from cyberthreat_ml.anomaly import get_anomaly_description
description = get_anomaly_description(analysis)
print(description)

# Get recommended actions
from cyberthreat_ml.anomaly import recommend_action
actions = recommend_action(analysis)
print(f"Priority: {actions['priority']}")
for action in actions['actions']:
    print(f"- {action}")
```

## Integrating with Signature-Based Detection

CyberThreat-ML's Zero-Day detection can complement traditional signature-based detection:

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.anomaly import ZeroDayDetector, RealTimeZeroDayDetector
from cyberthreat_ml.realtime import PacketStreamDetector

# Setup signature-based detector
signature_model = ThreatDetectionModel(input_shape=(25,), num_classes=2)
# ... train the model ...

signature_detector = PacketStreamDetector(
    signature_model,
    feature_extractor,
    threshold=0.5
)

# Setup zero-day detector
zero_day_detector = RealTimeZeroDayDetector(
    feature_extractor=feature_extractor,
    baseline_data=normal_baseline_data,
    feature_names=feature_names
)

# Process network data with both detectors
def process_packet(packet):
    # Check with signature detector
    signature_detector.process_packet(packet)
    
    # Check with zero-day detector
    anomaly_result = zero_day_detector.add_sample(packet)
    
    if anomaly_result:
        # Handle potential zero-day threat
        print(f"Zero-day threat detected: {get_anomaly_description(anomaly_result['analysis'])}")
```

## Analyzing Detected Anomalies

When an anomaly is detected, CyberThreat-ML provides tools to analyze why:

```python
# Get detailed analysis of an anomaly
analysis = detector.analyze_anomaly(anomalous_sample)

# Key information in the analysis
print(f"Anomaly score: {analysis['anomaly_score']}")
print(f"Severity: {analysis['severity']} ({analysis['severity_level']})")
print(f"Top contributors: {analysis['top_contributors']}")

# Feature details show which features contributed most to the anomaly
for feature, details in analysis['feature_details'].items():
    if details['deviation'] > 2.0:  # Features with high deviation
        print(f"Feature {feature}: value={details['value']}, z-score={details['z_score']}")
```

The analysis identifies which features deviate most from the baseline, helping security analysts understand the nature of the potential threat.

## Adaptive Learning

To adapt to evolving "normal" traffic patterns, the zero-day detector can be updated:

```python
# Train on recent normal traffic
realtime_detector.train_on_recent_normal(min_samples=100)

# Or update with new baseline data
new_normal_data = collect_verified_normal_traffic()
detector.fit(new_normal_data, feature_names)
```

The adaptive learning capabilities help reduce false positives over time as the detector learns from new normal traffic patterns.

## Best Practices

1. **Start with conservative thresholds** (low contamination values like 0.01) to minimize false positives
2. **Use ensemble methods** for more robust detection
3. **Establish a good baseline** with sufficient normal traffic data
4. **Regularly update the baseline** to adapt to evolving normal patterns
5. **Combine with signature-based detection** for a comprehensive security approach
6. **Investigate all high-severity anomalies**, especially those with high anomaly scores
7. **Focus on feature contributions** to understand the nature of the anomaly
8. **Consider the contextual information** when analyzing anomalies
9. **Use time-window analysis** to detect temporal patterns in anomalies

## Example Code

See the full zero-day detection example:

```bash
python examples/zero_day_detection.py
```

This example demonstrates:
- Setting up various types of zero-day detectors
- Comparing performance with signature-based detection
- Analyzing detected anomalies
- Implementing a hybrid detection approach
- Using adaptive learning techniques

The example also provides a realistic simulation of how zero-day detection can complement signature-based approaches in a production environment.