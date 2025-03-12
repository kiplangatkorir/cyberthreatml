# CyberThreat-ML Quick Start Guide

This guide will help you get started with the CyberThreat-ML library for cybersecurity threat detection using machine learning.

## Introduction
 
CyberThreat-ML is a Python library designed for detecting and analyzing cybersecurity threats using machine learning techniques. The library provides tools for:

- Real-time network traffic monitoring
- Multi-class threat classification
- Anomaly-based zero-day threat detection
- Model explainability and visualization
- Enterprise security integration

## Prerequisites

- Python 3.10 or higher
- Basic understanding of cybersecurity concepts
- Familiarity with Python programming

## Running the Examples

The examples directory contains several scripts demonstrating different aspects of the library. Here's how to run them:

### 1. Minimal Example

The minimal example demonstrates the basic structure of the library and checks your environment:

```bash
python examples/minimal_example.py
```

This example:
- Checks your Python environment
- Verifies the library structure
- Lists available modules and examples
- Simulates a basic threat detection

### 2. Simplified Real-Time Detection

This example demonstrates real-time cybersecurity threat detection without external dependencies:

```bash
python examples/simplified_realtime.py
```

This example:
- Creates a simulated network environment
- Detects various types of threats in real-time
- Generates alerts for detected threats
- Provides a summary report with recommendations

#### Understanding the Output

When running the real-time detection example, you'll see output like:

```
⚠️ Threat detected: Port Scan from 192.168.1.1:38998 to 203.0.113.14:23 (0.89 confidence)
```

This indicates:
- The type of threat (Port Scan)
- Source IP address and port (192.168.1.1:38998)
- Destination IP address and port (203.0.113.14:23)
- Detection confidence (0.89 or 89%)

The final report provides:
- Total packets processed
- Threat detection rate
- Breakdown of threats by type
- Detailed information for sample threats
- Security recommendations

## Core Components

### Feature Extraction

The `SimpleFeatureExtractor` class in the examples demonstrates how network packet data is transformed into features for machine learning models:

```python
features = [
    packet.get('packet_size', 0) / 1500.0,  # Normalize packet size
    packet.get('protocol') == 'TCP',  # Protocol binary feature
    packet.get('source_port', 0) / 65535.0,  # Normalize source port
    # Additional features...
]
```

### Threat Detection Model

The `SimpleDetectionModel` class shows how a machine learning model makes predictions based on extracted features:

```python
def predict(self, features):
    # Logic to analyze features and return predictions
    prediction = 0  # Default to normal traffic
    confidence = 0.0
    
    # Detection logic based on packet characteristics
    if features[4] > 0 and features[6] == 0 and features[5] == 0:
        # Potential port scan
        prediction = 1
        confidence = 0.7 + (random.random() * 0.2)
    
    # Additional detection rules...
    
    return prediction, confidence
```

### Real-Time Detector

The `SimpleRealTimeDetector` class processes packets in real-time:

```python
def process_packet(self, packet):
    # Extract features
    features = self.feature_extractor.transform(packet)
    
    # Make prediction
    prediction, confidence = self.model.predict(features)
    
    # Process results
    # ...
```

## Next Steps

After running these basic examples, you can explore more advanced features:

1. **Multi-class Classification**: Explore the multi-class example for classifying different types of threats.
2. **Zero-Day Detection**: Learn how anomaly-based detection can identify previously unknown threats.
3. **Model Explainability**: Understand why certain traffic is flagged as malicious.
4. **Enterprise Integration**: See how to integrate the library into enterprise security systems.

## Additional Resources

- [CyberML101](CyberML101.md): Introduction to machine learning for cybersecurity
- [Explainability Guide](Explainability_Guide.md): Understanding model decisions
- [Zero-Day Tutorial](ZeroDay_Tutorial.md): Detecting unknown threats
- [FAQ](FAQ.md): Frequently asked questions
