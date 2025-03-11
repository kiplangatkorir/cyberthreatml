# Real-Time Threat Detection Tutorial

This tutorial walks you through using the CyberThreat-ML library for real-time network traffic monitoring and threat detection.

## Introduction

Real-time threat detection is a critical capability for modern cybersecurity systems. This tutorial demonstrates how to use CyberThreat-ML to:

1. Process network packets in real-time
2. Extract meaningful features from network traffic
3. Detect various types of threats
4. Generate alerts and security reports

## The Simplified Real-Time Example

The `examples/simplified_realtime.py` script provides a standalone example of real-time threat detection that doesn't rely on external dependencies.

### Running the Example

To run the example:

```bash
python examples/simplified_realtime.py
```

The script will:
1. Start a simulated network environment
2. Process about 10 packets per second
3. Flag suspicious packets as potential threats
4. Generate a comprehensive security report

## Understanding the Components

Let's break down the key components of the real-time detection system:

### 1. Feature Extraction

The `SimpleFeatureExtractor` class transforms raw packet data into a format suitable for machine learning models:

```python
class SimpleFeatureExtractor:
    def transform(self, packet):
        features = [
            packet.get('packet_size', 0) / 1500.0,  # Normalize packet size
            packet.get('protocol') == 'TCP',  # Protocol binary feature
            packet.get('source_port', 0) / 65535.0,  # Normalize source port
            # Additional features...
        ]
        return features
```

Key features extracted from packets include:
- Packet size (normalized)
- Protocol type (TCP, UDP, ICMP)
- Source and destination ports
- TCP flags (SYN, ACK, FIN, RST)
- Payload size and presence

### 2. Threat Detection Model

The `SimpleDetectionModel` class simulates a machine learning model that analyzes packet features to identify threats:

```python
class SimpleDetectionModel:
    def predict(self, features):
        # Logic to determine if a packet represents a threat
        # Returns prediction (threat type) and confidence score
```

The model can detect various types of threats:
- **Port Scans**: Characterized by SYN packets without ACK or FIN flags
- **DDoS Attacks**: Large packets with significant payloads
- **Brute Force Attempts**: Multiple ACK packets to authentication services
- **Data Exfiltration**: Large outbound data transfers
- **Command & Control**: Small packets with payload to unusual destinations

### 3. Real-Time Detector

The `SimpleRealTimeDetector` class coordinates the detection process:

```python
class SimpleRealTimeDetector:
    def __init__(self, model, feature_extractor, threshold=0.5):
        # Initialize with model and feature extractor
        
    def start(self):
        # Start detection process
        
    def process_packet(self, packet):
        # Process a single packet
        # Extract features, make prediction, generate alert if needed
```

The detector maintains statistics about processed packets and detected threats, which are used to generate the final report.

## Simulating Network Traffic

The example uses the `generate_random_packet()` function to create synthetic network packets:

```python
def generate_random_packet():
    # Generate a random network packet with realistic characteristics
    # Returns a dictionary with packet attributes
```

Each packet includes:
- Source and destination IP addresses
- Source and destination ports
- Protocol (TCP, UDP, ICMP)
- TCP flags (for TCP packets)
- Packet and payload size

The `simulate_traffic()` function controls the simulation:

```python
def simulate_traffic(detector, duration=10, packet_rate=5):
    # Generate and process packets for the specified duration
```

## Understanding the Results

### 1. Real-Time Alerts

During execution, the script outputs alerts for detected threats:

```
⚠️ Threat detected: Port Scan from 192.168.1.1:38998 to 203.0.113.14:23 (0.89 confidence)
```

Each alert includes:
- Threat type
- Source IP and port
- Destination IP and port
- Confidence score

### 2. Summary Report

After the simulation completes, a summary report is generated:

```
Threat Detection Report
======================
Total packets processed: 299
Total threats detected: 88

Threats by type:
  - Port Scan: 38 (43.2%)
  - DDoS: 30 (34.1%)
  - Brute Force: 9 (10.2%)
  - Data Exfiltration: 4 (4.5%)
  - Command & Control: 7 (8.0%)
```

The report includes:
- Overall statistics
- Breakdown of threats by type
- Detailed examples of detected threats
- Security recommendations

## Extending the Example

You can extend the simplified example in several ways:

### 1. Custom Feature Extraction

Enhance the `SimpleFeatureExtractor` to extract more sophisticated features:

```python
def transform(self, packet):
    features = [
        # Basic features
        packet.get('packet_size', 0) / 1500.0,
        
        # Additional features
        self._calculate_entropy(packet.get('payload', b'')),
        self._is_suspicious_port(packet.get('dest_port', 0)),
        # More custom features...
    ]
    return features
```

### 2. Advanced Detection Logic

Improve the detection model with more sophisticated rules or machine learning models:

```python
def predict(self, features):
    # Use a trained model (if available)
    if hasattr(self, 'trained_model'):
        return self.trained_model.predict(features)
        
    # Fall back to rule-based detection
    # ...
```

### 3. Persistent Monitoring

Modify the detector to maintain state across multiple packets for detecting persistent threats:

```python
def __init__(self, model, feature_extractor):
    # ...
    self.connection_history = {}
    
def process_packet(self, packet):
    # ...
    connection_key = f"{packet['source_ip']}:{packet['source_port']}-{packet['dest_ip']}:{packet['dest_port']}"
    
    if connection_key in self.connection_history:
        # Consider previous packets in this connection
        # ...
```

## Conclusion

The simplified real-time detection example demonstrates the core concepts of using machine learning for cybersecurity threat detection. While it uses simulated data, the approach can be adapted for real network environments with appropriate feature extraction and model training.

By understanding these components, you can build more sophisticated threat detection systems using the full capabilities of the CyberThreat-ML library.