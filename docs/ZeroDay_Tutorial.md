# Building a Zero-Day Threat Detection System

This tutorial will guide you through building a complete zero-day threat detection system using CyberThreat-ML. By the end, you'll have a working system that can identify previously unknown threats in network traffic.

## Overview

Zero-day threats exploit previously unknown vulnerabilities, making them particularly dangerous. Traditional signature-based detection fails against these threats because there are no known signatures yet. This is where anomaly-based detection shines, by identifying activity that deviates significantly from normal patterns.

This tutorial will show you how to:
1. Set up baseline data of normal traffic
2. Configure and train the zero-day detector
3. Implement real-time monitoring
4. Analyze and interpret detection results
5. Adapt the system as network behavior changes

## Prerequisites

- Python 3.8+
- CyberThreat-ML library
- Basic understanding of network traffic concepts
- Sample network traffic data (or the ability to capture it)

## Step 1: Set Up Your Environment

First, let's set up a Python environment and install the necessary packages:

```bash
# Create a virtual environment
python -m venv cyberthreat-env
source cyberthreat-env/bin/activate  # On Windows: cyberthreat-env\Scripts\activate

# Install required packages
pip install cyberthreat-ml numpy pandas matplotlib
```

## Step 2: Prepare Your Data

For this tutorial, we'll need:
- A dataset of normal network traffic
- Some examples of known attacks (optional, for comparative testing)

If you don't have real data, the library includes methods to generate synthetic data for testing:

```python
import numpy as np
from cyberthreat_ml.anomaly import ZeroDayDetector

# Generate synthetic normal traffic data
def create_normal_dataset(n_samples=1000):
    # Create features typical of normal traffic
    X = np.random.random((n_samples, 10))  # 10 features
    
    # Add some statistical patterns resembling normal traffic
    # For real applications, these would be extracted from your network
    X[:, 0] = np.random.normal(0.5, 0.1, n_samples)  # e.g., packet size
    X[:, 1] = np.random.normal(0.3, 0.05, n_samples)  # e.g., inter-arrival time
    X[:, 2] = np.random.choice([0.2, 0.4, 0.6], n_samples)  # e.g., protocol type
    
    # Return features and "normal" labels (0)
    return X, np.zeros(n_samples)

# Generate normal traffic data
normal_data, normal_labels = create_normal_dataset(n_samples=1000)
```

In a real-world setting, you would replace this with data captured from your environment.

## Step 3: Configure the Zero-Day Detector

Now let's set up and train our anomaly detector:

```python
# Create a zero-day detector with ensemble methods
zero_day_detector = ZeroDayDetector(
    method='ensemble',       # Use multiple detection algorithms together
    contamination=0.01,      # Expect 1% false positives
    min_samples=100          # Need at least 100 samples before detection
)

# Create feature names for better interpretability
feature_names = [
    "Packet Size", 
    "Inter-arrival Time", 
    "Protocol", 
    "Source Port", 
    "Destination Port",
    "TCP Flags",
    "Packet Count", 
    "Flow Duration", 
    "Bytes In", 
    "Bytes Out"
]

# Fit the detector on normal traffic only
zero_day_detector.fit(normal_data, feature_names)
print("Zero-day detector trained successfully")
```

## Step 4: Set Up Real-Time Monitoring

For real-time monitoring, we need to:
1. Create a feature extractor for raw packets
2. Initialize a real-time detector
3. Set up callbacks for when threats are detected

```python
import time
from cyberthreat_ml.anomaly import RealTimeZeroDayDetector, get_anomaly_description, recommend_action

# Create a simple feature extractor
class NetworkFeatureExtractor:
    def transform(self, packet):
        """Extract features from a network packet."""
        if isinstance(packet, dict):
            # Extract and normalize features
            features = np.array([
                packet.get('size', 0) / 10000.0,                # Normalize packet size
                packet.get('time_delta', 0) / 1.0,              # Inter-arrival time
                packet.get('protocol', 0) / 255.0,              # Protocol
                packet.get('src_port', 0) / 65535.0,            # Source port
                packet.get('dst_port', 0) / 65535.0,            # Destination port
                packet.get('tcp_flags', 0) / 255.0,             # TCP flags
                packet.get('packet_count', 0) / 100.0,          # Packet count
                packet.get('duration', 0) / 10.0,               # Flow duration
                packet.get('bytes_in', 0) / 10000.0,            # Bytes in
                packet.get('bytes_out', 0) / 10000.0            # Bytes out
            ]).reshape(1, -1)
            return features
        else:
            # Handle if already a numpy array
            return packet.reshape(1, -1) if packet.ndim == 1 else packet

# Create a real-time zero-day detector
realtime_detector = RealTimeZeroDayDetector(
    feature_extractor=NetworkFeatureExtractor(),
    baseline_data=normal_data,
    feature_names=feature_names,
    method='isolation_forest',
    contamination=0.05,
    time_window=3600  # Consider data from the last hour
)

# Define a callback for when anomalies are detected
def on_anomaly_detected(result):
    """Handle detected anomalies."""
    print("\n*** ANOMALY DETECTED ***")
    print(f"Timestamp: {time.ctime(result['timestamp'])}")
    print(f"Anomaly score: {result['anomaly_score']:.4f}")
    print(f"Severity: {result['severity_level']} ({result['severity']:.4f})")
    
    # Get a human-readable description
    description = get_anomaly_description(result['analysis'])
    print(f"Description: {description}")
    
    # Get recommended actions
    actions = recommend_action(result['analysis'])
    print(f"Priority: {actions['priority']}")
    print("Recommended actions:")
    for action in actions['actions']:
        print(f"  - {action}")
        
    # Store the anomaly for further analysis
    store_anomaly(result)
    
def store_anomaly(anomaly_data):
    """Store anomaly data for further analysis."""
    # In a real system, you would store this in a database
    # For this tutorial, we'll just append to a list
    detected_anomalies.append(anomaly_data)
    
# List to store detected anomalies
detected_anomalies = []
```

## Step 5: Process Network Traffic

Now let's simulate processing network traffic:

```python
def process_packet_stream(duration=60, interval=0.1):
    """
    Simulate processing network packets for a specified duration.
    
    Args:
        duration (int): Duration to run in seconds
        interval (float): Time between packets in seconds
    """
    print(f"Starting packet processing for {duration} seconds...")
    start_time = time.time()
    packet_count = 0
    anomaly_count = 0
    
    while time.time() - start_time < duration:
        # Generate a packet (in real-world, you'd get this from the network)
        if np.random.random() < 0.05:  # 5% chance of abnormal packet
            packet = generate_abnormal_packet()
        else:
            packet = generate_normal_packet()
            
        # Process the packet
        result = realtime_detector.add_sample(packet)
        
        # If an anomaly was detected, it will be handled by the callback
        if result:
            anomaly_count += 1
            
        packet_count += 1
        time.sleep(interval)  # Simulate packet arrival interval
        
    print(f"\nProcessed {packet_count} packets in {duration} seconds")
    print(f"Detected {anomaly_count} potential zero-day threats")
    
def generate_normal_packet():
    """Generate a normal packet for simulation."""
    return {
        'size': np.random.normal(500, 200),
        'time_delta': np.random.exponential(0.1),
        'protocol': np.random.choice([6, 17]),  # TCP or UDP
        'src_port': np.random.randint(1024, 65535),
        'dst_port': np.random.choice([80, 443, 53, 22, 25]),
        'tcp_flags': np.random.randint(0, 64),
        'packet_count': np.random.randint(1, 10),
        'duration': np.random.exponential(1.0),
        'bytes_in': np.random.normal(500, 200),
        'bytes_out': np.random.normal(300, 100)
    }
    
def generate_abnormal_packet():
    """Generate an abnormal packet for simulation."""
    # Choose one of several anomaly types
    anomaly_type = np.random.choice(['size', 'port', 'flags', 'rate'])
    
    # Start with a normal packet
    packet = generate_normal_packet()
    
    # Modify based on anomaly type
    if anomaly_type == 'size':
        # Unusually large packet
        packet['size'] = np.random.normal(9000, 1000)
        packet['bytes_in'] = np.random.normal(9000, 1000)
    elif anomaly_type == 'port':
        # Unusual port combination
        packet['dst_port'] = np.random.choice([6667, 4444, 31337])
    elif anomaly_type == 'flags':
        # Unusual TCP flags
        packet['tcp_flags'] = np.random.randint(128, 255)
    elif anomaly_type == 'rate':
        # Unusual packet rate/timing
        packet['time_delta'] = np.random.exponential(0.001)
        packet['packet_count'] = np.random.randint(50, 100)
        
    return packet
    
# Run the simulation
process_packet_stream(duration=30, interval=0.1)
```

## Step 6: Analyze the Results

After collecting data, let's analyze what we've found:

```python
def analyze_detected_anomalies():
    """Analyze the anomalies we've detected."""
    if not detected_anomalies:
        print("No anomalies detected during this run.")
        return
        
    print(f"\nAnalyzing {len(detected_anomalies)} detected anomalies:")
    
    # Group by severity
    severity_counts = {}
    for anomaly in detected_anomalies:
        severity = anomaly['severity_level']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
    print("\nSeverity distribution:")
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count} anomalies")
        
    # Find the most severe anomaly
    most_severe = max(detected_anomalies, key=lambda x: x['severity'])
    print(f"\nMost severe anomaly (score: {most_severe['severity']:.4f}):")
    print(f"  Description: {get_anomaly_description(most_severe['analysis'])}")
    
    # Extract top contributing features across all anomalies
    feature_contributions = {}
    for anomaly in detected_anomalies:
        for feature in anomaly['analysis'].get('top_contributors', []):
            feature_contributions[feature] = feature_contributions.get(feature, 0) + 1
            
    print("\nTop contributing features:")
    sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
    for feature, count in sorted_features[:5]:
        print(f"  {feature}: implicated in {count} anomalies")
        
# Run the analysis
analyze_detected_anomalies()
```

## Step 7: Adaptive Learning

Networks evolve over time, so our detector should adapt:

```python
def adapt_detector():
    """Update the detector based on recent normal traffic."""
    print("\nAdapting detector based on recent normal traffic...")
    
    # Train on recent normal traffic (excluding detected anomalies)
    result = realtime_detector.train_on_recent_normal(min_samples=50)
    
    if result:
        print("Detector adapted successfully to recent traffic patterns")
        # Get detector stats after adaptation
        stats = realtime_detector.get_stats()
        print(f"Detector stats:")
        print(f"  Samples collected: {stats['samples_collected']}")
        print(f"  Anomalies detected: {stats['anomalies_detected']}")
        print(f"  Models used: {', '.join(stats['models_used'])}")
    else:
        print("Not enough recent normal samples to adapt detector")

# Adapt the detector
adapt_detector()
```

## Step 8: Putting It All Together

Here's a complete script that ties everything together:

```python
import numpy as np
import time
from cyberthreat_ml.anomaly import ZeroDayDetector, RealTimeZeroDayDetector
from cyberthreat_ml.anomaly import get_anomaly_description, recommend_action

# 1. Generate or load your data
normal_data, _ = create_normal_dataset(n_samples=1000)

# 2. Set up feature names
feature_names = [
    "Packet Size", "Inter-arrival Time", "Protocol", "Source Port", 
    "Destination Port", "TCP Flags", "Packet Count", "Flow Duration", 
    "Bytes In", "Bytes Out"
]

# 3. Create and train zero-day detector
zero_day_detector = ZeroDayDetector(method='ensemble', contamination=0.01)
zero_day_detector.fit(normal_data, feature_names)

# 4. Create feature extractor
class NetworkFeatureExtractor:
    def transform(self, packet):
        # (implementation as above)
        pass

# 5. Create real-time detector
realtime_detector = RealTimeZeroDayDetector(
    feature_extractor=NetworkFeatureExtractor(),
    baseline_data=normal_data,
    feature_names=feature_names,
    method='isolation_forest',
    contamination=0.05,
    time_window=3600
)

# 6. List to store anomalies
detected_anomalies = []

# 7. Define anomaly callback
def on_anomaly_detected(result):
    # (implementation as above)
    pass

# 8. Process traffic
process_packet_stream(duration=300, interval=0.1)  # Run for 5 minutes

# 9. Analyze results
analyze_detected_anomalies()

# 10. Adapt detector
adapt_detector()
```

## Extending the System

There are many ways to extend this basic zero-day detection system:

### 1. Integrate with Signature-Based Detection

Combine zero-day detection with traditional signature-based detection:

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.realtime import PacketStreamDetector

# Create and train a signature model (with known threat signatures)
signature_model = ThreatDetectionModel(input_shape=(10,), num_classes=2)
signature_model.train(known_data, known_labels)

# Create a signature-based detector
signature_detector = PacketStreamDetector(
    model=signature_model,
    feature_extractor=NetworkFeatureExtractor(),
    threshold=0.7
)

# Process with both detectors
def process_packet(packet):
    # Check for known threats
    signature_detector.process_packet(packet)
    
    # Check for zero-day threats
    realtime_detector.add_sample(packet)
```

### 2. Add Visualization

Visualize detections with the visualization module:

```python
from cyberthreat_ml.visualization import ThreatVisualizationDashboard

# Create and start a dashboard
dashboard = ThreatVisualizationDashboard()
dashboard.start()

# Add visualization to our anomaly callback
def on_anomaly_detected(result):
    # Format for the dashboard
    dashboard_result = {
        'timestamp': result['timestamp'],
        'is_threat': True,
        'threat_score': result['anomaly_score'],
        'class_name': f"Zero-Day ({result['severity_level']})",
        'data': result['raw_data']
    }
    
    # Add to dashboard
    dashboard.add_threat(dashboard_result)
    
    # Rest of the handler...
```

### 3. Implement Automatic Response

For critical threats, implement automatic response:

```python
def on_anomaly_detected(result):
    # ... existing code ...
    
    # If high severity, take action
    if result['severity_level'] in ['Medium-High', 'High']:
        src_ip = result['raw_data'].get('src_ip')
        if src_ip:
            print(f"AUTOMATIC RESPONSE: Blocking IP {src_ip}")
            # In a real system, you would call your firewall API:
            # firewall.block_ip(src_ip)
```

## Best Practices for Production Use

1. **Data Quality**: Use representative normal traffic for training.
2. **Tuning**: Tune the contamination rate based on your false positive tolerance.
3. **Regular Updates**: Retrain periodically with recent normal traffic.
4. **Monitor Performance**: Track false positive and false negative rates.
5. **Human Verification**: Have human analysts verify critical alerts.
6. **Resource Management**: Monitor resource usage for high-volume traffic.

## Conclusion

You've now built a complete zero-day threat detection system using anomaly detection. This system can identify previously unknown threats by recognizing deviations from normal behavior patterns.

Remember that zero-day detection works best as part of a comprehensive security approach that includes:
- Signature-based detection for known threats
- Zero-day detection for unknown threats  
- Regular security updates
- Security awareness training
- Incident response planning

By combining these approaches, you'll have a robust defense against both known and unknown cybersecurity threats.