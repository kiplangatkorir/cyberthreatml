# CyberThreat-ML Use Cases

This document outlines common use cases and integration patterns for the CyberThreat-ML library in cybersecurity environments.

## Table of Contents

1. [Network Security Monitoring](#network-security-monitoring)
2. [Intrusion Detection Systems Integration](#intrusion-detection-systems-integration)
3. [Security Operations Center (SOC)](#security-operations-center-soc)
4. [Automated Incident Response](#automated-incident-response)
5. [Threat Hunting](#threat-hunting)
6. [Security Analytics Platform](#security-analytics-platform)
7. [Customizing for Specific Threats](#customizing-for-specific-threats)

## Network Security Monitoring

### Overview

Integrate CyberThreat-ML with network monitoring solutions to detect threats in real-time from network traffic.

### Implementation

1. **Data Collection**: Capture network packets using packet capture libraries like `scapy` or by integrating with existing network monitoring tools
2. **Feature Extraction**: Process raw packets with the `preprocessing` module
3. **Real-time Detection**: Use `PacketStreamDetector` to analyze traffic in real-time
4. **Alerting**: Configure detection callbacks to trigger alerts

### Code Example

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.realtime import PacketStreamDetector
import scapy.all as scapy

# Load pre-trained model
model = load_model("models/network_threat_model")

# Configure feature extractor
extractor = FeatureExtractor(
    numeric_features=["packet_size", "window_size", "ttl"],
    ip_features=["src_ip", "dst_ip"],
    categorical_features=["protocol", "flags"]
)

# Create real-time detector
detector = PacketStreamDetector(
    model=model,
    feature_extractor=extractor,
    threshold=0.7,
    batch_size=32
)

# Configure alerting
def on_threat_detected(result):
    print(f"ALERT: {result['class_name']} detected with {result['confidence']:.2f} confidence")
    # Send alert to SIEM or alert management system
    # log_to_siem(result)

detector.register_threat_callback(on_threat_detected)

# Start detector
detector.start()

# Capture and process packets
def packet_callback(packet):
    if packet.haslayer(scapy.IP):
        # Convert Scapy packet to dictionary
        packet_dict = {
            "timestamp": float(packet.time),
            "src_ip": packet[scapy.IP].src,
            "dst_ip": packet[scapy.IP].dst,
            "protocol": packet[scapy.IP].proto,
            "ttl": packet[scapy.IP].ttl,
            "packet_size": len(packet),
            # Extract more features as needed
        }
        
        # Add TCP-specific fields if available
        if packet.haslayer(scapy.TCP):
            packet_dict.update({
                "src_port": packet[scapy.TCP].sport,
                "dst_port": packet[scapy.TCP].dport,
                "flags": packet[scapy.TCP].flags,
                "window_size": packet[scapy.TCP].window
            })
        
        # Process the packet
        detector.process_packet(packet_dict)

# Start packet capture
scapy.sniff(prn=packet_callback, store=0, count=0)
```

### Integration Points

- Network taps or SPAN ports
- Existing network monitoring tools
- SIEM systems for alert correlation
- Network flow analyzers

## Intrusion Detection Systems Integration

### Overview

Enhance existing IDS solutions with machine learning capabilities for better threat detection.

### Implementation

1. **Integration with IDS**: Connect with Suricata, Snort, or other IDS tools
2. **Alert Enhancement**: Add ML-based classification to IDS alerts
3. **False Positive Reduction**: Use ML to filter out false positives

### Code Example

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.preprocessing import FeatureExtractor
import json
import subprocess
import time

# Load or train model
model = ThreatDetectionModel(input_shape=(20,), num_classes=6)
model.train(X_train, y_train, epochs=10)

# Create feature extractor
extractor = FeatureExtractor()

# Function to process Suricata EVE JSON output
def process_ids_alerts(eve_json_file):
    # Command to follow the eve.json file
    cmd = ["tail", "-f", eve_json_file]
    
    # Start the process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    try:
        # Process each new line (alert) as it appears
        for line in proc.stdout:
            try:
                # Parse the JSON alert
                alert = json.loads(line.strip())
                
                # Only process alert events
                if "alert" in alert:
                    # Extract features from the alert
                    features = extract_features_from_alert(alert)
                    
                    # Transform features
                    X = extractor.transform([features])
                    
                    # Make prediction
                    prediction = model.predict(X)[0]
                    confidence = model.predict_proba(X)[0][prediction]
                    
                    # Define class names
                    class_names = ["Normal", "Port Scan", "DDoS", "Brute Force", 
                                 "Data Exfiltration", "Command & Control"]
                    
                    # Only forward high-confidence threats
                    if confidence > 0.7 and prediction > 0:  # Not normal traffic
                        # Enhance the alert with ML classification
                        enhanced_alert = {
                            "original_alert": alert,
                            "ml_classification": {
                                "class": class_names[prediction],
                                "confidence": float(confidence),
                                "timestamp": time.time()
                            }
                        }
                        
                        # Forward to SIEM or alert system
                        forward_enhanced_alert(enhanced_alert)
                        
                        print(f"Enhanced alert: {class_names[prediction]} with {confidence:.2f} confidence")
            
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
                
    except KeyboardInterrupt:
        # Clean up
        proc.terminate()
        proc.wait()

def extract_features_from_alert(alert):
    """Extract features from Suricata alert."""
    features = {}
    
    # Extract basic information
    if "src_ip" in alert:
        features["src_ip"] = alert["src_ip"]
    if "dest_ip" in alert:
        features["dst_ip"] = alert["dest_ip"]
    if "src_port" in alert:
        features["src_port"] = alert["src_port"]
    if "dest_port" in alert:
        features["dst_port"] = alert["dest_port"]
    
    # Extract alert details
    if "alert" in alert:
        features["signature_id"] = alert["alert"].get("signature_id", 0)
        features["category"] = alert["alert"].get("category", "")
        features["severity"] = alert["alert"].get("severity", 0)
    
    # Extract HTTP information if available
    if "http" in alert:
        features["http_method"] = alert["http"].get("http_method", "")
        features["status"] = alert["http"].get("status", 0)
        features["length"] = alert["http"].get("length", 0)
    
    # Add more feature extraction as needed
    
    return features

def forward_enhanced_alert(enhanced_alert):
    """Forward enhanced alert to SIEM or alert system."""
    # Implement integration with your SIEM or alert system
    # This might be an API call, a message to a queue, etc.
    pass

# Start processing alerts
process_ids_alerts("/var/log/suricata/eve.json")
```

### Integration Points

- Suricata or Snort EVE JSON output
- OSSEC alerts
- Wazuh integration
- Commercial IDS/IPS systems with API access

## Security Operations Center (SOC)

### Overview

Provide SOC analysts with advanced tools for threat detection and investigation.

### Implementation

1. **Dashboard Integration**: Display the `ThreatVisualizationDashboard` in SOC monitoring screens
2. **Alert Triage**: Use `ThreatInterpreter` to help analysts understand threat alerts
3. **Automation**: Integrate with SOAR platforms for automated response

### Code Example

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter
import threading
import time
import json
import requests

# Load pre-trained model
model = load_model("models/threat_model")

# Create visualization dashboard 
dashboard = ThreatVisualizationDashboard(max_history=500)
dashboard.start()

# Create interpreter for explaining threats
interpreter = ThreatInterpreter(
    model=model,
    feature_names=["src_ip", "dst_ip", "src_port", "dst_port", "protocol", 
                 "bytes_in", "bytes_out", "packets_in", "packets_out", 
                 "duration", "flags", "tcp_window", "ttl", "packet_size", 
                 "inter_packet_time", "entropy", "is_encrypted", "cert_valid", 
                 "http_method", "mime_type"],
    class_names=["Normal Traffic", "Port Scan", "DDoS", "Brute Force", 
               "Data Exfiltration", "Command & Control"]
)

# Initialize interpreter with background data
interpreter.initialize(X_background)

# Function to poll for new alerts from SIEM or alert queue
def poll_alerts(interval=5):
    while True:
        try:
            # Get new alerts (implementation depends on your SIEM)
            alerts = get_alerts_from_siem()
            
            for alert in alerts:
                # Extract features for the model
                features = extract_features_from_alert(alert)
                
                # Make prediction
                prediction = model.predict(features.reshape(1, -1))[0]
                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                
                class_idx = prediction
                class_name = interpreter.class_names[class_idx]
                confidence = probabilities[class_idx]
                
                # Create threat data for dashboard
                threat_data = {
                    "timestamp": alert["timestamp"],
                    "features": features,
                    "prediction": int(class_idx),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "source_ip": alert["source_ip"],
                    "destination_ip": alert["destination_ip"],
                    "alert_id": alert["id"],
                    "probabilities": {
                        name: float(prob) for name, prob in zip(interpreter.class_names, probabilities)
                    }
                }
                
                # Add to dashboard
                dashboard.add_threat(threat_data)
                
                # For high-confidence threats, generate explanation
                if confidence > 0.8 and class_idx > 0:  # Not normal traffic
                    explanation = interpreter.explain_prediction(
                        features,
                        method="shap",
                        target_class=class_idx,
                        top_features=10
                    )
                    
                    # Save explanation to file for analyst reference
                    save_explanation(alert["id"], explanation)
                    
                    # Add explanation to alert in SIEM
                    update_alert_with_explanation(alert["id"], explanation)
        
        except Exception as e:
            print(f"Error polling alerts: {e}")
        
        time.sleep(interval)

def get_alerts_from_siem():
    """Get new alerts from SIEM system."""
    # Implement integration with your SIEM
    # Example with a REST API:
    response = requests.get(
        "https://siem-api.example.com/alerts",
        params={"status": "new", "limit": 10},
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
    
    if response.status_code == 200:
        return response.json()["alerts"]
    else:
        print(f"Error fetching alerts: {response.status_code}")
        return []

def extract_features_from_alert(alert):
    """Extract model features from SIEM alert."""
    # Implementation depends on your SIEM alert format
    # This is just an example
    features = [
        # Convert IP to numeric representation
        int(alert["source_ip"].replace(".", "")),
        int(alert["destination_ip"].replace(".", "")),
        alert["source_port"],
        alert["destination_port"],
        {"tcp": 1, "udp": 2, "icmp": 3}.get(alert["protocol"].lower(), 0),
        alert["bytes_in"],
        alert["bytes_out"],
        alert["packets_in"],
        alert["packets_out"],
        alert["duration"],
        alert.get("flags", 0),
        alert.get("tcp_window", 0),
        alert.get("ttl", 0),
        alert.get("packet_size", 0),
        alert.get("inter_packet_time", 0),
        alert.get("entropy", 0),
        1 if alert.get("is_encrypted", False) else 0,
        1 if alert.get("cert_valid", False) else 0,
        {"get": 1, "post": 2, "put": 3, "delete": 4}.get(alert.get("http_method", "").lower(), 0),
        {"text": 1, "application": 2, "image": 3, "audio": 4, "video": 5}.get(
            alert.get("mime_type", "").split("/")[0].lower(), 0)
    ]
    
    return np.array(features)

def save_explanation(alert_id, explanation):
    """Save explanation to file for analyst reference."""
    filename = f"explanations/alert_{alert_id}.json"
    with open(filename, "w") as f:
        # Convert numpy values to native Python types
        serializable_explanation = {
            "feature_importances": [
                (feature, float(importance))
                for feature, importance in explanation["feature_importances"]
            ],
            "prediction": int(explanation["prediction"]),
            "class_name": explanation["class_name"],
            "confidence": float(explanation["confidence"]),
            "method": explanation["method"]
        }
        json.dump(serializable_explanation, f, indent=2)

def update_alert_with_explanation(alert_id, explanation):
    """Update alert in SIEM with explanation."""
    # Implement integration with your SIEM
    # Example with a REST API:
    payload = {
        "explanation": {
            "top_features": [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in explanation["feature_importances"]
            ],
            "confidence": float(explanation["confidence"]),
            "class": explanation["class_name"]
        }
    }
    
    response = requests.put(
        f"https://siem-api.example.com/alerts/{alert_id}",
        json=payload,
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )
    
    if response.status_code != 200:
        print(f"Error updating alert {alert_id}: {response.status_code}")

# Start alert polling in a background thread
alert_thread = threading.Thread(target=poll_alerts, daemon=True)
alert_thread.start()

# Main application loop
try:
    while True:
        # Save dashboard snapshot periodically
        time.sleep(300)  # Every 5 minutes
        dashboard.save_snapshot("soc_dashboard_latest.png")
except KeyboardInterrupt:
    print("Stopping dashboard...")
    dashboard.stop()
```

### Integration Points

- SIEM systems (Splunk, ELK Stack, QRadar, etc.)
- SOAR platforms
- Ticketing systems
- Knowledge bases for threat intelligence

## Automated Incident Response

### Overview

Use CyberThreat-ML to trigger and guide automated responses to security incidents.

### Implementation

1. **Detection**: Use the `RealTimeDetector` to identify threats
2. **Response Rules**: Define response actions based on threat type and confidence
3. **Automation**: Integrate with security automation tools

### Code Example

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.realtime import PacketStreamDetector
import requests
import json
import time
import threading

# Configuration
CONFIG = {
    "firewall_api": "https://firewall-api.internal/api/v1",
    "firewall_token": "YOUR_API_TOKEN",
    "response_thresholds": {
        "port_scan": 0.75,
        "ddos": 0.85,
        "brute_force": 0.80,
        "data_exfiltration": 0.70,
        "command_control": 0.65
    },
    "block_duration": 3600,  # 1 hour in seconds
    "notification_url": "https://alerts.internal/api/notify",
    "notification_token": "YOUR_NOTIFICATION_TOKEN"
}

# Load pre-trained model
model = load_model("models/threat_model")

# Create detector
detector = PacketStreamDetector(
    model=model,
    feature_extractor=SimpleFeatureExtractor(),
    threshold=0.6,  # Lower detection threshold, filter in callback
    batch_size=32
)

# Track blocked IPs to avoid redundant actions
blocked_ips = set()
block_timestamps = {}

def on_threat_detected(result):
    """Callback for threat detection with automated response."""
    # Check if threat confidence exceeds response threshold
    threat_type = result["class_name"].lower().replace(" ", "_").replace("&", "and")
    threshold = CONFIG["response_thresholds"].get(threat_type, 0.9)
    
    if result["confidence"] >= threshold:
        print(f"High confidence threat detected: {result['class_name']} ({result['confidence']:.2f})")
        
        # Extract relevant information
        source_ip = extract_source_ip(result)
        
        # Execute appropriate response based on threat type
        if threat_type == "port_scan":
            block_scanning_ip(source_ip)
            
        elif threat_type == "ddos":
            activate_ddos_protection(source_ip)
            
        elif threat_type == "brute_force":
            lock_targeted_accounts(result)
            
        elif threat_type == "data_exfiltration":
            block_data_transfer(result)
            
        elif threat_type == "command_control":
            isolate_infected_system(result)
        
        # Send notification
        send_notification(result)

def extract_source_ip(result):
    """Extract source IP from result features."""
    # Implementation depends on your feature format
    # This is just an example
    if hasattr(result, "packet") and "src_ip" in result["packet"]:
        return result["packet"]["src_ip"]
    elif "features" in result:
        # Example if source IP is encoded in the feature vector
        return "10.0.0.1"  # Placeholder, replace with actual extraction
    else:
        return None

def block_scanning_ip(ip_address):
    """Block an IP address at the firewall."""
    if not ip_address or ip_address in blocked_ips:
        return
    
    # Add to firewall block list
    try:
        response = requests.post(
            f"{CONFIG['firewall_api']}/blocks",
            json={
                "ip_address": ip_address,
                "reason": "Port scanning activity detected by ML",
                "duration": CONFIG["block_duration"]
            },
            headers={"Authorization": f"Bearer {CONFIG['firewall_token']}"}
        )
        
        if response.status_code == 200 or response.status_code == 201:
            print(f"Blocked IP address: {ip_address}")
            blocked_ips.add(ip_address)
            block_timestamps[ip_address] = time.time()
        else:
            print(f"Failed to block IP: {response.status_code}")
    
    except Exception as e:
        print(f"Error blocking IP: {e}")

def activate_ddos_protection(source_ip):
    """Activate DDoS protection measures."""
    try:
        # Enable DDoS protection mode
        response = requests.post(
            f"{CONFIG['firewall_api']}/ddos_protection",
            json={
                "enabled": True,
                "source_ips": [source_ip] if source_ip else [],
                "duration": CONFIG["block_duration"]
            },
            headers={"Authorization": f"Bearer {CONFIG['firewall_token']}"}
        )
        
        if response.status_code == 200:
            print("Activated DDoS protection")
        else:
            print(f"Failed to activate DDoS protection: {response.status_code}")
    
    except Exception as e:
        print(f"Error activating DDoS protection: {e}")

def lock_targeted_accounts(result):
    """Temporarily lock accounts targeted by brute force."""
    # Implementation depends on your environment
    # This is just an example
    try:
        # Extract targeted account information
        target_ip = "192.168.1.10"  # Example, extract from result
        target_port = 22  # Example, extract from result
        
        # Identify targeted service
        service = "ssh" if target_port == 22 else "unknown"
        
        # Lock accounts via API
        response = requests.post(
            "https://auth-api.internal/api/lockout",
            json={
                "target_ip": target_ip,
                "service": service,
                "reason": "Brute force attack detected by ML",
                "duration": 1800  # 30 minutes
            },
            headers={"Authorization": "Bearer YOUR_AUTH_API_TOKEN"}
        )
        
        if response.status_code == 200:
            print(f"Locked accounts on {target_ip} for service {service}")
        else:
            print(f"Failed to lock accounts: {response.status_code}")
    
    except Exception as e:
        print(f"Error locking accounts: {e}")

def block_data_transfer(result):
    """Block suspicious data transfer."""
    # Implementation depends on your environment
    # This is just an example
    try:
        # Extract information
        source_ip = extract_source_ip(result)
        destination_ip = "203.0.113.5"  # Example, extract from result
        
        # Block the connection
        response = requests.post(
            f"{CONFIG['firewall_api']}/connections/block",
            json={
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "reason": "Suspected data exfiltration detected by ML",
                "duration": CONFIG["block_duration"]
            },
            headers={"Authorization": f"Bearer {CONFIG['firewall_token']}"}
        )
        
        if response.status_code == 200:
            print(f"Blocked connection from {source_ip} to {destination_ip}")
        else:
            print(f"Failed to block connection: {response.status_code}")
    
    except Exception as e:
        print(f"Error blocking data transfer: {e}")

def isolate_infected_system(result):
    """Isolate a system infected with C&C malware."""
    # Implementation depends on your environment
    # This is just an example
    try:
        # Extract information
        infected_ip = extract_source_ip(result)
        
        # Isolate the system
        response = requests.post(
            "https://network-api.internal/api/isolate",
            json={
                "ip_address": infected_ip,
                "reason": "Command & Control communication detected by ML",
                "isolation_level": "full"  # Or "limited" for less restrictive
            },
            headers={"Authorization": "Bearer YOUR_NETWORK_API_TOKEN"}
        )
        
        if response.status_code == 200:
            print(f"Isolated infected system: {infected_ip}")
        else:
            print(f"Failed to isolate system: {response.status_code}")
    
    except Exception as e:
        print(f"Error isolating system: {e}")

def send_notification(result):
    """Send notification about the incident."""
    try:
        # Create notification payload
        notification = {
            "type": "security_incident",
            "severity": "high",
            "title": f"ML-detected threat: {result['class_name']}",
            "description": f"Confidence: {result['confidence']:.2f}",
            "source_ip": extract_source_ip(result),
            "timestamp": time.time(),
            "automated_response": "executed",
            "details": {
                "class_probabilities": result["probabilities"],
                "recommendation": get_recommendation(result["class_name"])
            }
        }
        
        # Send notification
        response = requests.post(
            CONFIG["notification_url"],
            json=notification,
            headers={"Authorization": f"Bearer {CONFIG['notification_token']}"}
        )
        
        if response.status_code != 200:
            print(f"Failed to send notification: {response.status_code}")
    
    except Exception as e:
        print(f"Error sending notification: {e}")

def get_recommendation(threat_class):
    """Get recommendation based on threat class."""
    recommendations = {
        "Port Scan": "Review firewall rules and confirm legitimate scanning activity",
        "DDoS": "Monitor network performance and verify DDoS protection measures",
        "Brute Force": "Enforce stronger password policies and implement account lockout",
        "Data Exfiltration": "Investigate data access and increase DLP monitoring",
        "Command & Control": "Run full malware scan and isolate affected systems"
    }
    
    return recommendations.get(threat_class, "Investigate the incident")

# Cleanup task for removing expired blocks
def cleanup_expired_blocks():
    """Remove expired IP blocks."""
    while True:
        current_time = time.time()
        expired = []
        
        for ip, timestamp in block_timestamps.items():
            if current_time - timestamp >= CONFIG["block_duration"]:
                expired.append(ip)
        
        for ip in expired:
            try:
                # Remove from firewall block list
                response = requests.delete(
                    f"{CONFIG['firewall_api']}/blocks/{ip}",
                    headers={"Authorization": f"Bearer {CONFIG['firewall_token']}"}
                )
                
                if response.status_code == 200:
                    print(f"Removed expired block for IP: {ip}")
                    blocked_ips.remove(ip)
                    del block_timestamps[ip]
                else:
                    print(f"Failed to remove expired block for IP {ip}: {response.status_code}")
            
            except Exception as e:
                print(f"Error removing expired block: {e}")
        
        time.sleep(60)  # Check every minute

# Start the detector with automated response
detector.register_threat_callback(on_threat_detected)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_expired_blocks, daemon=True)
cleanup_thread.start()

# Start the detector
detector.start()

# Example packet processing (replace with your actual packet source)
# ...

# Keep the script running
try:
    while True:
        time.sleep(10)
        # Display statistics periodically
        stats = detector.get_stats()
        print(f"Processed: {stats['packets_processed']}, Threats: {stats['threats_detected']}")
except KeyboardInterrupt:
    print("Stopping...")
    detector.stop()
```

### Integration Points

- Firewalls and network security appliances
- Identity and access management systems
- Endpoint security solutions
- SOAR platforms

## Threat Hunting

### Overview

Use the interpretability features to assist in proactive threat hunting.

### Implementation

1. **Targeted Analysis**: Use `ThreatInterpreter` to understand threat patterns
2. **Feature Importance**: Analyze which features are most indicative of specific threats
3. **Hypothesis Testing**: Use the model to test threat hunting hypotheses

### Code Example

```python
from cyberthreat_ml.model import load_model
from cyberthreat_ml.interpretability import ThreatInterpreter, get_threat_pattern_insights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load pre-trained model
model = load_model("models/threat_model")

# Load feature names and class names
with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)
    feature_names = metadata.get("feature_names", [f"feature_{i}" for i in range(25)])
    class_names = metadata.get("class_names", ["Normal", "Port Scan", "DDoS", "Brute Force", 
                                           "Data Exfiltration", "Command & Control"])

# Initialize interpreter
interpreter = ThreatInterpreter(model, feature_names, class_names)

# Load historical data for analysis
historical_data = pd.read_csv("data/historical_network_traffic.csv")

# Separate features and convert to numpy array
X = historical_data.drop(columns=["label", "timestamp"]).values

# Initialize interpreter with background data
interpreter.initialize(X[:100])  # Use a subset for background

# Define threat hunting scenarios
scenarios = [
    {
        "name": "Internal Port Scanning",
        "description": "Looking for signs of internal reconnaissance",
        "filters": {
            "src_ip": "10.0.0.0/8",    # Internal IP range
            "dst_port_range": [1, 1024] # Well-known ports
        },
        "target_class": "Port Scan"
    },
    {
        "name": "Data Exfiltration to Cloud Storage",
        "description": "Detecting data being moved to unauthorized cloud storage",
        "filters": {
            "dst_ip": ["34.0.0.0/8", "52.0.0.0/8"],  # Example cloud provider IP ranges
            "bytes_out_min": 10000000  # Large outbound transfers
        },
        "target_class": "Data Exfiltration"
    },
    {
        "name": "Beaconing Activity",
        "description": "Identifying C&C beaconing patterns",
        "filters": {
            "regular_interval": True,
            "dst_port": [443, 80, 8080, 8443]  # Common HTTP/HTTPS ports
        },
        "target_class": "Command & Control"
    }
]

# Run threat hunting analysis for each scenario
for scenario in scenarios:
    print(f"\nThreat Hunting Scenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print("=" * 50)
    
    # Filter data based on scenario criteria
    filtered_data = filter_data_for_scenario(historical_data, scenario["filters"])
    
    if filtered_data.empty:
        print("No data matches the scenario criteria.")
        continue
    
    print(f"Found {len(filtered_data)} events matching criteria")
    
    # Convert to numpy for model input
    X_filtered = filtered_data.drop(columns=["label", "timestamp"]).values
    
    # Predict on filtered data
    y_pred = model.predict(X_filtered)
    y_proba = model.predict_proba(X_filtered)
    
    # Get target class index
    target_class_idx = class_names.index(scenario["target_class"]) if scenario["target_class"] in class_names else -1
    
    if target_class_idx >= 0:
        # Find instances predicted as the target class
        matches = np.where(y_pred == target_class_idx)[0]
        
        if len(matches) > 0:
            print(f"Found {len(matches)} events classified as {scenario['target_class']}")
            
            # Get high confidence predictions
            high_conf_indices = [i for i in matches if y_proba[i, target_class_idx] > 0.7]
            
            if high_conf_indices:
                print(f"Including {len(high_conf_indices)} high-confidence events")
                
                # Analyze patterns in these events
                high_conf_data = X_filtered[high_conf_indices]
                
                # Get insights using the interpreter
                insights = get_threat_pattern_insights(
                    interpreter,
                    samples=high_conf_data,
                    threat_class_id=target_class_idx,
                    top_features=10,
                    method="shap"
                )
                
                # Print insights
                print("\nKey Indicators:")
                for feature, importance in insights["top_features"]:
                    print(f"  {feature}: {importance:.4f}")
                
                if "patterns" in insights:
                    print("\nObserved Patterns:")
                    for pattern in insights["patterns"]:
                        print(f"  {pattern}")
                
                # Create samples for further manual investigation
                sample_indices = np.random.choice(high_conf_indices, min(5, len(high_conf_indices)), replace=False)
                print("\nSample events for investigation:")
                for idx in sample_indices:
                    event = filtered_data.iloc[idx]
                    print(f"  Timestamp: {event['timestamp']}")
                    print(f"  Confidence: {y_proba[idx, target_class_idx]:.4f}")
                    
                    # Explain this specific prediction
                    explanation = interpreter.explain_prediction(
                        X_filtered[idx],
                        method="shap",
                        target_class=target_class_idx,
                        top_features=5
                    )
                    
                    # Plot and save the explanation
                    plt.figure(figsize=(10, 6))
                    interpreter.plot_explanation(
                        explanation,
                        plot_type="bar",
                        save_path=f"threat_hunting/{scenario['name'].replace(' ', '_')}_{idx}.png"
                    )
                    
                    # Generate hunting report
                    report = interpreter.create_feature_importance_report(
                        explanation,
                        output_path=f"threat_hunting/{scenario['name'].replace(' ', '_')}_{idx}.txt"
                    )
                    
                    print(f"  Explanation saved to threat_hunting/{scenario['name'].replace(' ', '_')}_{idx}.png")
                    print()
            else:
                print("No high-confidence events found")
        else:
            print(f"No events classified as {scenario['target_class']}")
    else:
        print(f"Target class {scenario['target_class']} not found in model classes")

def filter_data_for_scenario(data, filters):
    """Filter data based on scenario criteria."""
    # Start with all data
    mask = pd.Series([True] * len(data), index=data.index)
    
    # Apply filters
    for key, value in filters.items():
        if key == "src_ip" and "src_ip" in data.columns:
            # IP range filtering
            if isinstance(value, str) and "/" in value:
                network = ipaddress.ip_network(value)
                mask &= data["src_ip"].apply(lambda ip: ipaddress.ip_address(ip) in network)
            else:
                mask &= data["src_ip"].isin([value] if isinstance(value, str) else value)
                
        elif key == "dst_ip" and "dst_ip" in data.columns:
            # IP range filtering
            if isinstance(value, list):
                combined_mask = pd.Series([False] * len(data), index=data.index)
                for ip_range in value:
                    if "/" in ip_range:
                        network = ipaddress.ip_network(ip_range)
                        range_mask = data["dst_ip"].apply(lambda ip: ipaddress.ip_address(ip) in network)
                        combined_mask |= range_mask
                    else:
                        combined_mask |= (data["dst_ip"] == ip_range)
                mask &= combined_mask
            elif "/" in value:
                network = ipaddress.ip_network(value)
                mask &= data["dst_ip"].apply(lambda ip: ipaddress.ip_address(ip) in network)
            else:
                mask &= (data["dst_ip"] == value)
                
        elif key == "dst_port" and "dst_port" in data.columns:
            mask &= data["dst_port"].isin([value] if isinstance(value, int) else value)
            
        elif key == "dst_port_range" and "dst_port" in data.columns:
            min_port, max_port = value
            mask &= data["dst_port"].between(min_port, max_port)
            
        elif key == "bytes_out_min" and "bytes_out" in data.columns:
            mask &= (data["bytes_out"] >= value)
            
        elif key == "regular_interval" and value == True and "timestamp" in data.columns:
            # Group by source and destination, check for regular intervals
            # This is a simplified approach
            groups = data.groupby(["src_ip", "dst_ip"])
            
            regular_conn_mask = pd.Series([False] * len(data), index=data.index)
            
            for (src, dst), group in groups:
                if len(group) >= 3:  # Need at least 3 events to check for regularity
                    # Convert timestamps to datetime objects if they're strings
                    if isinstance(group["timestamp"].iloc[0], str):
                        timestamps = pd.to_datetime(group["timestamp"])
                    else:
                        timestamps = group["timestamp"]
                    
                    # Sort timestamps
                    sorted_timestamps = timestamps.sort_values()
                    
                    # Calculate intervals
                    intervals = sorted_timestamps.diff().dropna()
                    
                    # Check if intervals are regular (low standard deviation)
                    if intervals.std() / intervals.mean() < 0.1:  # Arbitrary threshold
                        regular_conn_mask |= data.index.isin(group.index)
            
            mask &= regular_conn_mask
    
    return data[mask]

# Run threat hunting analysis
print("Starting threat hunting analysis...")
# Code execution happens here
print("Threat hunting analysis complete")
```

### Integration Points

- Threat intelligence platforms
- Log management systems
- Security data lakes
- Historical security data repositories

## Security Analytics Platform

### Overview

Build a comprehensive security analytics platform with CyberThreat-ML.

### Implementation

1. **Data Pipeline**: Create a pipeline for processing security data
2. **Analytics Engine**: Use the library's components for advanced analytics
3. **Visualization**: Integrate dashboards for security insights

### Code Example

```python
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter
from cyberthreat_ml.evaluation import plot_confusion_matrix
import pandas as pd
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
import threading
from datetime import datetime, timedelta

class SecurityAnalyticsPlatform:
    """Comprehensive security analytics platform using CyberThreat-ML."""
    
    def __init__(self, config_file="config/platform_config.json"):
        """Initialize the security analytics platform."""
        # Load configuration
        with open(config_file, "r") as f:
            self.config = json.load(f)
        
        # Create output directories
        for directory in ["models", "reports", "visualizations", "alerts"]:
            os.makedirs(f"{self.config['output_dir']}/{directory}", exist_ok=True)
        
        # Initialize components
        self._init_models()
        self._init_extractors()
        self._init_dashboard()
        self._init_interpreters()
        
        # Track analytics state
        self.analytics_running = False
        self.scheduled_tasks = {}
        
        print("Security Analytics Platform initialized")
    
    def _init_models(self):
        """Initialize threat detection models."""
        self.models = {}
        
        for model_config in self.config["models"]:
            name = model_config["name"]
            model_path = model_config["path"]
            
            try:
                # Try to load existing model
                self.models[name] = load_model(model_path)
                print(f"Loaded model '{name}' from {model_path}")
            except (FileNotFoundError, ValueError):
                # Create new model if not found
                print(f"Creating new model '{name}'")
                input_shape = tuple(model_config["input_shape"])
                num_classes = model_config["num_classes"]
                
                self.models[name] = ThreatDetectionModel(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    model_config=model_config.get("architecture", {})
                )
                
                # Train if training data provided
                if "training_data" in model_config:
                    self._train_model(name, model_config["training_data"])
                    
                    # Save the model
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.models[name].save_model(model_path)
        
        # Set the default model
        self.default_model = self.models.get(self.config.get("default_model", ""), 
                                            next(iter(self.models.values())) if self.models else None)
    
    def _train_model(self, model_name, training_config):
        """Train a model with the specified configuration."""
        # Load training data
        data_path = training_config["data_path"]
        
        try:
            # Load training data
            if data_path.endswith(".csv"):
                data = pd.read_csv(data_path)
            elif data_path.endswith(".npz"):
                with np.load(data_path) as data:
                    X = data["X"]
                    y = data["y"]
                    X_train, X_val = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
                    y_train, y_val = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
            else:
                raise ValueError(f"Unsupported data format: {data_path}")
                
            # If loaded from CSV, process the data
            if 'data' in locals():
                # Identify feature and label columns
                label_col = training_config.get("label_column", "label")
                feature_cols = [col for col in data.columns if col != label_col]
                
                # Split data
                train_data = data.sample(frac=0.8, random_state=42)
                val_data = data.drop(train_data.index)
                
                X_train = train_data[feature_cols].values
                y_train = train_data[label_col].values
                X_val = val_data[feature_cols].values
                y_val = val_data[label_col].values
            
            # Train the model
            model = self.models[model_name]
            model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=training_config.get("epochs", 10),
                batch_size=training_config.get("batch_size", 32),
                early_stopping=training_config.get("early_stopping", True)
            )
            
            print(f"Model '{model_name}' trained successfully")
            
        except Exception as e:
            print(f"Error training model '{model_name}': {str(e)}")
    
    def _init_extractors(self):
        """Initialize feature extractors."""
        self.extractors = {}
        
        for extractor_config in self.config["extractors"]:
            name = extractor_config["name"]
            
            # Create the extractor
            self.extractors[name] = FeatureExtractor(
                categorical_features=extractor_config.get("categorical_features", []),
                numeric_features=extractor_config.get("numeric_features", []),
                ip_features=extractor_config.get("ip_features", []),
                scaling=extractor_config.get("scaling", "standard"),
                handle_missing=extractor_config.get("handle_missing", True)
            )
            
            # Load saved extractor state if available
            state_path = extractor_config.get("state_path")
            if state_path and os.path.exists(state_path):
                try:
                    with open(state_path, "rb") as f:
                        import pickle
                        extractor_state = pickle.load(f)
                        self.extractors[name].__dict__.update(extractor_state)
                    print(f"Loaded extractor '{name}' state from {state_path}")
                except Exception as e:
                    print(f"Error loading extractor state: {str(e)}")
        
        # Set the default extractor
        self.default_extractor = self.extractors.get(self.config.get("default_extractor", ""),
                                                    next(iter(self.extractors.values())) if self.extractors else None)
    
    def _init_dashboard(self):
        """Initialize the visualization dashboard."""
        dashboard_config = self.config.get("dashboard", {})
        
        self.dashboard = ThreatVisualizationDashboard(
            max_history=dashboard_config.get("max_history", 1000),
            update_interval=dashboard_config.get("update_interval", 1.0)
        )
        
        # Start the dashboard if configured
        if dashboard_config.get("autostart", False):
            self.dashboard.start()
            print("Visualization dashboard started")
    
    def _init_interpreters(self):
        """Initialize threat interpreters."""
        self.interpreters = {}
        
        for model_name, model in self.models.items():
            # Find associated metadata
            metadata = next((m for m in self.config["models"] if m["name"] == model_name), {})
            
            feature_names = metadata.get("feature_names", [])
            class_names = metadata.get("class_names", [])
            
            # Create interpreter
            self.interpreters[model_name] = ThreatInterpreter(
                model=model,
                feature_names=feature_names,
                class_names=class_names
            )
            
            print(f"Created interpreter for model '{model_name}'")
    
    def start_analytics(self):
        """Start the security analytics platform."""
        if self.analytics_running:
            print("Analytics already running")
            return
        
        self.analytics_running = True
        
        # Start scheduled tasks
        for task_config in self.config.get("scheduled_tasks", []):
            self._schedule_task(task_config)
        
        # Start data sources
        for source_config in self.config.get("data_sources", []):
            self._start_data_source(source_config)
        
        print("Security analytics platform started")
    
    def stop_analytics(self):
        """Stop the security analytics platform."""
        self.analytics_running = False
        
        # Stop the dashboard
        if hasattr(self, "dashboard"):
            self.dashboard.stop()
        
        # Cancel scheduled tasks
        for task_name, task_thread in self.scheduled_tasks.items():
            if task_thread.is_alive():
                # Can't directly stop threads, but we can signal them to stop
                print(f"Signaling task '{task_name}' to stop")
        
        print("Security analytics platform stopped")
    
    def _schedule_task(self, task_config):
        """Schedule a recurring task."""
        task_name = task_config["name"]
        interval = task_config["interval_seconds"]
        task_type = task_config["type"]
        
        def task_runner():
            while self.analytics_running:
                try:
                    if task_type == "model_evaluation":
                        self._run_model_evaluation(task_config)
                    elif task_type == "threat_report":
                        self._generate_threat_report(task_config)
                    elif task_type == "dashboard_snapshot":
                        self._save_dashboard_snapshot(task_config)
                    # Add more task types as needed
                    
                except Exception as e:
                    print(f"Error in task '{task_name}': {str(e)}")
                
                # Sleep for the specified interval
                for _ in range(interval):
                    if not self.analytics_running:
                        break
                    time.sleep(1)
        
        # Start the task thread
        task_thread = threading.Thread(target=task_runner, daemon=True)
        task_thread.start()
        
        self.scheduled_tasks[task_name] = task_thread
        print(f"Scheduled task '{task_name}' (every {interval} seconds)")
    
    def _start_data_source(self, source_config):
        """Start a data source for real-time analytics."""
        source_name = source_config["name"]
        source_type = source_config["type"]
        
        def source_runner():
            try:
                if source_type == "file":
                    self._process_file_source(source_config)
                elif source_type == "api":
                    self._process_api_source(source_config)
                elif source_type == "database":
                    self._process_database_source(source_config)
                # Add more source types as needed
                
            except Exception as e:
                print(f"Error in data source '{source_name}': {str(e)}")
        
        # Start the source thread
        source_thread = threading.Thread(target=source_runner, daemon=True)
        source_thread.start()
        
        print(f"Started data source '{source_name}'")
    
    def _process_file_source(self, source_config):
        """Process a file-based data source."""
        file_path = source_config["path"]
        model_name = source_config.get("model", self.config.get("default_model", ""))
        extractor_name = source_config.get("extractor", self.config.get("default_extractor", ""))
        
        # Get model and extractor
        model = self.models.get(model_name, self.default_model)
        extractor = self.extractors.get(extractor_name, self.default_extractor)
        
        if not model or not extractor:
            print(f"Invalid model or extractor for source '{source_config['name']}'")
            return
        
        # Track file position
        file_position = 0
        
        while self.analytics_running:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    time.sleep(5)
                    continue
                
                # Open the file and seek to the last position
                with open(file_path, "r") as f:
                    f.seek(file_position)
                    
                    # Read new lines
                    new_lines = f.readlines()
                    
                    if new_lines:
                        # Process new data
                        for line in new_lines:
                            try:
                                # Parse the line (assuming JSON format)
                                event = json.loads(line.strip())
                                
                                # Extract features
                                features = extractor.transform([event])[0]
                                
                                # Make prediction
                                prediction = model.predict(features.reshape(1, -1))[0]
                                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                                
                                # Get class name
                                interpreter = self.interpreters.get(model_name)
                                if interpreter and interpreter.class_names:
                                    class_name = interpreter.class_names[prediction]
                                else:
                                    class_name = f"Class_{prediction}"
                                
                                # Create threat data
                                threat_data = {
                                    "timestamp": event.get("timestamp", time.time()),
                                    "features": features,
                                    "prediction": int(prediction),
                                    "class_name": class_name,
                                    "confidence": float(probabilities[prediction]),
                                    "source": source_config["name"],
                                    "event_id": event.get("id", ""),
                                    "probabilities": {
                                        interpreter.class_names[i] if interpreter and i < len(interpreter.class_names) else f"Class_{i}": float(prob)
                                        for i, prob in enumerate(probabilities)
                                    }
                                }
                                
                                # Add to dashboard if it's a threat
                                if prediction > 0:  # Assuming class 0 is normal
                                    self.dashboard.add_threat(threat_data)
                                    
                                    # Process high confidence threats
                                    if probabilities[prediction] > source_config.get("alert_threshold", 0.8):
                                        self._process_threat_alert(threat_data)
                            
                            except json.JSONDecodeError:
                                # Skip invalid JSON
                                continue
                            
                            except Exception as e:
                                print(f"Error processing event: {str(e)}")
                    
                    # Update file position
                    file_position = f.tell()
            
            except Exception as e:
                print(f"Error processing file source '{source_config['name']}': {str(e)}")
            
            # Sleep before checking for new data
            time.sleep(source_config.get("poll_interval", 5))
    
    def _process_api_source(self, source_config):
        """Process an API-based data source."""
        import requests
        
        api_url = source_config["url"]
        auth_header = source_config.get("auth_header", {})
        poll_interval = source_config.get("poll_interval", 60)
        model_name = source_config.get("model", self.config.get("default_model", ""))
        extractor_name = source_config.get("extractor", self.config.get("default_extractor", ""))
        
        # Get model and extractor
        model = self.models.get(model_name, self.default_model)
        extractor = self.extractors.get(extractor_name, self.default_extractor)
        
        if not model or not extractor:
            print(f"Invalid model or extractor for source '{source_config['name']}'")
            return
        
        # Track last event timestamp
        last_timestamp = time.time()
        
        while self.analytics_running:
            try:
                # Query the API for new data
                response = requests.get(
                    api_url,
                    headers=auth_header,
                    params={"since": last_timestamp}
                )
                
                if response.status_code == 200:
                    events = response.json()
                    
                    if events:
                        # Process events
                        for event in events:
                            try:
                                # Extract features
                                features = extractor.transform([event])[0]
                                
                                # Make prediction
                                prediction = model.predict(features.reshape(1, -1))[0]
                                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                                
                                # Get class name
                                interpreter = self.interpreters.get(model_name)
                                if interpreter and interpreter.class_names:
                                    class_name = interpreter.class_names[prediction]
                                else:
                                    class_name = f"Class_{prediction}"
                                
                                # Create threat data
                                threat_data = {
                                    "timestamp": event.get("timestamp", time.time()),
                                    "features": features,
                                    "prediction": int(prediction),
                                    "class_name": class_name,
                                    "confidence": float(probabilities[prediction]),
                                    "source": source_config["name"],
                                    "event_id": event.get("id", ""),
                                    "probabilities": {
                                        interpreter.class_names[i] if interpreter and i < len(interpreter.class_names) else f"Class_{i}": float(prob)
                                        for i, prob in enumerate(probabilities)
                                    }
                                }
                                
                                # Add to dashboard if it's a threat
                                if prediction > 0:  # Assuming class 0 is normal
                                    self.dashboard.add_threat(threat_data)
                                    
                                    # Process high confidence threats
                                    if probabilities[prediction] > source_config.get("alert_threshold", 0.8):
                                        self._process_threat_alert(threat_data)
                                
                                # Update last timestamp
                                event_timestamp = event.get("timestamp", time.time())
                                if event_timestamp > last_timestamp:
                                    last_timestamp = event_timestamp
                            
                            except Exception as e:
                                print(f"Error processing event: {str(e)}")
                
                else:
                    print(f"API request failed: {response.status_code}")
            
            except Exception as e:
                print(f"Error processing API source '{source_config['name']}': {str(e)}")
            
            # Sleep before next poll
            time.sleep(poll_interval)
    
    def _process_database_source(self, source_config):
        """Process a database-based data source."""
        # Implementation would depend on your database library
        pass  # Implement based on your requirements
    
    def _process_threat_alert(self, threat_data):
        """Process a high-confidence threat alert."""
        # Generate alert file
        timestamp = datetime.fromtimestamp(threat_data["timestamp"]).strftime("%Y%m%d_%H%M%S")
        alert_id = f"{threat_data['class_name'].lower().replace(' ', '_')}_{timestamp}"
        
        alert_path = f"{self.config['output_dir']}/alerts/{alert_id}.json"
        
        # Create alert data
        alert = {
            "id": alert_id,
            "timestamp": threat_data["timestamp"],
            "class": threat_data["class_name"],
            "confidence": threat_data["confidence"],
            "source": threat_data["source"],
            "event_id": threat_data.get("event_id", ""),
            "probabilities": threat_data["probabilities"],
            "recommended_action": self._get_recommended_action(threat_data["class_name"])
        }
        
        # Save alert to file
        with open(alert_path, "w") as f:
            json.dump(alert, f, indent=2)
        
        print(f"Generated alert: {alert_id} - {threat_data['class_name']} ({threat_data['confidence']:.2f})")
        
        # Could integrate with external systems here
        # For example, sending to a SIEM, email, Slack, etc.
    
    def _get_recommended_action(self, threat_class):
        """Get recommended action for a threat class."""
        actions = {
            "Port Scan": "Review firewall rules and investigate source",
            "DDoS": "Activate DDoS protection measures and monitor network performance",
            "Brute Force": "Lock affected accounts and implement stronger authentication",
            "Data Exfiltration": "Isolate affected systems and investigate data access",
            "Command & Control": "Isolate infected system and block C&C domains/IPs"
        }
        
        return actions.get(threat_class, "Investigate the threat")
    
    def _run_model_evaluation(self, task_config):
        """Run model evaluation task."""
        model_name = task_config.get("model", self.config.get("default_model", ""))
        data_path = task_config.get("data_path")
        
        if not data_path or not os.path.exists(data_path):
            print(f"Data path not found for model evaluation: {data_path}")
            return
        
        model = self.models.get(model_name, self.default_model)
        
        if not model:
            print(f"Model not found for evaluation: {model_name}")
            return
        
        try:
            # Load evaluation data
            if data_path.endswith(".csv"):
                data = pd.read_csv(data_path)
                
                # Identify feature and label columns
                label_col = task_config.get("label_column", "label")
                feature_cols = [col for col in data.columns if col != label_col]
                
                X_test = data[feature_cols].values
                y_test = data[label_col].values
                
            elif data_path.endswith(".npz"):
                with np.load(data_path) as data:
                    X_test = data["X_test"]
                    y_test = data["y_test"]
            else:
                print(f"Unsupported data format for evaluation: {data_path}")
                return
            
            # Run evaluation
            from cyberthreat_ml.evaluation import evaluate_model, classification_report, plot_confusion_matrix
            
            # Basic metrics
            metrics = evaluate_model(model, X_test, y_test)
            
            # Classification report
            report = classification_report(model, X_test, y_test)
            
            # Confusion matrix
            cm_fig = plot_confusion_matrix(model, X_test, y_test, normalize=True)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics
            metrics_path = f"{self.config['output_dir']}/reports/evaluation_{model_name}_{timestamp}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save report
            report_path = f"{self.config['output_dir']}/reports/classification_{model_name}_{timestamp}.txt"
            with open(report_path, "w") as f:
                f.write(report)
            
            # Save confusion matrix
            cm_path = f"{self.config['output_dir']}/visualizations/confusion_matrix_{model_name}_{timestamp}.png"
            cm_fig.savefig(cm_path)
            
            print(f"Model evaluation completed for '{model_name}':")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Saved results to {self.config['output_dir']}/reports/ and {self.config['output_dir']}/visualizations/")
            
        except Exception as e:
            print(f"Error in model evaluation task: {str(e)}")
    
    def _generate_threat_report(self, task_config):
        """Generate a threat report."""
        # Configuration
        lookback_period = task_config.get("lookback_hours", 24)
        min_confidence = task_config.get("min_confidence", 0.7)
        
        # Determine time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_period)
        
        # Get alerts in the time range
        alerts = []
        alerts_dir = f"{self.config['output_dir']}/alerts/"
        
        if os.path.exists(alerts_dir):
            for filename in os.listdir(alerts_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(alerts_dir, filename), "r") as f:
                            alert = json.load(f)
                            
                            # Check if in time range
                            if "timestamp" in alert:
                                alert_time = datetime.fromtimestamp(alert["timestamp"])
                                if start_time <= alert_time <= end_time and alert.get("confidence", 0) >= min_confidence:
                                    alerts.append(alert)
                    except Exception as e:
                        print(f"Error reading alert file {filename}: {str(e)}")
        
        # Generate report
        if alerts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{self.config['output_dir']}/reports/threat_report_{timestamp}.txt"
            
            with open(report_path, "w") as f:
                f.write("SECURITY THREAT REPORT\n")
                f.write("=====================\n\n")
                f.write(f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary by threat type
                threat_types = {}
                for alert in alerts:
                    threat_class = alert.get("class", "Unknown")
                    if threat_class not in threat_types:
                        threat_types[threat_class] = []
                    threat_types[threat_class].append(alert)
                
                f.write(f"SUMMARY ({len(alerts)} high-confidence threats detected)\n")
                f.write("-" * 50 + "\n")
                
                for threat_class, class_alerts in threat_types.items():
                    f.write(f"{threat_class}: {len(class_alerts)}\n")
                
                f.write("\nDETAILED THREAT ANALYSIS\n")
                f.write("-" * 50 + "\n\n")
                
                # Detailed analysis by threat type
                for threat_class, class_alerts in threat_types.items():
                    f.write(f"{threat_class.upper()} THREATS\n")
                    f.write("=" * len(threat_class + " THREATS") + "\n\n")
                    
                    # Sort by confidence
                    class_alerts.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    
                    for alert in class_alerts:
                        alert_time = datetime.fromtimestamp(alert["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"Alert ID: {alert.get('id', 'Unknown')}\n")
                        f.write(f"Time: {alert_time}\n")
                        f.write(f"Confidence: {alert.get('confidence', 0):.4f}\n")
                        f.write(f"Source: {alert.get('source', 'Unknown')}\n")
                        
                        if "probabilities" in alert:
                            f.write("Threat Probabilities:\n")
                            for cls, prob in alert["probabilities"].items():
                                f.write(f"  {cls}: {prob:.4f}\n")
                        
                        if "recommended_action" in alert:
                            f.write(f"Recommended Action: {alert['recommended_action']}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 50 + "\n\n")
                
                # Add recommendations based on detected threats
                if "Port Scan" in threat_types:
                    f.write("- Review firewall rules and network segmentation\n")
                    f.write("- Implement network access controls\n")
                
                if "DDoS" in threat_types:
                    f.write("- Verify DDoS protection measures\n")
                    f.write("- Monitor network traffic patterns\n")
                
                if "Brute Force" in threat_types:
                    f.write("- Implement account lockout policies\n")
                    f.write("- Enable multi-factor authentication\n")
                
                if "Data Exfiltration" in threat_types:
                    f.write("- Review data loss prevention policies\n")
                    f.write("- Audit sensitive data access\n")
                
                if "Command & Control" in threat_types:
                    f.write("- Scan all systems for malware\n")
                    f.write("- Review outbound connection policies\n")
            
            print(f"Generated threat report: {report_path}")
        else:
            print("No high-confidence threats found for report")
    
    def _save_dashboard_snapshot(self, task_config):
        """Save a snapshot of the visualization dashboard."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = f"{self.config['output_dir']}/visualizations/dashboard_{timestamp}.png"
        
        try:
            self.dashboard.save_snapshot(snapshot_path)
            print(f"Saved dashboard snapshot: {snapshot_path}")
        except Exception as e:
            print(f"Error saving dashboard snapshot: {str(e)}")

# Example usage
if __name__ == "__main__":
    platform = SecurityAnalyticsPlatform("config/platform_config.json")
    platform.start_analytics()
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        platform.stop_analytics()
        print("Platform stopped")
```

### Integration Points

- Data ingestion from various security sources
- Visualization platforms
- Reporting systems
- Alerting mechanisms
- Security data lakes

## Customizing for Specific Threats

### Overview

Adapt CyberThreat-ML for specific threat types by customizing models and features.

### Implementation

1. **Custom Feature Engineering**: Define domain-specific features
2. **Specialized Models**: Train models for specific threat categories
3. **Post-processing**: Implement domain-specific result interpretation

### Code Example

```python
from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.preprocessing import FeatureExtractor
import pandas as pd
import numpy as np
import re
import ipaddress

class RansomwareDetector:
    """Custom detector for ransomware threats using CyberThreat-ML."""
    
    def __init__(self):
        """Initialize the ransomware detector."""
        # Create specialized model
        self.model = ThreatDetectionModel(
            input_shape=(25,),
            num_classes=2,  # Binary classification: ransomware or not
            model_config={
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.4,
                'activation': 'relu'
            }
        )
        
        # Define ransomware-specific features
        self.feature_names = [
            # Process behavior features
            "process_creation_rate",
            "file_modification_rate",
            "file_encryption_attempts",
            "registry_modification_rate",
            "unusual_process_activity",
            
            # File system features
            "file_extension_changes",
            "entropy_of_modified_files",
            "sequential_file_access",
            "sensitive_file_access",
            "shadow_copy_deletion",
            
            # Network features
            "connection_to_known_c2",
            "bitcoin_address_in_traffic",
            "tor_traffic_detected",
            "unusual_dns_requests",
            "outbound_connection_rate",
            
            # System features
            "credential_access_attempts",
            "privilege_escalation_attempts",
            "boot_record_modification",
            "service_creation",
            "scheduled_task_creation",
            
            # Misc
            "known_ransomware_patterns",
            "ransom_note_detection",
            "suspicious_powershell_commands",
            "suspicious_script_execution",
            "backup_deletion_attempts"
        ]
    
    def train(self, training_data_path):
        """Train the ransomware detection model."""
        # Load and preprocess data
        data = pd.read_csv(training_data_path)
        
        # Extract features
        X = self._extract_ransomware_features(data)
        
        # Get labels (assuming 1 for ransomware, 0 for benign)
        y = data["is_ransomware"].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=15,
            batch_size=32,
            early_stopping=True
        )
        
        print("Ransomware detection model trained successfully")
    
    def _extract_ransomware_features(self, data):
        """Extract ransomware-specific features from the data."""
        features = np.zeros((len(data), len(self.feature_names)))
        
        # Process behavior features
        features[:, 0] = data["process_creation_count"] / data["time_window"]
        features[:, 1] = data["file_modification_count"] / data["time_window"]
        features[:, 2] = data["encryption_api_calls"]
        features[:, 3] = data["registry_modification_count"] / data["time_window"]
        features[:, 4] = data["process_anomaly_score"]
        
        # File system features
        features[:, 5] = data["file_extension_change_count"]
        features[:, 6] = data["modified_files_entropy"]
        features[:, 7] = data["sequential_access_score"]
        features[:, 8] = data["sensitive_file_access_count"]
        features[:, 9] = data["shadow_copy_commands"].apply(lambda x: 1 if x > 0 else 0)
        
        # Network features
        features[:, 10] = data["known_c2_connections"].apply(lambda x: 1 if x > 0 else 0)
        features[:, 11] = data["bitcoin_address_count"].apply(lambda x: 1 if x > 0 else 0)
        features[:, 12] = data["tor_traffic"].apply(lambda x: 1 if x > 0 else 0)
        features[:, 13] = data["unusual_dns_score"]
        features[:, 14] = data["outbound_connection_count"] / data["time_window"]
        
        # System features
        features[:, 15] = data["credential_access_attempts"]
        features[:, 16] = data["privilege_escalation_attempts"]
        features[:, 17] = data["boot_record_access"].apply(lambda x: 1 if x > 0 else 0)
        features[:, 18] = data["service_creation_count"]
        features[:, 19] = data["scheduled_task_creation_count"]
        
        # Misc
        features[:, 20] = data["known_ransomware_pattern_matches"]
        features[:, 21] = data["ransom_note_detected"].apply(lambda x: 1 if x > 0 else 0)
        features[:, 22] = data["suspicious_powershell_commands"]
        features[:, 23] = data["suspicious_script_executions"]
        features[:, 24] = data["backup_deletion_attempts"]
        
        return features
    
    def detect(self, event_data):
        """
        Detect ransomware from event data.
        
        Args:
            event_data (dict): Event data from security monitoring.
            
        Returns:
            dict: Detection result.
        """
        # Extract features
        features = self._extract_features_from_event(event_data)
        
        # Make prediction
        prediction = self.model.predict(features.reshape(1, -1))[0]
        probability = self.model.predict_proba(features.reshape(1, -1))[0][1]  # Probability of ransomware
        
        # Add additional context
        result = {
            "is_ransomware": bool(prediction),
            "confidence": float(probability),
            "timestamp": event_data.get("timestamp", time.time()),
            "source": event_data.get("source", "unknown"),
            "event_id": event_data.get("id", ""),
            "indicators": self._extract_ransomware_indicators(event_data)
        }
        
        # Add remediation steps if ransomware detected
        if result["is_ransomware"]:
            result["remediation_steps"] = self._get_remediation_steps(result["indicators"])
        
        return result
    
    def _extract_features_from_event(self, event):
        """
        Extract features from a security event.
        
        Args:
            event (dict): Security event data.
            
        Returns:
            numpy.ndarray: Feature vector.
        """
        # Initialize features
        features = np.zeros(len(self.feature_names))
        
        # Time window for rate calculations (default 5 minutes)
        time_window = event.get("time_window", 300)
        
        # Extract features based on event type
        if event.get("type") == "process":
            # Process events
            process_events = event.get("process_events", [])
            features[0] = len(process_events) / time_window  # process_creation_rate
            
            # Check for suspicious processes
            suspicious_processes = [
                "vssadmin.exe delete shadows",
                "wbadmin delete catalog",
                "bcdedit",
                "powershell -enc",
                "wmic shadowcopy delete"
            ]
            
            for proc in process_events:
                cmd_line = proc.get("command_line", "").lower()
                
                # Check suspicious PowerShell commands
                if "powershell" in cmd_line:
                    features[22] += 1  # suspicious_powershell_commands
                    
                    if "-enc" in cmd_line or "-encodedcommand" in cmd_line:
                        features[22] += 2  # More suspicious
                
                # Check for shadow copy deletion
                if "vssadmin" in cmd_line and "delete" in cmd_line:
                    features[9] = 1  # shadow_copy_deletion
                
                # Check for boot record modification
                if "bcdedit" in cmd_line:
                    features[17] = 1  # boot_record_modification
                
                # Check for service creation
                if "sc create" in cmd_line or "new-service" in cmd_line:
                    features[18] += 1  # service_creation_count
                
                # Check for scheduled tasks
                if "schtasks" in cmd_line or "taskschd" in cmd_line:
                    features[19] += 1  # scheduled_task_creation_count
                
                # Check for backup deletion
                if ("wbadmin" in cmd_line and "delete" in cmd_line) or \
                   ("backup" in cmd_line and "delete" in cmd_line):
                    features[24] += 1  # backup_deletion_attempts
        
        elif event.get("type") == "file":
            # File events
            file_events = event.get("file_events", [])
            features[1] = len(file_events) / time_window  # file_modification_rate
            
            # Track extensions and entropy
            extension_changes = 0
            sensitive_accesses = 0
            
            for file_event in file_events:
                path = file_event.get("path", "")
                operation = file_event.get("operation", "")
                
                # Check for extension changes (potential encryption)
                if operation == "rename" and "old_path" in file_event:
                    old_ext = os.path.splitext(file_event["old_path"])[1]
                    new_ext = os.path.splitext(path)[1]
                    
                    if old_ext != new_ext:
                        extension_changes += 1
                        
                        # Check for known ransomware extensions
                        ransomware_extensions = [
                            ".encrypted", ".locked", ".crypto", ".wallet", ".ransom",
                            ".crypt", ".pay", ".wcry", ".wncry", ".locky", ".cerber"
                        ]
                        
                        if new_ext.lower() in ransomware_extensions:
                            features[20] += 1  # known_ransomware_patterns
                
                # Check for sensitive file access
                sensitive_patterns = [
                    r"\.doc$", r"\.docx$", r"\.xls$", r"\.xlsx$", r"\.pdf$", 
                    r"\.jpg$", r"\.jpeg$", r"\.png$", r"\.txt$", r"\.sql$",
                    r"\\Users\\.*\\Documents", r"\\Users\\.*\\Pictures"
                ]
                
                for pattern in sensitive_patterns:
                    if re.search(pattern, path, re.IGNORECASE):
                        sensitive_accesses += 1
                        break
                
                # Check for ransom notes
                ransom_note_patterns = [
                    "readme.txt", "howtodecrypt", "how_to_decrypt", 
                    "help_decrypt", "ransom", "recover_file"
                ]
                
                for pattern in ransom_note_patterns:
                    if pattern in path.lower():
                        features[21] = 1  # ransom_note_detection
                        break
            
            features[5] = extension_changes  # file_extension_changes
            features[8] = sensitive_accesses  # sensitive_file_access
            
            # File entropy (if available)
            if "file_entropy" in event:
                features[6] = event["file_entropy"]  # entropy_of_modified_files
        
        elif event.get("type") == "network":
            # Network events
            network_events = event.get("network_events", [])
            
            # Check for C2 connections
            known_c2_ips = event.get("known_c2_ips", [])
            bitcoin_patterns = [
                r"1[a-km-zA-HJ-NP-Z1-9]{25,34}",  # Bitcoin address
                r"www\.coinbase\.com",
                r"blockchain\.info"
            ]
            
            for net_event in network_events:
                dst_ip = net_event.get("dst_ip", "")
                dst_domain = net_event.get("dst_domain", "")
                http_content = net_event.get("http_content", "")
                
                # Check for known C2
                if dst_ip in known_c2_ips:
                    features[10] = 1  # connection_to_known_c2
                
                # Check for TOR traffic
                if dst_domain.endswith(".onion") or "tor" in dst_domain:
                    features[12] = 1  # tor_traffic_detected
                
                # Check for Bitcoin addresses in content
                for pattern in bitcoin_patterns:
                    if re.search(pattern, http_content, re.IGNORECASE):
                        features[11] = 1  # bitcoin_address_in_traffic
                        break
            
            # Outbound connections
            features[14] = len(network_events) / time_window  # outbound_connection_rate
        
        # Additional features if available
        features[2] = event.get("encryption_api_calls", 0)  # file_encryption_attempts
        features[3] = event.get("registry_modifications", 0) / time_window  # registry_modification_rate
        features[4] = event.get("process_anomaly_score", 0)  # unusual_process_activity
        features[7] = event.get("sequential_file_access_score", 0)  # sequential_file_access
        features[13] = event.get("unusual_dns_score", 0)  # unusual_dns_requests
        features[15] = event.get("credential_access_attempts", 0)  # credential_access_attempts
        features[16] = event.get("privilege_escalation_attempts", 0)  # privilege_escalation_attempts
        features[23] = event.get("suspicious_script_executions", 0)  # suspicious_script_execution
        
        return features
    
    def _extract_ransomware_indicators(self, event):
        """
        Extract specific ransomware indicators from the event.
        
        Args:
            event (dict): Security event data.
            
        Returns:
            list: List of ransomware indicators found.
        """
        indicators = []
        
        # Check for specific indicators based on event type
        if event.get("type") == "process":
            process_events = event.get("process_events", [])
            
            for proc in process_events:
                cmd_line = proc.get("command_line", "").lower()
                
                # Shadow copy deletion
                if "vssadmin" in cmd_line and "delete shadows" in cmd_line:
                    indicators.append("Shadow copy deletion attempt")
                
                # Backup deletion
                if "wbadmin delete catalog" in cmd_line:
                    indicators.append("Backup catalog deletion")
                
                # Boot configuration modification
                if "bcdedit" in cmd_line and ("recoveryenabled no" in cmd_line or 
                                           "bootstatuspolicy ignoreallfailures" in cmd_line):
                    indicators.append("Recovery settings modification")
                
                # Encoded PowerShell commands
                if "powershell" in cmd_line and ("-enc" in cmd_line or "-encodedcommand" in cmd_line):
                    indicators.append("Obfuscated PowerShell execution")
        
        elif event.get("type") == "file":
            file_events = event.get("file_events", [])
            
            # Track affected extensions
            affected_extensions = set()
            
            for file_event in file_events:
                path = file_event.get("path", "")
                operation = file_event.get("operation", "")
                
                # Extension changes
                if operation == "rename" and "old_path" in file_event:
                    old_ext = os.path.splitext(file_event["old_path"])[1]
                    new_ext = os.path.splitext(path)[1]
                    
                    if old_ext != new_ext:
                        affected_extensions.add(new_ext)
                        
                        # Known ransomware extensions
                        ransomware_extensions = [
                            ".encrypted", ".locked", ".crypto", ".wallet", ".ransom",
                            ".crypt", ".pay", ".wcry", ".wncry", ".locky", ".cerber"
                        ]
                        
                        if new_ext.lower() in ransomware_extensions:
                            indicators.append(f"Known ransomware file extension: {new_ext}")
                
                # Ransom notes
                ransom_note_patterns = [
                    "readme.txt", "howtodecrypt", "how_to_decrypt", 
                    "help_decrypt", "ransom", "recover_file"
                ]
                
                for pattern in ransom_note_patterns:
                    if pattern in path.lower():
                        indicators.append("Ransom note detected")
                        break
            
            if len(affected_extensions) > 3:
                indicators.append(f"Multiple file extensions changed ({len(affected_extensions)})")
        
        elif event.get("type") == "network":
            network_events = event.get("network_events", [])
            
            for net_event in network_events:
                dst_domain = net_event.get("dst_domain", "")
                http_content = net_event.get("http_content", "")
                
                # TOR traffic
                if dst_domain.endswith(".onion"):
                    indicators.append("TOR network communication")
                
                # Bitcoin address in traffic
                bitcoin_pattern = r"1[a-km-zA-HJ-NP-Z1-9]{25,34}"
                if re.search(bitcoin_pattern, http_content):
                    indicators.append("Bitcoin address in network traffic")
                
                # Known C2 communication
                known_c2_ips = event.get("known_c2_ips", [])
                if net_event.get("dst_ip", "") in known_c2_ips:
                    indicators.append("Communication with known command & control server")
        
        # Common ransomware IOCs from event metadata
        if event.get("encryption_api_calls", 0) > 5:
            indicators.append("Multiple encryption API calls")
        
        if event.get("file_ops_per_second", 0) > 10:
            indicators.append("High rate of file operations")
        
        if event.get("sequential_file_access_score", 0) > 0.7:
            indicators.append("Sequential file access pattern")
        
        return indicators
    
    def _get_remediation_steps(self, indicators):
        """
        Generate remediation steps based on detected indicators.
        
        Args:
            indicators (list): List of detected ransomware indicators.
            
        Returns:
            list: Recommended remediation steps.
        """
        steps = [
            "Immediately isolate affected systems from the network",
            "Stop any suspicious processes",
            "Preserve forensic evidence for investigation"
        ]
        
        # Add specific steps based on indicators
        if "Shadow copy deletion attempt" in indicators or "Backup catalog deletion" in indicators:
            steps.append("Check backup integrity and restore from offline backups if available")
        
        if "Communication with known command & control server" in indicators:
            steps.append("Block identified C2 domains/IPs at firewall and proxy")
        
        if "Multiple file extensions changed" in indicators:
            steps.append("Identify the scope of encryption and affected files")
        
        if "Ransom note detected" in indicators:
            steps.append("Document ransom note details for law enforcement")
            steps.append("Contact cybersecurity incident response team or law enforcement")
        
        if "Obfuscated PowerShell execution" in indicators:
            steps.append("Check for persistent malware using memory forensics")
        
        # Add standard steps
        steps.extend([
            "Scan all systems for indicators of compromise",
            "Reset compromised credentials",
            "Develop a recovery plan before proceeding",
            "Consider engaging ransomware specialists if critical systems are affected"
        ])
        
        return steps

# Example usage
def main():
    # Create ransomware detector
    detector = RansomwareDetector()
    
    # Train the detector (with your own data)
    # detector.train("data/ransomware_training_data.csv")
    
    # Simulate an event for detection
    event = {
        "type": "process",
        "timestamp": time.time(),
        "source": "endpoint_agent",
        "id": "evt-12345",
        "time_window": 300,  # 5 minutes
        "process_events": [
            {
                "process_id": 1234,
                "process_name": "vssadmin.exe",
                "command_line": "vssadmin.exe delete shadows /all /quiet",
                "user": "SYSTEM",
                "start_time": time.time() - 60
            },
            {
                "process_id": 1235,
                "process_name": "powershell.exe",
                "command_line": "powershell.exe -enc QWRkLU1wUHJlZmVyZW5jZSAta2V5ICdIb25leXBvdFg1MicgLXZhbHVlICdJdHMgdGltZSBmb3IgdGVhJw==",
                "user": "user1",
                "start_time": time.time() - 120
            }
        ],
        "encryption_api_calls": 12,
        "registry_modifications": 5,
        "process_anomaly_score": 0.85,
        "privilege_escalation_attempts": 1
    }
    
    # Detect ransomware
    result = detector.detect(event)
    
    # Display results
    print("Ransomware Detection Result:")
    print(f"  Is Ransomware: {result['is_ransomware']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    if result['is_ransomware']:
        print("\nDetected Indicators:")
        for indicator in result['indicators']:
            print(f"  - {indicator}")
        
        print("\nRecommended Remediation Steps:")
        for i, step in enumerate(result['remediation_steps'], 1):
            print(f"  {i}. {step}")

if __name__ == "__main__":
    import time
    main()
```

### Integration Points

- Specific threat intelligence feeds
- Domain-specific log sources
- Specialized security tools
- Industry-specific alert systems