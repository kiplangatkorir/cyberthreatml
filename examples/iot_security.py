"""
Example demonstrating how to apply CyberThreat-ML to IoT security scenarios.

This example shows:
1. How to detect anomalies in IoT device behavior
2. How to identify and classify specific IoT threats 
3. How to implement lightweight monitoring for resource-constrained environments
4. How to integrate with IoT-specific protocols and data formats
"""

import os
import time
import logging
import numpy as np
from datetime import datetime, timedelta
import json
import random

# Import from cyberthreat_ml library
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.realtime import RealTimeDetector
from cyberthreat_ml.interpretability import ThreatInterpreter
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.logger import CyberThreatLogger

# Configure logging
logger = CyberThreatLogger("iot_security", logging.INFO).get_logger()

def main():
    """
    Example of applying CyberThreat-ML to IoT security scenarios.
    """
    print("Starting IoT Security Example...")
    
    # Create output directories
    os.makedirs("security_output/iot", exist_ok=True)
    
    # Step 1: Initialize or create the model
    logger.info("Initializing IoT threat detection model")
    model_path = "models/iot_threat_model"
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Create and train a new model
    logger.info("Creating and training new IoT threat model")
    model = create_and_train_model()
    model.save_model(model_path)
    
    # Step 2: Define IoT-specific threat classes
    class_names = [
        "Normal Behavior", 
        "Botnet Activity", 
        "Firmware Tampering", 
        "Data Exfiltration", 
        "Command Injection", 
        "Replay Attack"
    ]
    
    feature_names = [
        "Power Consumption", 
        "CPU Usage", 
        "Memory Usage", 
        "Network Throughput", 
        "Packet Rate",
        "Connection Duration", 
        "Packet Size", 
        "Destination IP Entropy", 
        "Protocol Type", 
        "Time Pattern",
        "Authentication Failures", 
        "Encryption Level", 
        "Data Flow Direction", 
        "Command Frequency",
        "Firmware Checksum", 
        "Sensor Reading Variance", 
        "Temperature", 
        "Connection Count",
        "Outbound Data Volume", 
        "Inbound Data Volume"
    ]
    
    # Step 3: Setup visualization and interpretability
    logger.info("Setting up IoT security monitoring")
    dashboard = ThreatVisualizationDashboard()
    dashboard.start()
    
    interpreter = ThreatInterpreter(model, feature_names, class_names)
    interpreter.initialize(create_background_data())
    logger.info("IoT threat interpreter initialized")
    
    # Step 4: Create IoT device simulator and detector
    detector = IoTDeviceDetector(
        model,
        IoTFeatureExtractor(),
        threshold=0.5,
        batch_size=5,
        processing_interval=1.0
    )
    
    # Track detected threats
    detected_threats = {
        "botnet_activity": [],
        "firmware_tampering": [],
        "data_exfiltration": [],
        "command_injection": [],
        "replay_attack": []
    }
    
    # Register callback for threat detection
    def on_threat_detected(result):
        threat_type = result["class_name"].lower().replace(" ", "_")
        if threat_type != "normal_behavior" and threat_type in detected_threats:
            detected_threats[threat_type].append(result)
            
            # Add to dashboard
            dashboard.add_threat(result)
            
            # Log threat
            logger.warning(f"IoT Threat Detected: {result['class_name']} "
                         f"(Confidence: {result['confidence']:.4f})")
            
            # Generate alert for high-confidence threats
            if result["confidence"] > 0.7:
                generate_iot_alert(result)
    
    detector.register_threat_callback(on_threat_detected)
    detector.start()
    
    # Step 5: Simulate IoT devices and traffic
    logger.info("Simulating IoT device traffic...")
    simulate_iot_devices(detector, duration=10, device_count=5)
    
    # Step 6: Generate IoT security report
    logger.info("Generating IoT security report...")
    report_file = generate_iot_security_report(detected_threats, class_names)
    logger.info(f"IoT security report generated: {report_file}")
    
    # Step 7: Interpret threats for each threat type
    for cls_name, threats in detected_threats.items():
        if threats:
            # Get the first threat for analysis
            sample_threats = threats[:min(1, len(threats))]
            features = np.vstack([t["features"] for t in sample_threats])
            
            try:
                cls_idx = next(i for i, name in enumerate(class_names) 
                            if name.lower().replace(" ", "_") == cls_name)
                
                # Explain a sample threat
                explanation = interpreter.explain_prediction(
                    features[0],
                    method="rules",
                    target_class=cls_idx,
                    top_features=5
                )
                
                # Save explanation
                interpreter.plot_explanation(
                    explanation,
                    plot_type="bar",
                    save_path=f"security_output/iot/{cls_name}_explanation.png"
                )
                
                # Create text report
                report = interpreter.create_feature_importance_report(
                    explanation,
                    output_path=f"security_output/iot/{cls_name}_report.txt"
                )
                
            except (IndexError, StopIteration, ValueError) as e:
                logger.error(f"Error analyzing {cls_name} threats: {str(e)}")
    
    # Step 8: Demonstrate lightweight anomaly detection
    logger.info("Demonstrating lightweight anomaly detection...")
    print("\nLightweight IoT Anomaly Detection Results:")
    anomalies = detect_lightweight_anomalies(create_iot_telemetry(100), feature_names[:5])
    
    for i, anomaly in enumerate(anomalies[:5]):
        print(f"Anomaly {i+1}:")
        print(f"  Timestamp: {anomaly['timestamp']}")
        print(f"  Device ID: {anomaly['device_id']}")
        print(f"  Anomaly Score: {anomaly['anomaly_score']:.4f}")
        print(f"  Affected Metrics: {', '.join(anomaly['affected_metrics'])}")
    
    # Step 9: Clean up
    time.sleep(1)  # Allow time for processing
    detector.stop()
    
    # Save final dashboard view
    try:
        dashboard.save_snapshot("security_output/iot/iot_dashboard.png")
    except Exception as e:
        logger.error(f"Error saving dashboard snapshot: {str(e)}")
    
    dashboard.stop()
    logger.info("IoT security example completed")
    print("IoT security example completed successfully!")

def create_and_train_model():
    """
    Create and train a threat detection model for IoT security.
    
    Returns:
        ThreatDetectionModel: Trained model.
    """
    # Create synthetic dataset
    X, y, _ = create_synthetic_iot_dataset(n_samples=2000, n_features=20, n_classes=6)
    
    # Split into train/val
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    # Create and train model
    model = ThreatDetectionModel(
        input_shape=(20,),
        num_classes=6,
        model_config={
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.25
        }
    )
    
    model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=10,
        batch_size=32,
        early_stopping=True
    )
    
    return model

def create_background_data():
    """
    Create background data for the interpreter.
    
    Returns:
        numpy.ndarray: Background data for SHAP.
    """
    # Create a small synthetic dataset for background
    X, _, _ = create_synthetic_iot_dataset(n_samples=100, n_features=20, n_classes=6)
    return X

def create_synthetic_iot_dataset(n_samples=2000, n_features=20, n_classes=6, normal_prob=0.6):
    """
    Create a synthetic dataset for IoT threat detection.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_classes (int): Number of threat classes (including normal traffic).
        normal_prob (float): Probability of normal behavior samples.
        
    Returns:
        tuple: (X, y, class_names) - features, one-hot encoded labels, and class names.
    """
    X = np.random.rand(n_samples, n_features)
    
    # Generate class labels with a bias toward normal behavior
    y_classes = np.zeros(n_samples, dtype=int)
    normal_samples = int(n_samples * normal_prob)
    y_classes[:normal_samples] = 0  # Normal behavior
    
    # Distribute the rest among the threat classes
    threat_samples = n_samples - normal_samples
    for i in range(threat_samples):
        y_classes[normal_samples + i] = 1 + (i % (n_classes - 1))
    
    # Shuffle the dataset
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y_classes = y_classes[indices]
    
    # Create one-hot encoded labels
    y = np.zeros((n_samples, n_classes))
    for i, cls in enumerate(y_classes):
        y[i, cls] = 1
    
    class_names = [
        "Normal Behavior", 
        "Botnet Activity", 
        "Firmware Tampering", 
        "Data Exfiltration", 
        "Command Injection", 
        "Replay Attack"
    ]
    
    # Make features more realistic for IoT scenarios
    # Power consumption patterns (typically follows diurnal patterns)
    X[:, 0] = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.3 + X[:, 0] * 0.7
    
    # Memory usage (typically correlated with CPU usage)
    X[:, 2] = X[:, 1] * 0.6 + X[:, 2] * 0.4
    
    # Make botnet activity have specific signatures (features 3, 4 for network activity)
    botnet_indices = np.where(y_classes == 1)[0]
    X[botnet_indices, 3] *= 2.0  # Higher network throughput
    X[botnet_indices, 4] *= 3.0  # Higher packet rate
    
    # Firmware tampering affects checksum feature
    firmware_indices = np.where(y_classes == 2)[0]
    X[firmware_indices, 14] = np.random.rand(len(firmware_indices)) * 0.3  # Low checksums
    
    # Data exfiltration has high outbound data
    exfil_indices = np.where(y_classes == 3)[0]
    X[exfil_indices, 18] *= 3.0  # High outbound data volume
    
    # Command injection has specific command frequencies
    cmd_indices = np.where(y_classes == 4)[0]
    X[cmd_indices, 13] *= 2.5  # High command frequency
    
    # Replay attacks have repeating patterns in time feature
    replay_indices = np.where(y_classes == 5)[0]
    X[replay_indices, 9] = np.tile(np.linspace(0, 1, 5), len(replay_indices))[:len(replay_indices)]
    
    return X, y, class_names

class IoTFeatureExtractor:
    """
    Feature extractor for IoT device data.
    """
    def transform(self, data):
        """Transform IoT device data to feature vector."""
        if isinstance(data, dict):  # If it's a device reading
            # Extract features from the device data
            features = np.zeros(20)
            
            # Power consumption
            features[0] = data.get('power', 0) / 100.0
            
            # CPU and memory
            features[1] = data.get('cpu', 0) / 100.0
            features[2] = data.get('memory', 0) / 100.0
            
            # Network metrics
            features[3] = data.get('network_throughput', 0) / 1000.0
            features[4] = data.get('packet_rate', 0) / 100.0
            
            # Connection info
            features[5] = data.get('connection_duration', 0) / 3600.0
            features[6] = data.get('packet_size', 0) / 1500.0
            
            # Security metrics
            features[10] = data.get('auth_failures', 0) / 10.0
            features[11] = data.get('encryption', 0) / 5.0
            
            # Other IoT specific features
            features[14] = data.get('firmware_checksum', 1.0)
            features[16] = data.get('temperature', 25.0) / 100.0
            
            # Data volumes
            features[18] = data.get('outbound_data', 0) / 10000.0
            features[19] = data.get('inbound_data', 0) / 10000.0
            
            return features
            
        return data  # Return as is if already a feature vector

class IoTDeviceDetector(RealTimeDetector):
    """
    Extension of RealTimeDetector for IoT devices.
    """
    def __init__(self, model, feature_extractor, threshold=0.5, 
                batch_size=32, processing_interval=1.0):
        """Initialize the IoT device detector."""
        super().__init__(
            model, 
            feature_extractor, 
            threshold, 
            batch_size, 
            processing_interval
        )
        self.device_states = {}
    
    def process_device_reading(self, device_id, reading):
        """
        Process a reading from an IoT device.
        
        Args:
            device_id (str): Device identifier.
            reading (dict): Device reading data.
        """
        # Add device ID to the reading
        reading['device_id'] = device_id
        
        # Track device state
        self.device_states[device_id] = reading
        
        # Add to detection queue
        self.add_data(reading)
    
    def get_device_state(self, device_id):
        """
        Get the latest state of a device.
        
        Args:
            device_id (str): Device identifier.
            
        Returns:
            dict: Device state or None if not found.
        """
        return self.device_states.get(device_id)
    
    def get_all_device_ids(self):
        """
        Get all tracked device IDs.
        
        Returns:
            list: List of device IDs.
        """
        return list(self.device_states.keys())

def simulate_iot_devices(detector, duration=10, device_count=5):
    """
    Simulate IoT devices and generate readings.
    
    Args:
        detector (IoTDeviceDetector): The detector to feed readings to.
        duration (float): Duration to simulate in seconds.
        device_count (int): Number of devices to simulate.
    """
    start_time = time.time()
    device_types = ["thermostat", "camera", "smartlock", "lightbulb", "hub", 
                   "speaker", "refrigerator", "tv"]
    
    # Create a mix of devices
    devices = []
    for i in range(device_count):
        device_type = random.choice(device_types)
        device_id = f"{device_type}_{i+1}"
        normal = random.random() > 0.3  # 70% of devices are normal
        devices.append({
            "id": device_id,
            "type": device_type,
            "normal": normal,
            "attack_type": None if normal else random.choice(["botnet", "firmware", "exfil", "command", "replay"]),
            "attack_start": random.uniform(0, duration * 0.5) if not normal else None
        })
    
    # Simulate readings over time
    while time.time() - start_time < duration:
        for device in devices:
            device_id = device["id"]
            attack_active = False
            
            # Check if attack should be active
            if not device["normal"] and device["attack_start"] is not None:
                if time.time() - start_time > device["attack_start"]:
                    attack_active = True
            
            # Generate reading
            reading = generate_iot_reading(device["type"], attack_active, device["attack_type"])
            
            # Process through detector
            detector.process_device_reading(device_id, reading)
        
        # Small delay between readings
        time.sleep(0.2)

def generate_iot_reading(device_type, attack_active, attack_type):
    """
    Generate a realistic IoT device reading.
    
    Args:
        device_type (str): Type of IoT device.
        attack_active (bool): Whether an attack is active.
        attack_type (str): Type of attack if active.
        
    Returns:
        dict: Device reading data.
    """
    base_reading = {
        'timestamp': time.time(),
        'power': random.uniform(1, 15),
        'cpu': random.uniform(5, 30),
        'memory': random.uniform(10, 60),
        'network_throughput': random.uniform(1, 50),
        'packet_rate': random.uniform(1, 20),
        'connection_duration': random.uniform(10, 3600),
        'packet_size': random.uniform(100, 1200),
        'auth_failures': 0,
        'encryption': random.randint(3, 5),
        'firmware_checksum': 1.0,
        'temperature': random.uniform(20, 35),
        'outbound_data': random.uniform(10, 1000),
        'inbound_data': random.uniform(10, 500)
    }
    
    # Adjust for device type
    if device_type == "camera":
        base_reading['power'] *= 2
        base_reading['outbound_data'] *= 3
    elif device_type == "hub":
        base_reading['network_throughput'] *= 2
        base_reading['cpu'] *= 1.5
    elif device_type == "thermostat":
        base_reading['power'] *= 0.5
        base_reading['network_throughput'] *= 0.3
    
    # Add attack characteristics if active
    if attack_active:
        if attack_type == "botnet":
            base_reading['network_throughput'] *= 5
            base_reading['packet_rate'] *= 10
            base_reading['outbound_data'] *= 8
            base_reading['cpu'] *= 3
            
        elif attack_type == "firmware":
            base_reading['firmware_checksum'] = random.uniform(0.1, 0.5)
            base_reading['cpu'] *= 2
            base_reading['temperature'] *= 1.3
            
        elif attack_type == "exfil":
            base_reading['outbound_data'] *= 15
            base_reading['packet_size'] *= 2
            base_reading['connection_duration'] *= 0.2
            
        elif attack_type == "command":
            base_reading['auth_failures'] = random.randint(1, 5)
            base_reading['inbound_data'] *= 3
            base_reading['cpu'] *= random.uniform(1.5, 4)
            
        elif attack_type == "replay":
            # Replay attacks often look like normal traffic
            # but with repeating patterns
            pass
            
    return base_reading

def generate_iot_alert(threat_data):
    """
    Generate an alert for an IoT threat.
    
    Args:
        threat_data (dict): Threat detection result.
    """
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
    alert_file = f"security_output/iot/alert_{threat_data['class_name'].lower().replace(' ', '_')}_{timestamp}.txt"
    
    with open(alert_file, 'w') as f:
        f.write(f"IOT SECURITY ALERT: {threat_data['class_name']} DETECTED\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.fromtimestamp(threat_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if 'device_id' in threat_data:
            f.write(f"Device ID: {threat_data['device_id']}\n")
            
        f.write(f"Confidence: {threat_data['confidence']:.4f}\n\n")
        f.write("Threat Classification:\n")
        
        for cls, prob in threat_data['probabilities'].items():
            f.write(f"  {cls}: {prob:.4f}\n")
        
        f.write("\nRecommended Actions:\n")
        f.write(f"  {get_iot_recommended_action(threat_data['class_name'])}\n\n")
        f.write("This alert was generated automatically by CyberThreat-ML IoT Security\n")

def generate_iot_security_report(detected_threats, class_names):
    """
    Generate an IoT security report.
    
    Args:
        detected_threats (dict): Dictionary of detected threats by type.
        class_names (list): List of class names.
        
    Returns:
        str: Path to the generated report file.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = f"security_output/iot/iot_security_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("CYBERTHREAT-ML IOT SECURITY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Monitoring Period: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("THREAT SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        total_threats = sum(len(threats) for threats in detected_threats.values())
        f.write(f"Total IoT Threats Detected: {total_threats}\n\n")
        
        for key, threats in detected_threats.items():
            class_name = key.replace("_", " ").title()
            f.write(f"{class_name}: {len(threats)}\n")
        
        f.write("\nVULNERABLE DEVICES\n")
        f.write("-" * 50 + "\n")
        
        # Get unique device IDs from the threats
        vulnerable_devices = {}
        for key, threats in detected_threats.items():
            for threat in threats:
                if 'device_id' in threat:
                    device_id = threat['device_id']
                    if device_id not in vulnerable_devices:
                        vulnerable_devices[device_id] = []
                    vulnerable_devices[device_id].append(key)
        
        if vulnerable_devices:
            for device_id, threat_types in vulnerable_devices.items():
                unique_threats = list(set(threat_types))
                f.write(f"Device: {device_id}\n")
                f.write(f"  Threat Types: {', '.join(t.replace('_', ' ').title() for t in unique_threats)}\n")
                f.write(f"  Recommended Action: {get_iot_device_recommendation(unique_threats)}\n\n")
        else:
            f.write("No vulnerable devices detected.\n\n")
        
        f.write("RECOMMENDED SECURITY MEASURES\n")
        f.write("-" * 50 + "\n")
        
        if detected_threats["botnet_activity"]:
            f.write("- Implement network segmentation for IoT devices\n")
            f.write("- Install IoT-specific IDS/IPS for botnet detection\n")
        
        if detected_threats["firmware_tampering"]:
            f.write("- Establish secure boot processes for all IoT devices\n")
            f.write("- Implement digital signatures for firmware verification\n")
        
        if detected_threats["data_exfiltration"]:
            f.write("- Apply data egress filtering for IoT network segments\n")
            f.write("- Deploy IoT-specific DLP (Data Loss Prevention) solutions\n")
        
        f.write("\nIOT SECURITY BEST PRACTICES\n")
        f.write("-" * 50 + "\n")
        f.write("1. Maintain an inventory of all IoT devices\n")
        f.write("2. Change default credentials on all devices\n")
        f.write("3. Update firmware regularly\n")
        f.write("4. Segment IoT devices on a separate network\n")
        f.write("5. Monitor traffic patterns for anomalies\n")
        f.write("6. Implement encryption for all IoT communications\n")
        f.write("7. Disable unnecessary services and ports\n")
        f.write("8. Apply the principle of least privilege\n")
        
    return report_file

def get_iot_recommended_action(threat_class):
    """
    Get recommended action for an IoT threat.
    
    Args:
        threat_class (str): Threat class name.
        
    Returns:
        str: Recommended action.
    """
    actions = {
        "Botnet Activity": "ISOLATE the device immediately and INVESTIGATE the compromised firmware",
        "Firmware Tampering": "POWER DOWN the device, BACKUP data, and REINSTALL verified firmware",
        "Data Exfiltration": "BLOCK outbound connections and ANALYZE data flow patterns",
        "Command Injection": "ISOLATE the device and UPDATE firmware with patched version",
        "Replay Attack": "IMPLEMENT timestamping and REFRESH all authentication credentials"
    }
    
    return actions.get(threat_class, "INVESTIGATE the device and MONITOR for suspicious activity")

def get_iot_device_recommendation(threat_types):
    """
    Get a recommendation for a device based on its threat types.
    
    Args:
        threat_types (list): List of threat types affecting the device.
        
    Returns:
        str: Recommendation for the device.
    """
    if "firmware_tampering" in threat_types:
        return "Factory reset and reinstall verified firmware"
    elif "botnet_activity" in threat_types:
        return "Isolate device, factory reset, and update firmware"
    elif "command_injection" in threat_types:
        return "Update firmware to patched version"
    elif "data_exfiltration" in threat_types:
        return "Isolate device and audit data access patterns"
    elif "replay_attack" in threat_types:
        return "Update authentication mechanisms and refresh credentials"
    else:
        return "Monitor device and update firmware"

def create_iot_telemetry(n_samples=100):
    """
    Create synthetic IoT telemetry data for anomaly detection.
    
    Args:
        n_samples (int): Number of samples to generate.
        
    Returns:
        list: List of telemetry data points.
    """
    base_time = datetime.now() - timedelta(hours=24)
    device_ids = [f"device_{i}" for i in range(1, 6)]
    telemetry = []
    
    # Generate normal patterns
    for i in range(n_samples):
        timestamp = base_time + timedelta(minutes=i*15)
        device_id = random.choice(device_ids)
        
        # Create normal reading
        point = {
            "timestamp": timestamp,
            "device_id": device_id,
            "power": random.uniform(5, 20),
            "cpu": random.uniform(10, 40),
            "memory": random.uniform(20, 70),
            "network": random.uniform(5, 100),
            "temperature": random.uniform(25, 35)
        }
        
        # Introduce anomalies randomly (~10%)
        if random.random() < 0.1:
            anomaly_type = random.choice(["power", "cpu", "network"])
            if anomaly_type == "power":
                point["power"] *= random.uniform(3, 5)
            elif anomaly_type == "cpu":
                point["cpu"] *= random.uniform(2, 4) 
            elif anomaly_type == "network":
                point["network"] *= random.uniform(5, 10)
        
        telemetry.append(point)
    
    return telemetry

def detect_lightweight_anomalies(telemetry, metrics):
    """
    Demonstrate lightweight anomaly detection for IoT devices.
    
    Args:
        telemetry (list): List of telemetry data points.
        metrics (list): Metrics to analyze.
        
    Returns:
        list: Detected anomalies.
    """
    # Calculate baselines for each device and metric
    baselines = {}
    for point in telemetry:
        device_id = point["device_id"]
        if device_id not in baselines:
            baselines[device_id] = {metric: {'values': [], 'mean': 0, 'std': 0} for metric in metrics}
        
        for metric in metrics:
            if metric in point:
                baselines[device_id][metric]['values'].append(point[metric])
    
    # Calculate mean and standard deviation for each metric
    for device_id, metrics_dict in baselines.items():
        for metric, stats in metrics_dict.items():
            if stats['values']:
                stats['mean'] = np.mean(stats['values'])
                stats['std'] = np.std(stats['values']) or 1.0  # Avoid division by zero
    
    # Detect anomalies
    anomalies = []
    for point in telemetry:
        device_id = point["device_id"]
        if device_id in baselines:
            affected_metrics = []
            anomaly_score = 0
            
            for metric in metrics:
                if metric in point and metric in baselines[device_id]:
                    mean = baselines[device_id][metric]['mean']
                    std = baselines[device_id][metric]['std']
                    
                    # Calculate z-score
                    z_score = abs((point[metric] - mean) / std)
                    
                    # If z-score is greater than 3, consider it an anomaly
                    if z_score > 3:
                        affected_metrics.append(metric)
                        anomaly_score += z_score
            
            if affected_metrics:
                anomalies.append({
                    "timestamp": point["timestamp"],
                    "device_id": device_id,
                    "anomaly_score": anomaly_score / len(affected_metrics),
                    "affected_metrics": affected_metrics
                })
    
    # Sort anomalies by score
    anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
    
    return anomalies

if __name__ == "__main__":
    main()