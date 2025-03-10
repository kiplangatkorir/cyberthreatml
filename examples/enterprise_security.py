"""
Example demonstrating how to integrate CyberThreat-ML into an enterprise security environment.
This example shows a complete integration with:
- Real-time network traffic monitoring
- Threat detection and alerting
- Security dashboard visualization
- Threat pattern analysis
- Report generation
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.preprocessing import FeatureExtractor, extract_packet_features
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter, get_threat_pattern_insights
from cyberthreat_ml.logger import CyberThreatLogger

# Initialize logger
logger = CyberThreatLogger("enterprise_security", log_level="INFO").get_logger()

def main():
    """
    Example of integrating CyberThreat-ML into an enterprise security environment.
    """
    logger.info("Starting enterprise security integration example")
    
    # Create output directories
    os.makedirs("security_output/reports", exist_ok=True)
    os.makedirs("security_output/dashboards", exist_ok=True)
    os.makedirs("security_output/alerts", exist_ok=True)
    
    # Step 1: Initialize or load the threat detection model
    logger.info("Initializing threat detection model")
    model_path = "models/multiclass_threat_model"
    try:
        model = load_model(model_path)
        logger.info("Loaded existing model")
    except (FileNotFoundError, ValueError, TypeError):
        logger.info("Creating and training new model")
        model = create_and_train_model()
        model.save_model(model_path)
    
    # Step 2: Set up the security monitoring environment
    class_names = ["Normal Traffic", "Port Scan", "DDoS", "Brute Force", 
                  "Data Exfiltration", "Command & Control"]
    
    feature_names = [
        "Packet Size", "Duration", "Protocol Type", "Source Port", "Destination Port",
        "Flow Direction", "Bytes Transferred", "Packets Transferred", "Payload Entropy",
        "Connection State", "TCP Flags", "Window Size", "TTL", "Fragment Offset",
        "Header Length", "Encrypted Payload", "DNS Query Type", "HTTP Method",
        "Suspicious Port Combo", "SSL/TLS Version", "Certificate Validity",
        "Response Size", "Request Size", "Flow Interval", "Anomaly Score"
    ]
    
    # Step 3: Set up visualization and analytics components
    logger.info("Setting up visualization and analytics")
    dashboard = ThreatVisualizationDashboard()
    dashboard.start()
    logger.info("Threat visualization dashboard started")
    
    interpreter = ThreatInterpreter(model, feature_names, class_names)
    interpreter.initialize(create_background_data())
    logger.info("Threat interpreter initialized")
    
    # Step 4: Set up real-time detection
    detector = PacketStreamDetector(
        model,
        SimpleFeatureExtractor(),  # Simple feature extractor for simulation
        threshold=0.5,
        batch_size=10,
        processing_interval=1.0
    )
    logger.info("Real-time threat detector started")
    
    # Track detected threats for analysis
    detected_threats = {
        "port_scan": [],
        "ddos": [],
        "brute_force": [],
        "data_exfiltration": [],
        "command_control": []
    }
    
    # Register callback for threat detection
    def on_threat_detected(result):
        threat_type = result["class_name"].lower().replace(" ", "_")
        if threat_type != "normal_traffic" and threat_type in detected_threats:
            detected_threats[threat_type].append(result)
            
            # Generate alert for high-confidence threats
            if result["confidence"] > 0.7:
                generate_alert(result)
            
            # Add to dashboard
            dashboard.add_threat(result)
    
    detector.register_threat_callback(on_threat_detected)
    detector.start()
    
    # Step 5: Start monitoring
    logger.info("Starting network traffic monitoring...")
    logger.info("Simulating enterprise network traffic...")
    
    # Simulate network traffic for a short time
    simulate_enterprise_traffic(detector, duration=5)
    
    # Step 6: Generate security report
    logger.info("Generating security report...")
    report_file = generate_security_report(detected_threats, class_names)
    logger.info(f"Security report generated: {report_file}")
    
    # Step 7: Perform threat analysis
    for cls_name, threats in detected_threats.items():
        if threats:
            # Get the first few threats for analysis
            sample_threats = threats[:min(5, len(threats))]
            features = np.vstack([t["features"] for t in sample_threats])
            
            # Analyze the threats
            try:
                cls_idx = next(i for i, name in enumerate(class_names) 
                              if name.lower().replace(" & ", "_").replace(" ", "_") == cls_name)
                
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
                    save_path=f"security_output/reports/{cls_name}_explanation.png"
                )
                
                interpreter.create_feature_importance_report(
                    explanation,
                    output_path=f"security_output/reports/{cls_name}_report.txt"
                )
            except (IndexError, StopIteration, ValueError) as e:
                logger.error(f"Error analyzing {cls_name} threats: {str(e)}")
    
    # Step 8: Final cleanup
    time.sleep(1)  # Allow time for processing
    detector.stop()
    
    # Save final dashboard view
    try:
        dashboard.save_snapshot("security_output/dashboards/final_dashboard.png")
    except Exception as e:
        logger.error(f"Error saving dashboard snapshot: {str(e)}")
    
    dashboard.stop()
    logger.info("Enterprise security monitoring completed")

def create_and_train_model():
    """
    Create and train a new threat detection model.
    
    Returns:
        ThreatDetectionModel: Trained model.
    """
    # Create synthetic dataset for demonstration
    X, y, class_names = create_synthetic_dataset(n_samples=2000, n_features=25, n_classes=6)
    
    # Split into train/val/test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    
    # Create and train model
    model = ThreatDetectionModel(
        input_shape=(25,),
        num_classes=6,
        model_config={
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3
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
    X, _, _ = create_synthetic_dataset(n_samples=100, n_features=25, n_classes=6)
    return X

def create_synthetic_dataset(n_samples=2000, n_features=25, n_classes=6, normal_prob=0.4):
    """
    Create a synthetic dataset for demonstration purposes.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_classes (int): Number of threat classes (including normal traffic).
        normal_prob (float): Probability of normal traffic samples.
        
    Returns:
        tuple: (X, y, class_names) - features, one-hot encoded labels, and class names.
    """
    X = np.random.rand(n_samples, n_features)
    
    # Generate class labels with a bias toward normal traffic
    y_classes = np.zeros(n_samples, dtype=int)
    normal_samples = int(n_samples * normal_prob)
    y_classes[:normal_samples] = 0  # Normal traffic
    
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
    
    class_names = ["Normal Traffic", "Port Scan", "DDoS", "Brute Force", 
                  "Data Exfiltration", "Command & Control"]
    
    return X, y, class_names

class SimpleFeatureExtractor:
    """
    Simple feature extractor for demonstration.
    """
    def transform(self, x):
        """Transform input to feature vector."""
        if isinstance(x, dict):  # If it's a packet
            # Extract features from packet
            return np.random.rand(25)  # Random features for simulation
        
        return x  # Return as is if already a feature vector

def simulate_enterprise_traffic(detector, duration=10, max_packets=None):
    """
    Simulate enterprise network traffic for the detector.
    
    Args:
        detector (PacketStreamDetector): The detector to feed packets to.
        duration (float, optional): Duration to generate traffic for (in seconds).
        max_packets (int, optional): Maximum number of packets to generate.
    """
    start_time = time.time()
    packet_count = 0
    
    internal_ips = ['10.0.0.' + str(i) for i in range(1, 20)]
    external_ips = ['203.0.113.' + str(i) for i in range(1, 10)]
    suspicious_ips = ['185.13.45.' + str(i) for i in range(1, 5)]
    common_ports = [80, 443, 22, 25, 53, 3389]
    
    try:
        while True:
            if duration and time.time() - start_time > duration:
                break
                
            if max_packets and packet_count >= max_packets:
                break
            
            # Generate random packet
            packet = generate_random_packet(internal_ips, external_ips, suspicious_ips, common_ports)
            
            # Process packet
            detector.process_packet(packet)
            
            packet_count += 1
            
            # Add some delay to simulate real traffic
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Packet generation interrupted.")

def generate_random_packet(internal_ips, external_ips, suspicious_ips, common_ports):
    """
    Generate a random network packet for simulation.
    
    Args:
        internal_ips (list): List of internal IP addresses.
        external_ips (list): List of external IP addresses.
        suspicious_ips (list): List of suspicious IP addresses.
        common_ports (list): List of common port numbers.
        
    Returns:
        dict: Synthetic packet data.
    """
    # Choose traffic type with bias
    traffic_type = np.random.choice(['normal', 'suspicious', 'malicious'], 
                                   p=[0.7, 0.2, 0.1])
    
    if traffic_type == 'normal':
        src_ip = np.random.choice(internal_ips)
        dst_ip = np.random.choice(external_ips)
        src_port = np.random.randint(49152, 65535)
        dst_port = np.random.choice(common_ports)
        packet_size = np.random.randint(64, 1500)
        protocol = np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS'])
        
    elif traffic_type == 'suspicious':
        src_ip = np.random.choice(internal_ips)
        dst_ip = np.random.choice(external_ips + suspicious_ips, p=[0.7, 0.3])
        src_port = np.random.randint(1024, 65535)
        dst_port = np.random.randint(1, 65535)
        packet_size = np.random.randint(100, 9000)
        protocol = np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS', 'IRC'])
        
    else:  # malicious
        is_inbound = np.random.choice([True, False], p=[0.3, 0.7])
        
        if is_inbound:
            src_ip = np.random.choice(suspicious_ips)
            dst_ip = np.random.choice(internal_ips)
        else:
            src_ip = np.random.choice(internal_ips)
            dst_ip = np.random.choice(suspicious_ips)
            
        src_port = np.random.randint(1, 65535)
        dst_port = np.random.randint(1, 65535)
        packet_size = np.random.randint(40, 15000)
        protocol = np.random.choice(['TCP', 'UDP', 'HTTP', 'ICMP', 'IRC', 'FTP'])
    
    # Create packet
    packet = {
        'timestamp': time.time(),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'size': packet_size,
        'protocol': protocol,
        'ttl': np.random.randint(32, 255),
        'flags': np.random.randint(0, 0xFF) if protocol in ['TCP', 'HTTP', 'HTTPS'] else 0,
        'payload': np.random.bytes(min(packet_size - 40, 1000)) if packet_size > 40 else b''
    }
    
    return packet

def generate_alert(threat_data):
    """
    Generate a security alert for a detected threat.
    
    Args:
        threat_data (dict): Threat detection result.
    """
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
    alert_file = f"security_output/alerts/alert_{threat_data['class_name'].lower().replace(' ', '_')}_{timestamp}.txt"
    
    with open(alert_file, 'w') as f:
        f.write(f"SECURITY ALERT: {threat_data['class_name']} DETECTED\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.fromtimestamp(threat_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Confidence: {threat_data['confidence']:.4f}\n\n")
        f.write("Threat Classification:\n")
        
        for cls, prob in threat_data['probabilities'].items():
            f.write(f"  {cls}: {prob:.4f}\n")
        
        f.write("\nRecommended Actions:\n")
        f.write(f"  {get_recommended_action(threat_data['class_name'])}\n\n")
        f.write("This alert was generated automatically by CyberThreat-ML\n")

def generate_security_report(detected_threats, class_names):
    """
    Generate a comprehensive security report.
    
    Args:
        detected_threats (dict): Dictionary of detected threats by type.
        class_names (list): List of class names.
        
    Returns:
        str: Path to the generated report file.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = f"security_output/reports/security_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("CYBERTHREAT-ML SECURITY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Monitoring Period: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("THREAT SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        total_threats = sum(len(threats) for threats in detected_threats.values())
        f.write(f"Total Threats Detected: {total_threats}\n\n")
        
        for key, threats in detected_threats.items():
            class_name = key.replace("_", " ").title().replace("Ddos", "DDoS").replace("And", "&")
            f.write(f"{class_name}: {len(threats)}\n")
        
        f.write("\nHIGH CONFIDENCE THREATS\n")
        f.write("-" * 50 + "\n")
        
        high_confidence = []
        for key, threats in detected_threats.items():
            for threat in threats:
                if threat["confidence"] > 0.7:
                    high_confidence.append((key, threat))
        
        if high_confidence:
            for key, threat in high_confidence:
                class_name = key.replace("_", " ").title().replace("Ddos", "DDoS").replace("And", "&")
                f.write(f"{class_name} (Confidence: {threat['confidence']:.4f})\n")
                f.write(f"  Timestamp: {datetime.fromtimestamp(threat['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Recommended Action: {get_recommended_action(class_name)}\n\n")
        else:
            f.write("No high confidence threats detected.\n\n")
        
        f.write("RECOMMENDED SECURITY MEASURES\n")
        f.write("-" * 50 + "\n")
        
        if detected_threats["port_scan"]:
            f.write("- Review firewall rules and network segmentation\n")
            f.write("- Consider implementing port scan detection at the perimeter\n")
        
        if detected_threats["ddos"]:
            f.write("- Verify DDoS protection measures are in place\n")
            f.write("- Increase monitoring of network traffic patterns\n")
        
        if detected_threats["brute_force"]:
            f.write("- Implement account lockout policies\n")
            f.write("- Enable multi-factor authentication for all accounts\n")
        
        if detected_threats["data_exfiltration"]:
            f.write("- Review data loss prevention (DLP) policies\n")
            f.write("- Audit sensitive data access controls\n")
        
        if detected_threats["command_control"]:
            f.write("- Scan all systems for malware and backdoors\n")
            f.write("- Review outbound connection policies\n")
    
    return report_file

def get_recommended_action(threat_class):
    """
    Get recommended action for a threat class.
    
    Args:
        threat_class (str): Threat class name.
        
    Returns:
        str: Recommended action.
    """
    actions = {
        "Port Scan": "BLOCK scanning IP at firewall and INVESTIGATE source",
        "DDoS": "ACTIVATE ANTI-DDOS MEASURES and ALERT SECURITY TEAM",
        "Brute Force": "TEMPORARY ACCOUNT LOCKOUT and ENABLE 2FA",
        "Data Exfiltration": "ISOLATE affected systems and INVESTIGATE data access",
        "Command & Control": "ISOLATE infected system and BLOCK C&C domains/IPs"
    }
    
    # Clean up the threat class name for matching
    threat_class = threat_class.replace("_", " ").title().replace("Ddos", "DDoS").replace("And", "&")
    
    return actions.get(threat_class, "INVESTIGATE and follow incident response plan")

if __name__ == "__main__":
    main()