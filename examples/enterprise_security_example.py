"""
Example demonstrating how to use CyberThreat-ML in an enterprise security setting.

This example shows how to:
1. Set up real-time threat detection for an enterprise network
2. Integrate with SIEM systems and security alerts
3. Create a security dashboard for SOC analysts
4. Generate detailed reports for threat hunting
"""
import numpy as np
import pandas as pd
import time
import os
import logging
import json
from datetime import datetime

# Create output directories if they don't exist
os.makedirs('security_output/alerts', exist_ok=True)
os.makedirs('security_output/reports', exist_ok=True)
os.makedirs('security_output/dashboards', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_output/security_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enterprise_security')

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.preprocessing import FeatureExtractor
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter
from cyberthreat_ml.utils import split_data

def main():
    """
    Main function demonstrating enterprise security integration.
    """
    logger.info("Starting Enterprise Security Monitoring System")
    
    # Step 1: Load or create a model
    try:
        model = load_model('models/enterprise_threat_model')
        logger.info("Loaded existing enterprise threat model")
    except:
        logger.info("No existing model found. Creating and training a new model...")
        model = create_and_train_model()
        logger.info("Model training completed")
    
    # Step 2: Set up the feature extractor
    feature_extractor = EnterpriseFeatureExtractor()
    
    # Step 3: Set up the visualization dashboard
    dashboard = ThreatVisualizationDashboard(max_history=500)
    dashboard.start()
    logger.info("Threat visualization dashboard started")
    
    # Step 4: Define the class names for better interpretability
    class_names = [
        "Normal Traffic",
        "Port Scan",
        "DDoS",
        "Brute Force",
        "Data Exfiltration",
        "Command & Control"
    ]
    
    feature_names = [
        "Source Port", "Destination Port", "Packet Size", "Flow Duration",
        "Bytes Transferred", "Packet Count", "TCP Flags", "Time-to-live",
        "Inter-arrival Time", "Flow Direction", "Protocol Type", "Window Size",
        "Payload Length", "Payload Entropy", "Encrypted Payload", "Header Length",
        "Source IP Entropy", "Dest IP Entropy", "Connection State", "Suspicious Port Combo",
        "Rate of SYN Packets", "Unique Destinations", "Bytes per Packet", "Fragment Bits",
        "Packet Sequence"
    ]
    
    # Step 5: Initialize the interpreter for explainability
    interpreter = ThreatInterpreter(model, feature_names, class_names)
    interpreter.initialize(np.random.rand(100, 25))  # Background data
    logger.info("Threat interpreter initialized")
    
    # Step 6: Initialize threat data storage
    threat_storage = ThreatStorage()
    
    # Step 7: Set up the real-time detector
    detector = PacketStreamDetector(model, feature_extractor)
    
    # Define the SIEM integration function
    def send_to_siem(alert_data):
        """Simulate sending an alert to a SIEM system."""
        # In a real implementation, this would use the SIEM system's API
        # For this example, we'll just save it to a file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        filename = f"security_output/alerts/alert_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(alert_data, f, indent=2)
        logger.info(f"Alert sent to SIEM: {alert_data['alert_type']} from {alert_data['source_ip']}")
    
    # Define severity levels for different threat types
    severity_levels = {
        "Port Scan": "MEDIUM",
        "DDoS": "HIGH",
        "Brute Force": "HIGH",
        "Data Exfiltration": "CRITICAL",
        "Command & Control": "CRITICAL"
    }
    
    # Define callback for threat detection
    def on_threat_detected(result):
        if result['class_idx'] > 0:  # Skip normal traffic
            class_name = class_names[result['class_idx']]
            severity = severity_levels.get(class_name, "LOW")
            
            # Log the detection
            logger.warning(f"THREAT DETECTED: {class_name} (Confidence: {result['confidence']:.4f}, Severity: {severity})")
            
            # Add to the dashboard
            dashboard_data = {
                'timestamp': time.time(),
                'class_name': class_name,
                'class_idx': result['class_idx'],
                'confidence': result['confidence'],
                'source_ip': result.get('source_ip', '192.168.1.1'),
                'destination_ip': result.get('destination_ip', '10.0.0.1'),
                'features': result['features'],
                'severity': severity
            }
            dashboard.add_threat(dashboard_data)
            
            # Store the threat for analysis
            threat_storage.add_threat(dashboard_data)
            
            # Create and send a SIEM alert
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': class_name,
                'severity': severity,
                'confidence': float(result['confidence']),
                'source_ip': result.get('source_ip', '192.168.1.1'),
                'destination_ip': result.get('destination_ip', '10.0.0.1'),
                'detector': 'CyberThreat-ML',
                'description': f"ML-detected {class_name} threat with {result['confidence']:.2f} confidence",
                'recommended_action': get_recommended_action(class_name)
            }
            send_to_siem(alert_data)
            
            # For high-severity threats, generate an explanation immediately
            if severity in ("HIGH", "CRITICAL") and result['confidence'] > 0.3:
                explanation = interpreter.explain_prediction(
                    result['features'],
                    method="auto",
                    target_class=result['class_idx'],
                    top_features=5
                )
                
                # Save the explanation
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                report_path = f"security_output/reports/{class_name.lower().replace(' ', '_')}_{timestamp}_report.txt"
                interpreter.create_feature_importance_report(explanation, output_path=report_path)
                
                # Also save a visualization
                plot_path = f"security_output/reports/{class_name.lower().replace(' ', '_')}_{timestamp}_explanation.png"
                interpreter.plot_explanation(explanation, plot_type="bar", save_path=plot_path)
                
                logger.info(f"Generated threat explanation report: {report_path}")
    
    # Register the callback
    detector.register_threat_callback(on_threat_detected)
    
    # Define callback for batch processing
    def on_batch_processed(results):
        threats = sum(1 for r in results if r['class_idx'] > 0)
        logger.info(f"Processed batch: {len(results)} packets, {threats} threats")
    
    # Register the batch callback
    detector.register_processing_callback(on_batch_processed)
    
    # Start the detector
    detector.start()
    logger.info("Real-time threat detector started")
    
    # Step 8: Simulate network traffic (in a real system, this would be actual traffic)
    logger.info("Starting network traffic monitoring...")
    try:
        simulate_enterprise_traffic(detector)
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    
    # Step 9: Generate a security report based on collected threats
    generate_security_report(threat_storage, interpreter)
    
    # Step 10: Cleanup
    detector.stop()
    dashboard.save_snapshot('security_output/dashboards/final_dashboard.png')
    dashboard.stop()
    logger.info("Enterprise security monitoring completed")


def create_and_train_model():
    """
    Create and train a model for enterprise security monitoring.
    """
    # Create a synthetic dataset for demonstration
    X, y = create_enterprise_dataset()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create the model with enterprise configuration
    model = ThreatDetectionModel(
        input_shape=(25,),
        num_classes=6,
        model_config={
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'l2_regularization': 0.01
        }
    )
    
    # Train the model
    model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=15,
        batch_size=32,
        early_stopping=True
    )
    
    # Save the model for future use
    model.save_model('models/enterprise_threat_model')
    
    return model


def create_enterprise_dataset(n_samples=5000, n_features=25, n_classes=6):
    """
    Create a synthetic dataset for enterprise security modeling.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_classes (int): Number of threat classes.
        
    Returns:
        tuple: (X, y) - features and class labels.
    """
    # Generate features
    X = np.random.rand(n_samples, n_features)
    
    # Generate labels with a realistic class distribution
    # Normal traffic should be the majority
    class_probabilities = [0.75, 0.05, 0.05, 0.05, 0.05, 0.05]
    y = np.random.choice(n_classes, size=n_samples, p=class_probabilities)
    
    # Convert to categorical format for multi-class classification
    # This is needed for TensorFlow/Keras
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, n_classes)
    
    return X, y_categorical


class EnterpriseFeatureExtractor:
    """
    Custom feature extractor for enterprise network traffic.
    """
    def transform(self, packet):
        """
        Extract features from a network packet.
        
        Args:
            packet (dict): Raw packet data.
            
        Returns:
            numpy.ndarray: Extracted feature vector.
        """
        # In a real implementation, this would extract meaningful features
        # For this example, we'll create synthetic features
        
        # Basic feature extraction (simulated)
        if isinstance(packet, dict):
            # Convert packet attributes to a feature vector
            features = np.random.rand(25)  # Simulate extracted features
            
            # Add some structure to make it more realistic
            if 'source_port' in packet:
                features[0] = packet['source_port'] / 65535  # Normalize port number
            
            if 'destination_port' in packet:
                features[1] = packet['destination_port'] / 65535  # Normalize port number
            
            if 'size' in packet:
                features[2] = packet['size'] / 1500  # Normalize packet size
            
            # Add some bias toward the actual packet type if specified
            if 'type' in packet:
                if packet['type'] == 'port_scan':
                    features[1] = 0.95  # High destination port value
                    features[19] = 0.9  # High suspicious port combo
                elif packet['type'] == 'ddos':
                    features[2] = 0.9  # High packet size
                    features[5] = 0.95  # High packet count
                elif packet['type'] == 'brute_force':
                    features[0] = 0.1  # Low source port (often fixed)
                    features[3] = 0.9  # High flow duration
                elif packet['type'] == 'data_exfil':
                    features[4] = 0.95  # High bytes transferred
                    features[13] = 0.9  # High payload entropy
                elif packet['type'] == 'c2':
                    features[14] = 0.95  # High encrypted payload
                    features[21] = 0.1  # Low unique destinations
        else:
            # If not a dict, generate random features
            features = np.random.rand(25)
        
        return features


class ThreatStorage:
    """
    Class for storing and analyzing detected threats.
    """
    def __init__(self):
        """Initialize the threat storage."""
        self.threats = []
        self.class_data = {i: [] for i in range(1, 6)}  # Skip class 0 (normal)
    
    def add_threat(self, threat_data):
        """Add a threat to storage."""
        self.threats.append(threat_data)
        class_idx = threat_data['class_idx']
        if class_idx > 0:
            self.class_data[class_idx].append(threat_data)
    
    def get_threats_by_class(self, class_idx):
        """Get all threats of a specific class."""
        return self.class_data.get(class_idx, [])
    
    def get_threat_count(self):
        """Get the count of threats by class."""
        return {class_idx: len(threats) for class_idx, threats in self.class_data.items()}
    
    def get_threat_features(self, class_idx):
        """Get feature vectors for a specific threat class."""
        threats = self.class_data.get(class_idx, [])
        if not threats:
            return np.array([])
        return np.array([t['features'] for t in threats])


def get_recommended_action(threat_type):
    """
    Get recommended action based on threat type.
    
    Args:
        threat_type (str): Type of threat.
        
    Returns:
        str: Recommended action.
    """
    actions = {
        "Port Scan": "Block source IP and increase monitoring for the affected subnets",
        "DDoS": "Activate traffic scrubbing and contact upstream provider",
        "Brute Force": "Temporarily lock affected accounts and enable 2FA",
        "Data Exfiltration": "Isolate affected systems and conduct forensic investigation",
        "Command & Control": "Quarantine infected hosts and block C2 servers"
    }
    return actions.get(threat_type, "Investigate and monitor")


def simulate_enterprise_traffic(detector):
    """
    Simulate enterprise network traffic for demonstration.
    
    Args:
        detector (PacketStreamDetector): The real-time detector.
    """
    # Define some realistic IP ranges
    internal_ips = [f"10.0.{i}.{j}" for i in range(1, 5) for j in range(1, 10)]
    external_ips = [f"{i}.{j}.{k}.{l}" for i, j, k, l in [
        (203, 0, 113, 10), (198, 51, 100, 5), (172, 16, 0, 1),
        (8, 8, 8, 8), (1, 1, 1, 1), (93, 184, 216, 34)
    ]]
    
    # Define common ports
    common_ports = [80, 443, 22, 25, 53, 3389, 3306, 8080, 8443]
    
    # Packet type distribution
    packet_types = [
        'normal', 'normal', 'normal', 'normal', 'normal',
        'port_scan', 'ddos', 'brute_force', 'data_exfil', 'c2'
    ]
    
    logger.info("Simulating enterprise network traffic...")
    
    # Generate some packets
    for i in range(100):
        # Randomly select packet type, with bias toward normal
        packet_type = np.random.choice(packet_types)
        
        # Create a packet with appropriate characteristics
        packet = {
            'timestamp': time.time(),
            'source_ip': np.random.choice(internal_ips if packet_type != 'port_scan' else external_ips),
            'destination_ip': np.random.choice(internal_ips if packet_type == 'port_scan' else external_ips),
            'source_port': np.random.randint(1024, 65535),
            'destination_port': np.random.choice(common_ports) if packet_type != 'port_scan' else np.random.randint(1, 65535),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
            'size': np.random.randint(64, 1500),
            'type': packet_type
        }
        
        # Process the packet
        detector.process_packet(packet)
        
        # Sleep to simulate real-time traffic
        time.sleep(0.05)


def generate_security_report(threat_storage, interpreter):
    """
    Generate a comprehensive security report from collected threats.
    
    Args:
        threat_storage (ThreatStorage): Storage containing detected threats.
        interpreter (ThreatInterpreter): Interpreter for explaining threats.
    """
    logger.info("Generating security report...")
    
    # Get threat counts
    threat_counts = threat_storage.get_threat_count()
    
    # Create the report
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = f"security_output/reports/security_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENTERPRISE SECURITY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("THREAT SUMMARY\n")
        f.write("-"*80 + "\n")
        
        class_names = [
            "Normal Traffic",
            "Port Scan",
            "DDoS",
            "Brute Force",
            "Data Exfiltration",
            "Command & Control"
        ]
        
        for class_idx, count in threat_counts.items():
            f.write(f"{class_names[class_idx]}: {count} detections\n")
        
        f.write("\n")
        f.write("THREAT ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        # Analyze each threat class
        for class_idx, count in threat_counts.items():
            if count == 0:
                continue
            
            f.write(f"\n{class_names[class_idx]} Analysis:\n")
            
            # Get features for this class
            features = threat_storage.get_threat_features(class_idx)
            
            if len(features) >= 5:  # Need enough samples for meaningful analysis
                # Analyze patterns
                f.write("  Pattern Analysis:\n")
                
                # Calculate average confidence
                threats = threat_storage.get_threats_by_class(class_idx)
                avg_confidence = np.mean([t['confidence'] for t in threats])
                f.write(f"  - Average confidence: {avg_confidence:.4f}\n")
                
                # Get a representative sample
                sample_idx = np.random.randint(0, len(features))
                sample = features[sample_idx]
                
                # Explain this sample
                explanation = interpreter.explain_prediction(
                    sample,
                    method="rules",  # Use rules-based for simplicity
                    target_class=class_idx,
                    top_features=5
                )
                
                # Add top features to report
                f.write("  - Key indicators:\n")
                for feature, importance in explanation['top_features']:
                    f.write(f"    {feature}: {importance:.4f}\n")
                
                # Add recommended action
                f.write(f"  Recommended action: {get_recommended_action(class_names[class_idx])}\n")
        
        f.write("\n")
        f.write("SECURITY RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        
        # Add some generic recommendations
        f.write("1. Review firewall rules and ACLs for any unauthorized access patterns\n")
        f.write("2. Update IDS/IPS signatures based on detected threats\n")
        f.write("3. Conduct targeted vulnerability scanning on affected systems\n")
        f.write("4. Review user access controls and authentication mechanisms\n")
        f.write("5. Increase monitoring for high-risk assets and network segments\n")
    
    logger.info(f"Security report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    main()