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
import random   
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
import tensorflow as tf

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
    report = generate_security_report(threat_storage, interpreter)
    
    # Step 10: Cleanup
    detector.stop()
    dashboard.save_snapshot('security_output/dashboards/final_dashboard.png')
    dashboard.stop()
    logger.info("Enterprise security monitoring completed")


def create_and_train_model():
    """
    Create and train a model for enterprise security monitoring.
    """
    # Create synthetic dataset
    X_train, y_train = create_enterprise_dataset(n_samples=5000, n_features=25, n_classes=6)
    
    # Create model with enterprise-grade architecture
    model = ThreatDetectionModel(
        input_shape=(25,),  # 25 features
        num_classes=6,      # Normal + 5 threat types
        model_config={
            'hidden_layers': [256, 128, 64],  # Deeper network for complex threats
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'softmax',
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'metrics': ['accuracy', 'AUC', 'Precision', 'Recall']
        }
    )
    
    # Convert labels to one-hot encoding
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=6)
    
    # Train with early stopping and model checkpointing
    history = model.train(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/enterprise_threat_model_checkpoint.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
    )
    
    # Save the final model
    model.save_model(
        'models/enterprise_threat_model',
        'models/enterprise_threat_model_metadata.json'
    )
    
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
    # Initialize arrays
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int32)
    
    # Define class distribution (70% normal, 30% threats)
    n_normal = int(0.7 * n_samples)
    n_threats = n_samples - n_normal
    
    # Generate normal traffic patterns
    X[:n_normal] = np.random.normal(0.3, 0.1, (n_normal, n_features))
    y[:n_normal] = 0  # Class 0 is normal traffic
    
    # Generate threat patterns
    threat_samples_per_class = n_threats // (n_classes - 1)  # Excluding normal class
    
    for threat_class in range(1, n_classes):
        start_idx = n_normal + (threat_class - 1) * threat_samples_per_class
        end_idx = start_idx + threat_samples_per_class
        
        # Base pattern for this threat class
        X[start_idx:end_idx] = np.random.normal(0.6, 0.15, (threat_samples_per_class, n_features))
        
        # Add specific threat signatures
        if threat_class == 1:  # Port Scan
            # High port activity, low data transfer
            X[start_idx:end_idx, 0:2] = np.random.normal(0.9, 0.05, (threat_samples_per_class, 2))
            X[start_idx:end_idx, 4:6] = np.random.normal(0.1, 0.05, (threat_samples_per_class, 2))
            
        elif threat_class == 2:  # DDoS
            # High packet count, many source IPs
            X[start_idx:end_idx, 5] = np.random.normal(0.95, 0.03, threat_samples_per_class)
            X[start_idx:end_idx, 16] = np.random.normal(0.9, 0.05, threat_samples_per_class)
            
        elif threat_class == 3:  # Brute Force
            # Many failed auth attempts, consistent destination
            X[start_idx:end_idx, 18] = np.random.normal(0.85, 0.05, threat_samples_per_class)
            X[start_idx:end_idx, 17] = np.random.normal(0.1, 0.05, threat_samples_per_class)
            
        elif threat_class == 4:  # Data Exfiltration
            # Large data transfers, encrypted payload
            X[start_idx:end_idx, 4] = np.random.normal(0.9, 0.05, threat_samples_per_class)
            X[start_idx:end_idx, 14] = np.random.normal(0.95, 0.03, threat_samples_per_class)
            
        elif threat_class == 5:  # Command & Control
            # Periodic beacons, small encrypted packets
            X[start_idx:end_idx, 8] = np.random.normal(0.8, 0.05, threat_samples_per_class)
            X[start_idx:end_idx, 12:15] = np.random.normal(0.7, 0.1, (threat_samples_per_class, 3))
        
        y[start_idx:end_idx] = threat_class
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X.astype(np.float32), y


class EnterpriseFeatureExtractor:
    """Custom feature extractor for enterprise network traffic."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        # Initialize feature scaling parameters
        self.port_scaler = lambda x: x / 65535  # Scale ports to [0,1]
        self.size_scaler = lambda x: np.clip(x / 1500, 0, 1)  # MTU-based scaling
        self.time_scaler = lambda x: np.clip(x / 1000, 0, 1)  # Scale to seconds
        
        # Initialize running statistics for normalization
        self.running_stats = {
            'packet_count': {'mean': 0, 'std': 1, 'n': 0},
            'bytes_transferred': {'mean': 0, 'std': 1, 'n': 0}
        }
    
    def _update_running_stats(self, feature_name, value):
        """Update running mean and standard deviation."""
        stats = self.running_stats[feature_name]
        stats['n'] += 1
        delta = value - stats['mean']
        stats['mean'] += delta / stats['n']
        delta2 = value - stats['mean']
        stats['std'] = np.sqrt((stats['n'] - 1) * (stats['std'] ** 2) + delta * delta2) / stats['n']
    
    def _normalize(self, feature_name, value):
        """Normalize a value using running statistics."""
        stats = self.running_stats[feature_name]
        if stats['std'] == 0:
            return 0
        return (value - stats['mean']) / stats['std']
    
    def _compute_entropy(self, data):
        """Compute Shannon entropy of data."""
        if not data:
            return 0
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8))
        probabilities = counts[counts > 0] / len(data)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def transform(self, packet):
        """
        Extract features from a network packet.
        
        Args:
            packet (dict): Raw packet data.
            
        Returns:
            numpy.ndarray: Extracted feature vector.
        """
        features = np.zeros(25)
        
        # Basic packet features (0-2)
        features[0] = self.port_scaler(packet.get('source_port', 0))
        features[1] = self.port_scaler(packet.get('dest_port', 0))
        features[2] = self.size_scaler(packet.get('size', 0))
        
        # Flow features (3-5)
        features[3] = self.time_scaler(packet.get('duration', 0))
        bytes_transferred = packet.get('bytes', 0)
        features[4] = self._normalize('bytes_transferred', bytes_transferred)
        self._update_running_stats('bytes_transferred', bytes_transferred)
        
        packet_count = packet.get('packet_count', 1)
        features[5] = self._normalize('packet_count', packet_count)
        self._update_running_stats('packet_count', packet_count)
        
        # TCP-specific features (6-7)
        tcp_flags = packet.get('tcp_flags', 0)
        features[6] = tcp_flags / 255  # Normalize flags
        features[7] = packet.get('ttl', 64) / 255  # Normalize TTL
        
        # Timing features (8-9)
        features[8] = self.time_scaler(packet.get('inter_arrival_time', 0))
        features[9] = 1 if packet.get('direction') == 'outbound' else 0
        
        # Protocol features (10-11)
        protocol_type = packet.get('protocol', 'tcp').lower()
        features[10] = {'tcp': 0, 'udp': 1, 'icmp': 2}.get(protocol_type, 3) / 3
        features[11] = packet.get('window_size', 0) / 65535
        
        # Payload features (12-14)
        payload = packet.get('payload', b'')
        features[12] = len(payload) / 1500  # Normalize by MTU
        features[13] = self._compute_entropy(payload)
        features[14] = 1 if packet.get('is_encrypted', False) else 0
        
        # Header features (15-17)
        features[15] = len(packet.get('header', b'')) / 40  # Typical header size
        features[16] = self._compute_entropy(packet.get('source_ip', b''))
        features[17] = self._compute_entropy(packet.get('dest_ip', b''))
        
        # State features (18-19)
        conn_state = packet.get('connection_state', 'unknown').lower()
        features[18] = {'new': 0, 'established': 1, 'closed': 2}.get(conn_state, 3) / 3
        features[19] = 1 if self._is_suspicious_port_combo(
            packet.get('source_port', 0),
            packet.get('dest_port', 0)
        ) else 0
        
        # Advanced features (20-24)
        features[20] = packet.get('syn_rate', 0)
        features[21] = packet.get('unique_dests', 0) / 100  # Normalize
        features[22] = packet.get('bytes_per_packet', 0) / 1500
        features[23] = packet.get('fragment_bits', 0) / 8
        features[24] = packet.get('sequence_number', 0) % 100 / 100
        
        return features.astype(np.float32)
    
    def _is_suspicious_port_combo(self, src_port, dst_port):
        """Check if the port combination is suspicious."""
        suspicious_ports = {22, 23, 3389, 445, 135, 139}  # Common attack targets
        return src_port in suspicious_ports or dst_port in suspicious_ports


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
    recommendations = {
        "Port Scan": """
            1. Block scanning IP at firewall
            2. Review firewall rules and exposed ports
            3. Enable port scan detection and rate limiting
            4. Log and monitor for follow-up attacks
            """,
        "DDoS": """
            1. Enable DDoS mitigation services
            2. Scale infrastructure resources if needed
            3. Filter traffic using WAF rules
            4. Contact upstream providers if necessary
            """,
        "Brute Force": """
            1. Block attacking IP addresses
            2. Enable account lockout policies
            3. Implement multi-factor authentication
            4. Review and strengthen password policies
            """,
        "Data Exfiltration": """
            1. Block suspicious outbound connections
            2. Review data access logs and permissions
            3. Enable DLP monitoring and alerts
            4. Investigate compromised systems/accounts
            """,
        "Command & Control": """
            1. Isolate affected systems immediately
            2. Block C2 server IP addresses/domains
            3. Scan for and remove malware
            4. Review system logs for compromise scope
            """
    }
    
    return recommendations.get(
        threat_type,
        "1. Monitor system logs\n2. Review security alerts\n3. Update security policies"
    )


def simulate_enterprise_traffic(detector):
    """
    Simulate enterprise network traffic for demonstration.
    
    Args:
        detector (PacketStreamDetector): The real-time detector.
    """
    logger.info("Starting traffic simulation...")
    
    # Common enterprise ports and protocols
    common_ports = {
        'http': 80, 'https': 443, 'dns': 53, 'smb': 445,
        'ssh': 22, 'rdp': 3389, 'sql': 1433, 'ldap': 389
    }
    
    def generate_normal_packet():
        """Generate a normal traffic packet."""
        service = random.choice(list(common_ports.keys()))
        return {
            'source_port': random.randint(49152, 65535),
            'dest_port': common_ports[service],
            'size': random.randint(64, 1500),
            'protocol': 'tcp',
            'duration': random.randint(1, 1000),
            'bytes': random.randint(100, 1500),
            'packet_count': random.randint(1, 10),
            'tcp_flags': random.randint(0, 255),
            'ttl': random.randint(32, 128),
            'inter_arrival_time': random.randint(1, 100),
            'direction': random.choice(['inbound', 'outbound']),
            'window_size': random.randint(1024, 65535),
            'payload': os.urandom(random.randint(10, 100)),
            'is_encrypted': service == 'https',
            'header': os.urandom(20),
            'source_ip': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'dest_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'connection_state': random.choice(['new', 'established', 'closed']),
            'syn_rate': random.random() * 0.1,
            'unique_dests': random.randint(1, 5),
            'bytes_per_packet': random.randint(100, 1000),
            'fragment_bits': random.randint(0, 8),
            'sequence_number': random.randint(0, 1000000)
        }
    
    def generate_port_scan():
        """Generate a port scan packet."""
        packet = generate_normal_packet()
        packet.update({
            'source_port': random.randint(49152, 65535),
            'dest_port': random.randint(1, 1024),
            'size': random.randint(40, 60),
            'bytes': random.randint(40, 60),
            'packet_count': 1,
            'tcp_flags': 2,  # SYN
            'inter_arrival_time': random.randint(1, 10),
            'syn_rate': random.random() * 0.9 + 0.1,
            'unique_dests': random.randint(50, 100)
        })
        return packet
    
    def generate_ddos():
        """Generate a DDoS packet."""
        packet = generate_normal_packet()
        packet.update({
            'source_ip': f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
            'size': random.randint(500, 1500),
            'packet_count': random.randint(100, 1000),
            'bytes': random.randint(1000, 1500),
            'syn_rate': random.random() * 0.5 + 0.5,
            'unique_dests': 1,
            'bytes_per_packet': random.randint(500, 1500)
        })
        return packet
    
    def generate_brute_force():
        """Generate a brute force packet."""
        packet = generate_normal_packet()
        packet.update({
            'dest_port': random.choice([22, 3389, 445]),
            'size': random.randint(100, 200),
            'bytes': random.randint(100, 200),
            'tcp_flags': 24,  # PSH + ACK
            'connection_state': 'new',
            'syn_rate': random.random() * 0.3,
            'unique_dests': 1
        })
        return packet
    
    def generate_data_exfil():
        """Generate a data exfiltration packet."""
        packet = generate_normal_packet()
        packet.update({
            'dest_port': random.choice([443, 53, 6667]),
            'size': random.randint(1000, 1500),
            'bytes': random.randint(10000, 50000),
            'is_encrypted': True,
            'direction': 'outbound',
            'payload': os.urandom(random.randint(500, 1000)),
            'bytes_per_packet': random.randint(1000, 1500)
        })
        return packet
    
    def generate_c2():
        """Generate a command & control packet."""
        packet = generate_normal_packet()
        packet.update({
            'dest_port': random.choice([443, 53, 80]),
            'size': random.randint(50, 200),
            'bytes': random.randint(50, 200),
            'is_encrypted': True,
            'inter_arrival_time': random.randint(5000, 10000),
            'payload': os.urandom(random.randint(50, 100)),
            'connection_state': 'established'
        })
        return packet
    
    # Packet generation functions for each type
    packet_generators = {
        'normal': generate_normal_packet,
        'port_scan': generate_port_scan,
        'ddos': generate_ddos,
        'brute_force': generate_brute_force,
        'data_exfil': generate_data_exfil,
        'c2': generate_c2
    }
    
    # Simulate traffic for 60 seconds
    start_time = time.time()
    packet_count = 0
    attack_probability = 0.1  # 10% chance of attack packets
    
    try:
        while time.time() - start_time < 60:
            # Determine packet type
            if random.random() < attack_probability:
                packet_type = random.choice(['port_scan', 'ddos', 'brute_force', 'data_exfil', 'c2'])
            else:
                packet_type = 'normal'
            
            # Generate and process packet
            packet = packet_generators[packet_type]()
            packet['type'] = packet_type
            detector.process_packet(packet)
            
            packet_count += 1
            
            # Add some delay to simulate real traffic
            time.sleep(random.random() * 0.1)
    
    except KeyboardInterrupt:
        logger.info("Traffic simulation interrupted by user")
    
    logger.info(f"Traffic simulation completed. Processed {packet_count} packets")


def generate_security_report(threat_storage, interpreter):
    """
    Generate a comprehensive security report from detected threats.
    
    Args:
        threat_storage (ThreatStorage): Storage containing detected threats.
        interpreter (ThreatInterpreter): Model interpreter for threat analysis.
        
    Returns:
        dict: Security report with analysis and recommendations.
    """
    threats = threat_storage.get_threats()
    total_threats = len(threats)
    
    if total_threats == 0:
        return {
            'summary': 'No threats detected in the monitored time period.',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_threats': 0,
            'threat_categories': {},
            'recommendations': ['Continue monitoring and ensure all security controls are active.']
        }
    
    # Analyze threats by category
    threat_categories = {}
    high_risk_threats = []
    affected_systems = set()
    
    for threat in threats:
        # Get threat details
        threat_type = threat['type']
        confidence = threat['confidence']
        source_ip = threat.get('source_ip', 'Unknown')
        dest_ip = threat.get('dest_ip', 'Unknown')
        
        # Track affected systems
        if source_ip != 'Unknown':
            affected_systems.add(source_ip)
        if dest_ip != 'Unknown':
            affected_systems.add(dest_ip)
        
        # Update threat categories
        if threat_type not in threat_categories:
            threat_categories[threat_type] = {
                'count': 0,
                'avg_confidence': 0,
                'sources': set(),
                'targets': set(),
                'timestamps': []
            }
        
        cat = threat_categories[threat_type]
        cat['count'] += 1
        cat['avg_confidence'] = (cat['avg_confidence'] * (cat['count'] - 1) + confidence) / cat['count']
        cat['sources'].add(source_ip)
        cat['targets'].add(dest_ip)
        cat['timestamps'].append(threat['timestamp'])
        
        # Track high-risk threats (confidence > 0.8)
        if confidence > 0.8:
            high_risk_threats.append(threat)
    
    # Get model interpretability insights
    interpretability_insights = []
    for threat_type, stats in threat_categories.items():
        if stats['count'] >= 5:  # Only analyze patterns with sufficient samples
            insight = interpreter.explain_prediction({
                'type': threat_type,
                'count': stats['count'],
                'sources': list(stats['sources']),
                'targets': list(stats['targets'])
            })
            interpretability_insights.append({
                'threat_type': threat_type,
                'key_indicators': insight['key_indicators'],
                'confidence_factors': insight['confidence_factors']
            })
    
    # Generate attack timeline
    timeline = []
    for threat_type, stats in threat_categories.items():
        for timestamp in sorted(stats['timestamps']):
            timeline.append({
                'timestamp': timestamp,
                'type': threat_type,
                'sources': len(stats['sources']),
                'targets': len(stats['targets'])
            })
    
    # Calculate risk metrics
    risk_metrics = {
        'total_threats': total_threats,
        'high_risk_threats': len(high_risk_threats),
        'affected_systems': len(affected_systems),
        'unique_sources': len(set().union(*[cat['sources'] for cat in threat_categories.values()])),
        'unique_targets': len(set().union(*[cat['targets'] for cat in threat_categories.values()]))
    }
    
    # Generate recommendations
    recommendations = []
    for threat_type, stats in threat_categories.items():
        if stats['count'] > 0:
            action = get_recommended_action(threat_type)
            recommendations.append({
                'threat_type': threat_type,
                'priority': 'High' if stats['avg_confidence'] > 0.8 else 'Medium',
                'action': action
            })
    
    # Format threat categories for report
    formatted_categories = {}
    for threat_type, stats in threat_categories.items():
        formatted_categories[threat_type] = {
            'count': stats['count'],
            'confidence': round(stats['avg_confidence'], 2),
            'unique_sources': len(stats['sources']),
            'unique_targets': len(stats['targets']),
            'first_seen': min(stats['timestamps']),
            'last_seen': max(stats['timestamps'])
        }
    
    # Generate executive summary
    if total_threats > 0:
        most_common_threat = max(threat_categories.items(), key=lambda x: x[1]['count'])[0]
        highest_confidence_threat = max(threat_categories.items(), key=lambda x: x[1]['avg_confidence'])[0]
        summary = f"""
            Detected {total_threats} potential security threats affecting {risk_metrics['affected_systems']} systems.
            Most frequent attack type: {most_common_threat} ({threat_categories[most_common_threat]['count']} instances)
            Highest confidence threat: {highest_confidence_threat} ({threat_categories[highest_confidence_threat]['avg_confidence']:.2f} confidence)
            {risk_metrics['high_risk_threats']} high-risk threats require immediate attention.
        """.strip()
    else:
        summary = "No security threats detected during the monitoring period."
    
    # Compile final report
    report = {
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'risk_metrics': risk_metrics,
        'threat_categories': formatted_categories,
        'timeline': timeline,
        'interpretability_insights': interpretability_insights,
        'recommendations': recommendations,
        'high_risk_threats': [
            {
                'type': t['type'],
                'confidence': t['confidence'],
                'source': t.get('source_ip', 'Unknown'),
                'target': t.get('dest_ip', 'Unknown'),
                'timestamp': t['timestamp']
            }
            for t in high_risk_threats
        ]
    }
    
    return report


if __name__ == "__main__":
    main()