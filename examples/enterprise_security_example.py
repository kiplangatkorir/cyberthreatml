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
import threading
import keras
import tensorflow as tf

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
    """Main function to run the enterprise security monitoring system."""
    logger.info("Starting Enterprise Security Monitoring System")
    
    # Try to load existing model or create new one
    try:
        model = tf.keras.models.load_model('models/enterprise_threat_model.keras')
        
        # Add predict_proba method for SHAP
        def predict_proba(self, x):
            return self.predict(x)
        
        model.predict_proba = predict_proba.__get__(model)
        logger.info("Loaded existing model")
    except:
        logger.info("No existing model found. Creating and training a new model...")
        model = create_and_train_model()
        model.save('models/enterprise_threat_model.keras')
        logger.info("Model training completed")
    
    # Initialize components
    feature_extractor = EnterpriseFeatureExtractor()
    threat_storage = ThreatStorage()
    
    # Initialize threat interpreter with feature names
    feature_names = [
        "Source Port", "Destination Port", "Packet Size", "Flow Duration",
        "Bytes Transferred", "Packet Count", "TCP Flags", "Time-to-live",
        "Inter-arrival Time", "Flow Direction", "Protocol Type", "Window Size",
        "Payload Length", "Payload Entropy", "Encrypted Payload", "Header Length",
        "Source IP Entropy", "Dest IP Entropy", "Connection State", "Suspicious Port Combo",
        "Rate of SYN Packets", "Unique Destinations", "Bytes per Packet", "Fragment Bits",
        "Packet Sequence"
    ]
    interpreter = ThreatInterpreter(model, feature_names=feature_names)
    
    # Initialize interpreter with background data
    X_background, _ = create_enterprise_dataset(n_samples=1000)
    interpreter.initialize(X_background)
    logger.info("Threat interpreter initialized")
    
    # Start real-time detection
    stop_event = threading.Event()
    detector_thread = threading.Thread(
        target=real_time_detector,
        args=(model, feature_extractor, threat_storage, interpreter, stop_event)
    )
    detector_thread.start()
    logger.info("Real-time threat detector started")
    
    # Start visualization dashboard (in main thread)
    dashboard = ThreatVisualizationDashboard(max_history=500)
    dashboard.start()
    logger.info("Threat visualization dashboard started")
    
    # Start traffic monitoring
    logger.info("Starting network traffic monitoring...")
    logger.info("Starting traffic simulation...")
    
    try:
        # Simulate traffic for 60 seconds
        simulate_enterprise_traffic(60, threat_storage)
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        # Stop all threads
        stop_event.set()
        detector_thread.join()
        dashboard.stop()
        
        logger.info("Enterprise security monitoring completed")

def create_and_train_model():
    """Create and train the threat detection model."""
    # Define model architecture
    inputs = keras.Input(shape=(25,))
    x = keras.layers.Dense(256, activation='relu')(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Add predict_proba method for SHAP
    def predict_proba(self, x):
        return self.predict(x)
    
    model.predict_proba = predict_proba.__get__(model)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    # Create training data
    X_train, y_train = create_enterprise_dataset(n_samples=4000)
    X_val, y_val = create_enterprise_dataset(n_samples=1000)
    
    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=1
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
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            return 0
            
        if not data:
            return 0
            
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8))
        probabilities = counts[counts > 0] / len(data)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _ip_to_bytes(self, ip):
        """Convert IP address to bytes for entropy calculation."""
        try:
            # Split IP into octets and convert to bytes
            return bytes([int(x) for x in ip.split('.')])
        except:
            return b''
    
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
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        features[12] = len(payload) / 1500  # Normalize by MTU
        features[13] = self._compute_entropy(payload)
        features[14] = 1 if packet.get('is_encrypted', False) else 0
        
        # Header features (15-17)
        header = packet.get('header', b'')
        if isinstance(header, str):
            header = header.encode('utf-8')
        features[15] = len(header) / 40  # Typical header size
        features[16] = self._compute_entropy(self._ip_to_bytes(packet.get('source_ip', '')))
        features[17] = self._compute_entropy(self._ip_to_bytes(packet.get('dest_ip', '')))
        
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
    Storage for detected threats.
    """
    def __init__(self):
        """Initialize the threat storage."""
        self.threats = []
        self.lock = threading.Lock()
    
    def add_threat(self, threat):
        """Add a threat to storage."""
        with self.lock:
            self.threats.append({
                'timestamp': time.time(),
                **threat
            })
    
    def get_threats(self):
        """Get all stored threats."""
        with self.lock:
            return self.threats.copy()
    
    def get_threat_count(self):
        """Get count of threats by type."""
        with self.lock:
            counts = {}
            for threat in self.threats:
                threat_type = threat['type']
                counts[threat_type] = counts.get(threat_type, 0) + 1
            return counts
    
    def get_threats_by_class(self, class_idx):
        """Get threats of a specific class."""
        with self.lock:
            return [t for t in self.threats if t['type'] == class_idx]
    
    def clear(self):
        """Clear all stored threats."""
        with self.lock:
            self.threats.clear()


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


def simulate_enterprise_traffic(duration, threat_storage):
    """
    Simulate enterprise network traffic for demonstration.
    
    Args:
        duration (int): Duration of traffic simulation in seconds.
        threat_storage (ThreatStorage): Storage for detected threats.
    """
    logger.info("Starting traffic simulation...")
    
    # Packet generators for different types of traffic
    def generate_normal():
        return {
            'source_ip': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}',
            'dest_ip': f'10.0.{random.randint(1,255)}.{random.randint(1,255)}',
            'source_port': random.randint(1024, 65535),
            'dest_port': random.choice([80, 443, 22, 53]),
            'protocol': random.choice(['TCP', 'UDP']),
            'payload_size': random.randint(64, 1500),
            'flags': random.choice(['ACK', 'PSH-ACK', 'SYN', 'FIN'])
        }
    
    def generate_port_scan():
        return {
            'source_ip': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}',
            'dest_ip': '10.0.0.1',  # Fixed target
            'source_port': random.randint(1024, 65535),
            'dest_port': random.randint(1, 1024),  # Low ports
            'protocol': 'TCP',
            'payload_size': 64,  # Small packets
            'flags': 'SYN'  # SYN scanning
        }
    
    def generate_ddos():
        return {
            'source_ip': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
            'dest_ip': '10.0.0.1',  # Fixed target
            'source_port': random.randint(1024, 65535),
            'dest_port': 80,  # Web server
            'protocol': 'TCP',
            'payload_size': random.randint(500, 1500),
            'flags': random.choice(['SYN', 'ACK', 'PSH-ACK'])
        }
    
    def generate_brute_force():
        return {
            'source_ip': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}',
            'dest_ip': '10.0.0.2',  # Authentication server
            'source_port': random.randint(1024, 65535),
            'dest_port': 22,  # SSH
            'protocol': 'TCP',
            'payload_size': random.randint(100, 300),
            'flags': 'PSH-ACK'
        }
    
    def generate_data_exfil():
        return {
            'source_ip': '192.168.1.100',  # Compromised host
            'dest_ip': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
            'source_port': random.randint(1024, 65535),
            'dest_port': random.randint(1024, 65535),
            'protocol': 'TCP',
            'payload_size': random.randint(1000, 1500),  # Large packets
            'flags': 'PSH-ACK'
        }
    
    def generate_c2():
        return {
            'source_ip': '192.168.1.100',  # Compromised host
            'dest_ip': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
            'source_port': random.randint(1024, 65535),
            'dest_port': random.choice([53, 80, 443]),  # Common ports
            'protocol': 'TCP',
            'payload_size': random.randint(100, 500),  # Beaconing
            'flags': 'PSH-ACK'
        }
    
    # Map packet types to generators
    packet_generators = {
        'normal': generate_normal,
        'port_scan': generate_port_scan,
        'ddos': generate_ddos,
        'brute_force': generate_brute_force,
        'data_exfil': generate_data_exfil,
        'c2': generate_c2
    }
    
    # Simulate traffic for specified duration
    start_time = time.time()
    packet_count = 0
    attack_probability = 0.1  # 10% chance of attack packets
    
    try:
        while time.time() - start_time < duration:
            # Determine packet type
            if random.random() < attack_probability:
                packet_type = random.choice(['port_scan', 'ddos', 'brute_force', 'data_exfil', 'c2'])
            else:
                packet_type = 'normal'
            
            # Generate and process packet
            packet = packet_generators[packet_type]()
            packet['type'] = packet_type
            
            # Add some delay to simulate real traffic
            time.sleep(random.random() * 0.1)
            
            packet_count += 1
    
    except KeyboardInterrupt:
        logger.info("Traffic simulation interrupted by user")
    finally:
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


def _rules_based_explanation(features, feature_names):
    """Generate a rules-based explanation for the features."""
    # Define thresholds for common threat indicators (adjusted for normalized values)
    thresholds = {
        'Packet Size': 0.7,  # Large packet size relative to max
        'Flow Duration': 0.6,  # Long flow duration 
        'Bytes Transferred': 0.8,  # High data transfer
        'Packet Count': 0.7,  # High packet count
        'Rate of SYN Packets': 0.6,  # High SYN rate
        'Unique Destinations': 0.5,  # Many unique destinations
        'Payload Entropy': 0.8,  # High entropy (encrypted/compressed)
        'Inter-arrival Time': 0.7,  # Suspicious timing
        'Window Size': 0.9,  # Unusual window size
        'Payload Length': 0.8  # Large payload
    }
    
    explanation = {
        'method': 'rules',
        'feature_importance': {},
        'rules_triggered': []
    }
    
    # Check each feature against thresholds
    for i, feature_name in enumerate(feature_names):
        if feature_name in thresholds:
            feature_val = float(features[i])  # Convert to float for comparison
            threshold = thresholds[feature_name]
            
            if feature_val > threshold:
                importance = min((feature_val - threshold) / (1 - threshold), 1.0)
                explanation['feature_importance'][feature_name] = importance
                explanation['rules_triggered'].append(
                    f"{feature_name} ({feature_val:.2f}) exceeds threshold ({threshold:.2f})"
                )
    
    # Sort rules by importance
    if explanation['rules_triggered']:
        explanation['rules_triggered'].sort(
            key=lambda x: explanation['feature_importance'][x.split(' ')[0]], 
            reverse=True
        )
    
    return explanation

def process_batch(model, batch_features, threat_storage, interpreter):
    """Process a batch of network traffic features."""
    try:
        # Make predictions (ensure batch_features is 2D)
        features_2d = np.array([batch_features])
        predictions = model.predict(features_2d, verbose=0)
        
        # Get prediction probability (single value between 0 and 1)
        pred_prob = float(predictions[0][0])
        
        # Determine if it's a threat (threshold = 0.5)
        is_threat = pred_prob > 0.5
        
        if is_threat:
            try:
                # Get threat interpretation using rules-based method
                explanation = _rules_based_explanation(
                    features=batch_features,
                    feature_names=interpreter.feature_names
                )
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")
                explanation = {"method": "none", "error": str(e)}
            
            # Create threat object with severity
            severity = "HIGH" if pred_prob > 0.8 else "MEDIUM" if pred_prob > 0.6 else "LOW"
            threat = {
                'type': 'malicious',
                'confidence': pred_prob,
                'severity': severity,
                'features': batch_features.tolist(),
                'interpretation': explanation,
                'timestamp': time.time()
            }
            
            # Store the threat
            threat_storage.add_threat(threat)
            
            # Log the threat
            logging.warning(f"Threat detected! Confidence: {pred_prob:.2f}")
            
            # Generate alert
            generate_alert(threat)
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")

def real_time_detector(model, feature_extractor, threat_storage, interpreter, stop_event):
    """Run real-time threat detection."""
    try:
        while not stop_event.is_set():
            # Get latest network traffic
            packet = get_latest_traffic()
            if packet is None:
                time.sleep(0.1)
                continue
                
            # Extract features
            features = feature_extractor.transform(packet)
            
            # Process the features
            process_batch(model, features, threat_storage, interpreter)
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"Error in real-time detector: {str(e)}")
    finally:
        logging.info("Real-time detector stopped")


def get_latest_traffic():
    """Simulate getting the latest network traffic."""
    # In a real system, this would capture actual network packets
    # For simulation, we'll generate synthetic traffic
    traffic_types = ['normal', 'port_scan', 'ddos', 'brute_force', 'data_exfil', 'c2']
    traffic_type = random.choice(traffic_types)
    
    packet = {
        'timestamp': time.time(),
        'source_ip': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}',
        'dest_ip': f'10.0.{random.randint(1,255)}.{random.randint(1,255)}',
        'source_port': random.randint(1024, 65535),
        'dest_port': random.randint(1, 65535),
        'protocol': random.choice(['TCP', 'UDP', 'ICMP']),
        'payload_size': random.randint(64, 1500),
        'type': traffic_type
    }
    
    return packet


def generate_alert(threat_data):
    """Generate an alert for a detected threat."""
    severity = threat_data.get('severity', 'UNKNOWN')
    rules = threat_data['interpretation'].get('rules_triggered', [])
    
    alert_msg = (
        f"ALERT: {severity} Severity Threat Detected!\n"
        f"Type: {threat_data['type']}\n"
        f"Confidence: {threat_data['confidence']:.2f}\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(threat_data['timestamp']))}"
    )
    
    if rules:
        alert_msg += "\nRules Triggered:\n" + "\n".join(f"- {rule}" for rule in rules)
    else:
        alert_msg += "\nNo specific rules triggered - detected by ML model"
    
    logger.warning(alert_msg)


if __name__ == "__main__":
    main()