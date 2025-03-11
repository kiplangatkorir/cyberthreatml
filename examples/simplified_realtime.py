"""
Simplified example of real-time cybersecurity threat detection.
This example focuses on the core concepts without external dependencies.
"""

import sys
import os
import time
import random
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to sys.path to allow imports from the cyberthreat_ml package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class SimpleFeatureExtractor:
    """
    Simple feature extractor that simulates processing network packets.
    """
    def transform(self, packet):
        """
        Extract features from a packet.
        """
        # In a real implementation, this would apply sophisticated feature extraction
        # For this example, we'll just use the raw packet features
        features = [
            packet.get('packet_size', 0) / 1500.0,  # Normalize packet size
            packet.get('protocol') == 'TCP',  # Protocol binary feature
            packet.get('source_port', 0) / 65535.0,  # Normalize source port
            packet.get('dest_port', 0) / 65535.0,  # Normalize destination port
            packet.get('tcp_flags', {}).get('SYN', 0),
            packet.get('tcp_flags', {}).get('ACK', 0),
            packet.get('tcp_flags', {}).get('FIN', 0),
            packet.get('tcp_flags', {}).get('RST', 0),
            packet.get('payload_size', 0) / 1460.0,  # Normalize payload size
            packet.get('has_payload', False),
        ]
        return features


class SimpleDetectionModel:
    """
    Simple threat detection model that simulates ML-based detection.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.threat_types = [
            "Normal Traffic",
            "Port Scan",
            "DDoS",
            "Brute Force",
            "Data Exfiltration",
            "Command & Control"
        ]
        
    def predict(self, features):
        """
        Simulate prediction using a simple rule-based system.
        In a real implementation, this would use a trained ML model.
        """
        # Simulate predictions based on feature patterns
        # Feature 0: packet_size, Feature 1: is_tcp, Features 2-3: ports
        # Features 4-7: TCP flags, Feature 8: payload_size, Feature 9: has_payload
        
        prediction = 0  # Default to normal traffic
        confidence = 0.0
        
        # Simplified detection logic based on packet characteristics
        if features[4] > 0 and features[6] == 0 and features[5] == 0:  # SYN without ACK or FIN
            # Potential port scan
            prediction = 1
            confidence = 0.7 + (random.random() * 0.2)  # Random variation in confidence
            
        elif features[0] > 0.6 and features[8] > 0.7:  # Large packets with large payloads
            # Potential DDoS
            prediction = 2
            confidence = 0.75 + (random.random() * 0.2)
            
        elif features[5] > 0 and features[2] < 0.1:  # ACK with low port (like SSH or RDP)
            # Potential brute force
            prediction = 3
            confidence = 0.65 + (random.random() * 0.3)
            
        elif features[8] > 0.8 and features[9] and features[3] > 0.5:  # Large outbound data
            # Potential data exfiltration
            prediction = 4
            confidence = 0.7 + (random.random() * 0.25)
            
        elif features[0] < 0.2 and features[9] and random.random() > 0.7:  # Small packets with payloads
            # Potential command and control
            prediction = 5
            confidence = 0.6 + (random.random() * 0.3)
        else:
            # Normal traffic
            confidence = 0.8 + (random.random() * 0.15)
        
        # Add some randomness to make it interesting
        if random.random() < 0.05:  # 5% chance of a different detection
            alternate_prediction = random.randint(1, 5)  # Choose a random threat type
            return alternate_prediction, 0.55 + (random.random() * 0.2)
            
        return prediction, confidence


class SimpleRealTimeDetector:
    """
    Simple real-time detector that processes packets and detects threats.
    """
    def __init__(self, model, feature_extractor, threshold=0.5):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.detected_threats = []
        self.running = False
        self.packet_count = 0
        self.threat_count = 0
        
    def start(self):
        """
        Start the detector.
        """
        self.running = True
        print("Real-time detector started.")
        
    def stop(self):
        """
        Stop the detector.
        """
        self.running = False
        print("Real-time detector stopped.")
        
    def process_packet(self, packet):
        """
        Process a packet and detect threats.
        """
        if not self.running:
            return None
            
        self.packet_count += 1
        
        # Extract features
        features = self.feature_extractor.transform(packet)
        
        # Make prediction
        prediction, confidence = self.model.predict(features)
        
        result = {
            'timestamp': packet.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')),
            'source_ip': packet.get('source_ip', '0.0.0.0'),
            'dest_ip': packet.get('dest_ip', '0.0.0.0'),
            'source_port': packet.get('source_port', 0),
            'dest_port': packet.get('dest_port', 0),
            'protocol': packet.get('protocol', 'UNKNOWN'),
            'packet_size': packet.get('packet_size', 0),
            'prediction': prediction,
            'confidence': confidence,
            'is_threat': prediction > 0 and confidence > self.threshold,
            'threat_type': self.model.threat_types[prediction] if prediction < len(self.model.threat_types) else 'Unknown',
        }
        
        if result['is_threat']:
            self.threat_count += 1
            self.detected_threats.append(result)
            
        return result


def generate_random_packet():
    """
    Generate a random network packet for testing.
    """
    protocols = ['TCP', 'UDP', 'ICMP']
    internal_ips = ['192.168.1.' + str(i) for i in range(1, 20)]
    external_ips = ['203.0.113.' + str(i) for i in range(1, 50)]
    
    # Random packet properties
    protocol = random.choice(protocols)
    source_ip = random.choice(internal_ips + external_ips)
    dest_ip = random.choice(internal_ips + external_ips)
    
    # Ports only relevant for TCP/UDP
    source_port = random.randint(1024, 65535) if protocol in ['TCP', 'UDP'] else 0
    dest_port = random.choice([21, 22, 23, 25, 53, 80, 443, 3389, 8080] + 
                              [random.randint(1024, 65535) for _ in range(5)]) if protocol in ['TCP', 'UDP'] else 0
    
    # TCP flags only relevant for TCP
    tcp_flags = {}
    if protocol == 'TCP':
        flags = ['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG']
        # Most packets are SYN, ACK, or SYN-ACK
        if random.random() < 0.7:
            if random.random() < 0.4:  # SYN
                tcp_flags = {'SYN': 1, 'ACK': 0, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0}
            elif random.random() < 0.7:  # ACK
                tcp_flags = {'SYN': 0, 'ACK': 1, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0}
            else:  # SYN-ACK
                tcp_flags = {'SYN': 1, 'ACK': 1, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0}
        else:
            # Random combination of flags
            tcp_flags = {flag: 1 if random.random() > 0.7 else 0 for flag in flags}
    
    # Packet and payload size
    packet_size = random.randint(64, 1500)
    has_payload = random.random() > 0.3
    payload_size = random.randint(0, 1460) if has_payload else 0
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    return {
        'timestamp': timestamp,
        'source_ip': source_ip,
        'dest_ip': dest_ip,
        'source_port': source_port,
        'dest_port': dest_port,
        'protocol': protocol,
        'tcp_flags': tcp_flags,
        'packet_size': packet_size,
        'has_payload': has_payload,
        'payload_size': payload_size
    }


def simulate_traffic(detector, duration=10, packet_rate=5):
    """
    Simulate network traffic for a specified duration.
    
    Args:
        detector: The detector to process packets.
        duration: Duration to simulate in seconds.
        packet_rate: Approximate packets per second.
    """
    start_time = time.time()
    packet_count = 0
    
    print(f"Starting traffic simulation for {duration} seconds at {packet_rate} packets/sec...")
    
    while time.time() - start_time < duration:
        # Generate a random packet
        packet = generate_random_packet()
        
        # Process the packet
        result = detector.process_packet(packet)
        
        # Print threat detections
        if result and result['is_threat']:
            print(f"⚠️ Threat detected: {result['threat_type']} from {result['source_ip']}:{result['source_port']} "
                  f"to {result['dest_ip']}:{result['dest_port']} ({result['confidence']:.2f} confidence)")
        
        packet_count += 1
        
        # Sleep to control packet rate
        time.sleep(1.0 / packet_rate)
    
    elapsed = time.time() - start_time
    print(f"Simulation complete: {packet_count} packets processed in {elapsed:.2f} seconds "
          f"({packet_count / elapsed:.2f} packets/sec)")
    print(f"Detected {detector.threat_count} threats out of {detector.packet_count} packets "
          f"({detector.threat_count / detector.packet_count * 100:.2f}% threat rate)")


def generate_report(detector):
    """
    Generate a simple report of detected threats.
    """
    if not detector.detected_threats:
        print("No threats detected during the simulation.")
        return
        
    print("\nThreat Detection Report")
    print("======================")
    print(f"Total packets processed: {detector.packet_count}")
    print(f"Total threats detected: {detector.threat_count}")
    
    # Count threats by type
    threat_counts = {}
    for threat in detector.detected_threats:
        threat_type = threat['threat_type']
        if threat_type not in threat_counts:
            threat_counts[threat_type] = 0
        threat_counts[threat_type] += 1
    
    print("\nThreats by type:")
    for threat_type, count in threat_counts.items():
        print(f"  - {threat_type}: {count} ({count / detector.threat_count * 100:.1f}%)")
    
    # Show a few example threats
    print("\nExample threat detections:")
    for i, threat in enumerate(detector.detected_threats[:5]):
        print(f"\n[{i+1}] {threat['threat_type']} (Confidence: {threat['confidence']:.2f})")
        print(f"    Time: {threat['timestamp']}")
        print(f"    Source: {threat['source_ip']}:{threat['source_port']}")
        print(f"    Destination: {threat['dest_ip']}:{threat['dest_port']}")
        print(f"    Protocol: {threat['protocol']}")
        print(f"    Packet Size: {threat['packet_size']} bytes")
    
    print("\nRecommendations:")
    if 'Port Scan' in threat_counts:
        print("  - Monitor for reconnaissance activity and consider firewall rule updates")
    if 'DDoS' in threat_counts:
        print("  - Implement rate limiting and traffic filtering to mitigate DDoS attacks")
    if 'Brute Force' in threat_counts:
        print("  - Implement account lockout policies and consider IP blocking for repeated failures")
    if 'Data Exfiltration' in threat_counts:
        print("  - Review data loss prevention policies and monitor outbound data transfers")
    if 'Command & Control' in threat_counts:
        print("  - Investigate potentially compromised systems and isolate affected hosts")
    
    print("\nNote: This is a simulated report using synthetic data for demonstration purposes.")


def main():
    """
    Main function for the simplified real-time threat detection example.
    """
    print("CyberThreat-ML Simplified Real-Time Detection Example")
    print("-----------------------------------------------------")
    
    # Create components
    model = SimpleDetectionModel(threshold=0.6)
    feature_extractor = SimpleFeatureExtractor()
    detector = SimpleRealTimeDetector(model, feature_extractor, threshold=0.6)
    
    # Start the detector
    detector.start()
    
    # Simulate traffic
    try:
        # Run a short simulation
        simulate_traffic(detector, duration=30, packet_rate=10)
        
        # Generate report
        generate_report(detector)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Stop the detector
        detector.stop()
    
    print("\nSimplified real-time detection example completed successfully!")


if __name__ == "__main__":
    main()