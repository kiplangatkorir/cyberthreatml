#!/usr/bin/env python3
"""
Synthetic Attack Demo - CyberThreat-ML

This script simulates a multi-stage attack scenario in real-time and shows how CyberThreat-ML 
detects and responds to each stage of the attack. It's designed for educational purposes
to demonstrate the capabilities of the library.

Usage:
    python synthetic_attack_demo.py [--duration 300] [--verbose]

Options:
    --duration  Duration of the simulation in seconds (default: 300)
    --verbose   Show detailed output including packet data

Note: This is purely educational and simulates detection of synthetic attacks.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cyberthreat_ml.model import ThreatDetectionModel
from cyberthreat_ml.anomaly import ZeroDayDetector, get_anomaly_description, recommend_action
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.visualization import ThreatVisualizationDashboard
from cyberthreat_ml.interpretability import ThreatInterpreter

# Create output directory
os.makedirs("attack_demo_output", exist_ok=True)

# Terminal colors for better output visualization
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Attack scenario configuration
ATTACK_SCENARIO = [
    {
        "name": "Reconnaissance", 
        "duration": 30,
        "description": "Attacker performs port scanning and network enumeration",
        "attack_type": "Port Scan",
        "indicators": ["Unusual port access patterns", "Sequential port access", "Multiple connection attempts"],
        "severity": "Low"
    },
    {
        "name": "Initial Access", 
        "duration": 40,
        "description": "Attacker attempts to gain initial access through credential brute forcing",
        "attack_type": "Brute Force",
        "indicators": ["Multiple failed logins", "Rapid authentication attempts", "Password spray pattern"],
        "severity": "Medium"
    },
    {
        "name": "Command & Control", 
        "duration": 50,
        "description": "Attacker establishes command and control channel",
        "attack_type": "Command & Control",
        "indicators": ["Unusual DNS queries", "Periodic beaconing", "Encrypted traffic patterns"],
        "severity": "Medium-High"
    },
    {
        "name": "Lateral Movement", 
        "duration": 60,
        "description": "Attacker moves laterally through the network",
        "attack_type": "Unusual Internal Traffic",
        "indicators": ["Internal scanning", "Privileged account usage", "Unusual service access"],
        "severity": "High"
    },
    {
        "name": "Data Exfiltration", 
        "duration": 45,
        "description": "Attacker exfiltrates sensitive data from the network",
        "attack_type": "Data Exfiltration",
        "indicators": ["Large outbound transfers", "Unusual upload patterns", "Compressed data transfers"],
        "severity": "Critical"
    },
    {
        "name": "Novel Zero-Day Exploit", 
        "duration": 75,
        "description": "Attacker uses a previously unknown zero-day vulnerability",
        "attack_type": "Zero-Day",
        "indicators": ["Anomalous system behavior", "Unusual process activity", "Unexpected privilege escalation"],
        "severity": "Critical"
    }
]

class AttackSimulator:
    """Simulates an attack scenario and generates network traffic data."""
    
    def __init__(self, scenario=ATTACK_SCENARIO, verbose=False):
        """Initialize the attack simulator."""
        self.scenario = scenario
        self.verbose = verbose
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.normal_traffic_ratio = 0.7  # 70% normal traffic, 30% attack traffic
        self.internal_ips = [f"10.0.0.{i}" for i in range(1, 20)]
        self.external_ips = [f"192.168.1.{i}" for i in range(1, 10)] + [f"203.0.113.{i}" for i in range(1, 5)]
        self.common_ports = [80, 443, 22, 25, 53, 3389, 8080, 8443]
        self.suspicious_ports = [4444, 31337, 6667, 1337, 9001]
        self.attack_stats = {phase["name"]: 0 for phase in self.scenario}
        self.normal_packets = 0
        self.detected_threats = []
        self.log_file = open("attack_demo_output/attack_simulation.log", "w")
        self.log(f"{Colors.HEADER}Attack Simulation Started{Colors.ENDC}")
        self.log(f"{Colors.BOLD}Scenario:{Colors.ENDC} Multi-stage cyber attack simulation")
        
        # Threat detection status
        self.signature_detected = False
        self.anomaly_detected = False
        
        # Feature extractor for packets
        self.feature_names = [
            "Source Port", "Destination Port", "Packet Size", "Protocol Type",
            "TCP Flags", "TTL Value", "Inter-packet Time", "Flow Duration",
            "Payload Entropy", "Connection State", "Packet Count", "Bytes In",
            "Bytes Out", "Source IP Type", "Destination IP Type", "HTTP Method",
            "HTTP Response", "DNS Query Type", "Certificate Info", "Time of Day"
        ]

    def log(self, message):
        """Log a message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message.replace(Colors.HEADER, "").replace(Colors.BLUE, "")
                          .replace(Colors.CYAN, "").replace(Colors.GREEN, "")
                          .replace(Colors.WARNING, "").replace(Colors.RED, "")
                          .replace(Colors.ENDC, "").replace(Colors.BOLD, "")
                          .replace(Colors.UNDERLINE, "") + "\n")
        self.log_file.flush()

    def get_current_phase(self):
        """Get the current attack phase."""
        if self.current_phase >= len(self.scenario):
            return None
        
        current_time = time.time()
        elapsed = current_time - self.phase_start_time
        
        if elapsed > self.scenario[self.current_phase]["duration"]:
            self.current_phase += 1
            if self.current_phase < len(self.scenario):
                self.phase_start_time = current_time
                self.log(f"\n{Colors.HEADER}[ATTACK PHASE CHANGE]{Colors.ENDC}")
                self.log(f"{Colors.BOLD}Now entering: {self.scenario[self.current_phase]['name']} phase{Colors.ENDC}")
                self.log(f"{Colors.CYAN}Description: {self.scenario[self.current_phase]['description']}{Colors.ENDC}")
                self.log(f"{Colors.WARNING}Severity: {self.scenario[self.current_phase]['severity']}{Colors.ENDC}\n")
            else:
                self.log(f"\n{Colors.GREEN}Attack scenario completed.{Colors.ENDC}")
                return None
        
        return self.scenario[self.current_phase]

    def generate_packet(self):
        """Generate a synthetic network packet based on current attack phase."""
        phase = self.get_current_phase()
        
        if phase is None:
            return None
        
        # Decide if this is a normal or attack packet
        is_attack = random.random() > self.normal_traffic_ratio
        
        if is_attack:
            # Generate attack packet for current phase
            packet = self._generate_attack_packet(phase)
            self.attack_stats[phase["name"]] += 1
        else:
            # Generate normal packet
            packet = self._generate_normal_packet()
            self.normal_packets += 1
        
        return packet

    def _generate_normal_packet(self):
        """Generate a normal network packet."""
        src_ip = random.choice(self.internal_ips) if random.random() < 0.5 else random.choice(self.external_ips)
        dst_ip = random.choice(self.internal_ips) if src_ip in self.external_ips else random.choice(self.external_ips)
        
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.randint(49152, 65535),  # Ephemeral ports
            'dst_port': random.choice(self.common_ports),
            'size': int(random.normalvariate(500, 200)),  # Normal size
            'protocol': random.choice([6, 17]),  # TCP or UDP
            'flags': random.randint(0, 31),  # Random TCP flags
            'ttl': random.randint(50, 64),
            'payload_entropy': random.uniform(0.1, 0.5),  # Lower entropy for normal
            'payload': None,  # No need to generate actual payload
            'is_attack': False,
            'attack_type': None
        }
        
        return packet

    def _generate_attack_packet(self, phase):
        """Generate an attack packet based on the current phase."""
        attack_type = phase["attack_type"]
        
        if attack_type == "Port Scan":
            return self._generate_port_scan_packet()
        elif attack_type == "Brute Force":
            return self._generate_brute_force_packet()
        elif attack_type == "Command & Control":
            return self._generate_cnc_packet()
        elif attack_type == "Unusual Internal Traffic":
            return self._generate_lateral_movement_packet()
        elif attack_type == "Data Exfiltration":
            return self._generate_exfiltration_packet()
        elif attack_type == "Zero-Day":
            return self._generate_zero_day_packet()
        else:
            # Fallback to normal packet
            packet = self._generate_normal_packet()
            packet['is_attack'] = True
            packet['attack_type'] = attack_type
            return packet

    def _generate_port_scan_packet(self):
        """Generate a port scan packet."""
        src_ip = random.choice(self.external_ips)
        dst_ip = random.choice(self.internal_ips)
        
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.randint(49152, 65535),
            'dst_port': random.randint(1, 10000),  # Scanning a wide range of ports
            'size': int(random.normalvariate(60, 10)),  # Small packets
            'protocol': 6,  # TCP
            'flags': 2,  # SYN
            'ttl': random.randint(50, 64),
            'payload_entropy': random.uniform(0.0, 0.1),  # Very low entropy
            'payload': None,
            'is_attack': True,
            'attack_type': "Port Scan"
        }
        
        return packet

    def _generate_brute_force_packet(self):
        """Generate a brute force authentication packet."""
        src_ip = random.choice(self.external_ips)
        dst_ip = random.choice(self.internal_ips)
        
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.randint(49152, 65535),
            'dst_port': random.choice([22, 3389, 23, 21, 5900]),  # SSH, RDP, Telnet, FTP, VNC
            'size': int(random.normalvariate(200, 50)),
            'protocol': 6,  # TCP
            'flags': 24,  # ACK, PSH
            'ttl': random.randint(50, 64),
            'payload_entropy': random.uniform(0.5, 0.8),
            'payload': None,
            'is_attack': True,
            'attack_type': "Brute Force"
        }
        
        return packet

    def _generate_cnc_packet(self):
        """Generate a command and control packet."""
        src_ip = random.choice(self.internal_ips)
        dst_ip = random.choice(self.external_ips)
        
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.randint(49152, 65535),
            'dst_port': random.choice(self.suspicious_ports),
            'size': int(random.normalvariate(300, 100)),
            'protocol': random.choice([6, 17]),  # TCP or UDP
            'flags': 24 if packet['protocol'] == 6 else 0,  # ACK, PSH for TCP
            'ttl': random.randint(50, 64),
            'payload_entropy': random.uniform(0.8, 0.95),  # High entropy (encryption)
            'payload': None,
            'is_attack': True,
            'attack_type': "Command & Control"
        }
        
        return packet

    def _generate_lateral_movement_packet(self):
        """Generate a lateral movement packet."""
        src_ip = random.choice(self.internal_ips)
        
        # Choose a different internal IP
        available_ips = [ip for ip in self.internal_ips if ip != src_ip]
        dst_ip = random.choice(available_ips)
        
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.randint(49152, 65535),
            'dst_port': random.choice([445, 135, 139, 3389, 5985, 5986]),  # SMB, RPC, NetBIOS, RDP, WinRM
            'size': int(random.normalvariate(800, 200)),
            'protocol': 6,  # TCP
            'flags': 24,  # ACK, PSH
            'ttl': random.randint(50, 64),
            'payload_entropy': random.uniform(0.6, 0.9),
            'payload': None,
            'is_attack': True,
            'attack_type': "Lateral Movement"
        }
        
        return packet

    def _generate_exfiltration_packet(self):
        """Generate a data exfiltration packet."""
        src_ip = random.choice(self.internal_ips)
        dst_ip = random.choice(self.external_ips)
        
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.randint(49152, 65535),
            'dst_port': random.choice([80, 443, 8080, 8443, 21, 22]),  # HTTP, HTTPS, FTP, SSH
            'size': int(random.normalvariate(9000, 2000)),  # Large packet
            'protocol': 6,  # TCP
            'flags': 24,  # ACK, PSH
            'ttl': random.randint(50, 64),
            'payload_entropy': random.uniform(0.85, 0.99),  # Very high entropy (compressed/encrypted)
            'payload': None,
            'is_attack': True,
            'attack_type': "Data Exfiltration"
        }
        
        return packet

    def _generate_zero_day_packet(self):
        """Generate a zero-day exploit packet with unusual characteristics."""
        # This is a hypothetical packet that doesn't fit normal patterns
        # or known attack patterns - it should trigger anomaly detection
        
        src_ip = random.choice(self.external_ips)
        dst_ip = random.choice(self.internal_ips)
        
        # Unusual combination of characteristics
        packet = {
            'timestamp': time.time(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': random.choice(self.suspicious_ports),
            'dst_port': random.choice([6000, 1434, 5432, 27017, 9200]),  # Unusual service ports
            'size': int(random.uniform(5000, 15000)),  # Unusual size
            'protocol': random.choice([6, 17, 1, 47, 50]),  # Including unusual protocols
            'flags': random.randint(0, 255),  # Potentially invalid flag combinations
            'ttl': random.randint(10, 255),   # Unusual TTL
            'payload_entropy': random.uniform(0.7, 0.99),
            'payload': None,
            'is_attack': True,
            'attack_type': "Zero-Day"
        }
        
        # Add unusual fields that normal packets don't have
        packet['fragment_offset'] = random.randint(0, 8000) if random.random() < 0.5 else 0
        packet['options'] = {'unusual_option': random.randint(0, 255)} if random.random() < 0.7 else {}
        
        return packet

    def extract_features(self, packet):
        """Extract features from a packet for the ML model."""
        # Create feature vector from packet
        features = np.zeros(20)  # 20 features
        
        # Normalized source port
        features[0] = packet['src_port'] / 65535.0
        
        # Normalized destination port
        features[1] = packet['dst_port'] / 65535.0
        
        # Normalized packet size
        features[2] = min(packet['size'] / 15000.0, 1.0)
        
        # Protocol type (one-hot encoded)
        if packet['protocol'] == 6:  # TCP
            features[3] = 1
        elif packet['protocol'] == 17:  # UDP
            features[3] = 0.5
        else:  # Other
            features[3] = 0
            
        # TCP flags (if applicable)
        features[4] = packet['flags'] / 255.0 if packet['protocol'] == 6 else 0
        
        # TTL value
        features[5] = packet['ttl'] / 255.0
        
        # Inter-packet time (simulated)
        features[6] = random.uniform(0, 1) if not hasattr(self, 'last_time') else \
                     min((packet['timestamp'] - self.last_time) / 10.0, 1.0)
        self.last_time = packet['timestamp']
        
        # Flow duration (simulated)
        features[7] = random.uniform(0, 1)
        
        # Payload entropy
        features[8] = packet['payload_entropy']
        
        # Connection state (simulated)
        features[9] = random.uniform(0, 1)
        
        # Packet count (simulated)
        features[10] = random.uniform(0, 1)
        
        # Bytes in (simulated)
        features[11] = random.uniform(0, 1)
        
        # Bytes out (simulated)
        features[12] = random.uniform(0, 1)
        
        # Source IP type (internal/external)
        features[13] = 1.0 if packet['src_ip'] in self.internal_ips else 0.0
        
        # Destination IP type (internal/external)
        features[14] = 1.0 if packet['dst_ip'] in self.internal_ips else 0.0
        
        # HTTP method (simulated)
        features[15] = random.uniform(0, 1) if packet['dst_port'] in [80, 443, 8080, 8443] else 0
        
        # HTTP response (simulated)
        features[16] = random.uniform(0, 1) if packet['dst_port'] in [80, 443, 8080, 8443] else 0
        
        # DNS query type (simulated)
        features[17] = random.uniform(0, 1) if packet['dst_port'] == 53 else 0
        
        # Certificate info (simulated)
        features[18] = random.uniform(0, 1) if packet['dst_port'] == 443 else 0
        
        # Time of day (normalized hour)
        current_hour = datetime.now().hour + datetime.now().minute / 60.0
        features[19] = current_hour / 24.0
        
        return features

    def on_threat_detected(self, result):
        """Handle threat detection."""
        if result.get('is_anomaly', False):
            # Anomaly (zero-day) detection
            self.anomaly_detected = True
            severity = result.get('severity_level', 'Unknown')
            score = result.get('anomaly_score', 0)
            
            # Get description and recommendations
            analysis = result.get('analysis', {})
            description = get_anomaly_description(analysis)
            
            self.log(f"\n{Colors.RED}[ANOMALY DETECTED]{Colors.ENDC}")
            self.log(f"{Colors.BOLD}Severity: {severity} ({result.get('severity', 0):.4f}){Colors.ENDC}")
            self.log(f"{Colors.CYAN}Description: {description}{Colors.ENDC}")
            self.log(f"{Colors.BLUE}Anomaly Score: {score:.4f}{Colors.ENDC}")
            
            # Add to detected threats
            self.detected_threats.append({
                'timestamp': time.time(),
                'type': 'anomaly',
                'severity': severity,
                'score': score,
                'description': description
            })
            
            # Log recommendations
            if 'analysis' in result:
                actions = recommend_action(result['analysis'])
                self.log(f"{Colors.GREEN}Priority: {actions['priority']}{Colors.ENDC}")
                self.log(f"{Colors.GREEN}Recommended actions:{Colors.ENDC}")
                for action in actions['actions']:
                    self.log(f"  - {action}")
        else:
            # Signature-based detection
            self.signature_detected = True
            confidence = result.get('confidence', 0)
            class_name = result.get('class_name', 'Unknown')
            
            self.log(f"\n{Colors.RED}[THREAT DETECTED] {class_name}{Colors.ENDC}")
            self.log(f"{Colors.BOLD}Confidence: {confidence:.4f}{Colors.ENDC}")
            
            # Add to detected threats
            self.detected_threats.append({
                'timestamp': time.time(),
                'type': 'signature',
                'class': class_name,
                'confidence': confidence
            })
            
            # Log packet details if verbose
            if self.verbose and 'data' in result:
                packet = result['data']
                self.log(f"Source IP: {packet.get('src_ip')}, Destination IP: {packet.get('dst_ip')}")
                self.log(f"Source Port: {packet.get('src_port')}, Destination Port: {packet.get('dst_port')}")
                self.log(f"Size: {packet.get('size')} bytes, Protocol: {packet.get('protocol')}")

    def on_batch_processed(self, results):
        """Handle batch processing results."""
        threats = [r for r in results if r.get('is_threat', False) or r.get('is_anomaly', False)]
        if threats and self.verbose:
            self.log(f"Batch processed: {len(results)} packets, {len(threats)} threats detected")

    def generate_summary(self):
        """Generate a summary of the attack simulation."""
        total_packets = self.normal_packets + sum(self.attack_stats.values())
        
        summary = {
            'total_packets': total_packets,
            'normal_packets': self.normal_packets,
            'attack_packets': sum(self.attack_stats.values()),
            'attack_breakdown': self.attack_stats,
            'detection_rate': {
                'signature': len([t for t in self.detected_threats if t['type'] == 'signature']) / max(1, sum(self.attack_stats.values())),
                'anomaly': len([t for t in self.detected_threats if t['type'] == 'anomaly']) / max(1, self.attack_stats.get("Zero-Day", 0))
            },
            'detected_threats': self.detected_threats
        }
        
        # Log summary
        self.log(f"\n{Colors.HEADER}Attack Simulation Summary{Colors.ENDC}")
        self.log(f"Total packets generated: {total_packets}")
        self.log(f"Normal traffic: {self.normal_packets} packets ({self.normal_packets/max(1, total_packets)*100:.1f}%)")
        self.log(f"Attack traffic: {sum(self.attack_stats.values())} packets ({sum(self.attack_stats.values())/max(1, total_packets)*100:.1f}%)")
        
        self.log("\nAttack breakdown:")
        for phase, count in self.attack_stats.items():
            self.log(f"  {phase}: {count} packets ({count/max(1, sum(self.attack_stats.values()))*100:.1f}% of attacks)")
        
        self.log("\nDetection stats:")
        sig_threats = len([t for t in self.detected_threats if t['type'] == 'signature'])
        anom_threats = len([t for t in self.detected_threats if t['type'] == 'anomaly'])
        self.log(f"  Signature-based detections: {sig_threats}")
        self.log(f"  Anomaly-based detections: {anom_threats}")
        
        sig_rate = sig_threats / max(1, sum(self.attack_stats.values()) - self.attack_stats.get("Zero-Day", 0))
        anom_rate = anom_threats / max(1, self.attack_stats.get("Zero-Day", 0))
        self.log(f"  Signature detection rate: {sig_rate*100:.1f}%")
        self.log(f"  Zero-day detection rate: {anom_rate*100:.1f}%")
        
        # Save summary to file
        with open("attack_demo_output/attack_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def close(self):
        """Close the log file."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

class FeatureExtractor:
    """Extract features from packet data for the model."""
    
    def __init__(self, simulator):
        """Initialize with the simulator for consistent feature extraction."""
        self.simulator = simulator
    
    def transform(self, packet):
        """Transform packet data into feature vector."""
        features = self.simulator.extract_features(packet)
        return features.reshape(1, -1)  # Return as 2D array

class CustomModel:
    """Custom model for the demo that returns predefined outputs based on packet type."""
    
    def __init__(self, class_names):
        """Initialize with class names."""
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.is_binary = False if self.num_classes > 2 else True
    
    def predict(self, X):
        """Predict class based on input features."""
        predictions = []
        for i in range(X.shape[0]):
            # Extract key features that help determine the attack type
            dst_port = X[i, 1] * 65535  # Denormalize
            pkt_size = X[i, 2] * 15000  # Denormalize
            tcp_flags = X[i, 4] * 255   # Denormalize
            payload_entropy = X[i, 8]   # Already normalized
            
            # Simple rules to classify
            if dst_port < 1024 and pkt_size < 100:
                predictions.append(1)  # Port Scan
            elif dst_port in [22, 3389, 23, 21, 5900] and tcp_flags > 20:
                predictions.append(2)  # Brute Force
            elif payload_entropy > 0.8 and dst_port > 1024:
                predictions.append(3)  # Command & Control
            elif X[i, 13] > 0.5 and X[i, 14] > 0.5:  # Internal to internal
                predictions.append(4)  # Lateral Movement
            elif pkt_size > 5000 and payload_entropy > 0.8:
                predictions.append(5)  # Data Exfiltration
            else:
                predictions.append(0)  # Normal
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return predicted probabilities for each class."""
        if self.is_binary:
            # For binary classification
            probas = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                # Determine confidence based on how strongly it matches rules
                dst_port = X[i, 1] * 65535
                pkt_size = X[i, 2] * 15000
                tcp_flags = X[i, 4] * 255
                payload_entropy = X[i, 8]
                
                if dst_port < 1024 and pkt_size < 100:
                    probas[i, 0] = 0.85  # Port Scan
                elif dst_port in [22, 3389, 23, 21, 5900] and tcp_flags > 20:
                    probas[i, 0] = 0.78  # Brute Force
                elif payload_entropy > 0.8 and dst_port > 1024:
                    probas[i, 0] = 0.82  # Command & Control
                elif X[i, 13] > 0.5 and X[i, 14] > 0.5:
                    probas[i, 0] = 0.75  # Lateral Movement
                elif pkt_size > 5000 and payload_entropy > 0.8:
                    probas[i, 0] = 0.88  # Data Exfiltration
                else:
                    probas[i, 0] = 0.3   # Normal with some uncertainty
                    
            return probas
        else:
            # For multi-class
            probas = np.zeros((X.shape[0], self.num_classes))
            preds = self.predict(X)
            
            for i in range(X.shape[0]):
                pred_class = preds[i]
                
                # Base probability for predicted class
                base_prob = 0.7
                
                # Distribute remaining probability among other classes
                other_probs = (1 - base_prob) / (self.num_classes - 1)
                
                for j in range(self.num_classes):
                    if j == pred_class:
                        probas[i, j] = base_prob
                    else:
                        probas[i, j] = other_probs
                        
            return probas

def run_attack_demo(duration=300, verbose=False):
    """Run the attack demo for the specified duration."""
    
    # Create attack simulator
    simulator = AttackSimulator(verbose=verbose)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(simulator)
    
    # Define class names
    class_names = [
        "Normal",
        "Port Scan",
        "Brute Force",
        "Command & Control",
        "Lateral Movement",
        "Data Exfiltration"
    ]
    
    # Create custom model for demo
    model = CustomModel(class_names)
    
    # Create zero-day detector
    zero_day_detector = ZeroDayDetector(
        method='ensemble',
        contamination=0.05,  # 5% expected anomaly rate
        min_samples=20       # Need at least 20 samples
    )
    
    # Create initial normal data for training zero-day detector
    normal_data = np.random.random((100, 20))
    # Make the normal data actually look normal
    normal_data[:, 2] *= 0.3  # Normal packet sizes
    normal_data[:, 8] *= 0.6  # Normal entropy values
    zero_day_detector.fit(normal_data, simulator.feature_names)
    
    # Create packet stream detector
    detector = PacketStreamDetector(
        model=model,
        feature_extractor=feature_extractor,
        threshold=0.6,       # Detection threshold
        batch_size=5,        # Process in batches of 5
        processing_interval=1.0  # Process every second
    )
    
    # Register callbacks
    detector.register_threat_callback(simulator.on_threat_detected)
    detector.register_processing_callback(simulator.on_batch_processed)
    
    # Start the detector
    detector.start()
    
    # Start time and processed count
    start_time = time.time()
    processed_count = 0
    animation = None
    
    try:
        # Set up the visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Real-time Cyber Attack Simulation', fontsize=16)
        
        # Initialize data structures for plotting
        timestamps = []
        attack_counts = {phase["name"]: [] for phase in ATTACK_SCENARIO}
        attack_counts["Normal"] = []
        
        # Set up the first subplot for traffic breakdown
        ax1.set_title('Traffic Breakdown')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Packets')
        
        # Set up the second subplot for detection timeline
        ax2.set_title('Attack Detection Timeline')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Detection Count')
        ax2.set_ylim(0, 10)  # Adjust as needed
        
        detection_timestamps = []
        signature_detections = []
        anomaly_detections = []
        
        # Function to update the plot
        def update_plot(frame):
            # Update traffic breakdown
            ax1.clear()
            ax1.set_title('Traffic Breakdown')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Packets')
            ax1.stackplot(timestamps, 
                         [attack_counts[phase] for phase in attack_counts.keys()],
                         labels=list(attack_counts.keys()),
                         alpha=0.7)
            ax1.legend(loc='upper left')
            
            # Update detection timeline
            ax2.clear()
            ax2.set_title('Attack Detection Timeline')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Detection Count')
            ax2.plot(detection_timestamps, signature_detections, 'b-', label='Signature Detections')
            ax2.plot(detection_timestamps, anomaly_detections, 'r-', label='Anomaly Detections')
            ax2.legend(loc='upper left')
            
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
        
        # Set up the animation
        animation = FuncAnimation(fig, update_plot, interval=1000)
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
        # Main simulation loop
        while time.time() - start_time < duration:
            # Get current phase
            phase = simulator.get_current_phase()
            if phase is None:
                break
                
            # Generate packet
            packet = simulator.generate_packet()
            if packet is None:
                break
                
            # Process packet
            detector.process_packet(packet)
            processed_count += 1
            
            # Check for zero-day detection
            if packet['attack_type'] == "Zero-Day":
                features = simulator.extract_features(packet)
                # Process with zero-day detector
                prediction, score = zero_day_detector.detect(features.reshape(1, -1), return_scores=True)
                
                if prediction[0] == -1:  # Anomaly detected
                    # Analyze the anomaly
                    analysis = zero_day_detector.analyze_anomaly(features, score[0])
                    
                    # Create result for callback
                    result = {
                        'timestamp': packet['timestamp'],
                        'is_anomaly': True,
                        'anomaly_score': float(score[0]),
                        'severity': analysis["severity"],
                        'severity_level': analysis["severity_level"],
                        'analysis': analysis,
                        'features': features.tolist() if hasattr(features, "tolist") else features,
                        'raw_data': packet
                    }
                    
                    # Call the callback directly
                    simulator.on_threat_detected(result)
            
            # Update plotting data (every second to avoid overhead)
            current_time = time.time() - start_time
            if not timestamps or current_time - timestamps[-1] >= 1.0:
                timestamps.append(current_time)
                for name in attack_counts.keys():
                    if name == "Normal":
                        attack_counts[name].append(simulator.normal_packets)
                    else:
                        attack_counts[name].append(simulator.attack_stats.get(name, 0))
                
                # Update detection timeline
                detection_timestamps.append(current_time)
                signature_detections.append(len([t for t in simulator.detected_threats 
                                               if t['type'] == 'signature']))
                anomaly_detections.append(len([t for t in simulator.detected_threats 
                                             if t['type'] == 'anomaly']))
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
            
            # Print status update every 50 packets
            if processed_count % 50 == 0 and verbose:
                elapsed = time.time() - start_time
                simulator.log(f"Processed {processed_count} packets in {elapsed:.1f} seconds "
                             f"({processed_count/elapsed:.1f} pps)")
        
        # Save the final plot
        plt.savefig("attack_demo_output/attack_simulation_plot.png")
        
    except KeyboardInterrupt:
        simulator.log("Simulation interrupted by user.")
    finally:
        # Stop the detector
        detector.stop()
        
        # Generate and save summary
        summary = simulator.generate_summary()
        
        # Close log file
        simulator.close()
        
        # Close the plot
        if animation:
            animation.event_source.stop()
            plt.close()
        
        return summary

def main():
    """Main function to parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description='Run a synthetic attack demo with CyberThreat-ML')
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration of the simulation in seconds (default: 300)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}Starting Synthetic Attack Demo{Colors.ENDC}")
    print(f"Duration: {args.duration} seconds")
    print(f"Verbose mode: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"{Colors.BOLD}Note: This is an educational demonstration using synthetic data.{Colors.ENDC}")
    print("-" * 80)
    
    summary = run_attack_demo(duration=args.duration, verbose=args.verbose)
    
    print("-" * 80)
    print(f"{Colors.GREEN}Demo completed successfully!{Colors.ENDC}")
    print(f"Summary saved to attack_demo_output/attack_summary.json")
    print(f"Log saved to attack_demo_output/attack_simulation.log")
    print(f"Plot saved to attack_demo_output/attack_simulation_plot.png")

if __name__ == "__main__":
    main()