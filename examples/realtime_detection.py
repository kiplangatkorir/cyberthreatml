"""
Example of real-time cybersecurity threat detection.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the cyberthreat_ml package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import time
import threading
import random
import json
from datetime import datetime

# Import CyberThreat-ML components
from cyberthreat_ml.model import ThreatDetectionModel, load_model
from cyberthreat_ml.realtime import PacketStreamDetector
from cyberthreat_ml.utils import normalize_port_number, normalize_packet_size, calculate_entropy

def main():
    """
    Example of real-time threat detection using PacketStreamDetector.
    """
    print("CyberThreat-ML Real-time Detection Example")
    print("------------------------------------------")
    
    # Step 1: Load or create a trained model
    print("\nStep 1: Loading model...")
    try:
        # Try to load a saved model
        model = load_model('threat_detection_model', 'threat_detection_metadata.json')
        print("Loaded existing model")
    except:
        print("No existing model found. Creating and training a new model...")
        model = create_and_train_model()
    
    # Step 2: Initialize the packet stream detector
    print("\nStep 2: Initializing real-time detector...")
    detector = PacketStreamDetector(
        model=model,
        feature_extractor=None,  # We'll manually extract features in this example
        threshold=0.6,
        batch_size=10,
        processing_interval=1.0
    )
    
    # Step 3: Set up the threat callback
    print("\nStep 3: Setting up callbacks...")
    detector.register_threat_callback(on_threat_detected)
    detector.register_processing_callback(on_batch_processed)
    
    # Step 4: Start the detector
    print("\nStep 4: Starting real-time detection...")
    detector.start()
    
    # Step 5: Generate and process network packets in a separate thread
    packet_generator = threading.Thread(target=generate_packets, args=(detector,))
    packet_generator.daemon = True
    packet_generator.start()
    
    # Step 6: Display statistics periodically
    try:
        while True:
            time.sleep(5)
            stats = detector.get_stats()
            print(f"\nStatistics at {datetime.now().strftime('%H:%M:%S')}:")
            print(f"  Packets processed: {stats['packet_count']}")
            print(f"  Threats detected: {stats['threat_count']}")
            print(f"  Queue size: {stats['queue_size']}")
    except KeyboardInterrupt:
        print("\nShutting down...")
        detector.stop()
        print("Real-time detection stopped")


def create_and_train_model():
    """
    Create and train a threat detection model using synthetic data.
    
    Returns:
        ThreatDetectionModel: Trained model with multi-class capabilities.
    """
    # Generate synthetic training data
    n_samples = 2000
    n_features = 10  # Number of features we'll use for packet representation
    
    # Define the threat classes
    threat_classes = [
        "Normal Traffic",  # Class 0
        "Port Scan",       # Class 1
        "DDoS",            # Class 2
        "Brute Force",     # Class 3
        "Data Exfiltration" # Class 4
    ]
    n_classes = len(threat_classes)
    
    # Generate random feature data
    X = np.random.random((n_samples, n_features))
    
    # Create class-specific patterns
    patterns = {
        0: lambda x: 0.1 * x[0] + 0.1 * x[1] + 0.1 * x[8],  # Normal traffic - low influence features
        1: lambda x: 0.8 * x[0] + 0.8 * x[1] + 0.2 * x[9],  # Port scan - high source/dest port influence
        2: lambda x: 0.3 * x[2] + 0.9 * x[3] + 0.7 * x[4],  # DDoS - high protocol and packet size influence
        3: lambda x: 0.2 * x[0] + 0.9 * x[6] + 0.7 * x[7],  # Brute force - flags and TTL influence
        4: lambda x: 0.8 * x[2] + 0.7 * x[8] + 0.5 * x[9]   # Data exfiltration - size and entropy influence
    }
    
    # Initialize labels with normal traffic (class 0)
    y = np.zeros(n_samples, dtype=int)
    
    # Apply patterns to create synthetic multi-class data
    for i in range(n_samples):
        # Calculate scores for each class
        class_scores = []
        for class_idx in range(n_classes):
            score = patterns[class_idx](X[i]) + 0.05 * np.random.random()
            class_scores.append(score)
        
        # Add class imbalance (more normal traffic)
        class_probs = np.array(class_scores)
        
        # Make class 0 more likely (to get imbalanced dataset for realism)
        if i % 2 == 0:
            class_probs[0] *= 2.0
        
        # Normalize probabilities
        class_probs = class_probs / np.sum(class_probs)
        
        # Assign class based on probabilities
        y[i] = np.random.choice(n_classes, p=class_probs)
    
    # Print class distribution
    print("Class distribution in synthetic dataset:")
    for i, class_name in enumerate(threat_classes):
        count = np.sum(y == i)
        print(f"  Class {i} ({class_name}): {count} samples ({count/len(y)*100:.1f}%)")
    
    # Create and train multi-class model
    model = ThreatDetectionModel(
        input_shape=(n_features,),
        num_classes=n_classes,
        model_config={
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.3,
            'class_names': threat_classes,
            'loss': 'sparse_categorical_crossentropy',  # Use sparse categorical since we're not using one-hot labels
            'metrics': ['accuracy']
        }
    )
    
    # Train model with validation split
    X_train, X_val = X[:int(0.8*n_samples)], X[int(0.8*n_samples):]
    y_train, y_val = y[:int(0.8*n_samples)], y[int(0.8*n_samples):]
    
    # Log the shape of training data
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Classes in training data: {np.unique(y_train)}")
    
    model.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        epochs=15, 
        batch_size=32,
        early_stopping=True
    )
    
    # Save the model
    import os
    os.makedirs('models', exist_ok=True)
    model.save_model(os.path.join('models', 'threat_detection_model'), 
                     os.path.join('models', 'threat_detection_metadata.json'))
    
    return model


def on_threat_detected(result):
    """
    Callback function for when a threat is detected.
    
    Args:
        result (dict): Detection result.
    """
    packet = result['packet']
    
    # Define threat classes (must match those in create_and_train_model)
    threat_classes = [
        "Normal Traffic",
        "Port Scan",
        "DDoS",
        "Brute Force",
        "Data Exfiltration"
    ]
    
    # Check if this is a multi-class result
    if 'class_probabilities' in result and not result.get('is_binary', True):
        # Multi-class threat detection
        class_idx = result.get('predicted_class', 0)
        probs = result.get('class_probabilities', [])
        
        # Skip if it's normal traffic (class 0)
        if class_idx == 0:
            return
            
        confidence = probs[class_idx] if len(probs) > class_idx else 0.0
        threat_type = threat_classes[class_idx] if class_idx < len(threat_classes) else f"Unknown Threat (Class {class_idx})"
        
        print(f"\n🚨 THREAT DETECTED! - {threat_type.upper()} 🚨")
        print(f"  Timestamp: {datetime.fromtimestamp(result['timestamp']).strftime('%H:%M:%S.%f')[:-3]}")
        
        # Check if packet is a dictionary or numpy array
        if isinstance(packet, dict):
            # Access packet data as dictionary
            print(f"  Source: {packet.get('src_ip', 'Unknown')}:{packet.get('src_port', 'Unknown')}")
            print(f"  Destination: {packet.get('dst_ip', 'Unknown')}:{packet.get('dst_port', 'Unknown')}")
            print(f"  Protocol: {packet.get('protocol', 'Unknown')}")
        else:
            # Handle numpy arrays or other data types
            print(f"  Feature vector: {packet[:5]}... (truncated)")
            
        print(f"  Confidence: {confidence:.4f}")
        
        # Print top 3 class probabilities if available
        if len(probs) >= 2:
            print("  Class probabilities:")
            # Get indices of top 3 classes sorted by probability
            top_indices = np.argsort(probs)[::-1][:3]
            for i in top_indices:
                print(f"    {threat_classes[i]}: {probs[i]:.4f}")
        
        # Suggested actions based on threat type and confidence
        print("  Suggested action:", end=" ")
        
        if class_idx == 1:  # Port Scan
            if confidence > 0.8:
                print("BLOCK source IP and ALERT SECURITY TEAM")
            else:
                print("MONITOR source IP for additional scanning activity")
                
        elif class_idx == 2:  # DDoS
            print("ACTIVATE ANTI-DDOS MEASURES and ALERT SECURITY TEAM")
                
        elif class_idx == 3:  # Brute Force
            print("TEMPORARY ACCOUNT LOCKOUT and ENABLE 2FA")
                
        elif class_idx == 4:  # Data Exfiltration
            print("ISOLATE affected systems and INVESTIGATE data access")
                
        else:  # Unknown threat type
            if confidence > 0.7:
                print("ISOLATE and INVESTIGATE")
            else:
                print("LOG and MONITOR")
    
    else:
        # Binary threat detection (fallback for compatibility)
        score = result.get('threat_score', 0.0)
        
        print(f"\n🚨 BINARY THREAT DETECTED! 🚨")
        print(f"  Timestamp: {datetime.fromtimestamp(result['timestamp']).strftime('%H:%M:%S.%f')[:-3]}")
        
        # Check if packet is a dictionary or numpy array
        if isinstance(packet, dict):
            # Access packet data as dictionary
            print(f"  Source: {packet.get('src_ip', 'Unknown')}:{packet.get('src_port', 'Unknown')}")
            print(f"  Destination: {packet.get('dst_ip', 'Unknown')}:{packet.get('dst_port', 'Unknown')}")
            print(f"  Protocol: {packet.get('protocol', 'Unknown')}")
        else:
            # Handle numpy arrays or other data types
            print(f"  Feature vector: {packet[:5]}... (truncated)")
            
        print(f"  Threat score: {score:.4f}")
        
        # Add suggested actions based on threat severity
        if score > 0.9:
            print("  Suggested action: BLOCK and ALERT SECURITY TEAM")
        elif score > 0.7:
            print("  Suggested action: ISOLATE and INVESTIGATE")
        else:
            print("  Suggested action: LOG and MONITOR")


def on_batch_processed(results):
    """
    Callback function for when a batch of packets is processed.
    
    Args:
        results (list): List of detection results.
    """
    # Count threats in batch
    threat_count = sum(1 for result in results if result.get('is_threat', False))
    
    if threat_count > 0:
        print(f"Batch processed: {len(results)} packets, {threat_count} threats detected")


def generate_packets(detector, duration=None, max_packets=None):
    """
    Generate synthetic network packets for testing.
    
    Args:
        detector (PacketStreamDetector): Detector to process packets.
        duration (float, optional): Duration to generate packets for (None = indefinitely).
        max_packets (int, optional): Maximum number of packets to generate (None = indefinitely).
    """
    start_time = time.time()
    packet_count = 0
    
    # Common ports for services
    common_ports = [80, 443, 22, 23, 25, 53, 110, 143, 3306, 3389, 8080]
    
    # Define IP address ranges
    internal_ips = [f"192.168.1.{i}" for i in range(1, 20)]
    external_ips = [f"203.0.113.{i}" for i in range(1, 50)]
    
    # Suspicious IP addresses (for generating threats)
    suspicious_ips = [f"10.0.0.{i}" for i in range(1, 5)]
    
    print("Starting packet generation...")
    
    while True:
        # Check termination conditions
        if duration is not None and time.time() - start_time > duration:
            break
        if max_packets is not None and packet_count >= max_packets:
            break
        
        # Generate a packet
        packet = generate_random_packet(internal_ips, external_ips, suspicious_ips, common_ports)
        
        # Extract features from the packet
        features = extract_packet_features(packet)
        
        # Send features to detector
        detector.process_packet(features)
        
        packet_count += 1
        
        # Random delay between packets
        time.sleep(random.uniform(0.05, 0.2))


def generate_random_packet(internal_ips, external_ips, suspicious_ips, common_ports):
    """
    Generate a random network packet.
    
    Args:
        internal_ips (list): List of internal IP addresses.
        external_ips (list): List of external IP addresses.
        suspicious_ips (list): List of suspicious IP addresses.
        common_ports (list): List of common port numbers.
        
    Returns:
        dict: Synthetic packet data.
    """
    # Determine if this should be a suspicious packet (5% chance)
    is_suspicious = random.random() < 0.05
    
    # Generate source and destination IPs
    if is_suspicious and random.random() < 0.7:
        # Suspicious packet - use suspicious IP as source or destination
        if random.random() < 0.5:
            src_ip = random.choice(suspicious_ips)
            dst_ip = random.choice(internal_ips)
        else:
            src_ip = random.choice(internal_ips)
            dst_ip = random.choice(suspicious_ips)
    else:
        # Normal packet
        if random.random() < 0.7:
            # Internal to external or vice versa
            if random.random() < 0.5:
                src_ip = random.choice(internal_ips)
                dst_ip = random.choice(external_ips)
            else:
                src_ip = random.choice(external_ips)
                dst_ip = random.choice(internal_ips)
        else:
            # Internal to internal
            src_ip = random.choice(internal_ips)
            dst_ip = random.choice(internal_ips)
    
    # Generate ports
    if is_suspicious and random.random() < 0.6:
        # Suspicious packet - use unusual high ports
        src_port = random.randint(10000, 65000)
        dst_port = random.randint(10000, 65000)
    else:
        # Normal packet - use common port for at least one end
        if random.random() < 0.7:
            # Common port as destination
            src_port = random.randint(1024, 65000)
            dst_port = random.choice(common_ports)
        else:
            # Common port as source
            src_port = random.choice(common_ports)
            dst_port = random.randint(1024, 65000)
    
    # Select protocol (TCP=6, UDP=17, ICMP=1)
    protocol = random.choice([6, 17, 1])
    
    # Generate packet size
    if is_suspicious and random.random() < 0.5:
        # Suspicious packet - unusual size
        size = random.randint(1, 100) if random.random() < 0.5 else random.randint(1500, 9000)
    else:
        # Normal packet - typical size range
        size = random.randint(64, 1500)
    
    # Generate payload entropy
    if is_suspicious and random.random() < 0.8:
        # Suspicious packet - unusual entropy (very high or very low)
        entropy = random.uniform(0.0, 0.2) if random.random() < 0.5 else random.uniform(0.9, 1.0)
    else:
        # Normal packet - typical entropy range
        entropy = random.uniform(0.4, 0.8)
    
    # Create packet data
    packet = {
        'timestamp': time.time(),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol': protocol,
        'size': size,
        'entropy': entropy,
        # Add additional fields that might be useful
        'ttl': random.randint(32, 128),
        'flags': random.randint(0, 0x3F) if protocol == 6 else 0,  # TCP flags
    }
    
    return packet


def extract_packet_features(packet):
    """
    Extract and normalize features from a packet for the model.
    
    Args:
        packet (dict): Packet data.
        
    Returns:
        numpy.ndarray: Feature vector.
    """
    # Extract the following features:
    # 1. Normalized source port
    # 2. Normalized destination port
    # 3. Normalized packet size
    # 4. Protocol (one-hot): TCP, UDP, ICMP (using indices 3, 4, 5)
    # 5. Flags (normalized)
    # 6. TTL (normalized)
    # 7. Payload entropy
    # 8. Is public IP source (boolean)
    # 9. Is public IP destination (boolean)
    
    # Create feature vector
    features = np.zeros(10)
    
    # 1. Normalized source port
    features[0] = normalize_port_number(packet['src_port'])
    
    # 2. Normalized destination port
    features[1] = normalize_port_number(packet['dst_port'])
    
    # 3. Normalized packet size
    features[2] = normalize_packet_size(packet['size'])
    
    # 4-6. Protocol (one-hot)
    protocol = packet['protocol']
    if protocol == 6:  # TCP
        features[3] = 1.0
    elif protocol == 17:  # UDP
        features[4] = 1.0
    elif protocol == 1:  # ICMP
        features[5] = 1.0
    
    # 7. Normalized flags
    features[6] = packet['flags'] / 63.0 if 'flags' in packet else 0.0
    
    # 8. Normalized TTL
    features[7] = packet['ttl'] / 255.0 if 'ttl' in packet else 0.5
    
    # 9. Payload entropy
    features[8] = packet['entropy'] if 'entropy' in packet else 0.5
    
    # 10. Is suspicious port combination (heuristic)
    if (packet['src_port'] > 50000 and packet['dst_port'] > 50000):
        features[9] = 1.0
    else:
        features[9] = 0.0
    
    return features


if __name__ == "__main__":
    main()
