#!/usr/bin/env python3
"""
Text-Based Visualization Demo for CyberThreat-ML

This script demonstrates the text-based visualization capabilities of CyberThreat-ML,
which allow for visualizing security data without requiring external visualization libraries.

Features:
- Visualizing threat detection results using text-based graphs and charts
- Creating security reports with formatted tables and visualizations
- Displaying real-time security monitoring data in the terminal
- Visualizing attack patterns and network connections

No external libraries required.
"""

import os
import sys
import random
import time
from datetime import datetime, timedelta

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from cyberthreat_ml.text_visualization import TextVisualizer, SecurityReportGenerator
except ImportError:
    print("Error: Could not import cyberthreat_ml.text_visualization module.")
    print("This script requires the CyberThreat-ML library to be installed.")
    sys.exit(1)


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def generate_random_threats(num_threats=10):
    """
    Generate random threat data for demonstration.
    
    Args:
        num_threats (int): Number of threats to generate
        
    Returns:
        list: List of threat dictionaries
    """
    threat_types = ["Brute Force", "Data Exfiltration", "Port Scan", "DDoS", "Command & Control"]
    internal_ips = [f"192.168.1.{random.randint(1, 254)}" for _ in range(5)]
    external_ips = [f"203.0.113.{random.randint(1, 254)}" for _ in range(5)]
    
    threats = []
    
    for _ in range(num_threats):
        threat_type = random.choice(threat_types)
        severity = random.randint(1, 5)
        
        # Different sources/targets based on threat type
        if threat_type == "Data Exfiltration":
            source = random.choice(internal_ips)
            target = random.choice(external_ips)
        elif threat_type == "Port Scan" or threat_type == "Brute Force":
            source = random.choice(external_ips)
            target = random.choice(internal_ips)
        else:
            source = random.choice(internal_ips + external_ips)
            target = random.choice(internal_ips + external_ips)
        
        # Random time in the last 24 hours
        hours_ago = random.uniform(0, 24)
        timestamp = datetime.now() - timedelta(hours=hours_ago)
        
        # Create threat dictionary
        threat = {
            'type': threat_type,
            'severity': severity,
            'source': source,
            'target': target,
            'timestamp': timestamp,
            'confidence': random.uniform(0.6, 0.95),
            'details': f"Suspicious {threat_type.lower()} activity detected"
        }
        
        # Add additional details based on threat type
        if threat_type == "Brute Force":
            threat['details'] = f"Multiple failed login attempts from {source}"
            threat['recommendation'] = "Implement account lockout policies and monitor for credential theft"
        elif threat_type == "Data Exfiltration":
            threat['details'] = f"Unusual data transfer of {random.randint(50, 2000)}MB to external IP"
            threat['recommendation'] = "Implement data loss prevention controls and review outbound traffic"
        elif threat_type == "Port Scan":
            ports = random.sample(range(1, 65535), random.randint(10, 100))
            threat['details'] = f"Scan of {len(ports)} ports detected from {source}"
            threat['recommendation'] = "Review firewall rules and implement network segmentation"
        
        # Add feature importance for some threats
        if random.random() > 0.7:
            features = [
                ("Packet Size", random.uniform(0.05, 0.2)),
                ("Connection Frequency", random.uniform(0.1, 0.3)),
                ("Destination Port Entropy", random.uniform(0.2, 0.4)),
                ("Protocol Anomaly Score", random.uniform(0.3, 0.5)),
                ("Traffic Volume", random.uniform(0.1, 0.4))
            ]
            threat['feature_importance'] = features
        
        threats.append(threat)
    
    return threats


def generate_network_connections(num_nodes=10, num_connections=20):
    """
    Generate random network connection data for visualization.
    
    Args:
        num_nodes (int): Number of network nodes
        num_connections (int): Number of connections between nodes
        
    Returns:
        tuple: (connections, node_labels)
    """
    # Generate node IPs and labels
    nodes = []
    node_labels = {}
    
    # Create some internal IPs
    for i in range(1, num_nodes // 2 + 1):
        ip = f"192.168.1.{i}"
        nodes.append(ip)
        if i == 1:
            node_labels[ip] = "Gateway"
        elif i == 2:
            node_labels[ip] = "File Server"
        elif i == 3:
            node_labels[ip] = "Database"
        else:
            node_labels[ip] = f"Internal {i}"
    
    # Create some external IPs
    for i in range(1, num_nodes // 2 + 1):
        ip = f"203.0.113.{i}"
        nodes.append(ip)
        node_labels[ip] = f"External {i}"
    
    # Generate random connections
    connections = []
    for _ in range(num_connections):
        source = random.choice(nodes)
        target = random.choice(nodes)
        
        # Avoid self-connections
        while target == source:
            target = random.choice(nodes)
        
        # Generate a weight (connection strength)
        weight = round(random.uniform(1.0, 10.0), 2)
        
        connections.append((source, target, weight))
    
    return connections, node_labels


def simulate_real_time_monitoring(visualizer, duration=10, update_interval=1):
    """
    Simulate real-time security monitoring with text visualization.
    
    Args:
        visualizer (TextVisualizer): Text visualizer instance
        duration (int): Duration of simulation in seconds
        update_interval (int): Update interval in seconds
    """
    threat_types = ["Brute Force", "Data Exfiltration", "Port Scan", "DDoS", "Command & Control"]
    threat_counts = {t_type: 0 for t_type in threat_types}
    threat_history = []
    
    visualizer.print_section("Real-Time Security Monitoring Simulation")
    print(f"Monitoring network traffic for {duration} seconds...")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    end_time = start_time + duration
    
    try:
        while time.time() < end_time:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Simulate detecting a threat (30% probability per cycle)
            if random.random() < 0.3:
                threat_type = random.choice(threat_types)
                threat_counts[threat_type] += 1
                
                source_ip = f"192.168.1.{random.randint(1, 254)}" if random.random() < 0.5 else f"203.0.113.{random.randint(1, 254)}"
                dest_ip = f"192.168.1.{random.randint(1, 254)}" if random.random() < 0.5 else f"203.0.113.{random.randint(1, 254)}"
                severity = random.randint(1, 5)
                confidence = round(random.uniform(0.6, 0.95), 2)
                
                # Create the threat message
                severity_indicator = "*" * severity
                print(f"[{current_time}] ⚠️ {threat_type} detected - Severity: {severity_indicator} - {source_ip} → {dest_ip} ({confidence:.2f} confidence)")
                
                # Add to history
                threat_history.append({
                    'type': threat_type,
                    'timestamp': datetime.now(),
                    'severity': severity,
                    'source': source_ip,
                    'target': dest_ip,
                    'confidence': confidence
                })
            else:
                # Normal traffic
                print(f"[{current_time}] Normal traffic flow...")
            
            # Update the threat count display every few iterations
            if len(threat_history) % 3 == 0 and threat_history:
                # Get data for histogram
                data = [count for _, count in threat_counts.items()]
                labels = list(threat_counts.keys())
                
                # Update the histogram
                print("\nCurrent Threat Distribution:")
                visualizer.histogram(data, labels, title="Threats by Type")
            
            # Pause before the next update
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    # Display final statistics
    elapsed = time.time() - start_time
    print(f"\nMonitoring session completed after {elapsed:.1f} seconds")
    print(f"Total threats detected: {sum(threat_counts.values())}")
    
    # Display threat distribution
    data = [count for _, count in threat_counts.items()]
    labels = list(threat_counts.keys())
    visualizer.histogram(data, labels, title="Final Threat Distribution")
    
    return threat_history


def visualize_attack_pattern(visualizer, pattern_name="APT Campaign"):
    """
    Visualize a multi-stage attack pattern using text visualization.
    
    Args:
        visualizer (TextVisualizer): Text visualizer instance
        pattern_name (str): Name of the attack pattern
    """
    # Define attack stages with timestamps
    attack_stages = [
        {
            'stage': "Reconnaissance",
            'timestamp': datetime.now() - timedelta(hours=18),
            'details': "Initial scanning of network perimeter",
            'type': "Recon"
        },
        {
            'stage': "Initial Access",
            'timestamp': datetime.now() - timedelta(hours=16),
            'details': "Exploitation of vulnerable web application",
            'type': "Access"
        },
        {
            'stage': "Persistence",
            'timestamp': datetime.now() - timedelta(hours=14),
            'details': "Installation of backdoor for maintaining access",
            'type': "Persistence"
        },
        {
            'stage': "Privilege Escalation",
            'timestamp': datetime.now() - timedelta(hours=12),
            'details': "Exploitation of local vulnerability to gain admin rights",
            'type': "PrivEsc"
        },
        {
            'stage': "Lateral Movement",
            'timestamp': datetime.now() - timedelta(hours=8),
            'details': "Movement to additional systems within the network",
            'type': "Lateral"
        },
        {
            'stage': "Data Collection",
            'timestamp': datetime.now() - timedelta(hours=6),
            'details': "Identification and gathering of target data",
            'type': "Collection"
        },
        {
            'stage': "Command & Control",
            'timestamp': datetime.now() - timedelta(hours=4),
            'details': "Communication with external control server",
            'type': "C2"
        },
        {
            'stage': "Data Exfiltration",
            'timestamp': datetime.now() - timedelta(hours=2),
            'details': "Extraction of sensitive data to external server",
            'type': "Exfil"
        }
    ]
    
    visualizer.print_section(f"Attack Pattern Visualization: {pattern_name}")
    print(f"This visualization shows the progression of a {pattern_name} attack over time.\n")
    
    # Extract data for timeline visualization
    events = [stage['details'] for stage in attack_stages]
    timestamps = [stage['timestamp'] for stage in attack_stages]
    event_types = [stage['stage'] for stage in attack_stages]
    
    # Display timeline
    visualizer.timeline(events, timestamps, event_types, title=f"{pattern_name} Attack Timeline")
    
    # Display attack stages with more details
    visualizer.print_subsection("Attack Stages")
    
    for i, stage in enumerate(attack_stages):
        severity = min(5, 2 + i // 2)  # Severity increases as attack progresses
        severity_indicator = "*" * severity
        
        print(f"Stage {i+1}: {stage['stage']}")
        print(f"Time: {stage['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Severity: {severity_indicator}")
        print(f"Details: {stage['details']}")
        print(f"Type: {stage['type']}")
        print()
    
    # Display attack recommendations
    visualizer.print_subsection("Mitigation Recommendations")
    
    print("1. Implement network segmentation to limit lateral movement")
    print("2. Deploy endpoint detection and response (EDR) solutions")
    print("3. Monitor for unusual authentication patterns")
    print("4. Implement data loss prevention controls")
    print("5. Utilize threat intelligence to identify indicators of compromise")
    print()


def visualize_threat_intelligence(visualizer, report_generator):
    """
    Visualize threat intelligence data using text visualizations.
    
    Args:
        visualizer (TextVisualizer): Text visualizer instance
        report_generator (SecurityReportGenerator): Report generator instance
    """
    visualizer.print_section("Threat Intelligence Visualization")
    
    # Generate synthetic threat intelligence data
    threat_sources = {
        "Russia": 35,
        "China": 28,
        "North Korea": 15,
        "Iran": 12,
        "Criminal Groups": 25,
        "Hacktivists": 8,
        "Insider Threats": 5,
        "Unknown": 10
    }
    
    attack_vectors = {
        "Phishing": 40,
        "Vulnerability Exploit": 28,
        "Credential Theft": 22,
        "Supply Chain": 15,
        "Zero-Day": 10,
        "Social Engineering": 18,
        "Malware": 32,
        "DDoS": 20
    }
    
    targeted_sectors = {
        "Financial": 30,
        "Government": 25,
        "Healthcare": 20,
        "Energy": 18,
        "Technology": 22,
        "Manufacturing": 15,
        "Education": 12,
        "Retail": 10
    }
    
    # Visualize threat sources with histogram
    visualizer.print_subsection("Threat Sources")
    source_data = list(threat_sources.values())
    source_labels = list(threat_sources.keys())
    visualizer.histogram(source_data, source_labels, title="Threat Activity by Source")
    
    # Visualize attack vectors with histogram
    visualizer.print_subsection("Attack Vectors")
    vector_data = list(attack_vectors.values())
    vector_labels = list(attack_vectors.keys())
    visualizer.histogram(vector_data, vector_labels, title="Common Attack Vectors")
    
    # Create a heatmap for targeted sectors by threat actor
    visualizer.print_subsection("Targeted Sectors by Threat Actor")
    
    # Generate synthetic heatmap data
    heatmap_data = []
    row_labels = ["Russia", "China", "North Korea", "Iran", "Criminal"]
    col_labels = ["Financial", "Government", "Healthcare", "Energy", "Technology"]
    
    for _ in range(len(row_labels)):
        row = [random.uniform(0.1, 0.9) for _ in range(len(col_labels))]
        heatmap_data.append(row)
    
    visualizer.heatmap(heatmap_data, row_labels, col_labels, title="Attack Focus by Threat Actor & Sector")
    
    # Create a threat intelligence summary report
    visualizer.print_subsection("Threat Intelligence Summary")
    
    # Generate synthetic threat intelligence findings
    findings = [
        {
            'title': "Increased Phishing Campaigns",
            'description': "30% increase in sophisticated phishing campaigns targeting financial institutions",
            'severity': 4,
            'confidence': 0.85,
            'recommendation': "Implement DMARC and employee security awareness training"
        },
        {
            'title': "New Ransomware Variant",
            'description': "New ransomware targeting healthcare sector with enhanced encryption capabilities",
            'severity': 5,
            'confidence': 0.9,
            'recommendation': "Update endpoint protection and implement application whitelisting"
        },
        {
            'title': "Zero-Day Vulnerability",
            'description': "Critical zero-day vulnerability being exploited in popular VPN solution",
            'severity': 5,
            'confidence': 0.8,
            'recommendation': "Implement workaround and monitor for patches from vendor"
        },
        {
            'title': "Supply Chain Compromise",
            'description': "Evidence of compromise in software distribution system affecting technology vendors",
            'severity': 4,
            'confidence': 0.75,
            'recommendation': "Verify software integrity and implement vendor risk management"
        },
        {
            'title': "DDoS Campaigns",
            'description': "Coordinated DDoS campaigns targeting energy sector infrastructure",
            'severity': 3,
            'confidence': 0.7,
            'recommendation': "Implement DDoS mitigation services and traffic filtering"
        }
    ]
    
    # Display the findings in a formatted table
    headers = ["Finding", "Severity", "Confidence", "Action"]
    rows = []
    
    for finding in findings:
        severity_str = "*" * finding['severity']
        confidence_str = f"{finding['confidence']:.2f}"
        rows.append([
            finding['title'],
            severity_str,
            confidence_str,
            "Immediate" if finding['severity'] >= 4 else "High Priority"
        ])
    
    # Print the table
    table = report_generator.format_table(headers, rows, "Key Threat Intelligence Findings")
    print(table)
    print()
    
    # Display comprehensive recommendations
    visualizer.print_subsection("Strategic Recommendations")
    
    print("Based on current threat intelligence, consider the following strategic improvements:")
    print("1. Enhance email security with advanced anti-phishing capabilities")
    print("2. Implement a zero-trust security model for critical systems")
    print("3. Improve vulnerability management with accelerated patching for critical systems")
    print("4. Expand threat hunting capabilities to detect sophisticated adversaries")
    print("5. Strengthen supply chain security through vendor assessment and code verification")
    print()


def main():
    """Main function for text visualization demonstration."""
    print_section("CyberThreat-ML Text Visualization Demo")
    print("This script demonstrates text-based visualization capabilities for security data")
    print("without requiring external visualization libraries.")
    
    # Create visualizer and report generator
    visualizer = TextVisualizer()
    report_generator = SecurityReportGenerator()
    
    # Generate random threats for visualization
    threats = generate_random_threats(num_threats=15)
    
    # Demonstrate threat dashboard
    visualizer.threat_dashboard(threats, title="Security Threat Dashboard")
    
    # Demonstrate network connection visualization
    connections, node_labels = generate_network_connections(num_nodes=8, num_connections=15)
    visualizer.network_graph(connections, node_labels, title="Network Connection Analysis")
    
    # Demonstrate attack pattern visualization
    visualize_attack_pattern(visualizer, pattern_name="Advanced Persistent Threat")
    
    # Demonstrate threat intelligence visualization
    visualize_threat_intelligence(visualizer, report_generator)
    
    # Demonstrate security report generation
    report = report_generator.generate_threat_report(
        threats=threats,
        detection_time=datetime.now() - timedelta(hours=2),
        model_name="CyberThreat-ML Advanced Detection Model"
    )
    
    visualizer.print_section("Security Report Generation")
    print("Generated a comprehensive security report. First section preview:")
    # Show just the first part of the report to avoid overwhelming the console
    report_lines = report.split('\n')
    print('\n'.join(report_lines[:25]))
    print("...")
    print(f"Full report contains {len(report_lines)} lines\n")
    
    # Demonstrate real-time monitoring visualization
    if input("Would you like to see the real-time monitoring demo? (y/n): ").lower().startswith('y'):
        threat_history = simulate_real_time_monitoring(visualizer, duration=20, update_interval=1)
        # You could save the threat history to a file here if desired
    
    print_section("Conclusion")
    print("The text visualization capabilities demonstrated in this script allow for:")
    print("  1. Creating informative security dashboards without graphical libraries")
    print("  2. Visualizing complex security patterns and relationships using text")
    print("  3. Generating comprehensive security reports with formatted data")
    print("  4. Monitoring security events in real-time in text-based environments")
    print()
    print("These capabilities make CyberThreat-ML suitable for environments where")
    print("graphical visualization libraries are not available or not preferred.")


if __name__ == "__main__":
    main()