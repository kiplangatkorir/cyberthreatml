# CyberThreat-ML Synthetic Attack Demo

This demo simulates a multi-stage cyber attack scenario in real-time and demonstrates how CyberThreat-ML detects different types of threats using both signature-based and zero-day detection capabilities.

## Overview 

The demo simulates a complete attack lifecycle including:

1. **Reconnaissance** (Port Scanning)
2. **Initial Access** (Brute Force)
3. **Command & Control** 
4. **Lateral Movement**
5. **Data Exfiltration**
6. **Zero-Day Exploit**

For each stage, the script generates synthetic network packets with characteristics typical of that attack type, mixed with normal traffic. The CyberThreat-ML library is used to detect threats through both signature-based detection (for known threats) and anomaly-based detection (for zero-day threats).

## Features

- Real-time traffic generation simulating different attack phases
- Visualization of traffic patterns and detection results
- Integration of signature-based and anomaly-based detection
- Detailed logging of detection events with severity levels and recommendations
- Comprehensive summary statistics at the end

## Usage

```bash
python synthetic_attack_demo.py [--duration 300] [--verbose]
```

Options:
- `--duration`: Duration of the simulation in seconds (default: 300)
- `--verbose`: Show detailed output including packet information

## Output

The demo produces several outputs:

1. **Real-time terminal output** with color-coded detection events
2. **Live visualization** showing traffic breakdown and detection timeline
3. **Log file** with all events and detections (`attack_demo_output/attack_simulation.log`)
4. **Summary statistics** in JSON format (`attack_demo_output/attack_summary.json`)
5. **Visualization graph** saved as image (`attack_demo_output/attack_simulation_plot.png`)

## Educational Value

This demo is designed to help users understand:

- How different attack types manifest in network traffic
- How signature-based detection identifies known threats
- How anomaly-based detection identifies unknown (zero-day) threats
- The importance of explainability in threat detection
- How to integrate multiple detection methods for comprehensive security

## Implementation Details

The demo uses:

- `AttackSimulator` class to generate realistic attack and normal traffic
- `FeatureExtractor` class to transform packets into feature vectors
- `CustomModel` class to simulate signature-based detection
- `ZeroDayDetector` from CyberThreat-ML for anomaly detection
- Real-time visualization of attack progression and detection events

## Example Output

### Terminal Output

```
[2025-03-11 08:15:23] [ATTACK PHASE CHANGE]
[2025-03-11 08:15:23] Now entering: Reconnaissance phase
[2025-03-11 08:15:23] Description: Attacker performs port scanning and network enumeration
[2025-03-11 08:15:23] Severity: Low

[2025-03-11 08:15:30] [THREAT DETECTED] Port Scan
[2025-03-11 08:15:30] Confidence: 0.8532

[2025-03-11 08:15:52] [ATTACK PHASE CHANGE]
[2025-03-11 08:15:52] Now entering: Initial Access phase
[2025-03-11 08:15:52] Description: Attacker attempts to gain initial access through credential brute forcing
[2025-03-11 08:15:52] Severity: Medium
```

### Detection Recommendations

```
[2025-03-11 08:17:45] [ANOMALY DETECTED]
[2025-03-11 08:17:45] Severity: High (0.7842)
[2025-03-11 08:17:45] Description: Anomalous Payload Size, Source Port, and TTL Value activity detected with High severity
[2025-03-11 08:17:45] Anomaly Score: 0.8723
[2025-03-11 08:17:45] Priority: High
[2025-03-11 08:17:45] Recommended actions:
  - Alert security team immediately
  - Isolate affected systems if disruption is minimal
  - Collect forensic data for investigation
  - Implement temporary security controls
```

## Notes

- This is an educational demonstration using synthetic data
- The attack patterns are simplified representations of real attacks
- The detection mechanisms demonstrate the core capabilities of CyberThreat-ML
- In a real-world setting, you would integrate with actual network traffic sources and security infrastructure

## Requirements

- Python 3.8 or higher
- CyberThreat-ML library
- NumPy, Matplotlib, and other dependencies
- Sufficient terminal size for visualization (recommend at least 120x40)
