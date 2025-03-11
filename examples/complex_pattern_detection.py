#!/usr/bin/env python3
"""
Complex Pattern Detection for CyberThreat-ML

This script demonstrates advanced pattern detection techniques for identifying
sophisticated cyber attacks that unfold over time, such as Advanced Persistent Threats (APTs)
and multi-stage attacks.

Features:
- Temporal pattern analysis to detect attack sequences
- Behavioral correlation to identify coordinated attacks
- Heuristic-based detection for complex attack patterns
- Text-based visualization of attack patterns

No external libraries required.
"""

import os
import sys
import random
import time
from datetime import datetime, timedelta

# Add parent directory to path to import library modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define attack types
ATTACK_TYPES = [
    'Normal',
    'Reconnaissance',
    'Initial Access',
    'Execution',
    'Persistence',
    'Privilege Escalation',
    'Defense Evasion',
    'Credential Access',
    'Discovery',
    'Lateral Movement',
    'Collection',
    'Command & Control',
    'Exfiltration',
    'Impact'
]

# Attack chain patterns (based on MITRE ATT&CK framework)
ATTACK_CHAINS = {
    'APT Campaign': [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
    'Ransomware': [1, 2, 3, 5, 8, 9, 10, 13],
    'Data Theft': [1, 2, 3, 7, 8, 10, 11, 12],
    'DDoS Campaign': [1, 2, 3, 6, 13],
    'Insider Threat': [8, 9, 10, 11, 12]
}

class Event:
    """Class representing a security event with timestamp and properties."""
    
    def __init__(self, timestamp, source_ip, dest_ip, event_type, severity=1, metadata=None):
        """
        Initialize a security event.
        
        Args:
            timestamp: Event timestamp
            source_ip: Source IP address
            dest_ip: Destination IP address
            event_type: Type of event (index in ATTACK_TYPES)
            severity: Event severity (1-5)
            metadata: Additional event metadata
        """
        self.timestamp = timestamp
        self.source_ip = source_ip
        self.dest_ip = dest_ip
        self.event_type = event_type
        self.severity = severity
        self.metadata = metadata or {}
    
    def __str__(self):
        """String representation of the event."""
        return (f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{ATTACK_TYPES[self.event_type]} (Severity: {self.severity}) "
                f"{self.source_ip} → {self.dest_ip}")


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def generate_ip_address(internal=True):
    """
    Generate a random IP address.
    
    Args:
        internal: Whether to generate an internal or external IP
        
    Returns:
        str: IP address
    """
    if internal:
        return f"192.168.{random.randint(1, 5)}.{random.randint(1, 254)}"
    else:
        return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def generate_synthetic_events(num_events=1000, num_normal=700, duration_days=7):
    """
    Generate synthetic security events including normal traffic and attack patterns.
    
    Args:
        num_events: Total number of events to generate
        num_normal: Number of normal events
        duration_days: Time span for events in days
        
    Returns:
        list: List of Event objects
    """
    print_section("Generating Synthetic Security Events")
    
    # Create timestamp range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=duration_days)
    time_range = (end_time - start_time).total_seconds()
    
    # Create IP addresses
    internal_ips = [generate_ip_address(internal=True) for _ in range(20)]
    external_ips = [generate_ip_address(internal=False) for _ in range(30)]
    
    # Create events
    events = []
    
    # Generate normal traffic events
    print(f"Generating {num_normal} normal traffic events...")
    for _ in range(num_normal):
        event_time = start_time + timedelta(seconds=random.random() * time_range)
        source_ip = random.choice(internal_ips)
        dest_ip = random.choice(internal_ips + external_ips)
        event = Event(
            timestamp=event_time,
            source_ip=source_ip,
            dest_ip=dest_ip,
            event_type=0,  # Normal
            severity=1,
            metadata={
                'protocol': random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS']),
                'port': random.choice([80, 443, 8080, 22, 23, 25, 53, 3389])
            }
        )
        events.append(event)
    
    # Generate attack pattern events
    num_attack_events = num_events - num_normal
    print(f"Generating {num_attack_events} attack events in patterns...")
    
    # Select attack patterns to include
    attack_patterns = random.sample(list(ATTACK_CHAINS.keys()), min(3, len(ATTACK_CHAINS)))
    
    for pattern_name in attack_patterns:
        attack_chain = ATTACK_CHAINS[pattern_name]
        print(f"  Creating {pattern_name} pattern")
        
        # Number of instances of this attack pattern
        num_instances = max(1, num_attack_events // (len(attack_patterns) * len(attack_chain)))
        
        for instance in range(num_instances):
            # Create a unique "attacker" for this attack chain
            attacker_ip = random.choice(external_ips)
            target_network = random.choice(internal_ips).split('.')
            target_network = f"{target_network[0]}.{target_network[1]}.{target_network[2]}"
            
            # Time for this attack chain (random window within overall time range)
            chain_start = start_time + timedelta(seconds=random.random() * time_range * 0.7)
            chain_duration = timedelta(hours=random.randint(2, 48))
            
            # Compromised hosts (added as the attack progresses)
            compromised_hosts = []
            
            # Generate events for each step in the attack chain
            for step_idx, step in enumerate(attack_chain):
                # How far into the chain time window should this event occur
                progress_ratio = step_idx / len(attack_chain)
                event_time = chain_start + timedelta(seconds=chain_duration.total_seconds() * progress_ratio)
                
                # For early stages, attacker connects from outside
                if step_idx < 2:
                    source_ip = attacker_ip
                    dest_ip = f"{target_network}.{random.randint(1, 254)}"
                    # Add to compromised hosts after initial access
                    if step == 2:  # Initial Access
                        compromised_hosts.append(dest_ip)
                # For later stages, use compromised hosts
                else:
                    # Make sure we have at least one compromised host
                    if not compromised_hosts:
                        # Create a compromised host if none exist yet
                        dest_ip = f"{target_network}.{random.randint(1, 254)}"
                        compromised_hosts.append(dest_ip)
                        source_ip = attacker_ip
                    else:
                        if random.random() < 0.3 and len(compromised_hosts) < 5:
                            # Add another compromised host via lateral movement
                            source_ip = random.choice(compromised_hosts)
                            dest_ip = f"{target_network}.{random.randint(1, 254)}"
                            while dest_ip in compromised_hosts:
                                dest_ip = f"{target_network}.{random.randint(1, 254)}"
                            compromised_hosts.append(dest_ip)
                        else:
                            source_ip = random.choice(compromised_hosts)
                            if random.random() < 0.5:
                                dest_ip = random.choice(compromised_hosts if len(compromised_hosts) > 1 else [f"{target_network}.{random.randint(1, 254)}"])
                            else:
                                dest_ip = attacker_ip
                
                # Create the event
                severity = min(5, 2 + step_idx // 3)  # Severity increases as attack progresses
                event = Event(
                    timestamp=event_time,
                    source_ip=source_ip,
                    dest_ip=dest_ip,
                    event_type=step,
                    severity=severity,
                    metadata={
                        'attack_pattern': pattern_name,
                        'attack_stage': step_idx + 1,
                        'protocol': random.choice(['TCP', 'HTTP', 'HTTPS', 'SMB']),
                        'port': random.choice([80, 443, 445, 3389, 22, 23])
                    }
                )
                events.append(event)
    
    # Sort events by timestamp
    events.sort(key=lambda e: e.timestamp)
    
    print(f"Generated {len(events)} total events")
    return events


class TemporalPatternDetector:
    """
    Detector for identifying temporal patterns in security events.
    
    This detector analyzes sequences of events over time to identify
    attack patterns that unfold in stages.
    """
    
    def __init__(self, time_window=24, min_pattern_length=3):
        """
        Initialize the temporal pattern detector.
        
        Args:
            time_window: Time window in hours to consider events related
            min_pattern_length: Minimum number of steps to consider a pattern
        """
        self.time_window = time_window
        self.min_pattern_length = min_pattern_length
        self.known_patterns = ATTACK_CHAINS
    
    def detect_patterns(self, events):
        """
        Detect temporal attack patterns in events.
        
        Args:
            events: List of security events
            
        Returns:
            list: Detected attack patterns
        """
        # Group events by source IP and destination network
        ip_groups = {}
        for event in events:
            if event.event_type == 0:  # Skip normal events
                continue
            
            # Group by source IP
            if event.source_ip not in ip_groups:
                ip_groups[event.source_ip] = []
            ip_groups[event.source_ip].append(event)
            
            # Also group by destination IP
            if event.dest_ip not in ip_groups:
                ip_groups[event.dest_ip] = []
            ip_groups[event.dest_ip].append(event)
        
        # Analyze each group for patterns
        detected_patterns = []
        
        for ip, ip_events in ip_groups.items():
            # Skip groups with too few events
            if len(ip_events) < self.min_pattern_length:
                continue
            
            # Sort events by timestamp
            ip_events.sort(key=lambda e: e.timestamp)
            
            # Look for event sequences that match known patterns
            for pattern_name, pattern in self.known_patterns.items():
                matches = self._match_pattern(ip_events, pattern)
                if matches:
                    detected_patterns.append({
                        'pattern_name': pattern_name,
                        'ip_address': ip,
                        'events': matches,
                        'confidence': self._calculate_confidence(matches, pattern),
                        'start_time': matches[0].timestamp,
                        'end_time': matches[-1].timestamp
                    })
        
        return detected_patterns
    
    def _match_pattern(self, events, pattern):
        """
        Check if events match a specific attack pattern.
        
        Args:
            events: List of events to check
            pattern: Attack pattern (list of event types)
            
        Returns:
            list: Matching events or empty list if no match
        """
        matches = []
        pattern_idx = 0
        
        # Safety check - if pattern is empty, return empty matches
        if not pattern:
            return matches
        
        # Allow for some missing steps and non-pattern events in between
        for event in events:
            # Safety check - ensure pattern_idx is in bounds
            if pattern_idx >= len(pattern):
                break
                
            # If this event matches the next step in pattern
            if event.event_type == pattern[pattern_idx]:
                matches.append(event)
                pattern_idx += 1
                
                # If we've matched the complete pattern
                if pattern_idx >= len(pattern):
                    return matches
            # If this event matches a later step in the pattern, skip ahead
            elif pattern_idx < len(pattern) and event.event_type in pattern[pattern_idx:]:
                try:
                    new_idx = pattern.index(event.event_type, pattern_idx)
                    # Only allow skipping if not too many steps
                    if new_idx - pattern_idx <= 2:
                        pattern_idx = new_idx
                        matches.append(event)
                        pattern_idx += 1
                except ValueError:
                    # Handle case where the event type isn't found in the pattern
                    continue
        
        # If we matched enough steps, consider it a partial match
        if len(matches) >= self.min_pattern_length:
            return matches
        
        return []
    
    def _calculate_confidence(self, matches, pattern):
        """
        Calculate confidence score for a pattern match.
        
        Args:
            matches: Matching events
            pattern: Expected pattern
            
        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence on:
        # 1. How many steps were matched vs. expected
        # 2. Time consistency of events
        # 3. IP consistency
        
        # Calculate completeness
        completeness = len(matches) / len(pattern)
        
        # Calculate time consistency
        time_diffs = []
        for i in range(1, len(matches)):
            time_diff = (matches[i].timestamp - matches[i-1].timestamp).total_seconds() / 3600  # hours
            time_diffs.append(time_diff)
        
        # Penalize if time differences are too large or too small
        time_penalties = sum(1 for diff in time_diffs if diff > self.time_window or diff < 0.01)
        time_consistency = max(0, 1 - (time_penalties / max(1, len(time_diffs))))
        
        # Check IP consistency
        ips = set()
        for event in matches:
            ips.add(event.source_ip)
            ips.add(event.dest_ip)
        
        # If too many IPs, lower confidence (unless it's lateral movement)
        ip_factor = min(1.0, 5 / len(ips))
        
        # Calculate final confidence
        confidence = 0.4 * completeness + 0.4 * time_consistency + 0.2 * ip_factor
        
        return min(1.0, confidence)


class BehavioralCorrelationDetector:
    """
    Detector for correlating behaviors across multiple events to identify attacks.
    
    This detector looks for related activities that together suggest an attack,
    even if individual events seem benign.
    """
    
    def __init__(self, max_time_window=48):
        """
        Initialize the behavioral correlation detector.
        
        Args:
            max_time_window: Maximum time window in hours to correlate events
        """
        self.max_time_window = max_time_window
    
    def detect_correlated_behaviors(self, events):
        """
        Detect correlated behaviors that indicate attacks.
        
        Args:
            events: List of security events
            
        Returns:
            list: Detected attack behaviors
        """
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Group events by source and destination IPs
        ip_connections = {}
        
        for event in sorted_events:
            # Create connection key (source_ip, dest_ip)
            connection = (event.source_ip, event.dest_ip)
            
            if connection not in ip_connections:
                ip_connections[connection] = []
            
            ip_connections[connection].append(event)
        
        # Identify suspicious connections
        suspicious_connections = []
        
        for connection, conn_events in ip_connections.items():
            # Skip connections with very few events
            if len(conn_events) < 3:
                continue
            
            # Check time span
            time_span = (conn_events[-1].timestamp - conn_events[0].timestamp).total_seconds() / 3600
            if time_span > self.max_time_window:
                continue
            
            # Calculate behavioral metrics
            event_types = [e.event_type for e in conn_events]
            unique_types = len(set(event_types))
            
            # Skip if only normal traffic
            if set(event_types) == {0}:
                continue
            
            # Analyze event progression
            progression_score = self._analyze_progression(conn_events)
            
            # Calculate suspicion score
            suspicion_score = self._calculate_suspicion(conn_events, unique_types, progression_score)
            
            if suspicion_score > 0.6:
                suspicious_connections.append({
                    'connection': connection,
                    'events': conn_events,
                    'suspicion_score': suspicion_score,
                    'behavior_type': self._classify_behavior(conn_events),
                    'start_time': conn_events[0].timestamp,
                    'end_time': conn_events[-1].timestamp
                })
        
        return suspicious_connections
    
    def _analyze_progression(self, events):
        """
        Analyze the progression of events to identify patterns.
        
        Args:
            events: List of events
            
        Returns:
            float: Progression score
        """
        # Check if events follow a logical progression
        progression_score = 0.0
        
        # Known logical progressions (simplified)
        logical_sequences = [
            [1, 2, 3],  # Reconnaissance → Initial Access → Execution
            [2, 3, 4],  # Initial Access → Execution → Persistence
            [3, 4, 5],  # Execution → Persistence → Privilege Escalation
            [5, 9, 10], # Privilege Escalation → Discovery → Lateral Movement
            [8, 10, 11], # Credential Access → Lateral Movement → Collection
            [11, 12, 13] # Collection → Command & Control → Exfiltration
        ]
        
        event_types = [e.event_type for e in events]
        
        # Check each logical sequence
        for sequence in logical_sequences:
            # Check if this sequence appears in events (in order, but not necessarily consecutive)
            seq_idx = 0
            for event_type in event_types:
                if seq_idx < len(sequence) and event_type == sequence[seq_idx]:
                    seq_idx += 1
            
            # If we found the complete sequence
            if seq_idx == len(sequence):
                progression_score += 0.3
        
        return min(1.0, progression_score)
    
    def _calculate_suspicion(self, events, unique_types, progression_score):
        """
        Calculate suspicion score for a set of events.
        
        Args:
            events: List of events
            unique_types: Number of unique event types
            progression_score: Score for logical progression
            
        Returns:
            float: Suspicion score
        """
        # Base score on:
        # 1. Diversity of event types
        # 2. Presence of high-severity events
        # 3. Logical progression
        # 4. Time pattern (e.g., regular intervals may be suspicious)
        
        # Diversity factor
        diversity = min(1.0, unique_types / 5)
        
        # Severity factor
        max_severity = max(e.severity for e in events)
        severity_factor = max_severity / 5
        
        # Time pattern
        time_diffs = []
        for i in range(1, len(events)):
            time_diff = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            time_diffs.append(time_diff)
        
        if len(time_diffs) > 2:
            # Calculate variance in time differences
            mean_diff = sum(time_diffs) / len(time_diffs)
            variance = sum((diff - mean_diff) ** 2 for diff in time_diffs) / len(time_diffs)
            
            # Very regular or very irregular patterns are suspicious
            normalized_variance = min(1.0, variance / (mean_diff ** 2))
            regularity_factor = 1.0 - normalized_variance
            if regularity_factor > 0.8 or regularity_factor < 0.2:
                regularity_factor = 0.8
            else:
                regularity_factor = 0.4
        else:
            regularity_factor = 0.5
        
        # Calculate final score
        suspicion_score = (0.3 * diversity + 
                          0.2 * severity_factor + 
                          0.3 * progression_score + 
                          0.2 * regularity_factor)
        
        return suspicion_score
    
    def _classify_behavior(self, events):
        """
        Classify the type of suspicious behavior.
        
        Args:
            events: List of events
            
        Returns:
            str: Behavior classification
        """
        event_types = [e.event_type for e in events]
        
        # Count occurrences of each event type
        type_counts = {}
        for event_type in event_types:
            if event_type not in type_counts:
                type_counts[event_type] = 0
            type_counts[event_type] += 1
        
        # Determine the most common non-normal event type
        most_common = None
        max_count = 0
        for event_type, count in type_counts.items():
            if event_type != 0 and count > max_count:
                most_common = event_type
                max_count = count
        
        # Classify based on dominant event types
        if 13 in type_counts:  # Exfiltration
            return "Data Theft"
        elif 12 in type_counts and 3 in type_counts:  # Command & Control + Execution
            return "Botnet Activity"
        elif 10 in type_counts and 5 in type_counts:  # Lateral Movement + Privilege Escalation
            return "Network Infiltration"
        elif 1 in type_counts and 10 not in type_counts:  # Reconnaissance without Lateral Movement
            return "Network Scanning"
        elif most_common is not None:
            return f"{ATTACK_TYPES[most_common]} Activity"
        else:
            return "Unknown Suspicious Activity"


class TextVisualizer:
    """
    Visualize security events and detected patterns using text-based graphics.
    
    This visualizer creates ASCII-based timeline views, connection maps, and
    pattern visualizations without requiring external libraries.
    """
    
    def __init__(self, terminal_width=80):
        """
        Initialize the text visualizer.
        
        Args:
            terminal_width: Width of terminal in characters
        """
        self.width = terminal_width
    
    def visualize_timeline(self, events, title="Event Timeline"):
        """
        Create a text-based timeline of events.
        
        Args:
            events: List of events to visualize
            title: Timeline title
        """
        print_section(title)
        
        if not events:
            print("No events to visualize")
            return
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Calculate time range
        start_time = sorted_events[0].timestamp
        end_time = sorted_events[-1].timestamp
        time_range = (end_time - start_time).total_seconds()
        
        # Define timeline width
        timeline_width = self.width - 30
        
        print(f"Timeline from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {time_range / 3600:.1f} hours")
        print()
        
        # Print the timeline header
        print(" " * 28 + "┌" + "─" * timeline_width + "┐")
        print(" " * 28 + "│" + " " * timeline_width + "│")
        
        # Print hour markers
        hours = int(time_range / 3600) + 1
        if hours <= timeline_width:
            hour_marker = " " * 28 + "│"
            hour_label = " " * 28 + " "
            markers_per_hour = timeline_width / hours
            
            for i in range(hours):
                pos = int(i * markers_per_hour)
                if pos < timeline_width:
                    hour_marker = hour_marker[:29+pos] + "│" + hour_marker[30+pos:]
                    hour_time = start_time + timedelta(hours=i)
                    time_str = hour_time.strftime("%H:%M")
                    if pos + len(time_str) <= timeline_width:
                        hour_label = hour_label[:29+pos] + time_str + hour_label[29+pos+len(time_str):]
            
            print(hour_marker + "│")
            print(hour_label + " ")
        
        print(" " * 28 + "│" + " " * timeline_width + "│")
        
        # Print each event on the timeline
        for event in sorted_events:
            event_time = (event.timestamp - start_time).total_seconds()
            position = int((event_time / time_range) * timeline_width)
            
            # Create the event line
            line = f"{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} │ "
            if position > 0:
                line += " " * (position - 1) + "●"
            else:
                line += "●"
            
            line_len = len(line)
            line += " " * (28 + timeline_width - line_len) + "│"
            
            # Choose symbol based on event type and severity
            symbol = "●"
            if event.event_type == 0:  # Normal
                symbol = "·"
            elif event.severity >= 4:  # High severity
                symbol = "!"
            
            # Print line with event type
            print(line)
            print(f"{ATTACK_TYPES[event.event_type]:<26} │" + " " * timeline_width + "│")
        
        # Print timeline footer
        print(" " * 28 + "│" + " " * timeline_width + "│")
        print(" " * 28 + "└" + "─" * timeline_width + "┘")
    
    def visualize_connections(self, events, title="Network Connections"):
        """
        Visualize connections between IP addresses.
        
        Args:
            events: List of events
            title: Visualization title
        """
        print_section(title)
        
        if not events:
            print("No events to visualize")
            return
        
        # Count connections between IPs
        connections = {}
        for event in events:
            connection = (event.source_ip, event.dest_ip)
            if connection not in connections:
                connections[connection] = {
                    'count': 0,
                    'event_types': set(),
                    'max_severity': 0
                }
            
            connections[connection]['count'] += 1
            connections[connection]['event_types'].add(event.event_type)
            connections[connection]['max_severity'] = max(connections[connection]['max_severity'], event.severity)
        
        # Get unique IPs
        all_ips = set()
        for src, dst in connections.keys():
            all_ips.add(src)
            all_ips.add(dst)
        
        print(f"Network with {len(all_ips)} hosts and {len(connections)} connections")
        print()
        
        # Sort connections by count
        sorted_connections = sorted(connections.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Display the top connections
        print("Top connections:")
        for (src, dst), data in sorted_connections[:10]:
            event_types_str = ', '.join(ATTACK_TYPES[t] for t in data['event_types'] if t > 0)
            if not event_types_str:
                event_types_str = "Normal traffic only"
            
            # Create connection visualization
            severity_indicator = "!" * data['max_severity']
            print(f"{src} → {dst} ({data['count']} events, severity {severity_indicator})")
            print(f"  Events: {event_types_str}")
        
        # Create a simple text-based network graph for highly connected nodes
        if len(all_ips) <= 20:
            self._create_text_network_graph(connections)
    
    def _create_text_network_graph(self, connections):
        """
        Create a simple text-based network graph.
        
        Args:
            connections: Dictionary of connections
        """
        print("\nNetwork Graph:")
        
        # Count connections per IP
        ip_connections = {}
        for (src, dst), data in connections.items():
            if src not in ip_connections:
                ip_connections[src] = {'in': 0, 'out': 0, 'total': 0}
            if dst not in ip_connections:
                ip_connections[dst] = {'in': 0, 'out': 0, 'total': 0}
            
            ip_connections[src]['out'] += 1
            ip_connections[src]['total'] += 1
            ip_connections[dst]['in'] += 1
            ip_connections[dst]['total'] += 1
        
        # Sort IPs by connection count
        sorted_ips = sorted(ip_connections.items(), key=lambda x: x[1]['total'], reverse=True)
        
        # Display the network graph
        for ip, conn_data in sorted_ips[:10]:
            in_conn = "←" * min(10, conn_data['in'])
            out_conn = "→" * min(10, conn_data['out'])
            print(f"{in_conn} {ip} {out_conn}")
    
    def visualize_patterns(self, patterns, title="Detected Attack Patterns"):
        """
        Visualize detected attack patterns.
        
        Args:
            patterns: List of detected patterns
            title: Visualization title
        """
        print_section(title)
        
        if not patterns:
            print("No patterns to visualize")
            return
        
        # Sort patterns by confidence
        sorted_patterns = sorted(patterns, key=lambda p: p['confidence'], reverse=True)
        
        for i, pattern in enumerate(sorted_patterns):
            print(f"Pattern #{i+1}: {pattern['pattern_name']}")
            print(f"Confidence: {pattern['confidence']:.2f}")
            print(f"Time range: {pattern['start_time'].strftime('%Y-%m-%d %H:%M:%S')} to {pattern['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {(pattern['end_time'] - pattern['start_time']).total_seconds() / 3600:.1f} hours")
            print(f"Related IP: {pattern['ip_address']}")
            print()
            
            # Visualize the attack stages
            print("Attack stages:")
            for event in pattern['events']:
                time_str = event.timestamp.strftime('%H:%M:%S')
                severity_str = "*" * event.severity
                print(f"  {time_str} | {ATTACK_TYPES[event.event_type]:<20} | {severity_str}")
            
            print()
            
            # Create a pattern visualization
            self._create_pattern_visualization(pattern)
            
            print("\n" + "-" * 80 + "\n")
    
    def _create_pattern_visualization(self, pattern):
        """
        Create a visual representation of an attack pattern.
        
        Args:
            pattern: Attack pattern data
        """
        print("Pattern visualization:")
        
        # Create a timeline representation
        timeline_width = self.width - 20
        start_time = pattern['start_time']
        end_time = pattern['end_time']
        time_range = (end_time - start_time).total_seconds()
        
        # Print timeline header
        print("┌" + "─" * timeline_width + "┐")
        
        # Create event markers
        timeline = [" " for _ in range(timeline_width)]
        for event in pattern['events']:
            event_time = (event.timestamp - start_time).total_seconds()
            position = int((event_time / time_range) * (timeline_width - 1))
            position = min(timeline_width - 1, max(0, position))
            
            # Mark the position with a character based on event type
            if event.event_type == 1:  # Reconnaissance
                marker = "R"
            elif event.event_type == 2:  # Initial Access
                marker = "A"
            elif event.event_type == 3:  # Execution
                marker = "E"
            elif event.event_type == 4:  # Persistence
                marker = "P"
            elif event.event_type == 5:  # Privilege Escalation
                marker = "S"
            elif event.event_type == 10:  # Lateral Movement
                marker = "L"
            elif event.event_type == 12:  # Command & Control
                marker = "C"
            elif event.event_type == 13:  # Exfiltration
                marker = "X"
            else:
                marker = str(event.event_type % 10)
            
            timeline[position] = marker
        
        # Print the timeline
        print("│" + "".join(timeline) + "│")
        print("└" + "─" * timeline_width + "┘")
        
        # Print the legend
        print("\nLegend:")
        print("  R: Reconnaissance   A: Initial Access   E: Execution")
        print("  P: Persistence      S: Privilege Esc.   L: Lateral Movement")
        print("  C: Command & Control  X: Exfiltration")


def main():
    """Main function for complex pattern detection demo."""
    print_section("CyberThreat-ML Complex Pattern Detection Demo")
    print("This script demonstrates advanced pattern detection techniques for identifying")
    print("sophisticated cyber attacks that unfold over time, such as Advanced Persistent")
    print("Threats (APTs) and multi-stage attacks.")
    
    # Generate synthetic security events
    events = generate_synthetic_events(num_events=200, num_normal=140, duration_days=3)
    
    # Initialize the temporal pattern detector
    pattern_detector = TemporalPatternDetector(time_window=24, min_pattern_length=3)
    
    # Detect temporal patterns
    print_section("Temporal Pattern Detection")
    detected_patterns = pattern_detector.detect_patterns(events)
    
    print(f"Detected {len(detected_patterns)} temporal attack patterns")
    for i, pattern in enumerate(detected_patterns):
        print(f"\nPattern #{i+1}: {pattern['pattern_name']}")
        print(f"Confidence: {pattern['confidence']:.2f}")
        print(f"Time range: {pattern['start_time'].strftime('%Y-%m-%d %H:%M:%S')} to {pattern['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Related IP: {pattern['ip_address']}")
        print(f"Stages: {len(pattern['events'])}")
    
    # Initialize the behavioral correlation detector
    behavior_detector = BehavioralCorrelationDetector(max_time_window=48)
    
    # Detect correlated behaviors
    print_section("Behavioral Correlation Detection")
    suspicious_behaviors = behavior_detector.detect_correlated_behaviors(events)
    
    print(f"Detected {len(suspicious_behaviors)} suspicious behavior patterns")
    for i, behavior in enumerate(suspicious_behaviors):
        src_ip, dst_ip = behavior['connection']
        print(f"\nSuspicious Behavior #{i+1}: {behavior['behavior_type']}")
        print(f"Suspicion score: {behavior['suspicion_score']:.2f}")
        print(f"Connection: {src_ip} → {dst_ip}")
        print(f"Time range: {behavior['start_time'].strftime('%Y-%m-%d %H:%M:%S')} to {behavior['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Event count: {len(behavior['events'])}")
    
    # Initialize the text visualizer
    visualizer = TextVisualizer()
    
    # Visualize a subset of events
    visualizer.visualize_timeline(events[:50], title="Event Timeline (First 50 Events)")
    
    # Visualize network connections
    visualizer.visualize_connections(events, title="Network Connection Analysis")
    
    # Visualize detected patterns
    visualizer.visualize_patterns(detected_patterns, title="Detected Attack Patterns")
    
    print_section("Conclusion")
    print("The complex pattern detection capabilities demonstrated in this script enhance")
    print("CyberThreat-ML's ability to detect sophisticated attacks by looking at:")
    print("  1. Temporal sequences of events that match known attack patterns")
    print("  2. Behavioral correlations that indicate suspicious activity")
    print("  3. Network connection patterns that reveal attacker infrastructure")
    print()
    print("These capabilities are particularly effective for identifying Advanced Persistent")
    print("Threats (APTs) and multi-stage attacks that evolve over time.")


if __name__ == "__main__":
    main()