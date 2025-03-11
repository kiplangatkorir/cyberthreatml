"""
Module for text-based visualization of cybersecurity data.

This module provides classes and functions for creating text-based
visualizations of security data without requiring external libraries.
These visualizations can be used in environments where graphical
libraries are not available or when command-line output is preferred.
"""

import os
import math
import time
from datetime import datetime, timedelta

class TextVisualizer:
    """
    Class for creating text-based visualizations of security data.
    
    This class provides methods for creating ASCII/Unicode-based visualizations
    such as histograms, timelines, and heatmaps for security data.
    """
    
    def __init__(self, width=80, height=24):
        """
        Initialize the text visualizer.
        
        Args:
            width (int): Width of the visualization in characters.
            height (int): Height of the visualization in characters.
        """
        self.width = width
        self.height = height
        
        # Try to get terminal size
        try:
            terminal_size = os.get_terminal_size()
            self.width = terminal_size.columns
            self.height = terminal_size.lines - 5  # Leave some space for prompts
        except (OSError, AttributeError):
            # If we can't get terminal size, use the defaults
            pass
    
    def print_section(self, title):
        """
        Print a formatted section title.
        
        Args:
            title (str): The title of the section.
        """
        print("\n" + "=" * self.width)
        print(f" {title} ".center(self.width, "="))
        print("=" * self.width + "\n")
    
    def print_subsection(self, title):
        """
        Print a formatted subsection title.
        
        Args:
            title (str): The title of the subsection.
        """
        print("\n" + "-" * self.width)
        print(f" {title} ".center(self.width, "-"))
        print("-" * self.width + "\n")
    
    def histogram(self, data, labels=None, title="Histogram", max_width=None):
        """
        Display a text-based histogram.
        
        Args:
            data (list): List of numeric values to display.
            labels (list, optional): Labels for each bar.
            title (str, optional): Title of the histogram.
            max_width (int, optional): Maximum width of the histogram.
        """
        if not data:
            print("No data to visualize")
            return
        
        if max_width is None:
            max_width = self.width - 20  # Leave space for labels
        
        max_value = max(data)
        
        # If all values are the same, adjust to avoid division by zero
        if max_value == 0:
            max_value = 1
        
        # If no labels are provided, create generic ones
        if labels is None:
            labels = [f"Item {i+1}" for i in range(len(data))]
        
        # Calculate the maximum label length
        max_label_len = max(len(str(label)) for label in labels)
        
        # Print the histogram
        print(f"{title}:")
        for i, value in enumerate(data):
            # Calculate bar length
            bar_length = int((value / max_value) * max_width)
            if value > 0 and bar_length == 0:
                bar_length = 1
            
            # Format the bar
            bar = "█" * bar_length
            
            # Format the label
            label = str(labels[i]).ljust(max_label_len)
            
            # Print the bar
            print(f"{label} | {bar} {value}")
        
        print()
    
    def timeline(self, events, timestamps=None, event_types=None, title="Event Timeline"):
        """
        Display a text-based timeline of events.
        
        Args:
            events (list): List of event objects or descriptions.
            timestamps (list, optional): List of timestamps for each event. If None, uses current timestamps.
            event_types (list, optional): List of event types or categories.
            title (str, optional): Title of the timeline.
        """
        if not events:
            print("No events to visualize")
            return
        
        # Create timestamps if not provided
        if timestamps is None:
            now = datetime.now()
            timestamps = [now - timedelta(hours=i) for i in range(len(events)-1, -1, -1)]
        
        # Combine events, timestamps, and types
        timeline_data = []
        for i, event in enumerate(events):
            event_type = event_types[i] if event_types and i < len(event_types) else "Event"
            timestamp = timestamps[i]
            timeline_data.append((timestamp, event, event_type))
        
        # Sort by timestamp
        timeline_data.sort(key=lambda x: x[0])
        
        # Find the time range
        start_time = timeline_data[0][0]
        end_time = timeline_data[-1][0]
        time_range = (end_time - start_time).total_seconds()
        
        # If all events have the same timestamp, add a small range
        if time_range == 0:
            time_range = 3600  # 1 hour
            end_time = start_time + timedelta(seconds=time_range)
        
        # Calculate the timeline width
        timeline_width = self.width - 40  # Leave space for text
        
        # Print the timeline
        self.print_subsection(title)
        print(f"Timeline from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {time_range / 3600:.1f} hours\n")
        
        # Print the timeline header
        print(" " * 28 + "┌" + "─" * timeline_width + "┐")
        
        # Print each event on the timeline
        for timestamp, event, event_type in timeline_data:
            event_time = (timestamp - start_time).total_seconds()
            position = int((event_time / time_range) * timeline_width)
            
            # Create the event line
            line = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} │ "
            if position > 0:
                line += " " * (position - 1) + "●"
            else:
                line += "●"
            
            line_len = len(line)
            line += " " * (28 + timeline_width - line_len) + "│"
            
            # Print line with event text
            print(line)
            event_text = event_type + ": " + str(event)
            if len(event_text) > timeline_width + 20:
                event_text = event_text[:timeline_width + 17] + "..."
            print(f"{event_text:<{28 + timeline_width}}│")
            print(" " * 28 + "│" + " " * timeline_width + "│")
        
        # Print timeline footer
        print(" " * 28 + "└" + "─" * timeline_width + "┘")
    
    def heatmap(self, data, row_labels=None, col_labels=None, title="Heatmap"):
        """
        Display a text-based heatmap using ASCII block characters.
        
        Args:
            data (list): 2D list of numeric values.
            row_labels (list, optional): Labels for rows.
            col_labels (list, optional): Labels for columns.
            title (str, optional): Title of the heatmap.
        """
        if not data or not data[0]:
            print("No data to visualize")
            return
        
        # Generate default labels if not provided
        if row_labels is None:
            row_labels = [f"Row {i+1}" for i in range(len(data))]
        
        if col_labels is None:
            col_labels = [f"Col {i+1}" for i in range(len(data[0]))]
        
        # Find the maximum value for normalization
        max_value = max(max(row) for row in data)
        if max_value == 0:
            max_value = 1
        
        # Define heat characters (from low to high intensity)
        heat_chars = " ░▒▓█"
        
        # Calculate the maximum label length
        max_row_label_len = max(len(str(label)) for label in row_labels)
        
        # Print the heatmap title
        self.print_subsection(title)
        
        # Print column headers
        print(" " * max_row_label_len + " ", end="")
        for col_label in col_labels:
            print(f" {str(col_label)[:4]:4}", end="")
        print()
        
        # Print the heatmap
        for i, row in enumerate(data):
            # Print row label
            print(f"{str(row_labels[i])[:max_row_label_len]:{max_row_label_len}} ", end="")
            
            # Print each cell
            for value in row:
                # Normalize value and select character
                norm_value = value / max_value
                char_idx = min(len(heat_chars) - 1, int(norm_value * len(heat_chars)))
                char = heat_chars[char_idx]
                
                # Print cell (double width for better visibility)
                print(f" {char*2}  ", end="")
            
            print()
        
        # Print legend
        print("\nLegend:")
        for i, char in enumerate(heat_chars):
            min_val = i / len(heat_chars) * max_value
            max_val = (i + 1) / len(heat_chars) * max_value
            print(f"{char*2} = {min_val:.2f} to {max_val:.2f}")
        
        print()
    
    def network_graph(self, connections, node_labels=None, title="Network Connections"):
        """
        Display a simple text-based representation of a network graph.
        
        Args:
            connections (list): List of tuples (source, destination, weight).
            node_labels (dict, optional): Dictionary mapping node IDs to labels.
            title (str, optional): Title of the graph.
        """
        if not connections:
            print("No connections to visualize")
            return
        
        # Extract all unique nodes
        nodes = set()
        for src, dst, _ in connections:
            nodes.add(src)
            nodes.add(dst)
        
        # Create node labels if not provided
        if node_labels is None:
            node_labels = {node: f"Node {i+1}" for i, node in enumerate(nodes)}
        
        # Count connections per node
        node_connections = {node: {'in': 0, 'out': 0, 'total': 0} for node in nodes}
        for src, dst, weight in connections:
            node_connections[src]['out'] += 1
            node_connections[src]['total'] += 1
            node_connections[dst]['in'] += 1
            node_connections[dst]['total'] += 1
        
        # Sort nodes by connection count
        sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1]['total'], reverse=True)
        
        # Calculate the maximum node label length
        max_label_len = max(len(node_labels[node]) for node in nodes)
        
        # Print the graph title
        self.print_subsection(title)
        print(f"Network with {len(nodes)} nodes and {len(connections)} connections\n")
        
        # Print node connection summary
        print("Node Connections:")
        for node, conn_data in sorted_nodes[:10]:  # Show top 10 nodes
            label = node_labels[node]
            in_conn = conn_data['in']
            out_conn = conn_data['out']
            total_conn = conn_data['total']
            
            # Create visual indicators for incoming and outgoing connections
            in_ind = "←" * min(10, in_conn)
            out_ind = "→" * min(10, out_conn)
            
            print(f"{in_ind:10} {label:{max_label_len}} {out_ind:10} ({in_conn} in, {out_conn} out)")
        
        # Print top connections
        print("\nTop Connections:")
        sorted_connections = sorted(connections, key=lambda x: x[2], reverse=True)
        for src, dst, weight in sorted_connections[:10]:  # Show top 10 connections
            src_label = node_labels[src]
            dst_label = node_labels[dst]
            print(f"{src_label} → {dst_label} (weight: {weight:.2f})")
        
        print()
    
    def confusion_matrix(self, matrix, class_names=None, title="Confusion Matrix"):
        """
        Display a text-based confusion matrix.
        
        Args:
            matrix (list): 2D list of confusion matrix values.
            class_names (list, optional): Names of classes.
            title (str, optional): Title of the matrix.
        """
        if not matrix:
            print("No data to visualize")
            return
        
        # If class names are not provided, generate generic ones
        if class_names is None:
            class_names = [f"Class {i+1}" for i in range(len(matrix))]
        
        # Calculate the maximum class name length
        max_name_len = max(len(name) for name in class_names)
        
        # Calculate the maximum value in the matrix for formatting
        max_value = max(max(row) for row in matrix)
        value_width = max(len(str(int(max_value))), 4)
        
        # Print matrix title
        self.print_subsection(title)
        
        # Print header row with predicted class names
        print(f"{'':{max_name_len}} | ", end="")
        for name in class_names:
            print(f"{name[:value_width]:{value_width}} ", end="")
        print("| Recall")
        
        # Print separator
        print("-" * max_name_len + "-+-" + "-" * (value_width + 1) * len(class_names) + "-+-------")
        
        # Print each row with actual class names
        for i, row in enumerate(matrix):
            # Print row label (actual class)
            print(f"{class_names[i]:{max_name_len}} | ", end="")
            
            # Print confusion values
            for value in row:
                print(f"{int(value):{value_width}} ", end="")
            
            # Calculate and print recall
            row_sum = sum(row)
            recall = row[i] / row_sum if row_sum > 0 else 0
            print(f"| {recall:.2f}")
        
        # Print separator
        print("-" * max_name_len + "-+-" + "-" * (value_width + 1) * len(class_names) + "-+-------")
        
        # Print precision for each column
        print(f"{'Precision':{max_name_len}} | ", end="")
        for j in range(len(matrix[0])):
            col_sum = sum(matrix[i][j] for i in range(len(matrix)))
            precision = matrix[j][j] / col_sum if col_sum > 0 else 0
            print(f"{precision:.2f} ", end="")
        
        # Calculate overall accuracy
        total = sum(sum(row) for row in matrix)
        correct = sum(matrix[i][i] for i in range(len(matrix)))
        accuracy = correct / total if total > 0 else 0
        
        print(f"| {accuracy:.2f}")
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print()
    
    def progress_bar(self, current, total, title="Progress", width=None, auto_update=False):
        """
        Display a text-based progress bar.
        
        Args:
            current (int): Current progress value.
            total (int): Total value representing 100% progress.
            title (str, optional): Title of the progress bar.
            width (int, optional): Width of the progress bar in characters.
            auto_update (bool, optional): Whether to automatically update in place.
        
        Returns:
            function: Update function for updating the progress bar.
        """
        if width is None:
            width = min(self.width - 30, 50)
        
        def update(current_value, message=None):
            """Update the progress bar with a new value."""
            nonlocal current
            current = current_value
            
            # Calculate percentage and bar length
            percent = min(100, max(0, current * 100 / total))
            bar_length = int(width * percent / 100)
            
            # Format the progress bar
            bar = f"[{'=' * bar_length}{' ' * (width - bar_length)}] {percent:6.2f}%"
            
            # Format the message
            if message:
                message_str = f" {message}"
            else:
                message_str = ""
            
            # Print the progress bar
            if auto_update:
                print(f"\r{title}: {bar}{message_str}", end="", flush=True)
            else:
                print(f"{title}: {bar}{message_str}")
        
        # Initial update
        update(current)
        
        return update
    
    def animate_processing(self, duration=5, message="Processing", fps=10):
        """
        Display an animated processing indicator for a specified duration.
        
        Args:
            duration (int, optional): Duration in seconds.
            message (str, optional): Message to display.
            fps (int, optional): Frames per second for the animation.
        """
        # Animation frames
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Calculate total frames
        total_frames = duration * fps
        
        # Run the animation
        try:
            for i in range(total_frames):
                frame = frames[i % len(frames)]
                print(f"\r{message} {frame}", end="", flush=True)
                time.sleep(1 / fps)
            
            # Clear the line after animation
            print("\r" + " " * (len(message) + 2), end="", flush=True)
            print("\r", end="", flush=True)
        except KeyboardInterrupt:
            # Handle keyboard interrupt gracefully
            print("\r" + " " * (len(message) + 2), end="", flush=True)
            print("\r", end="", flush=True)
    
    def threat_dashboard(self, threats, title="Security Threat Dashboard"):
        """
        Display a simple text-based security threat dashboard.
        
        Args:
            threats (list): List of threat dictionaries with keys:
                - 'type': Type of threat
                - 'severity': Severity level (1-5)
                - 'source': Source of the threat
                - 'target': Target of the threat
                - 'timestamp': Timestamp of the detection
            title (str, optional): Title of the dashboard.
        """
        if not threats:
            print("No threats to display")
            return
        
        # Sort threats by severity (highest first)
        sorted_threats = sorted(threats, key=lambda x: x.get('severity', 0), reverse=True)
        
        # Count threats by type and severity
        threat_types = {}
        severity_counts = [0, 0, 0, 0, 0]  # For severities 1-5
        
        for threat in threats:
            threat_type = threat.get('type', 'Unknown')
            severity = min(5, max(1, threat.get('severity', 1)))
            
            if threat_type not in threat_types:
                threat_types[threat_type] = 0
            
            threat_types[threat_type] += 1
            severity_counts[severity - 1] += 1
        
        # Print dashboard title
        self.print_section(title)
        
        # Print summary statistics
        print(f"Total Threats: {len(threats)}")
        print(f"Critical Threats: {severity_counts[4]} (Severity 5)")
        print(f"High Severity Threats: {severity_counts[3]} (Severity 4)")
        print(f"Medium Severity Threats: {severity_counts[2]} (Severity 3)")
        print(f"Low Severity Threats: {severity_counts[1]} (Severity 2)")
        print(f"Informational: {severity_counts[0]} (Severity 1)")
        print()
        
        # Display threat type distribution
        self.print_subsection("Threat Types")
        
        # Sort threat types by count
        sorted_types = sorted(threat_types.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate maximum threat type length for formatting
        max_type_len = max(len(t_type) for t_type in threat_types.keys())
        
        # Create data for histogram
        histogram_data = [count for _, count in sorted_types]
        histogram_labels = [t_type for t_type, _ in sorted_types]
        
        # Display histogram
        self.histogram(histogram_data, histogram_labels, title="Threat Type Distribution")
        
        # Display most recent threats
        self.print_subsection("Most Recent Threats")
        
        # Sort by timestamp for recent threats
        recent_threats = sorted(threats, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        # Display the 10 most recent threats
        for i, threat in enumerate(recent_threats[:10]):
            threat_type = threat.get('type', 'Unknown')
            severity = threat.get('severity', 1)
            source = threat.get('source', 'Unknown')
            target = threat.get('target', 'Unknown')
            timestamp = threat.get('timestamp', datetime.now())
            
            # Format severity indicator
            severity_indicator = '!' * severity
            
            print(f"{i+1}. [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] " +
                  f"{threat_type} (Severity: {severity_indicator})")
            print(f"   {source} → {target}")
            
            # Add threat details if available
            if 'details' in threat:
                print(f"   Details: {threat['details']}")
            
            print()
        
        # Display critical threats in more detail
        critical_threats = [threat for threat in threats if threat.get('severity', 0) >= 4]
        
        if critical_threats:
            self.print_subsection("Critical Threats")
            
            for i, threat in enumerate(critical_threats[:5]):  # Show top 5 critical threats
                threat_type = threat.get('type', 'Unknown')
                severity = threat.get('severity', 4)
                source = threat.get('source', 'Unknown')
                target = threat.get('target', 'Unknown')
                timestamp = threat.get('timestamp', datetime.now())
                
                print(f"CRITICAL THREAT #{i+1}: {threat_type} (Severity: {'!' * severity})")
                print(f"Detected at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Source: {source}")
                print(f"Target: {target}")
                
                # Add threat details if available
                if 'details' in threat:
                    print(f"Details: {threat['details']}")
                
                # Add mitigation recommendation if available
                if 'recommendation' in threat:
                    print(f"Recommendation: {threat['recommendation']}")
                else:
                    print("Recommendation: Investigate immediately and isolate affected systems")
                
                print()


class SecurityReportGenerator:
    """
    Class for generating text-based security reports.
    
    This class provides methods for creating formatted security reports
    based on detection and analysis results.
    """
    
    def __init__(self, width=80):
        """
        Initialize the report generator.
        
        Args:
            width (int): Width of the report in characters.
        """
        self.width = width
        self.visualizer = TextVisualizer(width=width)
    
    def create_section_header(self, title):
        """
        Create a formatted section header.
        
        Args:
            title (str): Section title.
            
        Returns:
            str: Formatted section header.
        """
        return "\n" + "=" * self.width + "\n" + f" {title} ".center(self.width, "=") + "\n" + "=" * self.width + "\n"
    
    def create_subsection_header(self, title):
        """
        Create a formatted subsection header.
        
        Args:
            title (str): Subsection title.
            
        Returns:
            str: Formatted subsection header.
        """
        return "\n" + "-" * self.width + "\n" + f" {title} ".center(self.width, "-") + "\n" + "-" * self.width + "\n"
    
    def format_table(self, headers, rows, title=None):
        """
        Create a formatted text table.
        
        Args:
            headers (list): List of column headers.
            rows (list): List of rows, each a list of values.
            title (str, optional): Table title.
            
        Returns:
            str: Formatted table as a string.
        """
        if not headers or not rows:
            return "No data to display"
        
        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Create the table
        result = []
        
        # Add title if provided
        if title:
            result.append(title)
            result.append("-" * self.width)
        
        # Create header row
        header_row = " | ".join(f"{str(h):{w}}" for h, w in zip(headers, col_widths))
        result.append(header_row)
        
        # Create separator
        separator = "-+-".join("-" * w for w in col_widths)
        result.append(separator)
        
        # Create data rows
        for row in rows:
            data_row = " | ".join(f"{str(cell):{w}}" for cell, w in zip(row, col_widths))
            result.append(data_row)
        
        return "\n".join(result)
    
    def generate_threat_report(self, threats, detection_time, model_name=None):
        """
        Generate a comprehensive threat detection report.
        
        Args:
            threats (list): List of detected threats.
            detection_time (datetime): Time when detection was performed.
            model_name (str, optional): Name of the detection model.
            
        Returns:
            str: Formatted report as a string.
        """
        report = []
        
        # Add report header
        report.append(self.create_section_header("CYBERSECURITY THREAT DETECTION REPORT"))
        
        # Add report metadata
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Detection Time: {detection_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if model_name:
            report.append(f"Detection Model: {model_name}")
        report.append(f"Total Threats Detected: {len(threats)}")
        report.append("")
        
        # Add threat summary
        if threats:
            # Count threats by type and severity
            threat_types = {}
            severity_levels = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            for threat in threats:
                threat_type = threat.get('type', 'Unknown')
                severity = min(5, max(1, threat.get('severity', 1)))
                
                if threat_type not in threat_types:
                    threat_types[threat_type] = 0
                
                threat_types[threat_type] += 1
                severity_levels[severity] += 1
            
            # Add threat distribution
            report.append(self.create_subsection_header("THREAT DISTRIBUTION"))
            
            # Add threat types table
            threat_type_rows = [(threat_type, count) for threat_type, count in threat_types.items()]
            threat_type_table = self.format_table(
                ["Threat Type", "Count"],
                threat_type_rows,
                "Threats by Type"
            )
            report.append(threat_type_table)
            report.append("")
            
            # Add severity levels table
            severity_rows = [(f"Severity {level}", count) for level, count in severity_levels.items()]
            severity_table = self.format_table(
                ["Severity Level", "Count"],
                severity_rows,
                "Threats by Severity"
            )
            report.append(severity_table)
            report.append("")
            
            # Add detailed threat information
            report.append(self.create_subsection_header("DETAILED THREAT INFORMATION"))
            
            # Sort threats by severity (highest first)
            sorted_threats = sorted(threats, key=lambda x: x.get('severity', 0), reverse=True)
            
            for i, threat in enumerate(sorted_threats):
                threat_type = threat.get('type', 'Unknown')
                severity = threat.get('severity', 1)
                source = threat.get('source', 'Unknown')
                target = threat.get('target', 'Unknown')
                timestamp = threat.get('timestamp', detection_time)
                confidence = threat.get('confidence', 0.0)
                
                # Format severity indicator
                severity_indicator = '!' * severity
                
                report.append(f"THREAT #{i+1}: {threat_type}")
                report.append(f"Severity: {severity}/5 {severity_indicator}")
                report.append(f"Detection Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                report.append(f"Source: {source}")
                report.append(f"Target: {target}")
                report.append(f"Confidence: {confidence:.2f}")
                
                # Add threat details if available
                if 'details' in threat:
                    report.append(f"Details: {threat['details']}")
                
                # Add mitigation recommendation if available
                if 'recommendation' in threat:
                    report.append(f"Recommendation: {threat['recommendation']}")
                
                # Add feature importance if available
                if 'feature_importance' in threat:
                    report.append("Feature Importance:")
                    for feature, importance in threat['feature_importance']:
                        report.append(f"  - {feature}: {importance:.4f}")
                
                report.append("-" * 40)
                report.append("")
            
            # Add recommendations section
            report.append(self.create_subsection_header("SECURITY RECOMMENDATIONS"))
            
            # Generate recommendations based on threat types
            recommendations = []
            
            if any(threat.get('type') == 'Brute Force' for threat in threats):
                recommendations.append("1. Implement account lockout policies")
                recommendations.append("2. Use multi-factor authentication")
                recommendations.append("3. Strengthen password requirements")
            
            if any(threat.get('type') == 'Data Exfiltration' for threat in threats):
                recommendations.append("1. Implement data loss prevention (DLP) solutions")
                recommendations.append("2. Monitor and restrict outbound traffic")
                recommendations.append("3. Encrypt sensitive data at rest and in transit")
            
            if any(threat.get('type') == 'DoS' for threat in threats) or any(threat.get('type') == 'DDoS' for threat in threats):
                recommendations.append("1. Implement rate limiting and traffic filtering")
                recommendations.append("2. Use a content delivery network (CDN)")
                recommendations.append("3. Have a DDoS response plan")
            
            if any(threat.get('type') == 'Port Scan' for threat in threats):
                recommendations.append("1. Configure firewalls to block port scanning")
                recommendations.append("2. Close unnecessary open ports")
                recommendations.append("3. Implement intrusion detection systems")
            
            if any(threat.get('severity', 0) >= 4 for threat in threats):
                recommendations.append("CRITICAL RECOMMENDATION: Immediately investigate high-severity threats")
                recommendations.append("1. Isolate affected systems")
                recommendations.append("2. Preserve forensic evidence")
                recommendations.append("3. Execute incident response procedures")
            
            # Add general recommendations
            recommendations.append("General Security Recommendations:")
            recommendations.append("1. Keep all systems and software up to date with security patches")
            recommendations.append("2. Regularly back up critical data")
            recommendations.append("3. Conduct security awareness training for all users")
            recommendations.append("4. Implement the principle of least privilege")
            
            for recommendation in recommendations:
                report.append(recommendation)
        else:
            report.append("No threats detected during this scan.")
        
        # Add report footer
        report.append(self.create_section_header("END OF REPORT"))
        
        return "\n".join(report)
    
    def generate_anomaly_report(self, anomalies, normal_baselines=None):
        """
        Generate a report on detected anomalies.
        
        Args:
            anomalies (list): List of detected anomalies.
            normal_baselines (dict, optional): Baseline values for normal behavior.
            
        Returns:
            str: Formatted anomaly report as a string.
        """
        report = []
        
        # Add report header
        report.append(self.create_section_header("ANOMALY DETECTION REPORT"))
        
        # Add report metadata
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Anomalies Detected: {len(anomalies)}")
        report.append("")
        
        if not anomalies:
            report.append("No anomalies detected during this scan.")
            report.append(self.create_section_header("END OF REPORT"))
            return "\n".join(report)
        
        # Add anomaly summary
        report.append(self.create_subsection_header("ANOMALY SUMMARY"))
        
        # Collect anomaly types
        anomaly_types = {}
        features_implicated = {}
        
        for anomaly in anomalies:
            anomaly_type = anomaly.get('type', 'Unknown')
            
            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = 0
            
            anomaly_types[anomaly_type] += 1
            
            # Track implicated features
            if 'features' in anomaly:
                for feature in anomaly['features']:
                    if feature not in features_implicated:
                        features_implicated[feature] = 0
                    features_implicated[feature] += 1
        
        # Add anomaly types table
        anomaly_type_rows = [(a_type, count) for a_type, count in anomaly_types.items()]
        anomaly_type_table = self.format_table(
            ["Anomaly Type", "Count"],
            anomaly_type_rows,
            "Anomalies by Type"
        )
        report.append(anomaly_type_table)
        report.append("")
        
        # Add feature frequency table if available
        if features_implicated:
            sorted_features = sorted(features_implicated.items(), key=lambda x: x[1], reverse=True)
            feature_rows = [(feature, count) for feature, count in sorted_features[:10]]  # Top 10 features
            
            feature_table = self.format_table(
                ["Feature", "Frequency"],
                feature_rows,
                "Top Features in Anomalies"
            )
            report.append(feature_table)
            report.append("")
        
        # Add detailed anomaly information
        report.append(self.create_subsection_header("DETAILED ANOMALY INFORMATION"))
        
        # Sort anomalies by score (highest first)
        sorted_anomalies = sorted(anomalies, key=lambda x: x.get('score', 0), reverse=True)
        
        for i, anomaly in enumerate(sorted_anomalies):
            anomaly_type = anomaly.get('type', 'Unknown')
            score = anomaly.get('score', 0)
            timestamp = anomaly.get('timestamp', datetime.now())
            source = anomaly.get('source', 'Unknown')
            
            report.append(f"ANOMALY #{i+1}: {anomaly_type}")
            report.append(f"Anomaly Score: {score:.4f}")
            report.append(f"Detection Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Source: {source}")
            
            # Add feature details if available
            if 'features' in anomaly:
                report.append("Anomalous Features:")
                for feature in anomaly['features']:
                    value = anomaly.get('values', {}).get(feature, 'N/A')
                    baseline = 'N/A'
                    
                    if normal_baselines and feature in normal_baselines:
                        baseline = normal_baselines[feature]
                    
                    report.append(f"  - {feature}: Value {value}, Baseline {baseline}")
            
            # Add explanation if available
            if 'explanation' in anomaly:
                report.append(f"Explanation: {anomaly['explanation']}")
            
            report.append("-" * 40)
            report.append("")
        
        # Add interpretation guidance
        report.append(self.create_subsection_header("INTERPRETATION GUIDANCE"))
        report.append("How to interpret anomaly scores:")
        report.append("- Scores close to 1.0 indicate strong anomalies")
        report.append("- Higher anomaly scores warrant higher priority investigation")
        report.append("- Multiple anomalies with similar features may indicate coordinated activity")
        report.append("")
        
        # Add recommendations
        report.append(self.create_subsection_header("RECOMMENDATIONS"))
        report.append("1. Investigate high-scoring anomalies first")
        report.append("2. Look for patterns across multiple anomalies")
        report.append("3. Monitor systems with recurring anomalies more closely")
        report.append("4. Consider updating baseline behavior profiles regularly")
        report.append("5. Use these anomalies to improve signature-based detection")
        report.append("")
        
        # Add report footer
        report.append(self.create_section_header("END OF REPORT"))
        
        return "\n".join(report)
    
    def generate_analysis_report(self, analysis_results, model_info=None):
        """
        Generate a report on security analysis results.
        
        Args:
            analysis_results (dict): Results from security analysis.
            model_info (dict, optional): Information about the model used.
            
        Returns:
            str: Formatted analysis report as a string.
        """
        report = []
        
        # Add report header
        report.append(self.create_section_header("SECURITY ANALYSIS REPORT"))
        
        # Add report metadata
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if model_info:
            report.append(f"Model Name: {model_info.get('name', 'Unknown')}")
            report.append(f"Model Version: {model_info.get('version', 'Unknown')}")
            report.append(f"Model Type: {model_info.get('type', 'Unknown')}")
        
        report.append("")
        
        # Add analysis summary
        report.append(self.create_subsection_header("ANALYSIS SUMMARY"))
        
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            for key, value in summary.items():
                report.append(f"{key}: {value}")
        
        report.append("")
        
        # Add detailed results for different analysis types
        if 'model_performance' in analysis_results:
            report.append(self.create_subsection_header("MODEL PERFORMANCE"))
            
            perf = analysis_results['model_performance']
            for metric, value in perf.items():
                if isinstance(value, float):
                    report.append(f"{metric}: {value:.4f}")
                else:
                    report.append(f"{metric}: {value}")
            
            report.append("")
        
        if 'feature_importance' in analysis_results:
            report.append(self.create_subsection_header("FEATURE IMPORTANCE"))
            
            features = analysis_results['feature_importance']
            # Sort features by importance
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
            
            feature_rows = [(feature, f"{importance:.4f}") for feature, importance in sorted_features]
            feature_table = self.format_table(
                ["Feature", "Importance"],
                feature_rows,
                "Feature Importance Analysis"
            )
            
            report.append(feature_table)
            report.append("")
        
        if 'threat_patterns' in analysis_results:
            report.append(self.create_subsection_header("THREAT PATTERNS"))
            
            patterns = analysis_results['threat_patterns']
            for i, pattern in enumerate(patterns):
                report.append(f"Pattern #{i+1}: {pattern.get('name', 'Unnamed Pattern')}")
                report.append(f"Confidence: {pattern.get('confidence', 0):.2f}")
                report.append(f"Frequency: {pattern.get('frequency', 0)}")
                
                if 'description' in pattern:
                    report.append(f"Description: {pattern['description']}")
                
                if 'indicators' in pattern:
                    report.append("Indicators:")
                    for indicator in pattern['indicators']:
                        report.append(f"  - {indicator}")
                
                report.append("")
        
        if 'recommendations' in analysis_results:
            report.append(self.create_subsection_header("SECURITY RECOMMENDATIONS"))
            
            recommendations = analysis_results['recommendations']
            for i, rec in enumerate(recommendations):
                report.append(f"{i+1}. {rec}")
            
            report.append("")
        
        # Add report footer
        report.append(self.create_section_header("END OF REPORT"))
        
        return "\n".join(report)


def demo():
    """
    Demonstrate the text visualization capabilities.
    """
    visualizer = TextVisualizer()
    
    # Demonstration of histogram
    visualizer.print_section("Histogram Demonstration")
    data = [23, 45, 12, 67, 34, 56, 78, 90, 23, 45]
    labels = ["Web", "Email", "DNS", "FTP", "SSH", "HTTP", "HTTPS", "SMB", "Telnet", "RDP"]
    visualizer.histogram(data, labels, title="Network Traffic by Protocol")
    
    # Demonstration of timeline
    visualizer.print_section("Timeline Demonstration")
    events = [
        "Port scan detected from 192.168.1.100",
        "Brute force attack attempted on admin account",
        "Unusual data transfer to external IP",
        "Firewall rule updated",
        "New device connected to network",
        "Malware signature detected",
    ]
    timestamps = [
        datetime.now() - timedelta(hours=5),
        datetime.now() - timedelta(hours=4),
        datetime.now() - timedelta(hours=3),
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=1),
        datetime.now(),
    ]
    event_types = [
        "Warning",
        "Alert",
        "Alert",
        "Info",
        "Info",
        "Critical",
    ]
    visualizer.timeline(events, timestamps, event_types, title="Security Event Timeline")
    
    # Demonstration of heatmap
    visualizer.print_section("Heatmap Demonstration")
    data = [
        [0.2, 0.5, 0.8, 0.3, 0.1],
        [0.7, 0.9, 0.2, 0.4, 0.6],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    ]
    row_labels = ["Server 1", "Server 2", "Server 3", "Server 4"]
    col_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    visualizer.heatmap(data, row_labels, col_labels, title="Server Load Heatmap")
    
    # Demonstration of network graph
    visualizer.print_section("Network Graph Demonstration")
    connections = [
        ("192.168.1.1", "192.168.1.2", 5.0),
        ("192.168.1.1", "192.168.1.3", 3.0),
        ("192.168.1.2", "192.168.1.4", 2.0),
        ("192.168.1.3", "192.168.1.5", 4.0),
        ("192.168.1.4", "192.168.1.5", 1.0),
        ("192.168.1.2", "192.168.1.5", 6.0),
    ]
    node_labels = {
        "192.168.1.1": "Router",
        "192.168.1.2": "Server A",
        "192.168.1.3": "Server B",
        "192.168.1.4": "Client 1",
        "192.168.1.5": "Client 2",
    }
    visualizer.network_graph(connections, node_labels, title="Network Connection Graph")
    
    # Demonstration of confusion matrix
    visualizer.print_section("Confusion Matrix Demonstration")
    matrix = [
        [45, 5, 0, 0],
        [4, 38, 3, 0],
        [1, 7, 35, 2],
        [0, 1, 5, 39],
    ]
    class_names = ["Normal", "DoS", "Brute Force", "Data Theft"]
    visualizer.confusion_matrix(matrix, class_names, title="Threat Detection Confusion Matrix")
    
    # Demonstration of progress bar
    visualizer.print_section("Progress Bar Demonstration")
    update_func = visualizer.progress_bar(0, 100, title="Processing", auto_update=True)
    for i in range(0, 101, 10):
        update_func(i, f"Processing step {i}")
        time.sleep(0.1)
    print()  # Add a newline after progress bar
    
    # Demonstration of animation
    visualizer.print_section("Animation Demonstration")
    print("Running analysis...")
    visualizer.animate_processing(2, "Analyzing network traffic")
    print("Analysis complete!")
    
    # Demonstration of threat dashboard
    visualizer.print_section("Threat Dashboard Demonstration")
    threats = [
        {
            'type': 'Brute Force',
            'severity': 4,
            'source': '203.0.113.42',
            'target': '192.168.1.100',
            'timestamp': datetime.now() - timedelta(minutes=5),
            'details': 'Multiple failed login attempts',
            'recommendation': 'Enable account lockout policy'
        },
        {
            'type': 'Data Exfiltration',
            'severity': 5,
            'source': '192.168.1.53',
            'target': '198.51.100.23',
            'timestamp': datetime.now() - timedelta(minutes=15),
            'details': 'Large data transfer to external IP'
        },
        {
            'type': 'Port Scan',
            'severity': 3,
            'source': '203.0.113.17',
            'target': '192.168.1.0/24',
            'timestamp': datetime.now() - timedelta(minutes=30)
        },
        {
            'type': 'DDoS',
            'severity': 4,
            'source': 'Multiple',
            'target': '192.168.1.10',
            'timestamp': datetime.now() - timedelta(minutes=45)
        },
        {
            'type': 'Brute Force',
            'severity': 3,
            'source': '198.51.100.76',
            'target': '192.168.1.100',
            'timestamp': datetime.now() - timedelta(minutes=55)
        },
        {
            'type': 'Command & Control',
            'severity': 5,
            'source': '192.168.1.42',
            'target': '203.0.113.99',
            'timestamp': datetime.now() - timedelta(hours=1),
            'details': 'Communication with known malicious C2 server'
        },
        {
            'type': 'Malware',
            'severity': 4,
            'source': '192.168.1.37',
            'target': 'N/A',
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'type': 'Data Exfiltration',
            'severity': 3,
            'source': '192.168.1.22',
            'target': '198.51.100.17',
            'timestamp': datetime.now() - timedelta(hours=3)
        },
    ]
    visualizer.threat_dashboard(threats, title="Security Operations Center Dashboard")
    
    # Report generator demonstration
    report_gen = SecurityReportGenerator()
    
    visualizer.print_section("Report Generator Demonstration")
    report = report_gen.generate_threat_report(
        threats=threats,
        detection_time=datetime.now() - timedelta(hours=1),
        model_name="CyberThreat-ML Advanced Threat Detection Model"
    )
    print("Generated Threat Report (first 20 lines):")
    print("\n".join(report.split("\n")[:20]))
    print("...(report continues)...")


if __name__ == "__main__":
    demo()