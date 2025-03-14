"""
Module for visualization and dashboard capabilities for cybersecurity threat detection.
"""

import time
import logging
from collections import deque, Counter
import threading
import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
from cyberthreat_ml.logger import CyberThreatLogger

# Configure logger
logger = CyberThreatLogger(name="cyberthreat_ml.visualization").get_logger()

class ThreatVisualizationDashboard:
    """
    Dashboard for real-time visualization of cybersecurity threats.
    """
    
    def __init__(self, max_history=1000, update_interval=1.0):
        """
        Initialize the dashboard.
        
        Args:
            max_history (int): Maximum number of events to keep in history.
            update_interval (float): Interval between dashboard updates in seconds.
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization packages (matplotlib, seaborn) not available. Dashboard disabled.")
            return
            
        self.max_history = max_history
        self.update_interval = update_interval
        
        # Data storage
        self.threat_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.threat_types = deque(maxlen=max_history)
        self.confidence_scores = deque(maxlen=max_history)
        
        # Plotting
        self.fig = None
        self.axes = None
        self.animation = None
        
        # Threading
        self.running = False
        
        # Initialize plot in the main thread
        self._setup_plot()
        
    def _setup_plot(self):
        """
        Set up the plot layout.
        """
        if not VISUALIZATION_AVAILABLE:
            return
            
        # Use plt.ion() for interactive mode
        plt.ion()
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('CyberThreat-ML Real-time Monitoring')
        
        # Define subplot grid
        self.axes = {
            'timeline': self.fig.add_subplot(3, 2, (1, 2)),  # Spans 2 columns
            'pie': self.fig.add_subplot(3, 2, 3),
            'confidence': self.fig.add_subplot(3, 2, 4),
            'heatmap': self.fig.add_subplot(3, 2, (5, 6))  # Spans 2 columns
        }
        
        # Set up titles
        self.axes['timeline'].set_title('Threat Detection Timeline')
        self.axes['pie'].set_title('Threat Type Distribution')
        self.axes['confidence'].set_title('Confidence Score Distribution')
        self.axes['heatmap'].set_title('Threat Intensity Heatmap (Last Hour)')
        
        # Set tight layout
        self.fig.tight_layout()
        
    def add_threat(self, threat_data):
        """
        Add a new threat detection event to the dashboard.
        
        Args:
            threat_data (dict): Detection result from the RealTimeDetector.
        """
        if not VISUALIZATION_AVAILABLE:
            return
            
        # Extract relevant data
        timestamp = threat_data.get('timestamp', time.time())
        is_threat = threat_data.get('is_threat', False)
        
        if not is_threat:
            return
            
        # For multi-class classification
        if not threat_data.get('is_binary', True):
            threat_type = 'Unknown'
            class_idx = threat_data.get('predicted_class', 0)
            class_names = threat_data.get('class_names', None)
            
            if class_names and class_idx < len(class_names):
                threat_type = class_names[class_idx]
            else:
                threat_type = f"Class {class_idx}"
                
            confidence = threat_data.get('confidence', 0.0)
        
        # For binary classification
        else:
            threat_type = 'Binary Threat'
            confidence = threat_data.get('threat_score', 0.0)
        
        # Store data
        self.threat_history.append(threat_data)
        self.timestamps.append(timestamp)
        self.threat_types.append(threat_type)
        self.confidence_scores.append(confidence)
        
    def start(self):
        """
        Start the dashboard animation.
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization not available. Cannot start dashboard.")
            return
            
        if self.running:
            return
            
        self.running = True
        
        # Start animation in the main thread
        self.animation = animation.FuncAnimation(
            self.fig, 
            self._update_plots,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            blit=False,
            save_count=100  # Limit the cache to 100 frames
        )
        
        # Show the plot in interactive mode
        plt.show(block=False)
        logger.info("Threat visualization dashboard started")
        
    def stop(self):
        """
        Stop the dashboard animation.
        """
        if not VISUALIZATION_AVAILABLE or not self.running:
            return
            
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        
        # Close all figures
        plt.close('all')
        logger.info("Threat visualization dashboard stopped")
        
    def _show_dashboard(self):
        """
        Show the dashboard in a separate thread.
        """
        if not VISUALIZATION_AVAILABLE:
            return
            
        try:
            plt.show()
        except Exception as e:
            logger.error(f"Error showing dashboard: {e}")
            
    def _update_plots(self, frame):
        """
        Update all plots with the latest data.
        
        Args:
            frame: Frame parameter required by FuncAnimation.
        """
        if not VISUALIZATION_AVAILABLE or not self.running:
            return
            
        try:
            self._update_timeline()
            self._update_pie_chart()
            self._update_confidence_histogram()
            self._update_heatmap()
            
            # Update layout
            self.fig.tight_layout()
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            
    def _update_timeline(self):
        """
        Update the threat timeline plot.
        """
        if not self.timestamps:
            return
            
        ax = self.axes['timeline']
        ax.clear()
        
        # Convert timestamps to datetime for better visualization
        dates = [datetime.datetime.fromtimestamp(ts) for ts in self.timestamps]
        
        # Create a colormap for different threat types
        unique_threats = list(set(self.threat_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_threats)))
        color_map = dict(zip(unique_threats, colors))
        
        # Plot each threat type with a different color
        for threat_type in unique_threats:
            mask = [t == threat_type for t in self.threat_types]
            if any(mask):
                threat_dates = [d for d, m in zip(dates, mask) if m]
                threat_conf = [c for c, m in zip(self.confidence_scores, mask) if m]
                
                # Scale marker size by confidence score
                sizes = [max(20, 50 * conf) for conf in threat_conf]
                
                ax.scatter(threat_dates, [threat_type] * len(threat_dates), 
                          s=sizes, alpha=0.7, label=threat_type, 
                          color=color_map[threat_type])
        
        # Set labels and grid
        ax.set_title('Threat Detection Timeline')
        ax.set_xlabel('Time')
        ax.set_ylabel('Threat Type')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.xaxis_date()
        
        # Add legend if we have multiple threat types
        if len(unique_threats) > 1:
            ax.legend(loc='upper left')
            
    def _update_pie_chart(self):
        """
        Update the threat type distribution pie chart.
        """
        if not self.threat_types:
            return
            
        ax = self.axes['pie']
        ax.clear()
        
        # Count threat types
        threat_counts = Counter(self.threat_types)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            threat_counts.values(), 
            labels=threat_counts.keys(),
            autopct='%1.1f%%', 
            startangle=90,
            shadow=True
        )
        
        # Make text readable
        for text in texts + autotexts:
            text.set_fontsize(9)
            
        ax.set_title('Threat Type Distribution')
        
    def _update_confidence_histogram(self):
        """
        Update the confidence score histogram.
        """
        if not self.confidence_scores:
            return
            
        ax = self.axes['confidence']
        ax.clear()
        
        # Create histogram
        ax.hist(self.confidence_scores, bins=20, alpha=0.7, color='steelblue')
        
        # Set labels and grid
        ax.set_title('Confidence Score Distribution')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis limits
        ax.set_xlim(0, 1)
        
        # Use integer ticks for y-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
    def _update_heatmap(self):
        """
        Update the threat intensity heatmap.
        """
        if not self.timestamps or not self.threat_types:
            return
            
        ax = self.axes['heatmap']
        ax.clear()
        
        # Get the last hour of data
        now = time.time()
        hour_ago = now - 3600
        
        # Filter data from the last hour
        recent_data = [(ts, tt) for ts, tt in zip(self.timestamps, self.threat_types) 
                      if ts >= hour_ago]
        
        if not recent_data:
            ax.text(0.5, 0.5, 'No threat data in the last hour', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Threat Intensity Heatmap (Last Hour)')
            return
            
        # Split into timestamps and types
        recent_timestamps, recent_types = zip(*recent_data)
        
        # Convert to datetime for better visualization
        recent_dates = [datetime.datetime.fromtimestamp(ts) for ts in recent_timestamps]
        
        # Create time bins (5-minute intervals)
        time_edges = np.linspace(hour_ago, now, 13)  # 12 intervals of 5 minutes
        time_bins = np.digitize(recent_timestamps, time_edges)
        
        # Get unique threat types
        unique_threats = sorted(set(recent_types))
        
        # Create a 2D histogram
        heatmap_data = np.zeros((len(unique_threats), 12))
        
        for i, threat_type in enumerate(unique_threats):
            for j, tb in enumerate(time_bins):
                if recent_types[j] == threat_type and tb < 13:  # Ensure we're within bounds
                    heatmap_data[i, tb-1] += 1
        
        # Plot heatmap
        sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', linewidths=0.5, 
                   cbar_kws={'label': 'Detection Count'})
        
        # Set labels
        ax.set_title('Threat Intensity Heatmap (Last Hour)')
        ax.set_ylabel('Threat Type')
        ax.set_xlabel('Time (5-minute intervals)')
        
        # Set y-tick labels
        ax.set_yticks(np.arange(len(unique_threats)) + 0.5)
        ax.set_yticklabels(unique_threats)
        
        # Format x-ticks as timestamps
        time_labels = [datetime.datetime.fromtimestamp(t).strftime('%H:%M') 
                      for t in time_edges[:-1]]
        ax.set_xticks(np.arange(len(time_labels)) + 0.5)
        ax.set_xticklabels(time_labels, rotation=45)
        
    def save_snapshot(self, filename=None):
        """
        Save a snapshot of the current dashboard state.
        
        Args:
            filename (str, optional): Filename to save to. If None, uses a timestamp.
        """
        if not VISUALIZATION_AVAILABLE or not self.running:
            logger.warning("Visualization not available. Cannot save snapshot.")
            return
            
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threat_dashboard_{timestamp}.png"
            
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard snapshot saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving dashboard snapshot: {e}")
            return None
            
# Singleton instance that can be used across the application
dashboard = None

def get_dashboard():
    """
    Get the global dashboard instance.
    
    Returns:
        ThreatVisualizationDashboard: The dashboard instance.
    """
    global dashboard
    if dashboard is None:
        dashboard = ThreatVisualizationDashboard()
    return dashboard