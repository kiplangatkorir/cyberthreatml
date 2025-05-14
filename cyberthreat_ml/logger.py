"""
Logger module for the CyberThreat-ML library.
"""
 
import logging
import os
import sys
from datetime import datetime

class CyberThreatLogger:
    """
    Custom logger for the CyberThreat-ML library.
    """
    
    def __init__(self, name="cyberthreat_ml", log_level=logging.INFO, log_to_file=False):
        """
        Initialize the logger.
        
        Args:
            name (str): Name of the logger.
            log_level (int): Logging level.
            log_to_file (bool): Whether to log to file.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        self.logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            log_file = os.path.join(log_dir, f"cyberthreat_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """
        Get the logger instance.
        
        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger


# Create default logger instance
logger = CyberThreatLogger().get_logger()
